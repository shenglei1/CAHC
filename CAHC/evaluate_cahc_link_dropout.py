"""Frozen CAHC evaluation under deterministic execution-link dropout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from fair_costs import FairCosts
from formal_cahc import (
    build_args,
    evaluate_episode,
    file_sha256,
    initialize,
    seed_all,
    validate_workload_rows,
    write_csv,
)
from operational_metrics import write_comparison_csv
from runtime_support import runtime_metadata
from scenario_registry import load_registry, registry_sha256


PROJECT = Path(__file__).resolve().parent
REGISTRY = PROJECT / "scenario_registry.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one existing full CAHC checkpoint after dropping "
            "decoded execution links; no training or parameter update is run."
        )
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--link-dropout-rate", type=float, required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument(
        "--evaluation-role",
        choices=("zero_shot_execution_link_dropout", "refinement_reference"),
        default="zero_shot_execution_link_dropout",
    )
    parser.add_argument("--eval-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.0)
    parser.add_argument("--results-root", type=Path, required=True)
    return parser.parse_args()


def project_relative(path: Path) -> Path:
    if path.is_absolute():
        raise ValueError(
            f"Experiment paths must be relative to the CAHC directory: {path}"
        )
    candidate = (PROJECT / path).resolve()
    try:
        candidate.relative_to(PROJECT)
    except ValueError as error:
        raise ValueError(
            f"Experiment paths must remain inside the CAHC directory: {path}"
        ) from error
    return candidate


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    cli = parse_args()
    if not 0.0 <= cli.link_dropout_rate <= 1.0:
        raise ValueError("--link-dropout-rate must be in [0, 1].")
    if cli.evaluation_role == "refinement_reference" and cli.link_dropout_rate != 0.0:
        raise ValueError("The refinement reference must use zero link dropout.")
    if cli.eval_episodes < 1 or len(cli.eval_seeds) != len(set(cli.eval_seeds)):
        raise ValueError("Evaluation seeds must be unique and episode count positive.")
    if cli.gpu >= 0 and not torch.cuda.is_available():
        raise RuntimeError("A GPU was requested, but PyTorch cannot access CUDA.")
    if cli.gpu >= 0 and cli.gpu_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(cli.gpu_memory_fraction, cli.gpu)

    registry = load_registry(REGISTRY)
    if cli.scenario not in registry["formal_scenario_ids"]:
        raise ValueError(f"Not a registered formal scenario: {cli.scenario}")
    if cli.seed not in registry["formal_seeds"]:
        raise ValueError(f"Not a registered formal training seed: {cli.seed}")
    if set(cli.eval_seeds) & set(registry["formal_seeds"]):
        raise ValueError("Training and evaluation seeds must be disjoint.")

    source = project_relative(cli.source_run_dir)
    results_root = project_relative(cli.results_root)
    required = {
        "model.pt",
        "config.json",
        "training_metrics.csv",
        "training_protocol.json",
        "evaluation_protocol.json",
        "test_metrics.csv",
    }
    missing = sorted(name for name in required if not (source / name).is_file())
    if missing:
        raise FileNotFoundError(f"Source run is missing {missing}: {source}")
    source_config = read_json(source / "config.json")
    source_training = read_json(source / "training_protocol.json")
    source_evaluation = read_json(source / "evaluation_protocol.json")
    checkpoint_hash = file_sha256(source / "model.pt")
    source_checks = {
        "scenario": source_config.get("env_name") == f"lbf_{cli.scenario}-v0",
        "seed": int(source_config.get("seed", -1)) == cli.seed,
        "variant": source_config.get("variant") == "full",
        "method": source_config.get("method_label") == "cahc",
        "training_method": source_training.get("method") == "cahc",
        "full_capability": source_config.get("capability_aware") is True,
        "full_refinement": source_config.get("use_wave") is True,
        "learned_topology": source_config.get("complete_topology") is False,
        "trained": bool(source_training.get("optimizer_step_verified")),
        "finite": bool(source_training.get("parameters_finite")),
        "source_evaluation_frozen": bool(
            source_evaluation.get("parameters_unchanged")
        ),
        "source_checkpoint_loaded": bool(
            source_evaluation.get("checkpoint_load_verified")
        ),
        "checkpoint_identity": source_evaluation.get("checkpoint_sha256")
        == checkpoint_hash,
    }
    failed = [name for name, passed in source_checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Source checkpoint audit failed {failed}: {source}")

    source_cli = argparse.Namespace(
        scenario=cli.scenario,
        seed=cli.seed,
        gpu=cli.gpu,
        episodes_per_epoch=int(source_training["episodes_per_epoch"]),
        learning_rate=float(source_config["lrate"]),
        variant="full",
        run_label=cli.method_label,
        capability_weight=float(source_config["capability_weight"]),
        hypergraph_temperature=float(source_config["hypergraph_temperature"]),
        hyperedge_size=int(source_config["hyperedge_size"]),
        wave_iterations=int(source_config["wave_iterations"]),
        wave_tau=float(source_config["wave_tau"]),
        wave_damping=float(source_config["wave_damping"]),
        topology_temperature=float(source_config["topology_temperature"]),
    )
    torch.set_default_dtype(torch.float64)
    model_args = build_args(source_cli, registry)
    env, policy, _trainer = initialize(model_args)
    saved = torch.load(source / "model.pt", map_location=policy.device, weights_only=False)
    policy.load_state_dict(saved["policy_net"], strict=True)
    if file_sha256(source / "model.pt") != checkpoint_hash:
        raise RuntimeError("Source checkpoint changed between audit and load.")
    policy.eval()
    parameters_before = [
        parameter.detach().cpu().clone() for parameter in policy.parameters()
    ]

    output = (
        results_root
        / cli.method_label
        / model_args.env_name
        / f"seed_{cli.seed}"
        / "run1"
    )
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing dropout evaluation: {output}"
        )
    output.mkdir(parents=True)

    rows = []
    costs = FairCosts(policy, cli.method_label)
    for eval_seed in cli.eval_seeds:
        seed_all(eval_seed)
        env.seed(eval_seed)
        for episode in range(1, cli.eval_episodes + 1):
            row, _trace = evaluate_episode(
                policy,
                env,
                model_args,
                eval_seed,
                episode,
                False,
                cli.link_dropout_rate,
            )
            costs.episode(row, row["Episode_Steps"])
            rows.append(row)
    validate_workload_rows(
        rows, registry["scenarios"][cli.scenario], frozen_test=True
    )
    write_csv(output / "test_metrics.csv", rows)
    costs.finish(output)
    write_comparison_csv(output, rows)
    parameters_after = [parameter.detach().cpu() for parameter in policy.parameters()]
    parameters_unchanged = all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, parameters_after)
    )
    if not parameters_unchanged:
        raise RuntimeError("Frozen link-dropout evaluation modified policy parameters.")

    source_relative = str(source.relative_to(PROJECT))
    output_config = vars(model_args).copy()
    output_config.update(
        {
            "formal_protocol": registry["protocol_version"],
            "scenario_registry_sha256": registry_sha256(registry),
            "link_dropout_rate": cli.link_dropout_rate,
            "source_run_dir": source_relative,
            "source_checkpoint_sha256": checkpoint_hash,
            "runtime": runtime_metadata(),
        }
    )
    (output / "config.json").write_text(
        json.dumps(output_config, indent=2), encoding="utf-8"
    )
    (output / "training_protocol.json").write_text(
        json.dumps(
            {
                "method": cli.method_label,
                "seed": cli.seed,
                "training_reused": True,
                "new_optimizer_updates": 0,
                "source_method": "cahc",
                "source_run_dir": source_relative,
                "source_checkpoint_sha256": checkpoint_hash,
                "source_training_protocol_sha256": file_sha256(
                    source / "training_protocol.json"
                ),
                "epochs": int(source_training["epochs"]),
                "episodes_per_epoch": int(source_training["episodes_per_epoch"]),
                "protocol_version": registry["protocol_version"],
                "scenario_registry_sha256": registry_sha256(registry),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "evaluation_protocol.json").write_text(
        json.dumps(
            {
                "evaluation_role": cli.evaluation_role,
                "optimizer_updates": 0,
                "action_mode": "native_sample",
                "eval_seeds": cli.eval_seeds,
                "episodes_per_seed": cli.eval_episodes,
                "link_dropout_rate": cli.link_dropout_rate,
                "dropout_mask_rule": (
                    "deterministic nested masks from evaluation seed, episode "
                    "index, step index, and a fixed experiment namespace; the "
                    "local mask RNG does not alter policy or environment RNG state"
                ),
                "dropout_location": (
                    "after directed topology decoding and physical-support masking; "
                    "before GAT message aggregation"
                ),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_load_verified": True,
                "parameters_unchanged": parameters_unchanged,
                "source_run_dir": source_relative,
                "protocol_version": registry["protocol_version"],
                "scenario_registry_sha256": registry_sha256(registry),
                "runtime": runtime_metadata(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "source_checkpoint_provenance.json").write_text(
        json.dumps(
            {
                "source_run_dir": source_relative,
                "source_checkpoint_sha256": checkpoint_hash,
                "source_config_sha256": file_sha256(source / "config.json"),
                "source_training_protocol_sha256": file_sha256(
                    source / "training_protocol.json"
                ),
                "source_evaluation_protocol_sha256": file_sha256(
                    source / "evaluation_protocol.json"
                ),
                "reuse_policy": (
                    "existing full CAHC checkpoint; no retraining; fresh "
                    "frozen evaluation under the declared evaluation role"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env.close()
    print(output.relative_to(PROJECT))


if __name__ == "__main__":
    main()
