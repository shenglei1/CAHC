"""Evaluate one audited HyperWave-X checkpoint without retraining it.

This entry point is intentionally separate from ``formal_hyperwavex.py`` so a
Reviewer sensitivity reference can reuse a preregistered main-run checkpoint
while writing a new, immutable validation-evaluation artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from fair_costs import FairCosts
from formal_hyperwavex import (
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
PLAN_PATH = PROJECT / "hyperwavex_experiment_plan.json"
REGISTRY_PATH = PROJECT / "scenario_registry.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen validation of one audited HyperWave-X checkpoint."
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-training-protocol-sha256", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--expected-epochs", type=int, default=700)
    parser.add_argument("--expected-episodes-per-epoch", type=int, default=64)
    parser.add_argument("--eval-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.0)
    parser.add_argument("--results-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_source(
    cli: argparse.Namespace,
    plan: dict,
    registry: dict,
) -> tuple[Path, dict, dict, dict, str]:
    source = cli.source_run_dir.resolve()
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
        raise FileNotFoundError(f"Source checkpoint run is missing {missing}: {source}")

    config = read_json(source / "config.json")
    training_protocol = read_json(source / "training_protocol.json")
    evaluation_protocol = read_json(source / "evaluation_protocol.json")
    checkpoint = source / "model.pt"
    checkpoint_hash = file_sha256(checkpoint)
    training_protocol_hash = file_sha256(source / "training_protocol.json")
    defaults = plan["default_hyperparameters"]
    expected_flags = {
        "capability_aware": True,
        "use_wave": True,
        "complete_topology": False,
    }
    expected_parameters = {
        "capability_weight": defaults["capability_weight"],
        "hypergraph_temperature": defaults["hypergraph_temperature"],
        # Zero is the formal sentinel for the task-conditioned adaptive
        # cardinality rule; it is deliberately not a tuned scalar in the plan.
        "hyperedge_size": 0,
        "wave_iterations": defaults["wave_iterations"],
        "wave_tau": defaults["wave_tau"],
        "wave_damping": defaults["wave_damping"],
        "topology_temperature": defaults["topology_temperature"],
        "hid_size": defaults["hidden_size"],
        "qk_hid_size": defaults["query_key_size"],
        "gat_num_heads": defaults["gat_heads"],
        "gat_hid_size": defaults["gat_head_width"],
        "lrate": defaults["learning_rate"],
        "value_coeff": defaults["value_loss_coefficient"],
        "entr": defaults["entropy_coefficient"],
    }
    checks = {
        "scenario": config.get("env_name") == f"lbf_{cli.scenario}-v0",
        "seed": int(config.get("seed", -1)) == cli.seed,
        "variant": config.get("variant") == "full",
        "method": config.get("method_label") == "hyperwave_x",
        "training_method": training_protocol.get("method") == "hyperwave_x",
        "epochs": int(training_protocol.get("epochs", -1)) == cli.expected_epochs,
        "episodes_per_epoch": int(
            training_protocol.get("episodes_per_epoch", -1)
        ) == cli.expected_episodes_per_epoch,
        "backward_verified": bool(training_protocol.get("backward_verified")),
        "optimizer_step_verified": bool(
            training_protocol.get("optimizer_step_verified")
        ),
        "parameters_finite": bool(training_protocol.get("parameters_finite")),
        "registry_hash_recorded": bool(
            training_protocol.get("scenario_registry_sha256")
        ),
        "checkpoint_hash_argument": checkpoint_hash
        == cli.expected_checkpoint_sha256,
        "training_protocol_hash_argument": training_protocol_hash
        == cli.expected_training_protocol_sha256,
        "checkpoint_hash_record": evaluation_protocol.get("checkpoint_sha256")
        == checkpoint_hash,
        "checkpoint_was_loaded": bool(
            evaluation_protocol.get("checkpoint_load_verified")
        ),
        "source_eval_was_frozen": bool(
            evaluation_protocol.get("parameters_unchanged")
        ),
        **{
            f"flag_{name}": config.get(name) is value
            for name, value in expected_flags.items()
        },
        **{
            f"parameter_{name}": (
                isinstance(config.get(name), (int, float))
                and math.isclose(
                    float(config[name]), float(value), rel_tol=0.0, abs_tol=1e-12
                )
            )
            for name, value in expected_parameters.items()
        },
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Source checkpoint audit failed at {source}: {failed}")

    training_rows = csv_rows(source / "training_metrics.csv")
    epoch_ledger = [int(float(row.get("Epoch", -1))) for row in training_rows]
    episode_ledger = [
        int(float(row.get("Episodes", -1))) for row in training_rows
    ]
    if (
        epoch_ledger != list(range(1, cli.expected_epochs + 1))
        or any(
            episodes != cli.expected_episodes_per_epoch
            for episodes in episode_ledger
        )
    ):
        raise RuntimeError(
            "Source training ledger does not contain the complete ordered "
            f"{cli.expected_epochs} x {cli.expected_episodes_per_epoch} budget: "
            f"{source}"
        )
    source_test_rows = csv_rows(source / "test_metrics.csv")
    csv_hashes = {
        row.get("Checkpoint_SHA256") for row in source_test_rows
        if row.get("Checkpoint_SHA256")
    }
    if not source_test_rows or (csv_hashes and csv_hashes != {checkpoint_hash}):
        raise RuntimeError(f"Source test ledger/checkpoint mismatch: {source}")
    return source, config, training_protocol, evaluation_protocol, checkpoint_hash


def main() -> None:
    cli = parse_args()
    if cli.expected_epochs < 1 or cli.expected_episodes_per_epoch < 1:
        raise ValueError("Expected training budget must be positive.")
    if cli.eval_episodes < 1 or len(cli.eval_seeds) != len(set(cli.eval_seeds)):
        raise ValueError("Evaluation seeds must be unique and episode count positive.")
    if cli.gpu >= 0 and not torch.cuda.is_available():
        raise RuntimeError("A GPU was requested, but PyTorch cannot access CUDA.")
    if cli.gpu >= 0 and cli.gpu_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(cli.gpu_memory_fraction, cli.gpu)

    plan = read_json(PLAN_PATH)
    registry = load_registry(REGISTRY_PATH)
    if cli.scenario not in registry["formal_scenario_ids"]:
        raise ValueError(f"Not a registered formal scenario: {cli.scenario}")
    if cli.seed not in registry["formal_seeds"]:
        raise ValueError(f"Not a registered training seed: {cli.seed}")
    if set(cli.eval_seeds) & set(registry["formal_seeds"]):
        raise ValueError("Training and evaluation seeds must be disjoint.")

    source, source_config, source_training, source_evaluation, checkpoint_hash = (
        validate_source(cli, plan, registry)
    )
    defaults = plan["default_hyperparameters"]
    source_cli = argparse.Namespace(
        scenario=cli.scenario,
        seed=cli.seed,
        gpu=cli.gpu,
        episodes_per_epoch=cli.expected_episodes_per_epoch,
        learning_rate=float(source_config.get("lrate", 0.001)),
        variant="full",
        run_label=cli.run_label,
        capability_weight=defaults["capability_weight"],
        hypergraph_temperature=defaults["hypergraph_temperature"],
        hyperedge_size=0,
        wave_iterations=defaults["wave_iterations"],
        wave_tau=defaults["wave_tau"],
        wave_damping=defaults["wave_damping"],
        topology_temperature=defaults["topology_temperature"],
    )

    torch.set_default_dtype(torch.float64)
    model_args = build_args(source_cli, registry)
    env, policy, _trainer = initialize(model_args)
    saved = torch.load(
        source / "model.pt", map_location=policy.device, weights_only=False
    )
    policy.load_state_dict(saved["policy_net"])
    if file_sha256(source / "model.pt") != checkpoint_hash:
        raise RuntimeError("Source checkpoint changed between audit and load.")
    policy.eval()
    before_eval = [parameter.detach().cpu().clone() for parameter in policy.parameters()]

    seed_root = (
        cli.results_root.resolve()
        / cli.run_label
        / model_args.env_name
        / f"seed_{cli.seed}"
    )
    existing = sorted(path for path in seed_root.glob("run*") if path.is_dir())
    if existing:
        raise FileExistsError(
            f"Refusing to create a duplicate reused-reference evaluation: {existing}"
        )
    output = seed_root / "run1"
    output.mkdir(parents=True)

    evaluation_rows = []
    costs = FairCosts(policy, cli.run_label)
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
                False,
            )
            costs.episode(row, row["Episode_Steps"])
            evaluation_rows.append(row)

    validate_workload_rows(
        evaluation_rows, registry["scenarios"][cli.scenario], frozen_test=True
    )
    write_csv(output / "test_metrics.csv", evaluation_rows)
    costs.finish(output)
    write_comparison_csv(output, evaluation_rows)
    after_eval = [parameter.detach().cpu() for parameter in policy.parameters()]
    parameters_unchanged = all(
        torch.equal(before, after)
        for before, after in zip(before_eval, after_eval)
    )
    if not parameters_unchanged:
        raise RuntimeError("Frozen checkpoint evaluation modified policy parameters.")

    runtime = runtime_metadata()
    output_config = vars(model_args).copy()
    output_config.update({
        "formal_protocol": registry["protocol_version"],
        "scenario_registry_sha256": registry_sha256(registry),
        "runtime": runtime,
        "checkpoint_reuse": True,
        "source_run_dir": str(source),
        "source_checkpoint_sha256": checkpoint_hash,
    })
    (output / "config.json").write_text(
        json.dumps(output_config, indent=2), encoding="utf-8"
    )
    (output / "training_protocol.json").write_text(
        json.dumps(
            {
                "method": cli.run_label,
                "training_reused": True,
                "new_optimizer_updates": 0,
                "source_method": "hyperwave_x",
                "source_run_dir": str(source),
                "source_checkpoint_sha256": checkpoint_hash,
                "source_training_protocol_sha256": file_sha256(
                    source / "training_protocol.json"
                ),
                "source_training_protocol": source_training,
                "source_evaluation_protocol": source_evaluation,
                "epochs": cli.expected_epochs,
                "episodes_per_epoch": cli.expected_episodes_per_epoch,
                "seed": cli.seed,
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
                "evaluation_role": "sensitivity_validation_reused_default",
                "optimizer_updates": 0,
                "action_mode": "native_sample",
                "eval_seeds": cli.eval_seeds,
                "episodes_per_seed": cli.eval_episodes,
                "evaluation_seed_rule": (
                    "seed once per validation block; episodes advance environment "
                    "and policy RNG state without repeated reseeding"
                ),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_load_verified": True,
                "parameters_unchanged": parameters_unchanged,
                "source_run_dir": str(source),
                "protocol_version": registry["protocol_version"],
                "scenario_registry_sha256": registry_sha256(registry),
                "runtime": runtime,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "source_checkpoint_provenance.json").write_text(
        json.dumps(
            {
                "source_run_dir": str(source),
                "source_checkpoint": str(source / "model.pt"),
                "source_checkpoint_sha256": checkpoint_hash,
                "source_config_sha256": file_sha256(source / "config.json"),
                "source_training_protocol_sha256": file_sha256(
                    source / "training_protocol.json"
                ),
                "source_test_metrics_sha256": file_sha256(
                    source / "test_metrics.csv"
                ),
                "reuse_policy": (
                    "pre-existing preregistered default checkpoint; no retraining; "
                    "fresh frozen evaluation on sensitivity validation seeds"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env.close()
    print(output)


if __name__ == "__main__":
    main()
