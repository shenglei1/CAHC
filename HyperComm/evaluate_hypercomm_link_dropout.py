"""Frozen HyperComm evaluation under deterministic execution-link dropout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fair_costs import FairCosts
from formal_hypercomm import (
    build_args,
    evaluate_episode,
    file_sha256,
    initialize,
    write_csv,
)
from operational_metrics import write_comparison_csv
from scenario_registry import load_registry, registry_sha256


PROJECT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one existing HyperComm checkpoint after dropping decoded "
            "execution links; no training or parameter update is run."
        )
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--link-dropout-rate", type=float, required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--eval-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--results-root", type=Path, required=True)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    if path.is_absolute():
        raise ValueError(f"Experiment paths must be relative: {path}")
    resolved = (PROJECT / path).resolve()
    try:
        resolved.relative_to(PROJECT)
    except ValueError as error:
        raise ValueError(f"Experiment path leaves the HyperComm directory: {path}") from error
    return resolved


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    cli = parse_args()
    if not 0.0 <= cli.link_dropout_rate <= 1.0:
        raise ValueError("--link-dropout-rate must be in [0, 1].")
    if cli.eval_episodes < 1 or len(cli.eval_seeds) != len(set(cli.eval_seeds)):
        raise ValueError("Evaluation seeds must be unique and episode count positive.")
    if cli.gpu >= 0 and not torch.cuda.is_available():
        raise RuntimeError("A GPU was requested, but PyTorch cannot access CUDA.")

    registry = load_registry(PROJECT / "scenario_registry.json")
    if cli.scenario not in registry["formal_scenario_ids"]:
        raise ValueError(f"Not a registered formal scenario: {cli.scenario}")
    if cli.seed not in registry["formal_seeds"]:
        raise ValueError(f"Not a registered formal training seed: {cli.seed}")

    source = project_path(cli.source_run_dir)
    results_root = project_path(cli.results_root)
    required = {
        "model.pt", "config.json", "training_metrics.csv",
        "training_protocol.json", "evaluation_protocol.json", "test_metrics.csv",
    }
    missing = sorted(name for name in required if not (source / name).is_file())
    if missing:
        raise FileNotFoundError(f"Source run is missing {missing}: {source}")

    source_config = read_json(source / "config.json")
    source_training = read_json(source / "training_protocol.json")
    source_evaluation = read_json(source / "evaluation_protocol.json")
    checkpoint_hash = file_sha256(source / "model.pt")
    checks = {
        "scenario": source_config.get("env_name") == f"lbf_{cli.scenario}-v0",
        "seed": int(source_config.get("seed", -1)) == cli.seed,
        "method": source_training.get("method") == "hypercomm",
        "trained": bool(source_training.get("optimizer_step_verified")),
        "finite": bool(source_training.get("parameters_finite")),
        "source_frozen": bool(source_evaluation.get("parameters_unchanged")),
        "checkpoint_loaded": bool(source_evaluation.get("checkpoint_load_verified")),
        "checkpoint_identity": source_evaluation.get("checkpoint_sha256") == checkpoint_hash,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Source checkpoint audit failed {failed}: {source}")

    source_cli = argparse.Namespace(
        scenario=cli.scenario,
        seed=cli.seed,
        gpu=cli.gpu,
        epochs=int(source_training["epochs"]),
        episodes_per_epoch=int(source_training["episodes_per_epoch"]),
        learning_rate=float(source_config["lrate"]),
    )
    torch.set_default_dtype(torch.float64)
    args = build_args(source_cli, registry)
    env, policy, _trainer, diagnostics, handles = initialize(args)
    saved = torch.load(source / "model.pt", map_location=policy.device, weights_only=False)
    policy.load_state_dict(saved["policy_net"], strict=True)
    policy.eval()
    before = [parameter.detach().cpu().clone() for parameter in policy.parameters()]

    output = (
        results_root / cli.method_label / args.env_name /
        f"seed_{cli.seed}" / "run1"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {output}")
    output.mkdir(parents=True)

    rows = []
    costs = FairCosts(policy, cli.method_label)
    try:
        for eval_seed in cli.eval_seeds:
            for episode in range(1, cli.eval_episodes + 1):
                row, _trace = evaluate_episode(
                    policy, env, args, diagnostics, eval_seed, episode,
                    cli.link_dropout_rate,
                )
                row["Method"] = cli.method_label
                costs.episode(row, row["Episode_Steps"])
                rows.append(row)
        write_csv(output / "test_metrics.csv", rows)
        costs.finish(output)
    finally:
        for handle in handles:
            handle.remove()
    write_comparison_csv(output, rows)

    after = [parameter.detach().cpu() for parameter in policy.parameters()]
    unchanged = all(torch.equal(a, b) for a, b in zip(before, after))
    if not unchanged:
        raise RuntimeError("Frozen link-dropout evaluation modified policy parameters.")

    source_relative = str(source.relative_to(PROJECT))
    config = vars(args).copy()
    config.update({
        "formal_protocol": registry["protocol_version"],
        "scenario_registry_sha256": registry_sha256(registry),
        "link_dropout_rate": cli.link_dropout_rate,
        "source_run_dir": source_relative,
        "source_checkpoint_sha256": checkpoint_hash,
    })
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "training_protocol.json").write_text(json.dumps({
        "method": cli.method_label,
        "seed": cli.seed,
        "training_reused": True,
        "new_optimizer_updates": 0,
        "source_method": "hypercomm",
        "source_run_dir": source_relative,
        "source_checkpoint_sha256": checkpoint_hash,
        "protocol_version": registry["protocol_version"],
    }, indent=2), encoding="utf-8")
    (output / "evaluation_protocol.json").write_text(json.dumps({
        "evaluation_role": "zero_shot_execution_link_dropout",
        "optimizer_updates": 0,
        "eval_seeds": cli.eval_seeds,
        "episodes_per_seed": cli.eval_episodes,
        "link_dropout_rate": cli.link_dropout_rate,
        "dropout_mask_rule": (
            "deterministic nested mask from evaluation seed, episode, step, "
            "and the shared experiment namespace"
        ),
        "dropout_location": (
            "after native topology decoding and support masking; before GAT aggregation"
        ),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_load_verified": True,
        "parameters_unchanged": unchanged,
        "protocol_version": registry["protocol_version"],
    }, indent=2), encoding="utf-8")
    env.close()
    print(output.relative_to(PROJECT))


if __name__ == "__main__":
    main()
