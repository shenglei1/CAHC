"""Zero-shot cross-configuration evaluation of one HyperComm checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from action_utils import parse_action_args
from cross_configuration import (
    apply_to_environment,
    load_design,
    target_spec,
    validate_evaluation_rows,
)
from env_wrappers import LBFWrapper
from fair_costs import FairCosts
from formal_hypercomm import (
    attach_external_diagnostic_hooks,
    evaluate_episode,
    write_csv,
)
from HyperComm import HyperComm
from operational_metrics import write_comparison_csv
from scenario_registry import load_registry, registry_sha256
from shared_lbf_env import make_lbf_env


PROJECT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one trained HyperComm policy on a registered unseen configuration."
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--target-configuration", required=True)
    parser.add_argument("--eval-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.0)
    parser.add_argument("--results-root", type=Path, required=True)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    cli = parse_args()
    if cli.eval_episodes < 1 or len(cli.eval_seeds) != len(set(cli.eval_seeds)):
        raise ValueError("Evaluation seeds must be unique and episode count positive.")
    if cli.gpu >= 0 and not torch.cuda.is_available():
        raise RuntimeError("A GPU was requested, but PyTorch cannot access CUDA.")
    if cli.gpu >= 0 and cli.gpu_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(cli.gpu_memory_fraction, cli.gpu)

    scenario_registry = load_registry(PROJECT / "scenario_registry.json")
    design = load_design(scenario_registry)
    if cli.eval_seeds != design["evaluation_seeds"]:
        raise ValueError(
            "Formal cross-configuration evaluation seeds must exactly match the design."
        )
    if cli.eval_episodes != int(design["episodes_per_evaluation_seed"]):
        raise ValueError("Formal cross-configuration episode budget mismatch.")
    source_scenario = design["source_scenario"]
    target = target_spec(design, scenario_registry, cli.target_configuration)

    source = cli.source_run_dir
    required = {
        "model.pt", "config.json", "training_protocol.json", "evaluation_protocol.json"
    }
    missing = sorted(name for name in required if not (source / name).is_file())
    if missing:
        raise FileNotFoundError(f"Source run is missing {missing}: {source}")
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    train_seed = int(source_config.get("seed", -1))
    if train_seed not in design["training_seeds"]:
        raise ValueError(f"Source checkpoint has an unregistered training seed: {train_seed}")
    checks = {
        "scenario": source_config.get("env_name") == f"lbf_{source_scenario}-v0",
        "agent_count": int(source_config.get("nagents", -1))
        == int(target["agents"]["count"]),
        "horizon": int(source_config.get("max_steps", -1)) == int(target["horizon"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Source checkpoint/target compatibility failed: {failed}")

    torch.set_default_dtype(torch.float64)
    model_args = argparse.Namespace(**source_config)
    model_args.gpu = cli.gpu
    # Batch size affects only a discarded constructor-time hidden-state tensor.
    # The learned parameter shapes are independent of it.
    model_args.batch_size = 1
    raw_environment = make_lbf_env(source_scenario)
    apply_to_environment(raw_environment, target)
    env = LBFWrapper(raw_environment, model_args)
    if env.observation_dim != int(source_config["obs_size"]):
        raise RuntimeError(
            f"Target observation width {env.observation_dim} differs from source "
            f"width {source_config['obs_size']}."
        )
    model_args.obs_size = env.observation_dim
    model_args.num_actions = [env.num_actions]
    model_args.dim_actions = env.dim_actions
    model_args.nagents = env.n_agents
    model_args.nfriendly = env.n_agents
    parse_action_args(model_args)
    policy = HyperComm(model_args).double()
    diagnostics, handles = attach_external_diagnostic_hooks(policy)
    saved = torch.load(source / "model.pt", map_location=policy.device, weights_only=False)
    policy.load_state_dict(saved["policy_net"], strict=True)
    policy.eval()
    before = [parameter.detach().cpu().clone() for parameter in policy.parameters()]

    output = (
        cli.results_root
        / "hypercomm"
        / f"{source_scenario}_to_{cli.target_configuration}"
        / f"seed_{train_seed}"
        / "run1"
    )
    if output.exists():
        raise FileExistsError(f"Cross-configuration result already exists: {output}")
    output.mkdir(parents=True)

    rows: list[dict] = []
    costs = FairCosts(policy, "hypercomm")
    try:
        for eval_seed in cli.eval_seeds:
            seed_all(eval_seed)
            env.seed(eval_seed)
            for episode in range(1, cli.eval_episodes + 1):
                row, _trace = evaluate_episode(
                    policy, env, model_args, diagnostics, eval_seed, episode
                )
                row["Source_Configuration"] = source_scenario
                row["Target_Configuration"] = cli.target_configuration
                costs.episode(row, int(row["Episode_Steps"]))
                rows.append(row)
        validate_evaluation_rows(rows, target)
        write_csv(output / "test_metrics.csv", rows)
        costs.finish(output)
        write_comparison_csv(output, rows)
    finally:
        for handle in handles:
            handle.remove()

    after = [parameter.detach().cpu() for parameter in policy.parameters()]
    if any(not torch.equal(a, b) for a, b in zip(before, after)):
        raise RuntimeError("Zero-shot evaluation modified HyperComm parameters.")
    checkpoint = source / "model.pt"
    provenance = {
        "experiment_id": design["experiment_id"],
        "method": "hypercomm",
        "source_scenario": source_scenario,
        "target_configuration": cli.target_configuration,
        "training_seed": train_seed,
        "optimizer_updates": 0,
        "parameters_unchanged": True,
        "checkpoint_load_strict": True,
        "source_run_dir": relative_or_absolute(source, PROJECT),
        "source_checkpoint_sha256": sha256(checkpoint),
        "source_scenario_registry_sha256": registry_sha256(scenario_registry),
        "cross_configuration_design_sha256": design["semantic_sha256"],
        "target_configuration_sha256": target["cross_configuration_sha256"],
        "evaluation_seeds": cli.eval_seeds,
        "episodes_per_seed": cli.eval_episodes,
    }
    (output / "evaluation_protocol.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    (output / "target_configuration.json").write_text(
        json.dumps(target, indent=2), encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
