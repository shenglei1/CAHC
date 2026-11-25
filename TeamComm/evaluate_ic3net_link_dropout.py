"""Frozen IC3Net evaluation under deterministic execution-link dropout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from baselines import REGISTRY as agent_REGISTRY
from envs import REGISTRY as env_REGISTRY
from fair_costs import FairCosts
from operational_metrics import write_comparison_csv
from runner import REGISTRY as runner_REGISTRY
from scenario_registry import load_registry


PROJECT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one existing IC3Net checkpoint after dropping native "
            "communication links; no training or parameter update is run."
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
        raise ValueError(f"Experiment path leaves the TeamComm directory: {path}") from error
    return resolved


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cli = parse_args()
    if cli.gpu != -1:
        raise ValueError("The maintained IC3Net baseline evaluator currently uses CPU; set --gpu -1.")
    if not 0.0 <= cli.link_dropout_rate <= 1.0:
        raise ValueError("--link-dropout-rate must be in [0, 1].")
    if cli.eval_episodes < 1 or len(cli.eval_seeds) != len(set(cli.eval_seeds)):
        raise ValueError("Evaluation seeds must be unique and episode count positive.")
    registry = load_registry(PROJECT / "scenario_registry.json")
    if cli.scenario not in registry["formal_scenario_ids"]:
        raise ValueError(f"Not a registered formal scenario: {cli.scenario}")
    if cli.seed not in registry["formal_seeds"]:
        raise ValueError(f"Not a registered formal training seed: {cli.seed}")

    source = project_path(cli.source_run_dir)
    results_root = project_path(cli.results_root)
    required = {
        "model.pt", "config.json", "training_protocol.json",
        "evaluation_protocol.json", "test_metrics.csv",
    }
    missing = sorted(name for name in required if not (source / name).is_file())
    if missing:
        raise FileNotFoundError(f"Source run is missing {missing}: {source}")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    training = json.loads((source / "training_protocol.json").read_text(encoding="utf-8"))
    evaluation = json.loads((source / "evaluation_protocol.json").read_text(encoding="utf-8"))
    checkpoint_hash = sha256(source / "model.pt")
    checks = {
        "method": config.get("agent") == "ic3net" and training.get("method") == "ic3net",
        "scenario": config.get("map") == f"Foraging-{cli.scenario}",
        "seed": int(config.get("seed", -1)) == cli.seed,
        "trained": bool(training.get("optimizer_step_verified")),
        "finite": bool(training.get("parameters_finite")),
        "checkpoint_loaded": bool(evaluation.get("checkpoint_load_verified")),
        "checkpoint_identity": evaluation.get("checkpoint_sha256") == checkpoint_hash,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Source checkpoint audit failed {failed}: {source}")

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    env_config = dict(config)
    env_config["seed"] = cli.seed
    env = env_REGISTRY[config["env"]](env_config)
    env_info = env.get_env_info()
    agent_config = dict(config)
    agent_config.update({
        "obs_shape": env_info["obs_shape"],
        "n_actions": env_info["n_actions"],
        "n_agents": env_info["n_agents"],
        "comm_action_one": False,
    })
    runner_config = dict(config)
    runner_config.update({
        "episode_length": env_info["episode_length"],
        "n_agents": env_info["n_agents"],
        "n_actions": env_info["n_actions"],
        "hard_attn": True,
        "commnet": True,
        "comm_action_one": False,
    })
    agent = agent_REGISTRY["ic3net"](agent_config)
    state = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
    agent.load_state_dict(state, strict=True)
    agent.eval()
    runner = runner_REGISTRY["ic3net"](runner_config, env, agent)
    before = [parameter.detach().cpu().clone() for parameter in agent.parameters()]

    output = (
        results_root / cli.method_label / f"lbf_{cli.scenario}-v0" /
        f"seed_{cli.seed}" / "run1"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {output}")
    output.mkdir(parents=True)

    rows = []
    costs = FairCosts(agent, "ic3net")
    env.collect_operational_metrics = True
    for eval_seed in cli.eval_seeds:
        random.seed(eval_seed)
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        env.unwrapped.seed(eval_seed)
        with torch.no_grad():
            for episode in range(1, cli.eval_episodes + 1):
                runner.evaluation_link_dropout_rate = cli.link_dropout_rate
                runner.evaluation_link_dropout_rng = np.random.default_rng(
                    np.random.SeedSequence([int(eval_seed), int(episode), 0x485758])
                )
                _, log = runner.run_an_episode()
                row = {
                    "Method": cli.method_label,
                    "Scenario": f"lbf_{cli.scenario}-v0",
                    "Train_Seed": cli.seed,
                    "Eval_Seed": eval_seed,
                    "Episode": episode,
                    "Episode_Steps": int(log["episode_steps"][0]),
                    "Agent_Count": int(env.n_agents),
                    "Task_Completion_Ratio": log["task_completion_ratio"],
                    "Episode_Success": log["episode_success"],
                    "Throughput": log["throughput"],
                    "Workload_Total": log["workload_total"],
                    "Workload_Completed": log["workload_completed"],
                    "Capacity_Workload_Total": log["capacity_workload_total"],
                    "Capacity_Workload_Completed": log["capacity_workload_completed"],
                    "Capacity_Completion_Ratio": (
                        log["capacity_workload_completed"] / log["capacity_workload_total"]
                        if log["capacity_workload_total"] else 0.0
                    ),
                    "Episode_Success_Rate": log["episode_success"],
                    "Collisions": log["num_collisions"],
                    "Collision_Rate": (
                        log["num_collisions"] /
                        (int(log["episode_steps"][0]) * int(env.n_agents))
                        if int(log["episode_steps"][0]) and int(env.n_agents) else 0.0
                    ),
                    "Active_Link_Ratio": log.get("comm_sparsity", 0.0),
                    "Mean_Active_Link_Ratio": log.get("comm_sparsity", 0.0),
                    "Episode_Return": float(np.sum(log["episode_return"])),
                    "Requested_Link_Dropout_Rate": log["requested_link_dropout_rate"],
                    "Realized_Active_Link_Dropout_Rate": log["realized_active_link_dropout_rate"],
                }
                row.update(env.operational_metrics.summary(env.unwrapped))
                for key, value in row.items():
                    if isinstance(value, (int, float, np.number)) and not math.isfinite(float(value)):
                        raise FloatingPointError(f"Non-finite evaluation metric {key}={value}")
                rows.append(row)
                costs.episode(row, int(row["Episode_Steps"]))
    env.collect_operational_metrics = False
    costs.finish(output)
    write_csv(output / "test_metrics.csv", rows)
    write_comparison_csv(output, rows)

    after = [parameter.detach().cpu() for parameter in agent.parameters()]
    unchanged = all(torch.equal(a, b) for a, b in zip(before, after))
    if not unchanged:
        raise RuntimeError("Frozen link-dropout evaluation modified IC3Net parameters.")

    source_relative = str(source.relative_to(PROJECT))
    output_config = dict(config)
    output_config.update({
        "link_dropout_rate": cli.link_dropout_rate,
        "source_run_dir": source_relative,
        "source_checkpoint_sha256": checkpoint_hash,
    })
    (output / "config.json").write_text(json.dumps(output_config, indent=2), encoding="utf-8")
    (output / "training_protocol.json").write_text(json.dumps({
        "method": cli.method_label,
        "seed": cli.seed,
        "training_reused": True,
        "new_optimizer_updates": 0,
        "source_method": "ic3net",
        "source_run_dir": source_relative,
        "source_checkpoint_sha256": checkpoint_hash,
        "protocol_version": training.get("protocol_version", "V1"),
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
            "after IC3Net communication gating and non-self/alive masking; "
            "before CommNet aggregation"
        ),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_load_verified": True,
        "parameters_unchanged": unchanged,
        "protocol_version": evaluation.get("protocol_version", "V1"),
    }, indent=2), encoding="utf-8")
    env.close()
    print(output.relative_to(PROJECT))


if __name__ == "__main__":
    main()
