"""Zero-shot cross-configuration evaluation for one TeamComm-owned baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from baselines import REGISTRY as AGENT_REGISTRY
from cross_configuration import (
    apply_to_environment,
    load_design,
    target_spec,
    validate_evaluation_rows,
)
from envs.paper_wrapper import PaperWrapper
from fair_costs import FairCosts
from operational_metrics import write_comparison_csv
from qmix_main import AgentUtilities, MonotoneMixer, rollout as qmix_rollout
from runner import REGISTRY as RUNNER_REGISTRY
from scenario_registry import load_registry


PROJECT = Path(__file__).resolve().parent
METHODS = {"commnet", "ic3net", "magic", "qmix", "tarmac", "teamcomm", "tiecomm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one trained TeamComm baseline on a registered unseen configuration."
    )
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def make_target_environment(source_scenario: str, train_seed: int, target: dict) -> PaperWrapper:
    env = PaperWrapper({"map": f"Foraging-{source_scenario}", "seed": train_seed})
    before = env.get_env_info()
    apply_to_environment(env.unwrapped, target)
    env.n_agents = env.unwrapped.n_agents
    env.episode_limit = env.unwrapped._max_episode_steps
    env._obs = None
    after = env.get_env_info()
    for key in ("obs_shape", "state_shape", "n_actions", "n_agents", "episode_length"):
        if before[key] != after[key]:
            raise RuntimeError(
                f"Target configuration changed policy interface {key}: "
                f"{before[key]} -> {after[key]}"
            )
    return env


def common_row(method: str, source_scenario: str, train_seed: int,
               eval_seed: int, episode: int, log: dict, env: PaperWrapper) -> dict:
    steps = int(log["episode_steps"][0])
    row = {
        "Method": method,
        "Scenario": f"lbf_{source_scenario}-v0",
        "Train_Seed": train_seed,
        "Eval_Seed": eval_seed,
        "Episode": episode,
        "Episode_Return": float(np.sum(log["episode_return"])),
        "Episode_Steps": steps,
        "Agent_Count": int(env.n_agents),
        "Success_Rate": float(log["success_rate"]),
        "Task_Completion_Ratio": float(log["task_completion_ratio"]),
        "Episode_Success": float(log["episode_success"]),
        "Throughput": float(log["throughput"]),
        "Workload_Total": int(log["workload_total"]),
        "Workload_Completed": int(log["workload_completed"]),
        "Capacity_Workload_Total": int(log["capacity_workload_total"]),
        "Capacity_Workload_Completed": int(log["capacity_workload_completed"]),
        "Capacity_Completion_Ratio": (
            float(log["capacity_workload_completed"])
            / float(log["capacity_workload_total"])
            if log["capacity_workload_total"] else 0.0
        ),
        "Episode_Success_Rate": float(log["episode_success"]),
        "Collisions": int(log["num_collisions"]),
        "Collision_Rate": (
            float(log["num_collisions"]) / float(steps * env.n_agents)
            if steps and env.n_agents else 0.0
        ),
        "Active_Link_Ratio": float(log.get("comm_sparsity", 0.0)),
    }
    row.update(env.operational_metrics.summary(env.unwrapped))
    return row


def evaluate_communication_baseline(method: str, source_config: dict, checkpoint: dict,
                                    env: PaperWrapper, eval_seeds: list[int],
                                    eval_episodes: int, source_scenario: str,
                                    train_seed: int) -> tuple[list[dict], FairCosts, bool]:
    config = dict(source_config)
    info = env.get_env_info()
    config.update(
        obs_shape=info["obs_shape"], n_actions=info["n_actions"],
        n_agents=info["n_agents"], episode_length=info["episode_length"],
        device="cpu",
    )
    agent = AGENT_REGISTRY[method](config)
    agent.load_state_dict(checkpoint, strict=True)
    runner = RUNNER_REGISTRY[method](config, env, agent)
    agent.eval()
    before = [parameter.detach().cpu().clone() for parameter in agent.parameters()]
    costs = FairCosts(agent, method)
    rows: list[dict] = []
    env.collect_operational_metrics = True
    try:
        for eval_seed in eval_seeds:
            seed_all(eval_seed)
            env.unwrapped.seed(eval_seed)
            with torch.no_grad():
                for episode in range(1, eval_episodes + 1):
                    _, log = runner.run_an_episode()
                    row = common_row(
                        method, source_scenario, train_seed, eval_seed, episode, log, env
                    )
                    costs.episode(row, int(row["Episode_Steps"]))
                    rows.append(row)
    finally:
        env.collect_operational_metrics = False
    after = [parameter.detach().cpu() for parameter in agent.parameters()]
    unchanged = all(torch.equal(left, right) for left, right in zip(before, after))
    return rows, costs, unchanged


def evaluate_qmix(source_config: dict, checkpoint: dict, env: PaperWrapper,
                  eval_seeds: list[int], eval_episodes: int,
                  source_scenario: str, train_seed: int) -> tuple[list[dict], FairCosts, bool]:
    observations, state, _ = env.reset()
    agent = AgentUtilities(len(observations[0]))
    mixer = MonotoneMixer(env.n_agents, len(state))
    agent.load_state_dict(checkpoint["agent"], strict=True)
    mixer.load_state_dict(checkpoint["mixer"], strict=True)
    agent.eval()
    mixer.eval()
    parameters = list(agent.parameters()) + list(mixer.parameters())
    before = [parameter.detach().cpu().clone() for parameter in parameters]
    costs = FairCosts(agent, "qmix")
    rows: list[dict] = []
    env.collect_operational_metrics = True
    try:
        for eval_seed in eval_seeds:
            seed_all(eval_seed)
            env.unwrapped.seed(eval_seed)
            eval_rng = np.random.default_rng(eval_seed)
            for episode in range(1, eval_episodes + 1):
                _, metrics = qmix_rollout(env, agent, 0.0, eval_rng)
                metrics.update(env.operational_metrics.summary(env.unwrapped))
                metrics["Episode_Return"] = metrics.pop("Avg_Return")
                row = {
                    "Method": "qmix",
                    "Scenario": f"lbf_{source_scenario}-v0",
                    "Train_Seed": train_seed,
                    "Eval_Seed": eval_seed,
                    "Episode": episode,
                    **metrics,
                }
                costs.episode(row, int(row["Episode_Steps"]))
                rows.append(row)
    finally:
        env.collect_operational_metrics = False
    after = [parameter.detach().cpu() for parameter in parameters]
    unchanged = all(torch.equal(left, right) for left, right in zip(before, after))
    return rows, costs, unchanged


def main() -> None:
    cli = parse_args()
    if cli.gpu != -1:
        raise ValueError("TeamComm cross-configuration evaluation is CPU-only; use --gpu -1.")
    if cli.eval_episodes < 1 or len(cli.eval_seeds) != len(set(cli.eval_seeds)):
        raise ValueError("Evaluation seeds must be unique and episode count positive.")

    scenario_registry = load_registry(PROJECT / "scenario_registry.json")
    design = load_design(scenario_registry)
    if cli.method not in design["methods"]:
        raise ValueError(f"Method is absent from the registered design: {cli.method}")
    if cli.eval_seeds != design["evaluation_seeds"]:
        raise ValueError("Formal cross-configuration evaluation seeds must match the design.")
    if cli.eval_episodes != int(design["episodes_per_evaluation_seed"]):
        raise ValueError("Formal cross-configuration episode budget mismatch.")
    source_scenario = design["source_scenario"]
    target = target_spec(design, scenario_registry, cli.target_configuration)

    source = cli.source_run_dir
    required = {"model.pt", "config.json", "training_protocol.json", "evaluation_protocol.json"}
    missing = sorted(name for name in required if not (source / name).is_file())
    if missing:
        raise FileNotFoundError(f"Source run is missing {missing}: {source}")
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    train_seed = int(source_config.get("seed", -1))
    if train_seed not in design["training_seeds"]:
        raise ValueError(f"Source checkpoint has an unregistered training seed: {train_seed}")
    recorded_method = source_config.get("agent", "qmix" if cli.method == "qmix" else None)
    recorded_scenario = (
        source_config.get("scenario") if cli.method == "qmix"
        else str(source_config.get("map", "")).removeprefix("Foraging-")
    )
    if recorded_method != cli.method or recorded_scenario != source_scenario:
        raise RuntimeError(
            f"Source identity mismatch: method={recorded_method}, scenario={recorded_scenario}"
        )

    env = make_target_environment(source_scenario, train_seed, target)
    checkpoint_path = source / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if cli.method == "qmix":
        rows, costs, unchanged = evaluate_qmix(
            source_config, checkpoint, env, cli.eval_seeds, cli.eval_episodes,
            source_scenario, train_seed,
        )
        action_mode = "greedy"
    else:
        rows, costs, unchanged = evaluate_communication_baseline(
            cli.method, source_config, checkpoint, env, cli.eval_seeds,
            cli.eval_episodes, source_scenario, train_seed,
        )
        action_mode = "native_sample"
    if not unchanged:
        raise RuntimeError("Zero-shot evaluation modified policy parameters.")
    for row in rows:
        row["Source_Configuration"] = source_scenario
        row["Target_Configuration"] = cli.target_configuration
    validate_evaluation_rows(rows, target)

    output = (
        cli.results_root / cli.method
        / f"{source_scenario}_to_{cli.target_configuration}"
        / f"seed_{train_seed}" / "run1"
    )
    if output.exists():
        raise FileExistsError(f"Cross-configuration result already exists: {output}")
    output.mkdir(parents=True)
    write_csv(output / "test_metrics.csv", rows)
    costs.finish(output)
    write_comparison_csv(output, rows)

    protocol = {
        "experiment_id": design["experiment_id"],
        "method": cli.method,
        "source_scenario": source_scenario,
        "target_configuration": cli.target_configuration,
        "training_seed": train_seed,
        "optimizer_updates": 0,
        "parameters_unchanged": True,
        "checkpoint_load_strict": True,
        "action_mode": action_mode,
        "source_run_dir": relative_or_absolute(source, PROJECT),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "cross_configuration_design_sha256": design["semantic_sha256"],
        "target_configuration_sha256": target["cross_configuration_sha256"],
        "evaluation_seeds": cli.eval_seeds,
        "episodes_per_seed": cli.eval_episodes,
    }
    (output / "evaluation_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    (output / "target_configuration.json").write_text(
        json.dumps(target, indent=2), encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
