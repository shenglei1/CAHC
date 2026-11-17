from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

import data
from action_utils import parse_action_args, select_action, translate_action
from hyperwavex import HyperWaveX
from runtime_support import runtime_metadata
from scenario_registry import load_registry, registry_sha256
from trainer import Trainer
from operational_metrics import OperationalMetrics, write_comparison_csv
from fair_costs import FairCosts
from probe2_interlock import evaluation_columns


PROJECT = Path(__file__).resolve().parent
VARIANTS = {
    "full",
    "no_capability",
    "no_wave",
    "complete_topology",
    "wave_k1",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    # Paper-facing CSV files contain identifiers and measured quantities only.
    # Execution/provenance metadata lives in the adjacent JSON protocol files.
    excluded = {
        "Evaluation_Phase", "Action_Mode", "Checkpoint_SHA256",
        "Latency_Isolated", "Metrics_Schema_Version",
        "Mean_Evaluation_Step_ms", "Training_Entropy_Coefficient",
        "Epoch_Wall_Seconds",
    }
    clean_rows = [
        {
            key: value for key, value in row.items()
            if key not in excluded and not key.endswith("_ms")
        }
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(clean_rows[0]))
        writer.writeheader()
        writer.writerows(clean_rows)


def validate_workload_rows(rows: list[dict], spec: dict, *, frozen_test: bool) -> None:

    probe_rules = spec.get("probe_rules", {})
    if probe_rules.get("probe_version") in {
        "V1"
    }:
        width = int(probe_rules["station_member_count"])
        for row in rows:
            total, done = float(row["Workload_Total"]), float(row["Workload_Completed"])
            if not 0 <= done <= total <= 120:
                raise ValueError("Probe2 realized service workload is out of range.")
            if not math.isclose(float(row["Capacity_Workload_Total"]), width * total):
                raise ValueError("Probe2 capacity denominator is inconsistent.")
            if not math.isclose(float(row["Capacity_Workload_Completed"]), width * done):
                raise ValueError("Probe2 completed capacity is inconsistent.")
            if frozen_test:
                expected_ratio = done / total if total else 0.0
                if not math.isclose(float(row["Task_Completion_Ratio"]), expected_ratio):
                    raise ValueError("Probe2 completion ratio is inconsistent.")
                if float(row["Episode_Success"]) != float(done == total and total > 0):
                    raise ValueError("Probe2 episode success is inconsistent.")
        return
    task_total = int(spec["tasks"]["workload"]["task_count"])
    capacity_total = int(spec["tasks"]["workload"]["capacity_units"])
    requirements = tuple(int(value) for value in spec["tasks"]["requirements"])
    subset_sums = {
        completed: {
            sum(values) for values in itertools.combinations(requirements, completed)
        }
        for completed in range(task_total + 1)
    }
    for row in rows:
        total = float(row["Workload_Total"])
        completed = float(row["Workload_Completed"])
        cap_total = float(row["Capacity_Workload_Total"])
        cap_completed = float(row["Capacity_Workload_Completed"])
        ratio = float(row["Task_Completion_Ratio"])
        success = float(row["Episode_Success"])
        if not math.isclose(total, task_total, abs_tol=1e-9):
            raise ValueError(f"Incorrect Workload_Total in metrics row: {row}")
        if not math.isclose(cap_total, capacity_total, abs_tol=1e-9):
            raise ValueError(f"Incorrect Capacity_Workload_Total in metrics row: {row}")
        if not 0.0 <= completed <= total or not 0.0 <= cap_completed <= cap_total:
            raise ValueError(f"Out-of-range workload ledger in metrics row: {row}")
        if not math.isclose(ratio, completed / total, abs_tol=1e-9):
            raise ValueError(f"Task_Completion_Ratio disagrees with ledger: {row}")
        if not 0.0 <= success <= 1.0:
            raise ValueError(f"Episode_Success is outside [0, 1]: {row}")
        if frozen_test:
            completed_int = int(completed)
            if not math.isclose(completed, completed_int, abs_tol=1e-9):
                raise ValueError(f"Frozen-test completed workload is not integral: {row}")
            if int(cap_completed) not in subset_sums[completed_int] or not math.isclose(
                cap_completed, int(cap_completed), abs_tol=1e-9
            ):
                raise ValueError(
                    "Capacity_Workload_Completed is not a valid sum of the "
                    f"{completed_int} completed task requirements: {row}"
                )
            expected_success = float(completed_int == task_total)
            if success != expected_success:
                raise ValueError(f"Episode_Success disagrees with workload ledger: {row}")
            if not math.isclose(
                float(row["Throughput"]), completed / int(spec["horizon"]), abs_tol=1e-9
            ):
                raise ValueError(f"Throughput disagrees with fixed horizon: {row}")


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HyperWave-X under the frozen CAIE protocol."
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument("--episodes-per-epoch", type=int, default=64)
    parser.add_argument(
        "--eval-seeds", nargs="+", type=int, default=list(range(1001, 1011))
    )
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.0)
    parser.add_argument("--allow-probe", action="store_true")
    parser.add_argument("--development-probe", action="store_true",
                        help="Probe2-only development run; never evidence for the final claim.")
    parser.add_argument("--record-trace", action="store_true")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="full")
    parser.add_argument(
        "--run-label",
        default="",
        help="Optional unique result label for a preregistered sensitivity run.",
    )
    parser.add_argument("--capability-weight", type=float, default=0.1)
    parser.add_argument("--hypergraph-temperature", type=float, default=1.0)
    parser.add_argument("--hyperedge-size", type=int, default=0)
    parser.add_argument("--wave-iterations", type=int, default=2)
    parser.add_argument("--wave-tau", type=float, default=1.0)
    parser.add_argument("--wave-damping", type=float, default=0.8)
    parser.add_argument("--topology-temperature", type=float, default=1.0)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT / "results/FORMAL_6SCENARIOS_ORGANIZED_V1",
    )
    return parser.parse_args()


def build_args(cli: argparse.Namespace, registry: dict) -> argparse.Namespace:
    scenario = registry["scenarios"][cli.scenario]
    capability_aware = cli.variant != "no_capability"
    use_wave = cli.variant != "no_wave"
    wave_iterations = 1 if cli.variant == "wave_k1" else cli.wave_iterations
    method_label = cli.run_label or (
        "hyperwave_x" if cli.variant == "full" else cli.variant
    )
    return argparse.Namespace(
        env_name=f"lbf_{cli.scenario}-v0",
        seed=cli.seed,
        gpu=cli.gpu,
        batch_size=cli.episodes_per_epoch,
        max_steps=int(scenario["horizon"]),
        hid_size=64,
        qk_hid_size=64,
        directed=True,
        self_loop_type=2,
        gat_num_heads=2,
        gat_hid_size=32,
        nagents=int(scenario["agents"]["count"]),
        nfriendly=int(scenario["agents"]["count"]),
        mean_ratio=0.0,
        detach_gap=10,
        gamma=0.0 if cli.scenario == "probe_interaction" else 1.0,
        normalize_rewards=False,
        lrate=(getattr(cli, "learning_rate", None) or
               scenario.get("probe_rules", {}).get("training_learning_rate", 0.001)),
        lr_decay=float(scenario.get("probe_rules", {}).get("training_lr_decay", 1.0)),
        max_grad_norm=float(
            scenario.get("probe_rules", {}).get("training_gradient_clip_norm", 0.0)
        ),
        entr=float(
            scenario.get("probe_rules", {}).get(
                "training_entropy_coefficient", 0.0
            )
        ),
        value_coeff=0.01,
        teacher_loss_coefficient=float(
            scenario.get("probe_rules", {}).get("teacher_loss_coefficient", 0.0)
        ),
        teacher_only_training=bool(
            scenario.get("probe_rules", {}).get("teacher_only_training", False)
        ),
        num_actions=[6],
        dim_actions=1,
        nactions="1",
        action_scale=1.0,
        max_player_level=max(scenario["agents"]["capabilities"]),
        task_observation_width=1 + 2 * int(scenario["observation"]["sight"]),
        capability_aware=capability_aware,
        capability_weight=cli.capability_weight,
        hypergraph_temperature=cli.hypergraph_temperature,
        hyperedge_size=cli.hyperedge_size,
        use_wave=use_wave,
        wave_iterations=wave_iterations,
        wave_tau=cli.wave_tau,
        wave_damping=cli.wave_damping,
        complete_topology=cli.variant == "complete_topology",
        topology_temperature=cli.topology_temperature,
        variant=cli.variant,
        method_label=method_label,
    )


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize(args: argparse.Namespace):
    seed_all(args.seed)
    env = data.init(args.env_name, args)
    args.obs_size = env.observation_dim
    args.num_actions = [env.num_actions]
    args.dim_actions = env.dim_actions
    args.nagents = env.n_agents
    args.nfriendly = env.n_agents
    parse_action_args(args)
    policy = HyperWaveX(args).double()
    trainer = Trainer(args, policy, env)
    return env, policy, trainer


def normalize_actions(actual) -> list[int]:
    if isinstance(actual, (list, tuple)) and len(actual) == 1:
        actual = actual[0]
    return np.asarray(actual, dtype=np.int64).reshape(-1).tolist()


def diagnostic_snapshot(
    policy: HyperWaveX,
    env,
    actions: list[int],
    info: dict,
    previous_completed: int,
    step: int,
    positions_before: list[list[int]],
    tasks_before: list[dict],
    record_link_details: bool,
) -> dict:
    raw = env.env
    diagnostics = policy.last_diagnostics
    incidence = diagnostics["incidence"].squeeze(0).cpu().numpy()
    visible_requirements = (
        diagnostics["visible_task_requirement"].squeeze(0).cpu().numpy()
    )
    coalition_cardinalities = (
        diagnostics["coalition_cardinality"].squeeze(0).cpu().numpy()
    )
    target_partner_capabilities = (
        diagnostics["target_partner_capability"].squeeze(0).cpu().numpy()
    )
    topology = diagnostics["adjacency"].cpu().numpy()
    if incidence.shape != (raw.n_agents, raw.n_agents):
        raise ValueError(f"Unexpected incidence shape: {incidence.shape}")
    if topology.shape != (raw.n_agents, raw.n_agents):
        raise ValueError(f"Unexpected topology shape: {topology.shape}")

    levels = np.asarray(raw.fixed_levels, dtype=np.int64)
    active_requirements = [item["requirement"] for item in tasks_before]
    threshold = min(active_requirements or raw.spec["tasks"]["requirements"])
    coalitions = []
    for edge in range(incidence.shape[0]):
        members = np.flatnonzero(incidence[edge] > 0.5).tolist()
        capacity = int(levels[members].sum()) if members else 0
        coalitions.append(
            {
                "hyperedge": edge,
                "members": members,
                "visible_task_requirement": int(visible_requirements[edge]),
                "task_conditioned_cardinality": int(
                    coalition_cardinalities[edge]
                ),
                "target_partner_capability": float(
                    target_partner_capabilities[edge]
                ),
                "capability_sum": capacity,
                "candidate_threshold_feasible": capacity >= threshold,
                "capacity_gap_to_nearest_active_threshold": min(
                    (abs(capacity - value) for value in active_requirements),
                    default=abs(capacity - threshold),
                ),
            }
        )

    non_self = ~np.eye(raw.n_agents, dtype=bool)
    active_links = int(np.count_nonzero(topology[non_self] > 0.5))
    routing_links = []
    if record_link_details:
        attention = diagnostics["attention"].cpu().numpy()
        for receiver, sender in np.argwhere((topology > 0.5) & non_self):
            routing_links.append(
                {
                    "receiver": int(receiver),
                    "sender": int(sender),
                    "attention_by_head": [
                        float(value) for value in attention[receiver, sender].reshape(-1)
                    ],
                }
            )
    possible_links = int(
        info.get(
            "probe2_communication_scaffold_directed_links",
            raw.n_agents * (raw.n_agents - 1),
        )
    )
    completed = int(info["workload_completed"])
    load_count = int(sum(action == getattr(raw, "load_action", 5) for action in actions))
    probe_thresholds = set(int(value) for value in raw.spec["tasks"]["requirements"])
    exact_pairs = {
        frozenset((left, right))
        for left in range(raw.n_agents)
        for right in range(left + 1, raw.n_agents)
        if int(levels[left] + levels[right]) in probe_thresholds
    }
    selected_pairs = {
        frozenset(item["members"])
        for item in coalitions
        if len(item["members"]) == 2
    }
    wave = diagnostics.get("wave")
    if wave is None:
        wave_summary = {
            "lambda_min": 0.0,
            "lambda_max": 0.0,
            "lambda_max_is_bound": 1.0,
            "contraction_bound": 0.0,
            "stability_margin": 0.0,
            "initial_dirichlet_energy": 0.0,
            "final_dirichlet_energy": 0.0,
            "energy_ratio": 0.0,
            "update_norms": [],
        }
    else:
        wave_summary = {
            key: (
                [float(value) for value in wave[key].reshape(-1).tolist()]
                if key == "update_norms"
                else float(wave[key].reshape(-1)[0])
            )
            for key in wave
        }
    return {
        "step": step,
        "actions": actions,
        "agent_positions_before_action": positions_before,
        "active_tasks_before_action": tasks_before,
        "coalitions": coalitions,
        "candidate_feasible_fraction": float(
            np.mean([item["candidate_threshold_feasible"] for item in coalitions])
        ),
        "mean_candidate_capacity_gap": float(
            np.mean(
                [item["capacity_gap_to_nearest_active_threshold"] for item in coalitions]
            )
        ),
        "mean_coalition_cardinality": float(np.mean(coalition_cardinalities)),
        "visible_task_conditioned_edge_fraction": float(
            np.mean(visible_requirements > 0)
        ),
        "probe1_exact_pair_coverage": float(
            len(exact_pairs & selected_pairs) / len(exact_pairs)
            if raw.type == "probe_complementary"
            else 0.0
        ),
        "active_directed_links": active_links,
        "active_routing_links": routing_links,
        "active_link_ratio": active_links / possible_links if possible_links else 0.0,
        "load_action_count": load_count,
        "probe2_decision_audit": info.get("probe2_decision_audit", {}),
        "task_completion_delta": completed - previous_completed,
        "false_load_attempt": int(load_count > 0 and completed == previous_completed),
        "workload_total": int(info["workload_total"]),
        "workload_completed": completed,
        "capacity_workload_total": int(info["capacity_workload_total"]),
        "capacity_workload_completed": int(info["capacity_workload_completed"]),
        "task_completion_ratio": float(info["task_completion_ratio"]),
        "episode_success": float(info["episode_success"]),
        "probe_locked_agent_count": int(info["probe_locked_agent_count"]),
        "probe1_exact_service_pairs": int(info["probe1_exact_service_pairs"]),
        "probe1_exact_service_pair_coverage": float(
            info["probe1_exact_service_pair_coverage"]
        ),
        "probe2_premature_load_attempts": int(
            info["probe2_premature_load_attempts"]
        ),
        "probe2_ready_load_participations": int(
            info["probe2_ready_load_participations"]
        ),
        "probe2_incomplete_ready_load_attempts": int(
            info["probe2_incomplete_ready_load_attempts"]
        ),
        "probe2_safe_completions": int(info["probe2_safe_completions"]),
        "probe2_safe_completion_ratio": float(
            info["probe2_safe_completion_ratio"]
        ),
        "probe2_safe_episode_success": float(
            info["probe2_safe_episode_success"]
        ),
        "probe2_missed_service_windows": int(
            info["probe2_missed_service_windows"]
        ),
        "probe2_station_ready_steps": info["probe2_station_ready_steps"],
        "probe2_sensor_values": info["probe2_sensor_values"],
        "probe_load_cooldown_until": info["probe_load_cooldown_until"],
        "wave": wave_summary,
    }


def evaluate_episode(
    policy: HyperWaveX,
    env,
    args: argparse.Namespace,
    eval_seed: int,
    episode: int,
    record_spectral_diagnostics: bool,
):
    if hasattr(env.env, "set_evaluation_mode"):
        env.env.set_evaluation_mode(True)
    state = env.reset().to(policy.device)
    hidden = policy.init_hidden(batch_size=state.shape[0])
    info = (
        env.initial_info()
        if hasattr(env, "initial_info")
        else {"agent_level": env._agent_levels}
    )
    trace = []
    previous_completed = 0
    operational = OperationalMetrics(env.env)
    with torch.no_grad():
        for step in range(1, args.max_steps + 1):
            raw = env.env
            positions_before = [list(player.position) for player in raw.players]
            tasks_before = [
                {"position": list(position), "requirement": int(value)}
                for position, value in zip(
                    np.argwhere(raw.field > 0).tolist(), raw.field[raw.field > 0].tolist()
                )
            ]
            info["collect_model_diagnostics"] = True
            info["record_spectral_diagnostics"] = record_spectral_diagnostics
            action_out, _, hidden = policy([state, hidden], env.policy_info(info))
            if getattr(raw, "is_probe2", False):
                sampled = torch.stack(
                    [head.argmax(dim=-1, keepdim=True)[0] for head in action_out]
                )
            else:
                sampled = select_action(args, action_out)
            _, actual = translate_action(args, env, sampled)
            actions = normalize_actions(actual)
            operational.before_step(raw, actions)
            state, _, done, info = env.step(actual)
            info["agent_level"] = env._agent_levels
            operational.after_step(raw)
            trace.append(
                diagnostic_snapshot(
                    policy,
                    env,
                    actions,
                    info,
                    previous_completed,
                    step,
                    positions_before,
                    tasks_before,
                    record_spectral_diagnostics,
                )
            )
            previous_completed = int(info["workload_completed"])
            if done:
                break
    final = trace[-1]
    load_attempts = sum(item["load_action_count"] > 0 for item in trace)
    false_loads = sum(item["false_load_attempt"] for item in trace)
    synchronized_successes = sum(item["task_completion_delta"] for item in trace)
    element_bytes = next(policy.parameters()).element_size()
    mean_active_links = float(
        np.mean([item["active_directed_links"] for item in trace])
    )
    row = {
        "Method": args.method_label,
        "Scenario": args.env_name,
        "Train_Seed": args.seed,
        "Eval_Seed": eval_seed,
        "Episode": episode,
        "Episode_Steps": len(trace),
        "Agent_Count": int(args.nagents),
        "Workload_Total": final["workload_total"],
        "Workload_Completed": final["workload_completed"],
        "Capacity_Workload_Total": final["capacity_workload_total"],
        "Capacity_Workload_Completed": final["capacity_workload_completed"],
        "Capacity_Completion_Ratio": (
            final["capacity_workload_completed"] / final["capacity_workload_total"]
            if final["capacity_workload_total"] else 0.0
        ),
        "Task_Completion_Ratio": final["task_completion_ratio"],
        "Episode_Success": final["episode_success"],
        "Episode_Success_Rate": final["episode_success"],
        "Throughput": final["workload_completed"] / args.max_steps,
        "Collisions": int(info["num_collisions"]),
        "Collision_Rate": (
            int(info["num_collisions"]) / (len(trace) * int(args.nagents))
            if trace and int(args.nagents) else 0.0
        ),
        "Mean_Candidate_Feasible_Fraction": float(
            np.mean([item["candidate_feasible_fraction"] for item in trace])
        ),
        "Mean_Candidate_Capacity_Gap": float(
            np.mean([item["mean_candidate_capacity_gap"] for item in trace])
        ),
        "Mean_Probe1_Exact_Pair_Coverage": float(
            np.mean([item["probe1_exact_pair_coverage"] for item in trace])
        ),
        "Probe1_Exact_Service_Pairs": int(info["probe1_exact_service_pairs"]),
        "Probe1_Exact_Service_Pair_Coverage": float(
            info["probe1_exact_service_pair_coverage"]
        ),
        "Probe2_Premature_Load_Attempts": int(
            info["probe2_premature_load_attempts"]
        ),
        "Probe2_Ready_Load_Participations": int(
            info["probe2_ready_load_participations"]
        ),
        "Probe2_Incomplete_Ready_Load_Attempts": int(
            info["probe2_incomplete_ready_load_attempts"]
        ),
        "Probe2_Safe_Completions": int(info["probe2_safe_completions"]),
        "Probe2_Safe_Completion_Ratio": float(
            info["probe2_safe_completion_ratio"]
        ),
        "Probe2_Safe_Episode_Success": float(
            info["probe2_safe_episode_success"]
        ),
        "Probe2_Unsafe_Station_Count": int(
            info["probe2_unsafe_station_count"]
        ),
        "Probe2_Missed_Service_Windows": int(
            info["probe2_missed_service_windows"]
        ),
        "Probe2_Expired_Tasks": int(info["probe2_expired_tasks"]),
        "Probe2_Ready_To_Success_Delay_Sum": int(
            info["probe2_ready_to_success_delay_sum"]
        ),
        "Probe2_Ready_To_Success_Delay_Count": int(
            info["probe2_ready_to_success_delay_count"]
        ),
        "Probe2_Mean_Ready_To_Success_Delay": float(
            info["probe2_mean_ready_to_success_delay"]
        ),
        "Probe2_Required_Sensor_To_Actuator_Hops": int(
            info.get("probe2_required_sensor_to_actuator_hops", 0)
        ),
        "Probe2_Communication_Scaffold_Directed_Links": int(
            info.get("probe2_communication_scaffold_directed_links", 0)
        ),
        "Probe2_Actuator_Load_Attempts": int(
            info.get("probe2_actuator_load_attempts", 0)
        ),
        "Mean_Active_Link_Ratio": float(
            np.mean([item["active_link_ratio"] for item in trace])
        ),
        "Mean_Active_Links_Per_Step": mean_active_links,
        "Estimated_Payload_Bytes_Per_Step": (
            mean_active_links * args.hid_size * element_bytes
        ),
        "Synchronized_Load_Successes": synchronized_successes,
        "Load_Attempts": load_attempts,
        "False_Load_Attempts": false_loads,
        "Missing_Member_Rate": false_loads / load_attempts if load_attempts else 0.0,
        "Synchronization_Success_Rate": (
            synchronized_successes / load_attempts if load_attempts else 0.0
        ),
        "Mean_Wave_Lambda_Max_Bound": float(
            np.mean([item["wave"]["lambda_max"] for item in trace])
        ),
        "Wave_Lambda_Max_Is_Analytical_Bound": float(
            np.mean([item["wave"]["lambda_max_is_bound"] for item in trace])
        ),
        "Max_Wave_Contraction_Bound": float(
            max(item["wave"]["contraction_bound"] for item in trace)
        ),
        "Min_Wave_Stability_Margin": float(
            min(item["wave"]["stability_margin"] for item in trace)
        ),
        "Mean_Diffusion_Energy_Ratio": float(
            np.mean([item["wave"]["energy_ratio"] for item in trace])
        ),
    }
    row.update(operational.summary(env.env))
    row["Training_Entropy_Coefficient"] = float(args.entr)
    row.update(evaluation_columns(env.env, info))
    if not all(
        math.isfinite(float(value))
        for value in row.values()
        if isinstance(value, (int, float))
    ):
        raise FloatingPointError(f"Non-finite evaluation metric: {row}")
    return row, trace


def main() -> None:
    cli = parse_cli()
    registry = load_registry(PROJECT / "scenario_registry.json")
    allowed = (
        registry["reviewer_probe_ids"]
        if cli.allow_probe
        else registry["formal_scenario_ids"] + registry["supplementary_scenario_ids"]
    )
    if cli.scenario not in allowed:
        raise ValueError(f"Scenario must be one of {allowed}; got {cli.scenario}")
    if cli.development_probe and cli.scenario != "probe_interaction":
        raise ValueError("--development-probe is restricted to Probe2.")
    probe_rules = registry["scenarios"][cli.scenario].get("probe_rules", {})
    if cli.seed < 0:
        raise ValueError("Training seed must be a non-negative integer.")
    if cli.epochs < 1 or cli.episodes_per_epoch < 1 or cli.eval_episodes < 1:
        raise ValueError("Epoch and episode counts must be positive.")
    if cli.learning_rate is None:
        cli.learning_rate = registry["scenarios"][cli.scenario].get(
            "probe_rules", {}
        ).get("training_learning_rate", 0.001)
    if not math.isfinite(cli.learning_rate) or cli.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if len(cli.eval_seeds) != len(set(cli.eval_seeds)):
        raise ValueError("Evaluation seeds must be unique.")
    if cli.seed in set(cli.eval_seeds):
        raise ValueError("Training and evaluation seeds must be disjoint.")
    if cli.scenario == "probe_interaction":
        expected_eval = (
            probe_rules["development_eval_seeds"]
            if cli.development_probe else probe_rules["formal_eval_seeds"]
        )
        if cli.development_probe:
            if not set(cli.eval_seeds).issubset(expected_eval):
                raise ValueError("Development Probe2 evaluation seeds are not registered.")
        elif cli.eval_seeds != expected_eval:
            raise ValueError(f"Formal Probe2 evaluation seeds must be {expected_eval}.")
    if cli.topology_temperature <= 0:
        raise ValueError("topology_temperature must be positive.")
    if cli.wave_iterations < 1:
        raise ValueError("wave_iterations must be at least one.")

    torch.set_default_dtype(torch.float64)
    if cli.gpu >= 0 and not torch.cuda.is_available():
        raise RuntimeError("A GPU was requested, but PyTorch cannot access CUDA.")
    if cli.gpu >= 0 and cli.gpu_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(cli.gpu_memory_fraction, cli.gpu)
    args = build_args(cli, registry)
    env, policy, trainer = initialize(args)
    method_label = args.method_label
    seed_root = (
        cli.results_root.resolve()
        / method_label
        / args.env_name
        / f"seed_{args.seed}"
    )
    run_number = 1
    while (seed_root / f"run{run_number}").exists():
        run_number += 1
    output = seed_root / f"run{run_number}"
    output.mkdir(parents=True)

    runtime = runtime_metadata()
    config = vars(args).copy()
    config.update(
        {
            "formal_protocol": registry["protocol_version"],
            "scenario_registry_sha256": registry_sha256(registry),
            "runtime": runtime,
        }
    )
    (output / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    initial = [parameter.detach().cpu().clone() for parameter in policy.parameters()]
    training_rows = []
    for epoch in range(1, cli.epochs + 1):
        stat = trainer.train_batch(epoch - 1)
        training_rows.append(
            {
                "Epoch": epoch,
                "Episodes": int(stat["num_episodes"]),
                "Steps": int(stat["num_steps"]),
                "Workload_Total": float(
                    stat["workload_total"] / stat["num_episodes"]
                ),
                "Workload_Completed": float(
                    stat["workload_completed"] / stat["num_episodes"]
                ),
                "Capacity_Workload_Total": float(
                    stat["capacity_workload_total"] / stat["num_episodes"]
                ),
                "Capacity_Workload_Completed": float(
                    stat["capacity_workload_completed"] / stat["num_episodes"]
                ),
                "Task_Completion_Ratio": float(
                    stat["task_completion_ratio"] / stat["num_episodes"]
                ),
                "Episode_Success": float(
                    stat["episode_success"] / stat["num_episodes"]
                ),
                "Action_Loss": float(stat["action_loss"] / stat["num_steps"]),
                "Value_Loss": float(stat["value_loss"] / stat["num_steps"]),
                "Entropy": float(stat["entropy"] / stat["num_steps"]),
                "Teacher_Loss": float(stat.get("teacher_loss", 0.0)),
                "Teacher_Accuracy": float(stat.get("teacher_accuracy", 0.0)),
                "Gradient_Norm_Before_Clip": float(
                    stat.get("gradient_norm_before_clip", 0.0)
                ),
                "Learning_Rate": float(stat.get("learning_rate", args.lrate)),
            }
        )
    validate_workload_rows(training_rows, registry["scenarios"][cli.scenario], frozen_test=False)
    write_csv(output / "training_metrics.csv", training_rows)
    final_parameters = [parameter.detach().cpu() for parameter in policy.parameters()]
    optimizer_step_verified = any(
        not torch.equal(before, after)
        for before, after in zip(initial, final_parameters)
    )
    backward_verified = any(
        parameter.grad is not None for parameter in policy.parameters()
    )
    parameters_finite = all(
        torch.isfinite(parameter).all().item() for parameter in policy.parameters()
    )
    if not (optimizer_step_verified and backward_verified and parameters_finite):
        raise RuntimeError("Backward/optimizer/finite-parameter verification failed.")

    checkpoint = output / "model.pt"
    torch.save(
        {"policy_net": policy.state_dict(), "optimizer": trainer.state_dict()},
        checkpoint,
    )
    saved = torch.load(checkpoint, map_location=policy.device)
    policy.load_state_dict(saved["policy_net"])
    trainer.load_state_dict(saved["optimizer"])
    checkpoint_hash = file_sha256(checkpoint)

    policy.eval()
    before_eval = [parameter.detach().cpu().clone() for parameter in policy.parameters()]
    evaluation_rows = []
    costs = FairCosts(policy, args.method_label)
    trace_payload = None
    for eval_seed in cli.eval_seeds:
        seed_all(eval_seed)
        env.seed(eval_seed)
        for episode in range(1, cli.eval_episodes + 1):
            row, trace = evaluate_episode(
                policy,
                env,
                args,
                eval_seed,
                episode,
                cli.record_trace and trace_payload is None,
            )
            costs.episode(row, row['Episode_Steps'])
            evaluation_rows.append(row)
            if cli.record_trace and trace_payload is None:
                trace_payload = {
                    "eval_seed": eval_seed,
                    "episode": episode,
                    "steps": trace,
                }
    validate_workload_rows(evaluation_rows, registry["scenarios"][cli.scenario], frozen_test=True)
    write_csv(output / "test_metrics.csv", evaluation_rows)
    costs.finish(output)
    write_comparison_csv(output, evaluation_rows)
    if trace_payload is not None:
        (output / "episode_trace.jsonl").write_text(
            json.dumps(trace_payload) + "\n", encoding="utf-8"
        )
    after_eval = [parameter.detach().cpu() for parameter in policy.parameters()]
    if any(
        not torch.equal(before, after)
        for before, after in zip(before_eval, after_eval)
    ):
        raise RuntimeError("Frozen evaluation modified policy parameters.")

    source_names = [
        "hyperwavex.py",
        "gnn_layers.py",
        "trainer.py",
        "data.py",
        "env_wrappers.py",
        "scenario_registry.py",
        "scenario_registry.json",
        "hyperwavex_experiment_plan.json",
        "shared_lbf_env.py",
        "runtime_support.py",
        "formal_hyperwavex.py",
        "operational_metrics.py", "fair_costs.py",
        "probe2_interlock.py", "compare_probe2.py",
    ]
    source_manifest = {
        name: file_sha256(PROJECT / name) for name in source_names
    }
    probe_rules = registry["scenarios"][cli.scenario].get("probe_rules", {})
    is_interlock_probe = probe_rules.get("probe_version") == "V1"
    (output / "training_protocol.json").write_text(
        json.dumps(
            {
                "method": method_label,
                "algorithm_policy": (
                    "capability-conditioned anchored top-k coalitions; symmetric "
                    "normalized dissipative hypergraph diffusion; hard Gumbel "
                    "routing; GAT; RMSProp; Probe2 V1 uses a common binary-action "
                    "teacher-only objective, the same four-hop path in training and "
                    "evaluation, and all legal physical links forced active for both "
                    "methods to isolate same-decision propagation depth"
                ),
                "seed": args.seed,
                "epochs": cli.epochs,
                "episodes_per_epoch": cli.episodes_per_epoch,
                "backward_verified": backward_verified,
                "optimizer_step_verified": optimizer_step_verified,
                "parameters_finite": parameters_finite,
                "protocol_version": registry["protocol_version"],
                "scenario_registry_sha256": registry_sha256(registry),
                "probe_version": probe_rules.get("probe_version"),
                "registered_physical_scaffold_applied": is_interlock_probe,
                "probe2_training_support": probe_rules.get("training_support") if is_interlock_probe else None,
                "probe2_evaluation_support": probe_rules.get("evaluation_support") if is_interlock_probe else None,
                "probe2_force_physical_links": probe_rules.get("force_physical_links") if is_interlock_probe else None,
                "teacher_loss_coefficient": (
                    float(probe_rules.get("teacher_loss_coefficient", 0.0)) if is_interlock_probe else 0.0
                ),
                "teacher_only_training": (
                    bool(probe_rules.get("teacher_only_training", False)) if is_interlock_probe else False
                ),
                "learning_rate_decay": float(args.lr_decay),
                "gradient_clip_norm": float(args.max_grad_norm),
                "development_probe": cli.development_probe,
                "runtime": runtime,
                "source_sha256": source_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "evaluation_protocol.json").write_text(
        json.dumps(
            {
                "optimizer_updates": 0,
                "action_mode": "greedy" if is_interlock_probe else "native_sample",
                "eval_seeds": cli.eval_seeds,
                "episodes_per_seed": cli.eval_episodes,
                "evaluation_seed_rule": (
                    "seed once per evaluation block; episodes advance environment and "
                    "policy RNG state without repeated reseeding"
                ),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_load_verified": True,
                "parameters_unchanged": True,
                "representative_trace_recorded": cli.record_trace,
                "representative_trace_file": (
                    "episode_trace.jsonl" if cli.record_trace else None
                ),
                "protocol_version": registry["protocol_version"],
                "scenario_registry_sha256": registry_sha256(registry),
                "probe_version": probe_rules.get("probe_version"),
                "registered_physical_scaffold_applied": is_interlock_probe,
                "development_probe": cli.development_probe,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env.close()
    print(output)


if __name__ == "__main__":
    main()
