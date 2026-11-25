"""Formal HyperComm training/evaluation entry for the frozen CAIE protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from operational_metrics import OperationalMetrics, write_comparison_csv
from fair_costs import FairCosts
from probe2_interlock import evaluation_columns

import numpy as np
import torch

import data
from action_utils import parse_action_args, select_action, translate_action
from HyperComm import HyperComm
from trainer import Trainer


PROJECT = Path(__file__).resolve().parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scenario_registry import load_registry, registry_sha256  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    # Paper-facing CSV files contain identifiers and measured quantities only.
    # Execution/provenance metadata lives in the adjacent JSON protocol files.
    excluded = {
        "Evaluation_Phase", "Action_Mode", "Checkpoint_SHA256",
        "Latency_Isolated", "Metrics_Schema_Version",
        "Mean_Evaluation_Step_ms", "Training_Entropy_Coefficient",
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


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native HyperComm under the frozen CAIE protocol.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument("--episodes-per-epoch", type=int, default=64)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=list(range(1001, 1011)))
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--allow-probe", action="store_true")
    parser.add_argument("--development-probe", action="store_true",
                        help="Probe2-only development run; never evidence for the final claim.")
    parser.add_argument(
        "--record-trace", action="store_true",
        help="Persist step-level interpretability JSONL for selected representative runs.",
    )
    parser.add_argument("--results-root", type=Path, default=PROJECT / "results/FORMAL_6SCENARIOS")
    return parser.parse_args()


def build_args(cli: argparse.Namespace, registry: dict) -> argparse.Namespace:
    scenario = registry["scenarios"][cli.scenario]
    args = argparse.Namespace(
        env_name=f"lbf_{cli.scenario}-v0",
        seed=cli.seed,
        gpu=cli.gpu,
        display=False,
        random=False,
        nprocesses=1,
        num_epochs=cli.epochs,
        epoch_size=1,
        batch_size=cli.episodes_per_epoch * int(scenario["horizon"]),
        max_steps=int(scenario["horizon"]),
        hid_size=64,
        qk_hid_size=64,
        directed=True,
        self_loop_type1=2,
        self_loop_type2=2,
        gat_num_heads=2,
        gat_num_heads_out=1,
        gat_hid_size=32,
        ge_num_heads=4,
        first_gat_normalize=True,
        second_gat_normalize=True,
        gat_encoder_normalize=False,
        use_gat_encoder=False,
        gat_encoder_out_size=64,
        first_graph_complete=False,
        second_graph_complete=False,
        learn_second_graph=False,
        message_encoder=False,
        message_decoder=True,
        nagents=int(scenario["agents"]["count"]),
        nfriendly=int(scenario["agents"]["count"]),
        nenemies=0,
        enemy_comm=False,
        mean_ratio=0.0,
        detach_gap=10,
        comm_init="uniform",
        advantages_per_action=False,
        comm_mask_zero=False,
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
        entropy_loss_weight=0.0,
        teacher_loss_coefficient=float(
            scenario.get("probe_rules", {}).get("teacher_loss_coefficient", 0.0)
        ),
        teacher_only_training=bool(
            scenario.get("probe_rules", {}).get("teacher_only_training", False)
        ),
        nactions="1",
        action_scale=1.0,
    )
    return args


def attach_external_diagnostic_hooks(policy: HyperComm) -> tuple[dict, list]:
    """Observe native intermediate tensors without editing the algorithm path."""
    diagnostics: dict[str, np.ndarray] = {}

    def capture_hypergraph(_module, _inputs, output) -> None:
        diagnostics["incidence"] = output[1].detach().cpu().numpy().copy()

    def capture_distribution(_module, inputs, _output) -> None:
        diagnostics["topology"] = inputs[1].detach().cpu().numpy().copy()

    def capture_policy(module, _inputs, _output) -> None:
        before = getattr(module, "last_adjacency_before_dropout", None)
        if before is not None:
            diagnostics["topology_before_dropout"] = (
                before.detach().cpu().numpy().copy()
            )

    handles = [
        policy.Hypergraph_Encoder.register_forward_hook(capture_hypergraph),
        policy.distributer.register_forward_hook(capture_distribution),
        policy.register_forward_hook(capture_policy),
    ]
    return diagnostics, handles


def initialize(args: argparse.Namespace) -> tuple:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    env = data.init(args.env_name, args, False)
    args.obs_size = env.observation_dim
    args.num_actions = [env.num_actions]
    args.dim_actions = env.dim_actions
    args.nagents = env.n_agents
    args.nfriendly = args.nagents
    parse_action_args(args)
    policy = HyperComm(args).double()
    diagnostics, diagnostic_handles = attach_external_diagnostic_hooks(policy)
    trainer = Trainer(args, policy, env)
    return env, policy, trainer, diagnostics, diagnostic_handles


def normalize_actions(actual) -> list[int]:
    if isinstance(actual, (list, tuple)) and len(actual) == 1:
        actual = actual[0]
    return np.asarray(actual, dtype=np.int64).reshape(-1).tolist()


def diagnostic_snapshot(diagnostics: dict, env, actions: list[int], info: dict,
                        previous_completed: int, step: int) -> dict:
    raw = env.env
    missing = {"incidence", "topology"} - diagnostics.keys()
    if missing:
        raise RuntimeError(f"HyperComm diagnostic hooks did not capture: {sorted(missing)}")
    incidence = np.asarray(diagnostics["incidence"])
    topology = np.asarray(diagnostics["topology"])
    topology_before_dropout = np.asarray(
        diagnostics.get("topology_before_dropout", topology)
    )
    if incidence.ndim == 3 and incidence.shape[0] == 1:
        incidence = incidence[0]
    if incidence.ndim != 2 or incidence.shape != (raw.n_agents, raw.n_agents):
        raise ValueError(f"Unexpected HyperComm incidence shape: {incidence.shape}")
    if topology.ndim == 3 and topology.shape[-1] == 1:
        topology = topology[..., 0]
    if topology.shape != (raw.n_agents, raw.n_agents):
        raise ValueError(f"Unexpected HyperComm topology shape: {topology.shape}")
    if topology_before_dropout.ndim == 3 and topology_before_dropout.shape[-1] == 1:
        topology_before_dropout = topology_before_dropout[..., 0]
    if topology_before_dropout.shape != (raw.n_agents, raw.n_agents):
        raise ValueError(
            "Unexpected HyperComm pre-dropout topology shape: "
            f"{topology_before_dropout.shape}"
        )

    levels = np.asarray(raw.fixed_levels, dtype=np.int64)
    coalitions = []
    threshold_floor = min(raw.spec["tasks"]["requirements"])
    # Native HyperComm constructs H as edge-by-node: node2edge uses H @ X.
    # Each row is therefore one generated hyperedge and its selected members.
    for edge in range(incidence.shape[0]):
        members = np.flatnonzero(incidence[edge, :] > 0.5).tolist()
        coalitions.append({
            "hyperedge": edge,
            "members": members,
            "capability_sum": int(levels[members].sum()) if members else 0,
            "candidate_threshold_feasible": bool(
                members and levels[members].sum() >= threshold_floor
            ),
        })

    mask = ~np.eye(raw.n_agents, dtype=bool)
    active_links = int(np.count_nonzero(topology[mask] > 0.5))
    decoded_active_links = int(
        np.count_nonzero(topology_before_dropout[mask] > 0.5)
    )
    possible_links = int(
        info.get(
            "probe2_communication_scaffold_directed_links",
            raw.n_agents * (raw.n_agents - 1),
        )
    )
    completed = int(info["workload_completed"])
    load_count = int(sum(action == getattr(raw, "load_action", 5) for action in actions))
    exact_pairs = {
        frozenset((0, 7)), frozenset((1, 6)),
        frozenset((2, 5)), frozenset((3, 4)),
    }
    selected_pairs = {
        frozenset(item["members"])
        for item in coalitions if len(item["members"]) == 2
    }
    return {
        "step": step,
        "actions": actions,
        "agent_positions": [list(player.position) for player in raw.players],
        "coalitions": coalitions,
        "candidate_feasible_fraction": float(np.mean([
            item["candidate_threshold_feasible"] for item in coalitions
        ])),
        "probe1_exact_pair_coverage": float(
            len(exact_pairs & selected_pairs) / len(exact_pairs)
            if raw.type == "probe_complementary" else 0.0
        ),
        "active_directed_links": active_links,
        "decoded_active_directed_links": decoded_active_links,
        "dropped_active_directed_links": decoded_active_links - active_links,
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
    }


def evaluate_episode(policy: HyperComm, env, args: argparse.Namespace,
                     diagnostics: dict,
                     eval_seed: int, episode: int,
                     link_dropout_rate: float | None = None) -> tuple[dict, list[dict]]:
    if link_dropout_rate is not None and not 0.0 <= link_dropout_rate <= 1.0:
        raise ValueError("Evaluation link-dropout rate must be in [0, 1].")
    if episode == 1:
        random.seed(eval_seed)
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
        env.seed(eval_seed)
    if hasattr(env.env, "set_evaluation_mode"):
        env.env.set_evaluation_mode(True)
    state = env.reset().to(policy.device)
    hidden = policy.init_hidden(batch_size=state.shape[0])
    info = env.initial_info() if hasattr(env, "initial_info") else {}
    total_return = 0.0
    trace = []
    previous_completed = 0
    operational = OperationalMetrics(env.env)
    dropout_rng = (
        np.random.default_rng(
            np.random.SeedSequence([int(eval_seed), int(episode), 0x485758])
        )
        if link_dropout_rate is not None
        else None
    )
    with torch.no_grad():
        for step in range(1, args.max_steps + 1):
            if dropout_rng is not None:
                dropout_uniform = dropout_rng.random((args.nagents, args.nagents))
                info["evaluation_link_keep_mask"] = (
                    dropout_uniform >= link_dropout_rate
                )
            else:
                info.pop("evaluation_link_keep_mask", None)
            action_out, _, hidden = policy([state, hidden], env.policy_info(info))
            if getattr(env.env, "is_probe2", False):
                sampled = torch.stack(
                    [head.argmax(dim=-1, keepdim=True)[0] for head in action_out]
                )
            else:
                sampled = select_action(args, action_out)
            _, actual = translate_action(args, env, sampled)
            actions = normalize_actions(actual)
            operational.before_step(env.env, actions)
            state, reward, done, info = env.step(actual)
            operational.after_step(env.env)
            total_return += float(np.asarray(reward, dtype=float).sum())
            snapshot = diagnostic_snapshot(
                diagnostics, env, actions, info, previous_completed, step
            )
            trace.append(snapshot)
            previous_completed = int(info["workload_completed"])
            if done:
                break
    final = trace[-1]
    sync_attempts = sum(item["load_action_count"] > 0 for item in trace)
    false_starts = sum(item["false_load_attempt"] for item in trace)
    decoded_links = sum(item["decoded_active_directed_links"] for item in trace)
    dropped_links = sum(item["dropped_active_directed_links"] for item in trace)
    row = {
        "Method": "hypercomm",
        "Scenario": args.env_name,
        "Train_Seed": args.seed,
        "Eval_Seed": eval_seed,
        "Episode": episode,
        "Episode_Return": total_return,
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
        # At episode level ESR is the binary success indicator. Its mean over
        # frozen-test episodes is the paper-facing Episode Success Rate.
        "Episode_Success_Rate": final["episode_success"],
        "Throughput": final["workload_completed"] / args.max_steps,
        "Mean_Candidate_Feasible_Fraction": float(np.mean([
            item["candidate_feasible_fraction"] for item in trace
        ])),
        "Mean_Probe1_Exact_Pair_Coverage": float(np.mean([
            item["probe1_exact_pair_coverage"] for item in trace
        ])),
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
        "Mean_Active_Link_Ratio": float(np.mean([
            item["active_link_ratio"] for item in trace
        ])),
        "Synchronized_Load_Successes": sum(
            item["task_completion_delta"] > 0 for item in trace
        ),
        "Load_Attempts": sync_attempts,
        "False_Load_Attempts": false_starts,
        "Missing_Member_Rate": false_starts / sync_attempts if sync_attempts else 0.0,
        "Collisions": int(info['num_collisions']),
        "Collision_Rate": (
            int(info["num_collisions"]) / (len(trace) * int(args.nagents))
            if trace and int(args.nagents) else 0.0
        ),
        "Training_Entropy_Coefficient": float(args.entr),
    }
    if link_dropout_rate is not None:
        row.update({
            "Requested_Link_Dropout_Rate": float(link_dropout_rate),
            "Realized_Active_Link_Dropout_Rate": (
                dropped_links / decoded_links if decoded_links else 0.0
            ),
        })
    row.update(operational.summary(env.env))
    row.update(evaluation_columns(env.env, info))
    if not all(
        math.isfinite(float(value))
        for value in row.values() if isinstance(value, (int, float))
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
    if cli.learning_rate is not None and (
        not math.isfinite(cli.learning_rate) or cli.learning_rate <= 0
    ):
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

    torch.set_default_dtype(torch.float64)
    args = build_args(cli, registry)
    env, policy, trainer, diagnostics, diagnostic_handles = initialize(args)
    seed_root = cli.results_root.resolve() / "hypercomm" / args.env_name / f"seed_{args.seed}"
    run_number = 1
    while (seed_root / f"run{run_number}").exists():
        run_number += 1
    output = seed_root / f"run{run_number}"
    output.mkdir(parents=True)

    config = vars(args).copy()
    config.update({
        "formal_protocol": registry["protocol_version"],
        "scenario_registry_sha256": registry_sha256(registry),
        "project_root": str(PROJECT),
        "python_executable": sys.executable,
        "trainer_source": str(Path(sys.modules[Trainer.__module__].__file__).resolve()),
    })
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    initial = [parameter.detach().cpu().clone() for parameter in policy.parameters()]
    training_rows = []
    for epoch in range(1, cli.epochs + 1):
        stat = trainer.train_batch(epoch - 1)
        training_rows.append({
            "Epoch": epoch,
            "Episodes": int(stat["num_episodes"]),
            "Steps": int(stat["num_steps"]),
            "Episode_Return": float(np.asarray(stat["reward"]).sum() / stat["num_episodes"]),
            "Action_Loss": float(stat["action_loss"]),
            "Value_Loss": float(stat["value_loss"]),
            "Entropy": float(stat["entropy"]),
            "Teacher_Loss": float(stat.get("teacher_loss", 0.0)),
            "Teacher_Accuracy": float(stat.get("teacher_accuracy", 0.0)),
            "Gradient_Norm_Before_Clip": float(
                stat.get("gradient_norm_before_clip", 0.0)
            ),
            "Learning_Rate": float(stat.get("learning_rate", args.lrate)),
        })
    write_csv(output / "training_metrics.csv", training_rows)

    final_parameters = [parameter.detach().cpu() for parameter in policy.parameters()]
    optimizer_step_verified = any(
        not torch.equal(before, after)
        for before, after in zip(initial, final_parameters)
    )
    backward_verified = any(parameter.grad is not None for parameter in policy.parameters())
    parameters_finite = all(torch.isfinite(parameter).all().item() for parameter in policy.parameters())
    if not (optimizer_step_verified and backward_verified and parameters_finite):
        raise RuntimeError("HyperComm backward/optimizer/finite-parameter verification failed.")

    checkpoint = output / "model.pt"
    torch.save({
        "policy_net": policy.state_dict(),
        "optimizer": trainer.state_dict(),
    }, checkpoint)
    checkpoint_data = torch.load(checkpoint, map_location=policy.device, weights_only=False)
    policy.load_state_dict(checkpoint_data["policy_net"])
    trainer.load_state_dict(checkpoint_data["optimizer"])
    checkpoint_hash = file_sha256(checkpoint)

    policy.eval()
    before_eval = [parameter.detach().cpu().clone() for parameter in policy.parameters()]
    evaluation_rows = []
    costs = FairCosts(policy, 'hypercomm')
    trace_stream = None
    if cli.record_trace:
        trace_stream = (output / "episode_trace.jsonl").open("w", encoding="utf-8")
    try:
        for eval_seed in cli.eval_seeds:
            for episode in range(1, cli.eval_episodes + 1):
                row, trace = evaluate_episode(
                    policy, env, args, diagnostics, eval_seed, episode
                )
                costs.episode(row, row['Episode_Steps'])
                evaluation_rows.append(row)
                if (
                    trace_stream is not None
                    and eval_seed == cli.eval_seeds[0]
                    and episode == 1
                ):
                    trace_stream.write(json.dumps({
                        "eval_seed": eval_seed,
                        "episode": episode,
                        "steps": trace,
                    }) + "\n")
    finally:
        if trace_stream is not None:
            trace_stream.close()
    write_csv(output / "test_metrics.csv", evaluation_rows)
    costs.finish(output)
    write_comparison_csv(output, evaluation_rows)
    after_eval = [parameter.detach().cpu() for parameter in policy.parameters()]
    if any(not torch.equal(before, after) for before, after in zip(before_eval, after_eval)):
        raise RuntimeError("Frozen evaluation modified HyperComm parameters.")

    source_manifest = {
        name: file_sha256(PROJECT / name)
        for name in [
            "HyperComm.py", "gnn_layers.py", "trainer.py", "data.py",
            "env_wrappers.py", "scenario_registry.py", "scenario_registry.json",
            "shared_lbf_env.py", "formal_hypercomm.py", "operational_metrics.py",
            "fair_costs.py",
            "probe2_interlock.py",
        ]
    }
    probe_rules = registry["scenarios"][cli.scenario].get("probe_rules", {})
    is_interlock_probe = probe_rules.get("probe_version") == "V1"
    (output / "training_protocol.json").write_text(json.dumps({
        "method": "hypercomm",
        "algorithm_policy": "native HyperComm hypergraph encoder and GAT with RMSProp; Probe2 V1 uses the common teacher-only binary target, the same four-hop path in training and evaluation, and all legal physical links forced active for both methods to isolate same-decision propagation depth",
        "seed": args.seed,
        "epochs": cli.epochs,
        "episodes_per_epoch": cli.episodes_per_epoch,
        "transition_batch_size": args.batch_size,
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
        "source_sha256": source_manifest,
    }, indent=2), encoding="utf-8")
    (output / "evaluation_protocol.json").write_text(json.dumps({
        "optimizer_updates": 0,
        "action_mode": "greedy" if is_interlock_probe else "native_sample",
        "eval_seeds": cli.eval_seeds,
        "episodes_per_seed": cli.eval_episodes,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_load_verified": True,
        "parameters_unchanged": True,
        "representative_trace_recorded": cli.record_trace,
        "representative_trace_file": "episode_trace.jsonl" if cli.record_trace else None,
        "representative_trace_selection": {
            "eval_seed": cli.eval_seeds[0], "episode": 1
        } if cli.record_trace else None,
        "trace_fields": [
            "coalition members/capability", "candidate feasibility",
            "active communication links", "LOAD attempts", "task completion process",
        ],
        "protocol_version": registry["protocol_version"],
        "scenario_registry_sha256": registry_sha256(registry),
        "probe_version": probe_rules.get("probe_version"),
        "registered_physical_scaffold_applied": is_interlock_probe,
        "development_probe": cli.development_probe,
    }, indent=2), encoding="utf-8")
    for handle in diagnostic_handles:
        handle.remove()
    print(output)


if __name__ == "__main__":
    main()
