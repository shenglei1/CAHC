"""Genuine QMIX baseline using the shared TeamComm paper environment.

Execution is decentralized: each agent selects from its local observation.
The monotone mixer and centralized state are used only for training.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from envs.paper_wrapper import PaperWrapper


class AgentUtilities(nn.Module):
    def __init__(self, obs_dim, n_actions=6, width=64):
        super().__init__()
        self.encoder = nn.Linear(obs_dim, width)
        self.recurrent = nn.GRUCell(width, width)
        self.head = nn.Linear(width, n_actions)
        self.width = width

    def forward(self, observations, hidden):
        hidden = self.recurrent(torch.relu(self.encoder(observations)), hidden)
        return self.head(hidden), hidden


class MonotoneMixer(nn.Module):
    def __init__(self, n_agents, state_dim, width=32):
        super().__init__()
        self.n_agents = n_agents
        self.width = width
        self.weight_1 = nn.Linear(state_dim, n_agents * width)
        self.bias_1 = nn.Linear(state_dim, width)
        self.weight_2 = nn.Linear(state_dim, width)
        self.bias = nn.Sequential(nn.Linear(state_dim, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, agent_values, state):
        weight_1 = self.weight_1(state).abs().reshape(-1, self.n_agents, self.width)
        hidden = torch.nn.functional.elu(
            torch.bmm(agent_values.unsqueeze(1), weight_1) + self.bias_1(state).unsqueeze(1)
        )
        return (
            torch.bmm(hidden, self.weight_2(state).abs().unsqueeze(-1)).flatten()
            + self.bias(state).flatten()
        )


def rollout(env, agent, epsilon, rng):
    observations, state, _ = env.reset()
    observations = np.asarray(observations, dtype=np.float32)
    hidden = torch.zeros(env.n_agents, agent.width)
    transitions = []
    total_return = 0.0
    final_info = None
    for _ in range(env.episode_limit):
        with torch.no_grad():
            utilities, hidden = agent(torch.from_numpy(observations), hidden)
        actions = utilities.argmax(-1).cpu().numpy()
        explore = rng.random(env.n_agents) < epsilon
        actions = np.where(explore, rng.integers(0, 6, env.n_agents), actions)
        rewards, done, final_info = env.step([actions])
        next_observations = np.asarray(env.get_obs(), dtype=np.float32)
        next_state = env.get_state()
        transitions.append(
            (observations, state, actions.astype(np.int64), float(np.sum(rewards)), float(done))
        )
        total_return += float(np.sum(rewards))
        observations, state = next_observations, next_state
        if done:
            break
    transitions.append((observations, state, None, None, None))
    if final_info is None:
        raise RuntimeError("QMIX rollout produced no environment step")
    horizon = env.episode_limit
    metrics = {
        "Avg_Return": total_return,
        "Episode_Steps": len(transitions) - 1,
        "Agent_Count": int(env.n_agents),
        "Success_Rate": float(final_info["task_completion_ratio"]),
        "Task_Completion_Ratio": float(final_info["task_completion_ratio"]),
        "Episode_Success": float(final_info["episode_success"]),
        "Throughput": float(final_info["workload_completed"] / horizon),
        "Workload_Total": int(final_info["workload_total"]),
        "Workload_Completed": int(final_info["workload_completed"]),
        "Capacity_Workload_Total": int(final_info["capacity_workload_total"]),
        "Capacity_Workload_Completed": int(final_info["capacity_workload_completed"]),
        "Capacity_Completion_Ratio": (
            float(final_info["capacity_workload_completed"])
            / float(final_info["capacity_workload_total"])
            if final_info["capacity_workload_total"] else 0.0
        ),
        "Episode_Success_Rate": float(final_info["episode_success"]),
        "Collisions": int(final_info["num_collisions"]),
        "Collision_Rate": (
            float(final_info["num_collisions"])
            / float((len(transitions) - 1) * env.n_agents)
            if len(transitions) > 1 and env.n_agents else 0.0
        ),
        "Active_Link_Ratio": 0.0,
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise FloatingPointError(f"Non-finite QMIX rollout metric: {metrics}")
    return transitions, metrics


def update(batch, agent, mixer, target_agent, target_mixer, optimizer, gamma=0.99):
    batch_size = len(batch)
    horizon = len(batch[0]) - 1
    n_agents = batch[0][0][0].shape[0]
    observations = torch.tensor(
        np.stack([[row[0] for row in episode] for episode in batch]), dtype=torch.float32
    )
    states = torch.tensor(
        np.stack([[row[1] for row in episode] for episode in batch]), dtype=torch.float32
    )
    actions = torch.tensor(
        np.stack([[row[2] for row in episode[:-1]] for episode in batch]), dtype=torch.long
    )
    rewards = torch.tensor(
        [[row[3] for row in episode[:-1]] for episode in batch], dtype=torch.float32
    )
    dones = torch.tensor(
        [[row[4] for row in episode[:-1]] for episode in batch], dtype=torch.float32
    )
    hidden = torch.zeros(batch_size * n_agents, agent.width)
    target_hidden = torch.zeros_like(hidden)
    predictions, targets = [], []
    for step in range(horizon + 1):
        utilities, hidden = agent(
            observations[:, step].reshape(batch_size * n_agents, -1), hidden
        )
        utilities = utilities.reshape(batch_size, n_agents, 6)
        with torch.no_grad():
            target_q, target_hidden = target_agent(
                observations[:, step].reshape(batch_size * n_agents, -1), target_hidden
            )
            best_actions = utilities.detach().argmax(-1, keepdim=True)
            selected = target_q.reshape(batch_size, n_agents, 6).gather(
                -1, best_actions
            ).squeeze(-1)
            targets.append(target_mixer(selected, states[:, step]))
        if step < horizon:
            chosen = utilities.gather(-1, actions[:, step].unsqueeze(-1)).squeeze(-1)
            predictions.append(mixer(chosen, states[:, step]))
    prediction = torch.stack(predictions, dim=1)
    target = rewards + gamma * (1 - dones) * torch.stack(targets[1:], dim=1)
    loss = (prediction - target.detach()).square().mean()
    optimizer.zero_grad()
    loss.backward()
    parameters = list(agent.parameters()) + list(mixer.parameters())
    if not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in parameters
    ):
        raise FloatingPointError("QMIX produced a non-finite gradient")
    nn.utils.clip_grad_norm_(parameters, 10.0)
    optimizer.step()
    if not math.isfinite(float(loss.detach())):
        raise FloatingPointError("QMIX produced a non-finite loss")
    return float(loss.detach())


def build_components(scenario, seed, learning_rate=0.0005):
    env = PaperWrapper({"map": f"Foraging-{scenario}", "seed": seed})
    observations, state, _ = env.reset()
    agent = AgentUtilities(len(observations[0]))
    mixer = MonotoneMixer(env.n_agents, len(state))
    target_agent = copy.deepcopy(agent)
    target_mixer = copy.deepcopy(mixer)
    optimizer = torch.optim.RMSprop(
        list(agent.parameters()) + list(mixer.parameters()), lr=learning_rate, alpha=0.99, eps=1e-5
    )
    return env, agent, mixer, target_agent, target_mixer, optimizer


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument("--episodes-per-epoch", type=int, default=64)
    parser.add_argument("--replay-batch", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=list(range(1001, 1011)))
    parser.add_argument("--results-root", type=Path, default=Path("results/FORMAL_6SCENARIOS_V1"))
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent
    import sys
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    from scenario_registry import load_registry
    registry = load_registry(workspace / "scenario_registry.json")
    allowed_scenarios = (
        registry["formal_scenario_ids"] + registry["supplementary_scenario_ids"]
    )
    if args.scenario not in allowed_scenarios:
        raise ValueError(f"QMIX comparison scenario must be one of {allowed_scenarios}")
    if args.seed not in registry["formal_seeds"]:
        raise ValueError(f"Formal training seed must be one of {registry['formal_seeds']}")

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env, agent, mixer, target_agent, target_mixer, optimizer = build_components(
        args.scenario, args.seed, args.learning_rate
    )
    initial_parameters = [
        parameter.detach().cpu().clone()
        for parameter in list(agent.parameters()) + list(mixer.parameters())
    ]
    replay = deque(maxlen=5000)
    seed_root = args.results_root / "qmix" / f"lbf_{args.scenario}-v0" / f"seed_{args.seed}"
    run_id = 1
    while (seed_root / f"run{run_id}").exists():
        run_id += 1
    output = seed_root / f"run{run_id}"
    output.mkdir(parents=True)
    resolved_config = {**vars(args), 'lr': args.learning_rate, 'optim_alpha': 0.99,
                       'optim_eps': 1e-5, 'gamma': 0.99, 'buffer_size': 5000,
                       'target_update_episodes': 200, 'epsilon_anneal_steps': 50000,
                       'update_cadence': 'one per collected episode once replay batch is available',
                       'implementation': 'local monotone QMIX reimplementation; not bitwise PyMARL'}
    (output / "config.json").write_text(json.dumps(resolved_config, indent=2, default=str))
    training_rows = []
    episodes = 0
    environment_steps = 0
    optimizer_updates = 0
    for epoch in range(1, args.epochs + 1):
        epoch_metrics = []
        epoch_losses = []
        for _ in range(args.episodes_per_epoch):
            epsilon = max(0.05, 1 - 0.95 * environment_steps / 50000)
            episode, metrics = rollout(env, agent, epsilon, rng)
            replay.append(episode)
            epoch_metrics.append(metrics)
            episodes += 1
            environment_steps += len(episode) - 1
            if len(replay) >= args.replay_batch:
                batch = random.sample(list(replay), args.replay_batch)
                epoch_losses.append(update(batch, agent, mixer, target_agent, target_mixer, optimizer))
                optimizer_updates += 1
            if episodes % 200 == 0:
                target_agent.load_state_dict(agent.state_dict())
                target_mixer.load_state_dict(mixer.state_dict())
        loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        averaged = {
            key: float(np.mean([metrics[key] for metrics in epoch_metrics]))
            for key in epoch_metrics[0]
        }
        training_rows.append({"Epoch": epoch, "Loss": loss, "Episodes": episodes,
                              "Environment_Steps": environment_steps,
                              "Optimizer_Updates": optimizer_updates, **averaged})
        write_csv(output / "metrics.csv", training_rows)

    trained_parameters = [
        parameter.detach().cpu()
        for parameter in list(agent.parameters()) + list(mixer.parameters())
    ]
    optimizer_step_verified = any(
        not torch.equal(before, after)
        for before, after in zip(initial_parameters, trained_parameters)
    )
    backward_verified = any(
        parameter.grad is not None
        for parameter in list(agent.parameters()) + list(mixer.parameters())
    ) and all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in list(agent.parameters()) + list(mixer.parameters())
    )
    parameters_finite = all(
        torch.isfinite(parameter).all().item()
        for parameter in list(agent.parameters()) + list(mixer.parameters())
    )
    if not (optimizer_step_verified and backward_verified and parameters_finite):
        raise RuntimeError(
            f"QMIX training verification failed: optimizer={optimizer_step_verified}, "
            f"backward={backward_verified}, finite={parameters_finite}"
        )
    (output / "training_protocol.json").write_text(
        json.dumps(
            {
                "method": "qmix",
                "scenario": args.scenario,
                "seed": args.seed,
                "optimizer_updates": optimizer_updates,
                "backward_verified": backward_verified,
                "optimizer_step_verified": optimizer_step_verified,
                "parameters_finite": parameters_finite,
                "workload_metric_source": "shared environment episode ledger",
                "protocol_version": env.unwrapped.protocol_version,
                "scenario_registry_sha256": env.unwrapped.registry_sha256,
                "algorithm_policy": "standalone QMIX reference implementation; no HyperWave-X mechanism",
            },
            indent=2,
        )
    )

    checkpoint = output / "model.pt"
    torch.save({"agent": agent.state_dict(), "mixer": mixer.state_dict()}, checkpoint)
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    project = Path(__file__).resolve().parent
    (output / 'source_manifest.json').write_text(json.dumps({str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest() for p in project.rglob('*.py') if not set(p.relative_to(project).parts) & {'results', '.git', '__pycache__'}}, indent=2))
    agent.load_state_dict(checkpoint_state["agent"])
    mixer.load_state_dict(checkpoint_state["mixer"])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    agent.eval()
    evaluation_rows = []
    from fair_costs import FairCosts
    costs = FairCosts(agent, 'qmix')
    env.collect_operational_metrics = True
    for seed in args.eval_seeds:
        env.unwrapped.seed(seed)
        eval_rng = np.random.default_rng(seed)
        for episode_index in range(1, args.eval_episodes + 1):
            _, metrics = rollout(env, agent, 0.0, eval_rng)
            metrics.update(env.operational_metrics.summary(env.unwrapped))
            metrics['Episode_Return'] = metrics.pop('Avg_Return')
            evaluation_rows.append(
                {
                    "Method": "qmix",
                    "Scenario": f"lbf_{args.scenario}-v0",
                    "Train_Seed": args.seed,
                    "Eval_Seed": seed,
                    "Episode": episode_index,
                    **metrics,
                }
            )
            costs.episode(evaluation_rows[-1], int(metrics['Episode_Steps']))
    costs.finish(output)
    write_csv(output / "test_metrics.csv", evaluation_rows)
    from operational_metrics import write_comparison_csv
    write_comparison_csv(output, evaluation_rows)
    (output / "evaluation_protocol.json").write_text(
        json.dumps(
            {
                "optimizer_updates": 0,
                "action_mode": "greedy",
                "eval_seeds": args.eval_seeds,
                "episodes_per_seed": args.eval_episodes,
                "checkpoint_sha256": digest,
                "checkpoint_load_verified": True,
                "protocol_version": env.unwrapped.protocol_version,
                "scenario_registry_sha256": env.unwrapped.registry_sha256,
            },
            indent=2,
        )
    )
    print(output)


if __name__ == "__main__":
    main()
