"""Frozen native-policy tests; no optimizer calls and seed-level outputs."""
import csv
import hashlib
import json
import random
import math
from pathlib import Path
import numpy as np
import torch
from operational_metrics import write_comparison_csv
from fair_costs import FairCosts


def evaluate(runner, agent, env, directory, args):
    directory = Path(directory)
    project = Path(__file__).resolve().parent
    source_hashes = {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in project.rglob('*.py')
                     if not set(p.relative_to(project).parts) & {'results', '.git', '__pycache__'}}
    (directory / 'source_manifest.json').write_text(json.dumps(source_hashes, indent=2))
    checkpoint = directory / 'model.pt'
    torch.save(agent.state_dict(), checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    loaded_state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    agent.load_state_dict(loaded_state)
    agent.eval()
    rows = []
    costs = FairCosts(agent, args.agent)
    env.collect_operational_metrics = True
    for seed in args.eval_seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.unwrapped.seed(seed)
        with torch.no_grad():
            for episode in range(args.eval_episodes_per_seed):
                _, log = runner.run_an_episode()
                rows.append(dict(Method=args.agent,
                    Scenario='lbf_' + args.map.removeprefix('Foraging-') + '-v0',
                    Train_Seed=args.seed, Eval_Seed=seed, Episode=episode + 1,
                    Episode_Return=float(np.sum(log['episode_return'])),
                    Episode_Steps=int(log['episode_steps'][0]),
                    Agent_Count=int(env.n_agents),
                    Success_Rate=log['success_rate'],
                    Task_Completion_Ratio=log['task_completion_ratio'],
                    Episode_Success=log['episode_success'],
                    Throughput=log['throughput'],
                    Workload_Total=log['workload_total'],
                    Workload_Completed=log['workload_completed'],
                    Capacity_Workload_Total=log['capacity_workload_total'],
                    Capacity_Workload_Completed=log['capacity_workload_completed'],
                    Capacity_Completion_Ratio=(
                        log['capacity_workload_completed'] / log['capacity_workload_total']
                        if log['capacity_workload_total'] else 0.0),
                    Episode_Success_Rate=log['episode_success'],
                    Collisions=log['num_collisions'],
                    Collision_Rate=(
                        log['num_collisions'] /
                        (int(log['episode_steps'][0]) * int(env.n_agents))
                        if int(log['episode_steps'][0]) and int(env.n_agents) else 0.0),
                    Active_Link_Ratio=log.get('comm_sparsity', 0.0)))
                rows[-1].update(env.operational_metrics.summary(env.unwrapped))
                costs.episode(rows[-1], int(rows[-1]['Episode_Steps']))
                for key, value in rows[-1].items():
                    if isinstance(value, (int, float, np.number)) and not math.isfinite(float(value)):
                        raise FloatingPointError(f"Non-finite evaluation metric {key}={value}")
    env.collect_operational_metrics = False
    costs.finish(directory)
    write_comparison_csv(directory, rows)
    with (directory/'test_metrics.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    (directory/'evaluation_protocol.json').write_text(json.dumps(dict(
        phase='frozen_test',optimizer_updates=0,train_seed=args.seed,
        eval_seeds=args.eval_seeds,episodes_per_seed=args.eval_episodes_per_seed,
        action_mode='native_sample',checkpoint_sha256=digest,checkpoint_load_verified=True,
        protocol_version=env.unwrapped.protocol_version,
        scenario_registry_sha256=env.unwrapped.registry_sha256,
        communication_metric='active directed link ratio over N*(N-1)'),indent=2))
