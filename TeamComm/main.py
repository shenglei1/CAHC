import os
import numpy as np
import random
import time
import csv
import json
import math
import argparse
import sys
import torch
import signal
from os.path import dirname, abspath
from envs import REGISTRY as env_REGISTRY
from baselines import REGISTRY as agent_REGISTRY
from runner import REGISTRY as runner_REGISTRY
from configs.utils import get_config, recursive_dict_update, signal_handler, merge_dict

from pathlib import Path
import warnings
os.environ['GYM_NO_DEPRECATION_WARNING'] = '1'
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def main(args):
    default_config = get_config('experiment')
    env_config = get_config(args.env, 'envs')
    agent_config = get_config(args.agent, 'agents')
    if args.paper_protocol and args.agent == 'magic':
        # Official MAGIC default: enable its learned scheduler, not the
        # complete-graph ablation stored in the inherited local YAML.
        agent_config['first_graph_complete'] = False
        agent_config['second_graph_complete'] = False
    if args.paper_protocol and args.agent == 'ic3net':
        # Keep model and runner configuration consistent with learned gating.
        agent_config['comm_action_one'] = False

    if args.seed == None:
        args.seed = np.random.randint(0, 10000)
    if args.agent == 'tiecomm':
        args.block = 'no'

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    env_config['seed'] = args.seed

    # update configs
    exp_config = recursive_dict_update(default_config, vars(args))
    env_config = recursive_dict_update(env_config, vars(args))
    agent_config = recursive_dict_update(agent_config, vars(args))

    # merge config
    config = {}
    config.update(default_config)
    config.update(env_config)
    config.update(agent_config)

    use_cuda = default_config.get('use_cuda', False)
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    # 更新所有配置
    default_config['device'] = device
    env_config['device'] = device
    agent_config['device'] = device

    exp_config['device'] = device
    env_config['device'] = device
    agent_config['device'] = device

    args = argparse.Namespace(**config)
    results_path = os.path.join(dirname(abspath(__file__)), "results")
    args.exp_id = f"{args.map}_{args.agent}_{args.block}_seed{args.seed}"
    save_dir = Path(results_path) / (("major_revision_v1_" if args.paper_protocol else "") + args.exp_id)
    if args.paper_protocol:
        scenario = 'lbf_' + args.map.removeprefix('Foraging-') + '-v0'
        seed_root = Path(args.paper_results_root) / args.agent / scenario / f'seed_{args.seed}'
        run_id = 1
        while (seed_root / f'run{run_id}').exists():
            run_id += 1
        save_dir = seed_root / f'run{run_id}'
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'config.json').write_text(json.dumps(vars(args), indent=2, default=str))

    # CSV表头（通用+算法特有）
    csv_header = [
        'epoch', 'total_episodes', 'total_steps',
        'avg_episode_return', 'avg_episode_steps',
        'action_loss', 'value_loss', 'total_loss',
        'task_success_rate', 'episode_success', 'throughput_per_step',
        'workload_total', 'workload_completed',
        'capacity_workload_total', 'capacity_workload_completed',
        'avg_collisions_per_episode', 'avg_comm_sparsity'
    ]

    if args.agent == 'tiecomm':
        csv_header.extend(['god_action_loss', 'god_value_loss', 'god_total_loss', 'avg_num_groups'])
    elif args.agent in ['teamcomm']:
        csv_header.extend(
            ['team_action_loss', 'team_value_loss', 'team_total_loss', 'avg_num_groups', 'vib_loss', 'moduarity_loss'])

    csv_metrics = []
    csv_path = save_dir / f"{args.agent}_{args.map}_metrics.csv"
    if not args.paper_protocol:
        raise ValueError("Use --paper_protocol: legacy environment definitions are archived for audit.")
    workspace_root = Path(__file__).resolve().parent
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from scenario_registry import load_registry
    registry = load_registry(workspace_root / 'scenario_registry.json')
    scenario_id = args.map.removeprefix('Foraging-')
    allowed_scenarios = (
        registry['formal_scenario_ids'] + registry['supplementary_scenario_ids']
    )
    if scenario_id not in allowed_scenarios:
        raise ValueError(
            f"TeamComm paper run requires one of "
            f"{allowed_scenarios}; got {scenario_id}. "
            "Reviewer probes use their separate protocol."
        )
    if args.seed not in registry['formal_seeds']:
        raise ValueError(f"Formal training seed must be one of {registry['formal_seeds']}")
    env = env_REGISTRY[args.env](env_config)
    env_info = env.get_env_info()
    agent_config['obs_shape'] = env_info["obs_shape"]
    agent_config['n_actions'] = env_info["n_actions"]
    agent_config['n_agents'] = env_info["n_agents"]
    exp_config['episode_length'] = env_info["episode_length"]
    exp_config['n_agents'] = env_info["n_agents"]
    exp_config['n_actions'] = env_info["n_actions"]

    agent = agent_REGISTRY[args.agent](agent_config)
    initial_parameters = [parameter.detach().cpu().clone() for parameter in agent.parameters()]

    if args.agent == 'ic3net':
        exp_config['hard_attn'] = True
        exp_config['commnet'] = True
        exp_config['detach_gap'] = 10
        exp_config['comm_action_one'] = False
    elif args.agent == 'commnet':
        exp_config['hard_attn'] = False
        exp_config['commnet'] = True
        exp_config['detach_gap'] = 10
    elif args.agent == 'tarmac':
        exp_config['hard_attn'] = False
        exp_config['commnet'] = True
        exp_config['detach_gap'] = 10
    elif args.agent == 'magic':
        exp_config['hard_attn'] = False
        exp_config['hid_size'] = 64
        exp_config['detach_gap'] = 10
    elif args.agent == 'tiecomm':
        exp_config['interval'] = agent_config['time_interval']
    else:
        pass

    epoch_size = 1 if args.paper_protocol else exp_config['epoch_size']
    batch_size = exp_config['batch_size']
    run = runner_REGISTRY[args.agent]
    runner = run(exp_config, env, agent)

    total_num_episodes = 0
    total_num_steps = 0

    print(f"start train：{args.agent} on {args.map} (seed={args.seed})")
    print(f"save：{csv_path.resolve()}")
    print("=" * 80)

    for epoch in range(1, args.total_epoches + 1):
        epoch_begin_time = time.time()
        log = {}

        for i in range(epoch_size):
            batch_log = runner.train_batch(batch_size)
            merge_dict(batch_log, log)

        total_num_episodes += log['num_episodes']
        total_num_steps += log['num_steps']
        epoch_time = time.time() - epoch_begin_time
        num_eps = log['num_episodes'] if log['num_episodes'] > 0 else 1.0
        avg_return = log['episode_return'] / num_eps
        avg_success_rate = log.get('success_rate', 0.0) / num_eps
        avg_throughput = log.get('throughput', 0.0) / num_eps
        avg_collisions = log.get('num_collisions', 0.0) / num_eps
        avg_comm_sparsity = log.get('comm_sparsity', 0.0) / num_eps
        avg_episode_steps = np.mean(log['episode_steps'])
        csv_row = {
            'epoch': epoch,
            'total_episodes': total_num_episodes,
            'total_steps': total_num_steps,
            'avg_episode_return': avg_return,
            'avg_episode_steps': avg_episode_steps,
            'action_loss': log['action_loss'],
            'value_loss': log['value_loss'],
            'total_loss': log['total_loss'],
            'task_success_rate': avg_success_rate,
            'episode_success': log.get('episode_success', 0.0) / num_eps,
            'throughput_per_step': avg_throughput,
            'workload_total': log.get('workload_total', 0.0) / num_eps,
            'workload_completed': log.get('workload_completed', 0.0) / num_eps,
            'capacity_workload_total': log.get('capacity_workload_total', 0.0) / num_eps,
            'capacity_workload_completed': log.get('capacity_workload_completed', 0.0) / num_eps,
            'avg_collisions_per_episode': avg_collisions,
            'avg_comm_sparsity': avg_comm_sparsity
        }

        if args.agent == 'tiecomm':
            csv_row['god_action_loss'] = log.get('god_action_loss', 0.0)
            csv_row['god_value_loss'] = log.get('god_value_loss', 0.0)
            csv_row['god_total_loss'] = log.get('god_total_loss', 0.0)
            csv_row['avg_num_groups'] = log.get('num_groups', 0.0) / num_eps
        elif args.agent in ['teamcomm']:
            csv_row['team_action_loss'] = log.get('team_action_loss', 0.0)
            csv_row['team_value_loss'] = log.get('team_value_loss', 0.0)
            csv_row['team_total_loss'] = log.get('team_total_loss', 0.0)
            csv_row['avg_num_groups'] = log.get('num_groups', 0.0) / num_eps
            csv_row['vib_loss'] = log.get('vib_loss', 0.0)
            csv_row['moduarity_loss'] = log.get('moduarity_loss', 0.0)

        csv_metrics.append(csv_row)
        for key, value in csv_row.items():
            if isinstance(value, (int, float, np.number)) and not math.isfinite(float(value)):
                raise FloatingPointError(f"Non-finite metric {key}={value}")

        print(f"Epoch {epoch:4d}/{args.total_epoches} | Time: {epoch_time:6.2f}s | Episodes: {total_num_episodes:5d}")
        print(f"  Return: {avg_return:8.2f} | Success: {avg_success_rate:6.4f} | Throughput: {avg_throughput:8.6f}")
        print(
            f"  Collisions: {avg_collisions:6.2f} | Comm Sparsity: {avg_comm_sparsity:6.4f}")
        print("-" * 80)

        if epoch % 1 == 0:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_header)
                writer.writeheader()
                writer.writerows(csv_metrics)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(csv_metrics)

    print("=" * 80)
    print(f"train have done！save：{csv_path.resolve()}")
    print(f"train time：{(time.time() - training_begin_time) / 3600:.2f}h")
    print('current epoch: {}/{}'.format(epoch, args.total_epoches))
    final_parameters = [parameter.detach().cpu() for parameter in agent.parameters()]
    optimizer_step_verified = any(
        not torch.equal(before, after)
        for before, after in zip(initial_parameters, final_parameters)
    )
    backward_verified = any(
        parameter.grad is not None for parameter in agent.parameters()
    ) and all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in agent.parameters()
    )
    parameters_finite = all(
        torch.isfinite(parameter).all().item() for parameter in agent.parameters()
    )
    if not (optimizer_step_verified and backward_verified and parameters_finite):
        raise RuntimeError(
            f"Training verification failed: optimizer={optimizer_step_verified}, "
            f"backward={backward_verified}, finite={parameters_finite}"
        )
    (save_dir / 'training_protocol.json').write_text(json.dumps({
        'method': args.agent,
        'scenario': scenario_id,
        'seed': args.seed,
        'optimizer_updates': args.total_epoches * epoch_size,
        'backward_verified': backward_verified,
        'optimizer_step_verified': optimizer_step_verified,
        'parameters_finite': parameters_finite,
        'workload_metric_source': 'shared environment episode ledger',
        'protocol_version': env.unwrapped.protocol_version,
        'scenario_registry_sha256': env.unwrapped.registry_sha256,
        'algorithm_policy': 'original TeamComm model, communication, loss, optimizer, and native action selection preserved',
    }, indent=2))
    if args.paper_protocol:
        from paper_evaluation import evaluate
        evaluate(runner, agent, env, save_dir, args)
    print("=====Done!!!=====")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HierComm')

    parser.add_argument("--paper_protocol", action="store_true", help="five-seed shared-environment protocol; batch_size retains the original transition-count meaning")
    parser.add_argument('--paper_results_root', default='results/FORMAL_6SCENARIOS_V1')
    parser.add_argument('--eval_seeds', nargs='+', type=int, default=list(range(1001,1011)))
    parser.add_argument('--eval_episodes_per_seed', type=int, default=20)
    # Env settings
    parser.add_argument('--use_cuda', action='store_true', help='use cuda')  # 修改
    parser.add_argument('--env', type=str, default='lbf', choices=['lbf'])
    parser.add_argument(
        '--map', type=str, default='Foraging-hetero_mixed',
        choices=[
            'Foraging-hetero_mixed', 'Foraging-hetero_strong',
            'Foraging-consensus_wide', 'Foraging-fog_sight1_level',
            'Foraging-dynamic_overload', 'Foraging-scale_20',
        ],
    )

    # Agent settings
    parser.add_argument('--agent', type=str, default="teamcomm", help='algorithm name',
                        choices=['commnet', 'ic3net', 'magic', 'tarmac', 'teamcomm', 'tiecomm'])
    parser.add_argument('--block', type=str, default="no", help='block type', choices=['no', 'inter', 'intra'])
    parser.add_argument('--moduarity', action='store_true', help='use moduarity_loss')  # 修改
    parser.add_argument('--vib', action='store_true', default=True, help='use vib_loss')  # 修改

    # Experiment settings
    parser.add_argument('--seed', type=int, default=20, help='random seed')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--total_epoches', type=int, default=1000, help='total number of training epochs')
    parser.add_argument('--hid_size', type=int, default=64, help='hidden size')
    parser.add_argument('--note', type=str, default="ap", help='note')

    args = parser.parse_args()

    # 新增：处理vib参数的默认值（因为action='store_true'默认是False）
    if not hasattr(args, 'vib'):
        args.vib = True
    training_begin_time = time.time()
    signal.signal(signal.SIGINT, signal_handler)
    main(args)
    # training_time = time.time() - training_begin_time
    # print('training time: {} h'.format(training_time / 3600))
