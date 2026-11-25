import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam
from modules.utils import merge_dict
import time
import argparse
from .runner import Runner
from .paper_metrics import apply_paper_metrics
import time
import numpy as np

Transition = namedtuple('Transition', ('action_outs', 'actions', 'rewards', 'values', 'episode_masks', 'episode_agent_masks'))
class RunnerBaseline(Runner):
    def __init__(self, config, env, agent):
        super(RunnerBaseline, self).__init__(config, env, agent)


    def run_an_episode(self):

        memory = []
        info = dict()
        log = dict()
        # episode_return = np.zeros(self.n_agents)
        episode_return = 0

        # 重置环境，获取初始观测和状态
        obs, state, _ = self.env.reset()
        # 优先使用环境提供的初始食物总值
        if hasattr(self.env.unwrapped, 'get_initial_food_value'):
            initial_food = self.env.unwrapped.get_initial_food_value()
        else:
            initial_food = self.env.unwrapped.field.sum()

        step = 1
        done = False

        # 新增：初始化episode级统计变量
        n_agents = self.args.n_agents
        max_comm_edges = n_agents * (n_agents - 1)
        episode_attentions = []
        episode_comm_sparsities = []

        while not done and step <= self.args.episode_length:
            if step == 1 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.n_agents, dtype=int)
            if self.args.commnet and self.args.recurrent:

                self.args.rnn_type = 'LSTM'
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and step == 1:
                    # prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])
                    prev_hid = self.agent.init_hidden(batch_size=1)

                obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
                obs_tensor = obs_tensor.unsqueeze(0)  #
                obs_tensor = [obs_tensor, prev_hid]  #

                action_outs, values, prev_hid = self.agent(obs_tensor, info)

                if (step) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()

        #prev_hid = self.agent.init_hidden(batch_size=state.shape[0])

        # for t in range(self.args.episode_length):
        #     if t == 0 and self.args.hard_attn and self.args.commnet:
        #         info['comm_action'] = np.zeros(self.args.n_agents, dtype=int)
            else:
                obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
                action_outs, values = self.agent(obs_tensor,info)#

            # 新增：收集每步数据
            if 'attn_matrix' in info and info['attn_matrix'] is not None:
                episode_attentions.append(info['attn_matrix'])
            elif 'comm_sparsity' in info:
                episode_comm_sparsities.append(info['comm_sparsity'])

            actions = self.choose_action(action_outs)
            rewards, done, env_info = self.env.step(actions)
            next_obs = self.env.get_obs()
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = actions[-1] if not self.args.comm_action_one else np.ones(self.args.n_agents,
                                                                                                dtype=int)

            # episode_mask = np.zeros(np.array(rewards).shape)
            # episode_agent_mask = np.array(dones) + 0
            #
            # if dones or t == self.args.episode_length - 1:
            # # if all(dones) or t == self.args.episode_length - 1:
            #     episode_mask = np.ones(np.array(rewards).shape)
            done = done or step == self.args.episode_length
            episode_mask = np.ones(rewards.shape)
            episode_agent_mask = np.ones(rewards.shape)
            if done:
                episode_mask = np.zeros(rewards.shape)
            else:
                if 'is_completed' in env_info:
                    episode_agent_mask = 1 - env_info['is_completed'].reshape(-1)


            trans = Transition(action_outs, actions, rewards, values, episode_mask, episode_agent_mask)
            memory.append(trans)


            #state = next_state
            obs = next_obs
            # episode_return += rewards.astype(episode_return.dtype)
            episode_return += float(np.sum(rewards))
            step += 1


            # episode_return += float(sum(rewards))
            # self.total_steps += 1
        log['episode_return'] = episode_return
        log['episode_steps'] = [step - 1]
        num_steps = step - 1
        # ====================== 新增：计算所有5个指标 ======================
        # 1. 任务成功率 & 吞吐量
        if hasattr(self.env.unwrapped, 'get_collected_initial_food'):
            success_food = self.env.unwrapped.get_collected_initial_food()
        else:
            remaining_food = self.env.unwrapped.field.sum()
            success_food = initial_food - remaining_food if initial_food > 0 else 0.0

        log['success_food'] = success_food
        log['success_rate'] = success_food / initial_food if initial_food > 0 else 0.0
        log['throughput'] = success_food / num_steps if num_steps > 0 else 0.0
        apply_paper_metrics(log, env_info, self.args.episode_length, episode_return, num_steps)

        # 2. 碰撞数
        log['num_collisions'] = env_info.get('num_collisions', 0)

        # 3. 通信稀疏度
        if len(episode_attentions) > 0:
            # 基于注意力矩阵计算（TarMAC）
            avg_sparsity = 0.0
            for attn in episode_attentions:
                if attn.dim() == 3:  # 多头注意力
                    attn = attn.mean(dim=-1)
                mask = ~torch.eye(n_agents, dtype=torch.bool, device=attn.device)
                attn_no_self = attn[mask]
                active_edges = (attn_no_self > 1e-6).sum().item()
                sparsity = active_edges / max_comm_edges
                avg_sparsity += sparsity
            log['comm_sparsity'] = avg_sparsity / len(episode_attentions)
        elif len(episode_comm_sparsities) > 0:
            # 直接使用返回的稀疏度（CommNet）
            log['comm_sparsity'] = sum(episode_comm_sparsities) / len(episode_comm_sparsities)
        else:
            log['comm_sparsity'] = 0.0

        return memory, log


