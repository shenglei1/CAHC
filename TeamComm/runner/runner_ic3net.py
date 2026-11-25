import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
from torch.optim import Adam

from modules.utils import merge_dict, multinomials_log_density
import time
import argparse
from .runner import Runner
from .paper_metrics import apply_paper_metrics
import time
import numpy as np

Transition = namedtuple('Transition', ('action_outs', 'actions', 'rewards', 'values', 'episode_masks', 'episode_agent_masks'))
class RunnerIcnet(Runner):
    def __init__(self, config, env, agent):
        super(RunnerIcnet, self).__init__(config, env, agent)


    def run_an_episode(self):

        memory = []
        info = dict()
        log = dict()
        # episode_return = np.zeros(self.n_agents)
        episode_return = 0

        # 重置环境，获取初始观测和状态
        obs, state, _ = self.env.reset()
        if hasattr(self.env.unwrapped, 'get_initial_food_value'):
            initial_food = self.env.unwrapped.get_initial_food_value()
        else:
            initial_food = self.env.unwrapped.field.sum()

        step = 1
        done = False

        # 新增：初始化episode级统计变量
        n_agents = self.args.n_agents
        max_comm_edges = n_agents * (n_agents - 1)
        episode_comm_sparsities = []
        dropout_rate = getattr(self, 'evaluation_link_dropout_rate', None)
        dropout_rng = getattr(self, 'evaluation_link_dropout_rng', None)
        decoded_links = 0
        dropped_links = 0

        while not done and step <= self.args.episode_length:
            if step == 1:
                info['comm_action'] = np.zeros(self.args.n_agents, dtype=int)
            if self.args.recurrent:

                self.args.rnn_type = 'LSTM'
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and step == 1:
                    # prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])
                    prev_hid = self.agent.init_hidden(batch_size=1)

                obs_tensor = torch.tensor(np.array(obs), dtype=torch.float)
                obs_tensor = obs_tensor.unsqueeze(0)  #
                obs_tensor = [obs_tensor, prev_hid]  #

                if dropout_rate is not None:
                    if dropout_rng is None:
                        raise RuntimeError('Evaluation dropout RNG was not configured.')
                    info['evaluation_link_keep_mask'] = (
                        dropout_rng.random((n_agents, n_agents)) >= dropout_rate
                    )

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
                if dropout_rate is not None:
                    if dropout_rng is None:
                        raise RuntimeError('Evaluation dropout RNG was not configured.')
                    info['evaluation_link_keep_mask'] = (
                        dropout_rng.random((n_agents, n_agents)) >= dropout_rate
                    )
                action_outs, values = self.agent(obs_tensor,info)#

            decoded_links += int(
                getattr(self.agent, 'last_decoded_active_links', 0)
            )
            dropped_links += int(
                getattr(self.agent, 'last_dropped_active_links', 0)
            )

            # IC3Net通信稀疏度：通信动作的平均比例
            comm_action = info.get('comm_action', np.zeros(n_agents))
            active_agents = np.sum(comm_action)
            actual_edges = active_agents * (active_agents - 1) if active_agents > 1 else 0
            sparsity = actual_edges / max_comm_edges if max_comm_edges > 0 else 0.0
            if 'evaluation_link_keep_mask' in info:
                sparsity = float(
                    getattr(self.agent, 'last_active_link_ratio', sparsity)
                )
            episode_comm_sparsities.append(sparsity)

            actions = self.choose_action(action_outs)
                # action_outs = action_outs.unsqueeze(0)
            rewards, done, env_info = self.env.step(actions)
            next_obs = self.env.get_obs()

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
        # 修改后
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
        log['num_collisions'] = int(env_info.get('num_collisions', 0))

        # 3. 通信稀疏度
        log['comm_sparsity'] = sum(episode_comm_sparsities) / len(
            episode_comm_sparsities) if episode_comm_sparsities else 0.0
        if dropout_rate is not None:
            log['requested_link_dropout_rate'] = float(dropout_rate)
            log['realized_active_link_dropout_rate'] = (
                dropped_links / decoded_links if decoded_links else 0.0
            )

        return memory, log




    def compute_grad(self, batch):
        return self.compute_agent_grad(batch)



    def compute_agent_grad(self, batch):

        log = dict()

        n = self.n_agents
        batch_size = len(batch.actions)
        rewards = torch.Tensor(batch.rewards)
        actions = torch.Tensor(batch.actions)
        actions = actions.transpose(1, 2)
        # actions = actions.reshape(-1, n, self.args.dim_actions)
        actions = actions.reshape(-1, n, 2)

        episode_masks = torch.Tensor(batch.episode_masks)
        episode_agent_masks = torch.Tensor(batch.episode_agent_masks)


        values = torch.cat(batch.values, dim=0)  # (batch, n, 1)
        action_outs = list(zip(*batch.action_outs))
        # action_outs = torch.Tensor(batch.action_outs)
        # # action_outs = batch.action_outs
        # action_outs = action_outs.transpose(1, 2).view(-1, n, 2)
        # for tnsr in action_outs:
        #     tnsr=list(tnsr)
        #     for b in tnsr:
        #         b=b.unsqueeze(0)
        #         c=tnsr
        action_outs = [torch.cat(a, dim=0) for a in action_outs]
        action_outs = [a.view(batch_size, -1, a.shape[1]) for a in action_outs]

        returns = torch.Tensor(batch_size, n)
        advantages = torch.Tensor(batch_size, n)
        values = values.view(batch_size, n)
        prev_returns = 0



        if self.args.normalize_rewards:
            mean_reward = rewards.mean()
            std_reward = rewards.std() + 1e-5  # Add a small value to prevent division by zero
            rewards = (rewards - mean_reward) / std_reward




        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + self.args.gamma * prev_returns * episode_masks[i] * episode_agent_masks[i]
            prev_returns = returns[i].clone()



        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        # if self.args.normalize_rewards:
        #     advantages = (advantages - advantages.mean()) / advantages.std()

        # element of log_p_a: [(batch_size*n) * num_actions[i]]
        # log_p_a = [action_outs.view(-1, self.n_actions)]
        # log_p_a = [a.view(-1, self.n_actions) for a in action_outs]
        num_actions=[self.n_actions, 2]
        log_p_a = [action_outs[i].view(-1, num_actions[i]) for i in range(2)]
        # actions: [(batch_size*n) * dim_actions]
        actions = actions.contiguous().view(-1, 2)
        log_prob = multinomials_log_density(actions, log_p_a)
        action_loss = -advantages.view(-1) * log_prob.squeeze()
        actor_loss = action_loss.sum()


        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        critic_loss = value_loss.sum()


        total_loss = actor_loss + self.args.value_coeff * critic_loss
        total_loss.backward()


        log['action_loss'] = actor_loss.item()
        log['value_loss'] = critic_loss.item()
        log['total_loss'] = total_loss.item()

        return log



    def choose_action(self, log_p_a):
        log_p_a = [a.unsqueeze(0) for a in log_p_a]
        p_a = [[z.exp() for z in x] for x in log_p_a]
        ret = torch.stack([torch.stack([torch.multinomial(x, 1).detach() for x in p]) for p in p_a])
        action = [x.squeeze().data.numpy() for x in ret]
        return action
