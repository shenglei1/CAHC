from __future__ import annotations

from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from action_utils import select_action, translate_action
from utils import merge_stat


Transition = namedtuple(
    "Transition",
    ("action", "action_out", "value", "episode_mask", "reward", "alive_mask", "teacher"),
)


class Trainer:
    def __init__(self, args, policy_net, env) -> None:
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.optimizer = optim.RMSprop(
            policy_net.parameters(), lr=args.lrate, alpha=0.97, eps=1e-6
        )
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=float(getattr(args, "lr_decay", 1.0))
        )
        self.parameters = list(policy_net.parameters())
        self.device = policy_net.device

    def get_episode(self, epoch: int):

        if hasattr(self.env.env, "set_evaluation_mode"):
            self.env.env.set_evaluation_mode(False)
        transitions = []
        state = self.env.reset().to(self.device)
        hidden = self.policy_net.init_hidden(batch_size=state.shape[0])
        info = (
            self.env.initial_info()
            if hasattr(self.env, "initial_info")
            else {"agent_level": self.env._agent_levels}
        )
        stat = {}

        for step in range(self.args.max_steps):
            action_out, value, next_hidden = self.policy_net(
                [state, hidden], self.env.policy_info(info)
            )
            action = select_action(self.args, action_out)
            stored_action, actual = translate_action(self.args, self.env, action)
            next_state, reward, done, env_info = self.env.step(actual)
            alive = np.asarray(
                env_info.get("alive_mask", np.ones_like(reward)), dtype=np.float64
            ).reshape(reward.shape)
            episode_mask = np.zeros_like(reward) if done else np.ones_like(reward)
            teacher = np.asarray(
                env_info.get("probe2_teacher_actions", np.zeros_like(reward, dtype=np.int64)),
                dtype=np.int64,
            )
            transitions.append(
                Transition(
                    stored_action,
                    action_out,
                    value,
                    episode_mask,
                    reward,
                    alive,
                    teacher,
                )
            )
            state = next_state.to(self.device)
            hidden = next_hidden




            info = {**env_info, "agent_level": self.env._agent_levels}
            if (step + 1) % self.args.detach_gap == 0:
                hidden = (hidden[0].detach(), hidden[1].detach())
            if done:
                break

        stat["num_steps"] = len(transitions)
        stat["workload_total"] = int(info["workload_total"])
        stat["workload_completed"] = int(info["workload_completed"])
        stat["capacity_workload_total"] = int(info["capacity_workload_total"])
        stat["capacity_workload_completed"] = int(
            info["capacity_workload_completed"]
        )
        stat["task_completion_ratio"] = float(info["task_completion_ratio"])
        stat["episode_success"] = float(info["episode_success"])
        stat["collisions"] = int(info["num_collisions"])
        return transitions, stat

    def _backward_episode(self, transitions):
        steps = len(transitions)
        n_agents = self.args.nagents
        rewards = torch.as_tensor(
            np.asarray([item.reward for item in transitions]),
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        masks = torch.as_tensor(
            np.asarray([item.episode_mask for item in transitions]),
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        alive = torch.as_tensor(
            np.asarray([item.alive_mask for item in transitions]),
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        values = torch.stack([item.value for item in transitions]).view(
            steps, n_agents
        )
        actions = torch.as_tensor(
            np.asarray([item.action for item in transitions]),
            device=self.device,
            dtype=torch.long,
        ).transpose(1, 2).reshape(steps, n_agents, self.args.dim_actions)
        action_out = [
            torch.cat([item.action_out[head] for item in transitions], dim=0)
            for head in range(self.args.dim_actions)
        ]

        cooperative_returns = torch.empty_like(rewards)
        individual_returns = torch.empty_like(rewards)
        next_cooperative = torch.zeros(n_agents, device=self.device)
        next_individual = torch.zeros(n_agents, device=self.device)
        for index in reversed(range(steps)):
            cooperative_returns[index] = (
                rewards[index]
                + self.args.gamma * next_cooperative * masks[index]
            )
            individual_returns[index] = (
                rewards[index]
                + self.args.gamma * next_individual * masks[index]
            )
            next_cooperative = cooperative_returns[index]
            next_individual = individual_returns[index]
        targets = (
            self.args.mean_ratio
            * cooperative_returns.mean(dim=1, keepdim=True)
            + (1.0 - self.args.mean_ratio) * individual_returns
        )
        advantages = targets - values.detach()
        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std().clamp(
                min=1e-8
            )

        selected_log_probability = torch.zeros_like(advantages)
        entropy = torch.zeros((), device=self.device)
        for head, log_probabilities in enumerate(action_out):
            chosen = actions[:, :, head].unsqueeze(-1)
            selected_log_probability = selected_log_probability + log_probabilities.gather(
                dim=-1, index=chosen
            ).squeeze(-1)
            entropy = entropy - (
                log_probabilities.exp() * log_probabilities
            ).sum()
        action_loss = -(
            advantages * selected_log_probability * alive
        ).sum()
        value_loss = (((values - targets) ** 2) * alive).sum()
        loss = (
            action_loss
            + self.args.value_coeff * value_loss
            - self.args.entr * entropy
        )




        teacher_coefficient = float(
            getattr(self.args, "teacher_loss_coefficient", 0.0) or 0.0
        )
        teacher_loss_value = 0.0
        if teacher_coefficient > 0:
            teachers = torch.as_tensor(
                np.asarray([item.teacher for item in transitions]),
                device=self.device,
                dtype=torch.long,
            ).view(-1)
            teacher_loss = F.nll_loss(
                action_out[0].reshape(-1, self.args.num_actions[0]),
                teachers,
                reduction="sum",
            )
            if bool(getattr(self.args, 'teacher_only_training', False)):



                loss = teacher_coefficient * teacher_loss
            else:
                loss = loss + teacher_coefficient * teacher_loss
            teacher_loss_value = float(teacher_loss.detach())
            teacher_accuracy = float(
                (action_out[0].reshape(-1, self.args.num_actions[0]).argmax(dim=-1)
                 == teachers).double().mean().item()
            )
        loss.backward()
        return {
            "action_loss": float(action_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()),
            "teacher_loss": teacher_loss_value,
            "teacher_accuracy": locals().get("teacher_accuracy", 0.0),
            "loss": float(loss.detach()),
        }

    def train_batch(self, epoch: int):
        self.policy_net.train()
        self.optimizer.zero_grad()
        stat = {}
        episodes = 0
        while episodes < self.args.batch_size:
            transitions, episode_stat = self.get_episode(epoch)
            merge_stat(episode_stat, stat)
            merge_stat(self._backward_episode(transitions), stat)
            episodes += 1
        total_steps = int(stat["num_steps"])
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.div_(total_steps)
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in self.parameters
        ):
            raise FloatingPointError("Non-finite HyperWave-X gradient.")
        max_norm = float(getattr(self.args, "max_grad_norm", 0.0) or 0.0)
        if max_norm > 0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters, max_norm=max_norm
            )
            stat["gradient_norm_before_clip"] = float(gradient_norm)
        self.optimizer.step()
        self.scheduler.step()
        stat["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        stat["num_episodes"] = episodes
        return stat

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state) -> None:
        self.optimizer.load_state_dict(state)
