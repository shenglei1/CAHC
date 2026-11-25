import numpy as np
import torch


class LBFWrapper:
    def __init__(self, env, args):
        self.env = env
        if getattr(args, 'seed', -1) >= 0 and hasattr(env, 'seed'):
            env.seed(args.seed)
        self.n_agents = env.n_agents
        self.device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
        # 新增：保存 max_player_level 等属性，方便外部访问
        self.max_player_level = getattr(env, 'max_player_level', 5)
        # 如果有其他属性也可一并保存
        self.sight = env.sight
        self.force_coop = env.force_coop

        self.observation_dim = env.observation_space[0].shape[0]   # 例如 74
        self.num_actions = int(env.action_space[0].n)
        self.dim_actions = 1
        self.action_space = env.action_space
        # 获取真实的最大 agent level（考虑 fixed_levels 的存在）
        if hasattr(env, 'fixed_levels'):
            self.max_player_level = max(env.fixed_levels)
        else:
            self.max_player_level = getattr(env, 'max_player_level', 1)

    def reset(self):
        obs_list = self.env.reset()                     # list of arrays, each shape (obs_dim, 1)
        # 去掉最后一维，变为 (obs_dim,)
        obs_list = [o.squeeze(-1) for o in obs_list]
        obs = np.stack(obs_list)                        # (n_agents, obs_dim)
        obs = np.expand_dims(obs, axis=0)               # (1, n_agents, obs_dim)
        obs = torch.from_numpy(obs).double().to(self.device)
        # 记录 agent levels
        self._agent_levels = np.array([p.level for p in self.env.players])
        return obs

    def policy_info(self, info):
        """Keep probe audit targets/future schedules outside the policy API.

        The physical communication graph is public environment metadata (the
        same radio-range constraint for both methods), so it is supplied to the
        policy. Hidden permits and future audit targets are still withheld.
        """
        if not getattr(self.env, "is_probe2", False):
            return info
        return {key: info[key] for key in (
            "agent_level",
            "probe2_fixed_incidence",
            "probe2_force_physical_links",
            "probe2_required_sensor_to_actuator_hops",
            "probe2_communication_scaffold_directed_links",
            "collect_model_diagnostics",
            "record_spectral_diagnostics",
        ) if key in info}

    def initial_info(self):
        info = {'agent_level': self._agent_levels}
        if hasattr(self.env, 'probe2_public_communication_info'):
            info.update(self.env.probe2_public_communication_info())
        return info

    def seed(self, seed):
        if not hasattr(self.env, 'seed'):
            raise AttributeError("The wrapped LBF environment does not expose seed().")
        return self.env.seed(seed)

    def step(self, actions):
        # 动作可能以 [array([...])] 形式传入，转换为普通列表
        if isinstance(actions, (list, tuple)) and len(actions) == 1 and isinstance(actions[0], np.ndarray):
            actions = actions[0].tolist()
        obs_list, rewards, dones, info = self.env.step(actions)

        # 去掉观测的最后一维
        obs_list = [o.squeeze(-1) for o in obs_list]
        obs = np.stack(obs_list)
        obs = np.expand_dims(obs, axis=0)
        obs = torch.from_numpy(obs).double().to(self.device)

        if isinstance(info, dict):
            info['agent_level'] = self._agent_levels
        else:
            info = {'agent_level': self._agent_levels}

        # 将 done 列表转为单个布尔值（所有 agent 同步结束）
        done = any(dones)
        return (obs, np.array(rewards), done, info)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()

    def get_stat(self):
        return {}
