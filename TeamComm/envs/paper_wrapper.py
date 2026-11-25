import argparse
import numpy as np
from .paper_lbf import PaperForagingEnv

class PaperWrapper:
    def __init__(self, config, **kwargs):
        self.args = argparse.Namespace(**config)
        self._env = PaperForagingEnv(self.args.map.removeprefix('Foraging-'))
        self._env.seed(self.args.seed)
        self.n_agents = self._env.n_agents
        self.episode_limit = self._env._max_episode_steps
        self._obs = None
    @property
    def unwrapped(self):
        return self._env
    def reset(self):
        self._obs = [o.reshape(-1) for o in self._env.reset()]
        if getattr(self, 'collect_operational_metrics', False):
            from operational_metrics import OperationalMetrics
            self.operational_metrics = OperationalMetrics(self._env)
        return self._obs, self.get_state(), {'initial_food': self._env.initial_food}
    def step(self, actions):
        values = np.asarray(actions[0]).reshape(-1).tolist()
        if getattr(self, 'collect_operational_metrics', False):
            self.operational_metrics.before_step(self._env, values)
        obs, reward, dones, info = self._env.step(values)
        if getattr(self, 'collect_operational_metrics', False):
            self.operational_metrics.after_step(self._env)
        self._obs = [o.reshape(-1) for o in obs]
        return np.asarray(reward), bool(all(dones)), info
    def get_obs(self):
        return self._obs
    def get_state(self):
        raw = self._env
        rows, cols = raw.field.shape
        task_scale = max(1, raw.max_food_level)
        field = raw.field.astype(np.float32).reshape(-1) / task_scale
        agents = np.asarray(
            [
                (
                    player.position[0] / max(1, rows - 1),
                    player.position[1] / max(1, cols - 1),
                    player.level / max(1, raw.max_player_level),
                )
                for player in raw.players
            ],
            dtype=np.float32,
        ).reshape(-1)
        info = raw._info()
        total = max(1, info['workload_total'])
        clock_and_ledger = np.asarray(
            [
                raw.current_step / raw._max_episode_steps,
                info['workload_arrived'] / total,
                info['workload_completed'] / total,
                info['workload_backlog'] / total,
                info['workload_unreleased'] / total,
            ],
            dtype=np.float32,
        )
        return np.concatenate([field, agents, clock_and_ledger]).astype(np.float32)
    def get_env_info(self):
        d = self._env.observation_space[0].shape[0]
        return dict(state_shape=self.get_state().size, obs_shape=d, n_actions=6,
                    n_agents=self.n_agents, episode_length=self.episode_limit)
    def close(self):
        return self._env.close()
    def get_graph(self):
        # Original TieComm radius-3 prior. Positions are private observation fields.
        import networkx as nx
        graph = nx.Graph()
        graph.add_nodes_from(range(self.n_agents))
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(np.asarray(self._env.players[i].position) -
                                  np.asarray(self._env.players[j].position)) <= 3.0:
                    graph.add_edge(i,j)
        return graph
