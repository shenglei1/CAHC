"""Registry-driven LBF environment shared by all three formal projects.

This module contains the six frozen comparison scenarios, one supplementary
restricted-perception control, and two separate Reviewer probes. Legacy
environments remain outside the revision protocol.
"""

from __future__ import annotations

from collections import defaultdict, deque
from enum import IntEnum
from typing import Deque, Dict, Iterable, List, Sequence, Tuple

import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np

from scenario_registry import load_registry, normalize_scenario_id, registry_sha256


class Action(IntEnum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class Player:
    def __init__(self, player_id: int) -> None:
        self.id = int(player_id)
        self.position: Tuple[int, int] | None = None
        self.level = 0
        self.field_size: Tuple[int, int] | None = None
        self.score = 0.0
        self.reward = 0.0

    def setup(
        self, position: Tuple[int, int], level: int, field_size: Tuple[int, int]
    ) -> None:
        self.position = tuple(position)
        self.level = int(level)
        self.field_size = tuple(field_size)
        self.score = 0.0
        self.reward = 0.0


class FormalForagingEnv(gym.Env):
    """Shared semantics for the main and probe suites."""

    metadata = {"render.modes": []}

    def __new__(cls, scenario_id):
        if cls is FormalForagingEnv and normalize_scenario_id(scenario_id) == "probe_interaction":
            from probe2_interlock import Probe2InterlockEnv
            return object.__new__(Probe2InterlockEnv)
        return object.__new__(cls)

    def __init__(self, scenario_id: str):
        registry = load_registry()
        self.registry_sha256 = registry_sha256(registry)
        self.type = normalize_scenario_id(scenario_id)
        experiment_ids = (
            registry["formal_scenario_ids"]
            + registry["supplementary_scenario_ids"]
            + registry["reviewer_probe_ids"]
        )
        if self.type not in experiment_ids:
            allowed = ", ".join(experiment_ids)
            raise ValueError(
                f"Unknown comparison/probe scenario '{self.type}'. Allowed: {allowed}"
            )

        self.registry = registry
        self.spec = registry["scenarios"][self.type]
        self.probe_rules = dict(self.spec.get("probe_rules", {}))
        self.protocol_version = registry["protocol_version"]
        self.fixed_levels = [int(x) for x in self.spec["agents"]["capabilities"]]
        self.n_agents = len(self.fixed_levels)
        self.players = [Player(i) for i in range(self.n_agents)]
        self.max_player_level = max(self.fixed_levels)
        self.field = np.zeros(tuple(self.spec["map"]["shape"]), dtype=np.int32)
        self.sight = int(self.spec["observation"]["sight"])
        self._max_episode_steps = int(self.spec["horizon"])
        self.max_food = int(self.spec["tasks"]["workload"]["task_count"])
        self.max_food_level = max(self.spec["tasks"]["requirements"])
        self.food_level_dist = [int(x) for x in self.spec["tasks"]["requirements"]]
        self.force_coop = True
        self._normalize_reward = False
        self.excess_capacity_penalty = 0.25
        self.coalition_member_penalty = 0.05
        self.collision_penalty = 0.0
        self.communication_penalty = 0.0

        self.action_space = gym.spaces.Tuple(
            tuple(gym.spaces.Discrete(len(Action)) for _ in self.players)
        )
        self.observation_space = gym.spaces.Tuple(
            tuple(self._get_observation_space() for _ in self.players)
        )
        self.np_random = None
        self.seed()
        self.reset()

    @property
    def field_size(self) -> Tuple[int, int]:
        return tuple(self.field.shape)

    @property
    def rows(self) -> int:
        return self.field.shape[0]

    @property
    def cols(self) -> int:
        return self.field.shape[1]

    @property
    def game_over(self) -> bool:
        return bool(self._game_over)

    def seed(self, seed: int | None = None):
        self.np_random, actual_seed = seeding.np_random(seed)
        # Gymnasium returns a NumPy integer here; keep the legacy MARL adapter
        # JSON-safe without changing the seeded random stream.
        return [int(actual_seed)]

    def _randint(self, low: int, high: int | None = None) -> int:
        if hasattr(self.np_random, "integers"):
            return int(self.np_random.integers(low, high))
        return int(self.np_random.randint(low, high))

    def _shuffle(self, values: Sequence[int]) -> List[int]:
        values_array = np.asarray(values, dtype=np.int64)
        return [int(x) for x in self.np_random.permutation(values_array)]

    def _uniform(self, low: float, high: float) -> float:
        return float(self.np_random.uniform(low, high))

    def _get_observation_space(self):
        crop_width = 1 + 2 * self.sight
        crop_cells = crop_width * crop_width
        axis_cells = self.rows + self.cols
        roster_cells = self.n_agents
        probe_status_cells = 1 if self.type in {
            "probe_complementary", "probe_interaction"
        } else 0
        size = 2 * crop_cells + axis_cells + roster_cells + 1 + probe_status_cells
        low = np.zeros(size, dtype=np.float32)
        high = np.concatenate(
            [
                np.full(crop_cells, self.max_food_level, dtype=np.float32),
                np.ones(crop_cells, dtype=np.float32),
                np.full(axis_cells, self.max_player_level, dtype=np.float32),
                np.full(roster_cells, self.max_player_level, dtype=np.float32),
                np.ones(1, dtype=np.float32),
                np.ones(probe_status_cells, dtype=np.float32),
            ]
        )
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _fixed_positions(self) -> List[Tuple[int, int]] | None:
        if self.type == "probe_complementary":
            positions = [
                (4, 1), (4, 2), (4, 4), (4, 5),
            ]
            # Capabilities remain attached to agent identities, while their
            # initial cells are seeded. This prevents a fixed identity-to-route
            # lookup without changing the public capability roster.
            return [positions[index] for index in self._shuffle(range(len(positions)))]
        if self.type == "probe_interaction":
            # Probe 2 registers role-preserving sensor/relay/actuator cells.
            positions = [
                tuple(position) for position in self.probe_rules["agent_positions"]
            ]
            if len(positions) != self.n_agents:
                raise RuntimeError("Probe 2 agent-position registry is incomplete.")
            return positions
        return None

    def _fixed_task_positions(self) -> List[Tuple[int, int]] | None:
        if self.type == "probe_complementary":
            return [(2, 2), (2, 4)]
        if self.type == "probe_interaction":
            return [
                tuple(position) for position in self.probe_rules["task_positions"]
            ]
        return None

    def _initialize_probe2_readiness_process(self) -> None:
        """Create the seeded station interlock and private sensor streams."""
        self._probe2_station_ready_steps = {}
        self._probe2_agent_station = {}
        self._probe2_agent_role = {}
        self._probe2_station_chains = {}
        self._probe2_sensor_noise = {}
        self._probe2_fixed_incidence = np.zeros(
            (self.n_agents, self.n_agents), dtype=np.float32
        )
        if self.type != "probe_interaction":
            return
        minimum = int(self.probe_rules["station_ready_step_min"])
        maximum = int(self.probe_rules["station_ready_step_max"])
        noise = float(self.probe_rules["sensor_noise_half_width"])
        if minimum < 1 or maximum < minimum:
            raise ValueError("Invalid Probe 2 station-readiness range.")
        if not 0.0 <= noise <= 1.0:
            raise ValueError("Invalid Probe 2 sensor-noise range.")
        roles = list(self.probe_rules["agent_roles"])
        chains = [tuple(chain) for chain in self.probe_rules["station_chains"]]
        edges = [tuple(edge) for edge in self.probe_rules["fixed_incidence_edges"]]
        if not (len(roles) == len(edges) == self.n_agents):
            raise RuntimeError("Probe 2 roles/incidence do not match the agent count.")
        for edge_index, members in enumerate(edges):
            self._probe2_fixed_incidence[edge_index, list(members)] = 1.0
        for task_position, chain in zip(self._fixed_task_positions() or [], chains):
            adjacent = sorted(
                player.id for player in self.players
                if self._adjacent_to(player.position, task_position)
            )
            expected_actuators = sorted(chain[-2:])
            if adjacent != expected_actuators:
                raise RuntimeError(
                    "Probe 2 requires exactly the two registered actuators at each task."
                )
            ready_step = self._randint(minimum, maximum + 1)
            self._probe2_station_ready_steps[task_position] = ready_step
            self._probe2_station_chains[task_position] = chain
            for player_id in chain:
                self._probe2_agent_station[player_id] = task_position
                self._probe2_agent_role[player_id] = roles[player_id]
            sensor_id = chain[0]
            self._probe2_sensor_noise[sensor_id] = np.asarray(
                [
                    self._uniform(-noise, noise)
                    for _ in range(self._max_episode_steps + 1)
                ],
                dtype=np.float32,
            )

    def _probe2_sensor_value(self, player: Player) -> float:
        if self.type != "probe_interaction":
            return 0.0
        if self._probe2_agent_role[player.id] != "sensor":
            return float(self.probe_rules["relay_neutral_level"])
        station = self._probe2_agent_station[player.id]
        true_ready = (
            self.current_step >= self._probe2_station_ready_steps[station]
        )
        base = float(
            self.probe_rules[
                "sensor_ready_level" if true_ready else "sensor_not_ready_level"
            ]
        )
        noise_stream = self._probe2_sensor_noise[player.id]
        noise = float(noise_stream[min(self.current_step, len(noise_stream) - 1)])
        return float(np.clip(base + noise, 0.0, 1.0))

    def probe2_public_communication_info(self) -> Dict[str, object]:
        """Return only the public, method-neutral Probe 2 scaffold."""
        if self.type != "probe_interaction":
            return {}
        return {
            "probe2_fixed_incidence": self._probe2_fixed_incidence.copy(),
            "probe2_required_sensor_to_actuator_hops": int(
                self.probe_rules["required_sensor_to_actuator_hops"]
            ),
        }

    def _is_empty_location(self, row: int, col: int) -> bool:
        if self.field[row, col] != 0:
            return False
        return not any(player.position == (row, col) for player in self.players)

    @staticmethod
    def _respects_task_spacing(
        candidate: Tuple[int, int], reserved: Iterable[Tuple[int, int]]
    ) -> bool:
        row, col = candidate
        for other_row, other_col in reserved:
            dr, dc = abs(row - other_row), abs(col - other_col)
            if dr <= 1 and dc <= 1:
                return False
            if (dr == 0 and dc <= 2) or (dc == 0 and dr <= 2):
                return False
        return True

    def _spawn_players(self) -> None:
        fixed_positions = self._fixed_positions()
        if fixed_positions is not None:
            for player, position, level in zip(
                self.players, fixed_positions, self.fixed_levels
            ):
                player.setup(position, level, self.field_size)
            return

        occupied = set()
        for player, level in zip(self.players, self.fixed_levels):
            for _ in range(10000):
                position = (self._randint(0, self.rows), self._randint(0, self.cols))
                if position not in occupied:
                    occupied.add(position)
                    player.setup(position, level, self.field_size)
                    break
            else:
                raise RuntimeError(f"{self.type}: could not place all agents.")

    def _reserve_task_positions(self, count: int) -> List[Tuple[int, int]]:
        fixed_positions = self._fixed_task_positions()
        if fixed_positions is not None:
            if len(fixed_positions) != count:
                raise RuntimeError(f"{self.type}: fixed task layout/count mismatch.")
            return fixed_positions

        reserved: List[Tuple[int, int]] = []
        player_positions = {player.position for player in self.players}
        attempts = 0
        while len(reserved) < count and attempts < 50000:
            attempts += 1
            position = (
                self._randint(1, self.rows - 1),
                self._randint(1, self.cols - 1),
            )
            if position in player_positions or position in reserved:
                continue
            if not self._respects_task_spacing(position, reserved):
                continue
            reserved.append(position)
        if len(reserved) != count:
            raise RuntimeError(
                f"{self.type}: declared {count} tasks but only reserved "
                f"{len(reserved)} legal task cells."
            )
        return reserved

    def _build_task_plan(self) -> List[Dict[str, object]]:
        requirements = list(self.spec["tasks"]["requirements"])
        positions = self._reserve_task_positions(len(requirements))
        arrival = self.spec["tasks"]["arrival_process"]
        releases: List[int] = [0] * len(arrival["initial_requirements"])
        for event in arrival["events"]:
            nominal_step = int(event["after_step"])
            jitter_steps = int(event.get("jitter_steps", 0))
            realized_step = nominal_step
            if jitter_steps:
                realized_step += self._randint(-jitter_steps, jitter_steps + 1)
            releases.extend([realized_step] * len(event["requirements"]))
        if len(releases) != len(requirements):
            raise RuntimeError(f"{self.type}: arrival plan length mismatch.")
        plan = [
            {
                "task_id": index,
                "requirement": int(requirement),
                "position": tuple(position),
                "release_after_step": int(release_step),
                "released": False,
                "completed": False,
            }
            for index, (requirement, position, release_step) in enumerate(
                zip(requirements, positions, releases)
            )
        ]
        if self.type == "probe_interaction":
            for task in plan:
                station = tuple(task["position"])
                task.update(
                    expired=False,
                    ready_step=None,
                    service_deadline_step=None,
                    scheduled_ready_step=int(
                        self._probe2_station_ready_steps[station]
                    ),
                )
        return plan

    def _release_due_tasks(self, after_step: int) -> None:
        for task in self._task_plan:
            if not task["released"] and task["release_after_step"] == after_step:
                task["released"] = True
                self._released_tasks += 1
                self._placement_backlog.append(task)
        self._place_backlog()

    def _place_backlog(self) -> None:
        retained: Deque[Dict[str, object]] = deque()
        while self._placement_backlog:
            task = self._placement_backlog.popleft()
            row, col = task["position"]
            if self._is_empty_location(row, col):
                self.field[row, col] = int(task["requirement"])
                self._active_task_by_position[(row, col)] = task
            else:
                retained.append(task)
        self._placement_backlog = retained

    def _gen_valid_moves(self) -> None:
        self._valid_actions = {
            player: [action for action in Action if self._is_valid_action(player, action)]
            for player in self.players
        }

    def _is_valid_action(self, player: Player, action: Action) -> bool:
        if player.id in self._probe_locked_agents:
            return action == Action.NONE
        row, col = player.position
        if action == Action.NONE:
            return True
        if self.type == "probe_interaction" and action in {
            Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST
        }:
            # Probe 2 controls navigation so its outcome isolates LOAD consensus.
            return False
        if action == Action.NORTH:
            return row > 0 and self.field[row - 1, col] == 0
        if action == Action.SOUTH:
            return row < self.rows - 1 and self.field[row + 1, col] == 0
        if action == Action.WEST:
            return col > 0 and self.field[row, col - 1] == 0
        if action == Action.EAST:
            return col < self.cols - 1 and self.field[row, col + 1] == 0
        if action == Action.LOAD:
            if self.current_step < self._load_cooldown_until.get(player.id, 0):
                return False
            return self._adjacent_task_location(player.position) is not None
        return False

    def _adjacent_task_location(
        self, position: Tuple[int, int]
    ) -> Tuple[int, int] | None:
        row, col = position
        candidates = ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
        for task_row, task_col in candidates:
            if 0 <= task_row < self.rows and 0 <= task_col < self.cols:
                if self.field[task_row, task_col] > 0:
                    return task_row, task_col
        return None

    def _make_observations(self) -> List[np.ndarray]:
        pad = self.sight
        food_layer = np.pad(self.field.astype(np.float32), pad, mode="constant")
        access = np.ones(self.field_size, dtype=np.float32)
        access[self.field > 0] = 0.0
        access_layer = np.pad(access, pad, mode="constant", constant_values=0.0)
        roster = np.asarray(self.fixed_levels, dtype=np.float32)
        clock = np.asarray(
            [min(1.0, self.current_step / self._max_episode_steps)], dtype=np.float32
        )

        observations = []
        width = 1 + 2 * self.sight
        for player in self.players:
            row, col = player.position
            local_food = food_layer[row : row + width, col : col + width].reshape(-1)
            local_access = access_layer[row : row + width, col : col + width].reshape(-1)
            axis = np.zeros(self.rows + self.cols, dtype=np.float32)
            axis[row] = player.level
            axis[self.rows + col] = player.level
            pieces = [local_food, local_access, axis, roster, clock]
            if self.type == "probe_complementary":
                pieces.append(
                    np.asarray(
                        [float(player.id in self._probe_locked_agents)],
                        dtype=np.float32,
                    )
                )
            elif self.type == "probe_interaction":
                pieces.append(
                    np.asarray([self._probe2_sensor_value(player)], dtype=np.float32)
                )
            observations.append(
                np.concatenate(pieces)
                .astype(np.float32)
                .reshape(-1, 1)
            )
        return observations

    def _info(self) -> Dict[str, object]:
        workload_total = len(self._task_plan)
        active_tasks = int(np.count_nonzero(self.field))
        backlog_tasks = len(self._placement_backlog)
        unreleased_tasks = workload_total - self._released_tasks
        remaining = workload_total - self._completed_tasks
        info = {
            "scenario_id": self.type,
            "protocol_version": self.protocol_version,
            "scenario_registry_sha256": self.registry_sha256,
            "num_collisions": int(self.num_collisions),
            "initial_food": int(workload_total),
            "remaining_food": int(remaining),
            "workload_total": int(workload_total),
            "workload_arrived": int(self._released_tasks),
            "workload_completed": int(self._completed_tasks),
            "workload_active": int(active_tasks),
            "workload_backlog": int(backlog_tasks),
            "workload_unreleased": int(unreleased_tasks),
            "capacity_workload_total": int(self._capacity_workload_total),
            "capacity_workload_completed": int(self._capacity_workload_completed),
            "task_completion_ratio": float(
                self._completed_tasks / workload_total if workload_total else 0.0
            ),
            "episode_success": float(
                self.current_step >= self._max_episode_steps
                and self._completed_tasks == workload_total
            ),
            "probe_locked_agents": sorted(self._probe_locked_agents),
            "probe_locked_agent_count": len(self._probe_locked_agents),
            "probe1_exact_service_pairs": int(self._probe1_exact_service_pairs),
            "probe1_exact_service_pair_coverage": float(
                self._probe1_exact_service_pairs / max(1, len(self._task_plan))
                if self.type == "probe_complementary" else 0.0
            ),
            "probe2_premature_load_attempts": int(
                self._probe2_premature_load_attempts
            ),
            "probe2_ready_load_participations": int(
                self._probe2_ready_load_participations
            ),
            "probe2_incomplete_ready_load_attempts": int(
                self._probe2_incomplete_ready_load_attempts
            ),
            "probe2_safe_completions": int(self._probe2_safe_completions),
            "probe2_safe_completion_ratio": float(
                self._probe2_safe_completions / workload_total
                if self.type == "probe_interaction" and workload_total else 0.0
            ),
            "probe2_safe_episode_success": float(
                self.type == "probe_interaction"
                and self.current_step >= self._max_episode_steps
                and self._probe2_safe_completions == workload_total
            ),
            "probe2_unsafe_station_count": int(len(self._probe2_unsafe_stations)),
            "probe2_missed_service_windows": int(
                self._probe2_missed_service_windows
            ),
            "probe2_expired_tasks": int(self._probe2_expired_tasks),
            "probe2_ready_to_success_delay_sum": int(
                sum(self._probe2_ready_to_success_delays)
            ),
            "probe2_ready_to_success_delay_count": int(
                len(self._probe2_ready_to_success_delays)
            ),
            "probe2_mean_ready_to_success_delay": float(
                np.mean(self._probe2_ready_to_success_delays)
                if self._probe2_ready_to_success_delays else 0.0
            ),
            "probe2_station_ready_steps": {
                f"{row},{col}": int(ready_step)
                for (row, col), ready_step in self._probe2_station_ready_steps.items()
            },
            "probe2_sensor_values": {
                str(player.id): self._probe2_sensor_value(player)
                for player in self.players
                if self.type == "probe_interaction"
            },
            "probe_load_cooldown_until": {
                str(player_id): int(until_step)
                for player_id, until_step in self._load_cooldown_until.items()
                if until_step > self.current_step
            },
            "arrival_schedule": [
                {
                    "task_id": int(task["task_id"]),
                    "requirement": int(task["requirement"]),
                    "release_after_step": int(task["release_after_step"]),
                }
                for task in self._task_plan
            ],
        }
        if self.type == "probe_interaction":
            support = self._probe2_fixed_incidence.T @ self._probe2_fixed_incidence
            directed_links = int(
                np.count_nonzero(support > 0)
                - np.count_nonzero(np.diag(support) > 0)
            )
            info.update(
                {
                    "probe2_fixed_incidence": self._probe2_fixed_incidence.tolist(),
                    "probe2_agent_roles": {
                        str(agent): role
                        for agent, role in self._probe2_agent_role.items()
                    },
                    "probe2_station_chains": {
                        f"{row},{col}": list(chain)
                        for (row, col), chain in self._probe2_station_chains.items()
                    },
                    "probe2_required_sensor_to_actuator_hops": int(
                        self.probe_rules["required_sensor_to_actuator_hops"]
                    ),
                    "probe2_communication_scaffold_directed_links": directed_links,
                    "probe2_actuator_load_attempts": int(
                        self._probe2_actuator_load_attempts
                    ),
                }
            )
        return info

    def reset(self):
        self.field.fill(0)
        self.num_collisions = 0
        self.current_step = 0
        self._game_over = False
        self._released_tasks = 0
        self._completed_tasks = 0
        self._capacity_workload_completed = 0
        self._probe_locked_agents: set[int] = set()
        self._load_cooldown_until: Dict[int, int] = {}
        self._probe2_station_ready_steps: Dict[Tuple[int, int], int] = {}
        self._probe2_agent_station: Dict[int, Tuple[int, int]] = {}
        self._probe2_agent_role: Dict[int, str] = {}
        self._probe2_station_chains: Dict[Tuple[int, int], Tuple[int, ...]] = {}
        self._probe2_sensor_noise: Dict[int, np.ndarray] = {}
        self._probe2_fixed_incidence = np.zeros(
            (self.n_agents, self.n_agents), dtype=np.float32
        )
        self._probe1_exact_service_pairs = 0
        self._probe2_premature_load_attempts = 0
        self._probe2_ready_load_participations = 0
        self._probe2_incomplete_ready_load_attempts = 0
        self._probe2_safe_completions = 0
        self._probe2_unsafe_stations: set[Tuple[int, int]] = set()
        self._probe2_ready_credited_agents: set[int] = set()
        self._probe2_missed_service_windows = 0
        self._probe2_expired_tasks = 0
        self._probe2_ready_to_success_delays: List[int] = []
        self._probe2_actuator_load_attempts = 0
        self._placement_backlog: Deque[Dict[str, object]] = deque()
        self._active_task_by_position: Dict[Tuple[int, int], Dict[str, object]] = {}
        for player in self.players:
            player.position = None
            player.reward = 0.0
            player.score = 0.0
        self._spawn_players()
        self._initialize_probe2_readiness_process()
        self._task_plan = self._build_task_plan()
        self._capacity_workload_total = sum(
            int(task["requirement"]) for task in self._task_plan
        )
        self.initial_food = len(self._task_plan)
        self._release_due_tasks(after_step=0)
        if self._placement_backlog:
            raise RuntimeError(f"{self.type}: initial tasks could not be placed.")
        self.remaining_food = len(self._task_plan)
        self._food_spawned = self.initial_food
        self._gen_valid_moves()
        return self._make_observations()

    @staticmethod
    def _normalized_actions(actions: Sequence[object]) -> List[int]:
        normalized = []
        for action in actions:
            normalized.append(int(np.asarray(action).reshape(-1)[0]))
        return normalized

    def step(self, actions: Sequence[object]):
        if self._game_over:
            raise RuntimeError("step() called after the fixed episode horizon.")
        if len(actions) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} actions, received {len(actions)}.")

        self.current_step += 1
        for player in self.players:
            player.reward = 0.0

        parsed_actions: List[Action] = []
        for player, raw_action in zip(self.players, self._normalized_actions(actions)):
            try:
                action = Action(raw_action)
            except ValueError:
                action = Action.NONE
            if action not in self._valid_actions[player]:
                action = Action.NONE
            parsed_actions.append(action)

        intended_destinations: Dict[Tuple[int, int], List[Player]] = defaultdict(list)
        loaders: List[Player] = []
        for player, action in zip(self.players, parsed_actions):
            row, col = player.position
            destination = (row, col)
            if action == Action.NORTH:
                destination = (row - 1, col)
            elif action == Action.SOUTH:
                destination = (row + 1, col)
            elif action == Action.WEST:
                destination = (row, col - 1)
            elif action == Action.EAST:
                destination = (row, col + 1)
            elif action == Action.LOAD:
                loaders.append(player)
            intended_destinations[destination].append(player)

        for destination, destination_players in intended_destinations.items():
            if len(destination_players) > 1:
                self.num_collisions += 1
            else:
                destination_players[0].position = destination

        if self.type == "probe_interaction":
            service_window = int(self.probe_rules["readiness_window_steps"])
            for task_position, task in self._active_task_by_position.items():
                if task["ready_step"] is not None:
                    continue
                if self.current_step >= int(task["scheduled_ready_step"]):
                    task["ready_step"] = int(task["scheduled_ready_step"])
                    task["service_deadline_step"] = (
                        int(task["ready_step"]) + service_window - 1
                    )

        loaders_by_task: Dict[Tuple[int, int], List[Player]] = defaultdict(list)
        if self.type == "probe_interaction":
            self._probe2_actuator_load_attempts += len(loaders)
        for player in loaders:
            task_position = self._adjacent_task_location(player.position)
            if task_position is not None:
                loaders_by_task[task_position].append(player)

        for task_position in sorted(loaders_by_task):
            coalition = sorted(loaders_by_task[task_position], key=lambda p: p.id)
            requirement = int(self.field[task_position])
            capacity = sum(player.level for player in coalition)
            task = self._active_task_by_position.get(task_position)
            before_station_ready = bool(
                self.type == "probe_interaction"
                and task is not None
                and self.current_step < int(task["scheduled_ready_step"])
            )
            if requirement <= 0:
                continue
            if self.type == "probe_interaction":
                if before_station_ready:
                    self._probe2_premature_load_attempts += 1
                    self._probe2_unsafe_stations.add(task_position)
                    cooldown = int(
                        self.probe_rules["premature_load_cooldown_steps"]
                    )
                    penalty = float(self.probe_rules["false_start_penalty"])
                    for player in coalition:
                        self._load_cooldown_until[player.id] = max(
                            self._load_cooldown_until.get(player.id, 0),
                            self.current_step + cooldown,
                        )
                        player.reward -= penalty
                    continue

                # Supply individual credit for recognizing a ready station, but
                # only once per agent so incomplete coalitions cannot farm reward.
                self._probe2_ready_load_participations += len(coalition)
                ready_credit = float(self.probe_rules["ready_load_credit"])
                for player in coalition:
                    if player.id not in self._probe2_ready_credited_agents:
                        player.reward += ready_credit
                        self._probe2_ready_credited_agents.add(player.id)
                if capacity < requirement:
                    self._probe2_incomplete_ready_load_attempts += 1
                    continue
            elif capacity < requirement:
                continue
            excess_capacity = capacity - requirement
            task_reward = max(
                0.0,
                requirement
                - self.excess_capacity_penalty * excess_capacity
                - self.coalition_member_penalty * max(0, len(coalition) - 1),
            )
            individual_reward = task_reward / len(coalition)
            for player in coalition:
                player.reward += individual_reward
            self.field[task_position] = 0
            task = self._active_task_by_position.pop(task_position)
            task["completed"] = True
            if self.type == "probe_interaction" and task["ready_step"] is not None:
                self._probe2_ready_to_success_delays.append(
                    self.current_step - int(task["ready_step"])
                )
                if task_position not in self._probe2_unsafe_stations:
                    self._probe2_safe_completions += 1
            self._completed_tasks += 1
            self._capacity_workload_completed += requirement
            if self.type == "probe_complementary":
                if len(coalition) == 2 and capacity == requirement:
                    self._probe1_exact_service_pairs += 1
                if self.probe_rules.get("service_lock_after_success", False):
                    self._probe_locked_agents.update(player.id for player in coalition)

        if self.type == "probe_interaction":
            for task_position, task in list(self._active_task_by_position.items()):
                deadline = task["service_deadline_step"]
                if deadline is None or self.current_step < int(deadline):
                    continue
                self.field[task_position] = 0
                self._active_task_by_position.pop(task_position)
                task["expired"] = True
                self._probe2_missed_service_windows += 1
                self._probe2_expired_tasks += 1

        self._release_due_tasks(after_step=self.current_step)
        self.remaining_food = len(self._task_plan) - self._completed_tasks
        self._game_over = self.current_step >= self._max_episode_steps
        self._gen_valid_moves()
        for player in self.players:
            player.score += player.reward

        observations = self._make_observations()
        rewards = [float(player.reward) for player in self.players]
        dones = [self._game_over for _ in self.players]
        return observations, rewards, dones, self._info()

    @staticmethod
    def _adjacent_to(
        player_position: Tuple[int, int], task_position: Tuple[int, int]
    ) -> bool:
        return (
            abs(player_position[0] - task_position[0])
            + abs(player_position[1] - task_position[1])
            == 1
        )

    def close(self):
        return None

    def render(self, mode="human"):
        return None


ForagingEnv = FormalForagingEnv


def make_lbf_env(env_type: str) -> FormalForagingEnv:
    if normalize_scenario_id(env_type) == "probe_interaction":
        from probe2_interlock import Probe2InterlockEnv
        return Probe2InterlockEnv(env_type)
    return FormalForagingEnv(env_type)
