

from __future__ import annotations

from collections import deque
import gymnasium as gym
import numpy as np

from shared_lbf_env import FormalForagingEnv


PRIMARY_METRIC = "Probe2_Balanced_Safe_Coordination"


class Probe2InterlockEnv(FormalForagingEnv):
    load_action = 1
    is_probe2 = True

    def __init__(self, scenario_id="probe_interaction"):

        self._evaluation_mode = True
        super().__init__(scenario_id)
        self.action_space = gym.spaces.Tuple(
            tuple(gym.spaces.Discrete(2) for _ in self.players)
        )

    def _get_observation_space(self):
        original = super()._get_observation_space()

        return gym.spaces.Box(
            low=np.insert(original.low, len(original.low) - 1, np.zeros(9)),
            high=np.insert(original.high, len(original.high) - 1, np.ones(9)),
            dtype=np.float32,
        )

    def _fixed_positions(self):
        positions = [tuple(p) for p in self.probe_rules["agent_positions"]]
        return [positions[i] for i in self._shuffle(range(len(positions)))]

    def _station_member_count(self) -> int:
        return int(self.probe_rules["station_member_count"])

    def set_evaluation_mode(self, value: bool) -> None:

        self._evaluation_mode = bool(value)

    def _build_communication_incidence(self) -> np.ndarray:

        n = self.n_agents
        width = self._station_member_count()
        incidence = np.zeros((n, n), dtype=np.float32)
        for members in self._station_members:
            for slot, member in enumerate(members):


                incidence[member, member] = 1.0
                if slot + 1 < width:
                    incidence[member, members[slot + 1]] = 1.0
        return incidence

    def probe2_public_communication_info(self):


        return {
            "probe2_fixed_incidence": self._communication_incidence.copy(),
            "probe2_force_physical_links": True,
            "probe2_required_sensor_to_actuator_hops": int(
                self.probe_rules["max_propagation_hops"]
            ),
            "probe2_communication_scaffold_directed_links": int(
                np.count_nonzero(self._communication_incidence.T
                                 @ self._communication_incidence)
                - np.count_nonzero(np.diag(
                    self._communication_incidence.T
                    @ self._communication_incidence))
            ),
        }

    def _probe2_sensor_value(self, player):
        t = min(self.current_step, self._max_episode_steps - 1)
        station, member = self._membership[player.id]
        if member != 0:
            return float(self.probe_rules["relay_neutral_level"])
        bit = self._signals[t, station]
        base = self.probe_rules["sensor_ready_level" if bit else "sensor_not_ready_level"]
        return float(base + self._noise[t, station])

    def _make_observations(self):
        observations = super()._make_observations()
        for player, observation in zip(self.players, observations):
            station, hop = self._membership[player.id]
            public_identity = np.concatenate([np.eye(4)[station], np.eye(5)[hop]])
            observations[player.id] = np.insert(
                observation.reshape(-1), observation.size - 1, public_identity
            ).astype(np.float32).reshape(-1, 1)
        return observations

    def reset(self):
        self.current_step = 0
        self._game_over = False
        self.num_collisions = 0
        self.field.fill(0)
        self._probe_locked_agents = set()
        self._placement_backlog = deque()
        self._load_cooldown_until = {}
        self._station_positions = [tuple(p) for p in self.probe_rules["task_positions"]]
        positions = self._fixed_positions()
        for player, position in zip(self.players, positions):
            player.setup(position, 1, self.field_size)
        width = self._station_member_count()
        stations = self.n_agents // width
        if len(self._station_positions) != stations:
            raise RuntimeError("Probe2 V1 station and controller counts disagree.")


        membership_by_position = {
            tuple(position): (index // width, index % width)
            for index, position in enumerate(self.probe_rules["agent_positions"])
        }
        self._station_members = [[None] * width for _ in range(stations)]
        self._membership = {}
        for player in self.players:
            station, slot = membership_by_position[tuple(player.position)]
            self._station_members[station][slot] = player.id
            self._membership[player.id] = (station, slot)
        for station, members in enumerate(self._station_members):
            if any(member is None for member in members):
                raise RuntimeError("Probe2 V1 requires five ordered controllers per station.")
        for position in self._station_positions:


            self.field[position] = width
        self._communication_incidence = self._build_communication_incidence()
        horizon = self._max_episode_steps
        ready = self.np_random.random((horizon, stations)) < self.probe_rules["station_ready_probability"]
        self._signals = ready.astype(np.int8)
        noise = self.probe_rules["sensor_noise_half_width"]
        self._noise = self.np_random.uniform(
            -noise, noise, size=(horizon, stations)
        )
        self._trial_plan = []
        for t in range(horizon):
            trials = []
            for s, position in enumerate(self._station_positions):
                trials.append(dict(
                    task_id=t * stations + s, requirement=width,
                    position=position, release_after_step=t, released=t == 0,
                    completed=False, eligible=bool(ready[t, s]), expired=False,
                ))
            self._trial_plan.append(trials)

        self._task_plan = [task for trials in self._trial_plan for task in trials if task["eligible"]]
        self._capacity_workload_total = len(self._task_plan) * width
        self.initial_food = self._food_spawned = len(self._task_plan)
        self.remaining_food = len(self._task_plan)
        self._completed_tasks = self._capacity_workload_completed = 0
        self._open_seen = self._blocked_seen = self._correct_holds = 0
        self._unsafe_decisions = self._unsafe_individual = self._incomplete = 0
        self._correct_individual = self._load_attempts = self._ready_participations = 0
        self._hop_correct = np.zeros(width, dtype=np.int64)
        self._hop_decisions = np.zeros(width, dtype=np.int64)
        self._unsafe_stations = set()
        self._last_decision = {}
        self._activate_slot()
        return self._make_observations()

    def _activate_slot(self):
        if self.current_step >= self._max_episode_steps:
            self._active_task_by_position = {}
        else:
            trials = self._trial_plan[self.current_step]
            for task in trials:
                task["released"] = True
            self._active_task_by_position = {task["position"]: task for task in trials}
        self._released_tasks = sum(task["released"] for task in self._task_plan)

    def step(self, actions):
        if self._game_over:
            raise RuntimeError("step() called after the fixed episode horizon.")
        values = self._normalized_actions(actions)
        width = self._station_member_count()
        if len(values) != self.n_agents or any(value not in (0, 1) for value in values):
            raise ValueError(
                "Probe2 V1 actions must be twenty binary WAIT=0 / LOAD=1 choices."
            )
        rewards = np.zeros(self.n_agents, dtype=np.float64)
        outcomes = []
        for s, task in enumerate(self._trial_plan[self.current_step]):
            members = self._station_members[s]
            chosen = np.asarray([values[i] for i in members])
            ready = bool(task["eligible"])
            correct = chosen == int(ready)
            joint_correct = bool(correct.all())
            joint_reward = (
                self.probe_rules["decision_correct_reward"]
                if joint_correct
                else self.probe_rules["decision_incorrect_reward"]
            )
            shaping = float(self.probe_rules.get("individual_shaping_weight", 0.0))
            rewards[members] = joint_reward + shaping * (2.0 * correct - 1.0)
            self._correct_individual += int(correct.sum())
            self._hop_correct += correct.astype(np.int64)
            self._hop_decisions += 1
            self._load_attempts += int(chosen.sum())
            if ready:
                self._open_seen += 1
                self._ready_participations += int(chosen.sum())
                task["completed"] = joint_correct
                task["expired"] = not joint_correct
                self._completed_tasks += int(joint_correct)
                self._incomplete += int(0 < chosen.sum() < width)
            else:
                self._blocked_seen += 1
                self._correct_holds += int(joint_correct)
                self._unsafe_decisions += int(not joint_correct)
                self._unsafe_individual += int(chosen.sum())
                if not joint_correct:
                    self._unsafe_stations.add(s)
            outcomes.append(dict(
                station=s, members=members, station_permit=int(ready),
                eligible=ready, actions=chosen.tolist(), joint_correct=joint_correct,
                service_completed=bool(task["completed"]), rewards=rewards[members].tolist(),
            ))
        self.current_step += 1
        self._capacity_workload_completed = width * self._completed_tasks
        self.remaining_food = len(self._task_plan) - self._completed_tasks
        self._game_over = self.current_step == self._max_episode_steps
        self._last_decision = {"step": self.current_step, "stations": outcomes}
        for player, reward in zip(self.players, rewards):
            player.reward = float(reward)
            player.score += float(reward)
        self._activate_slot()
        return (
            self._make_observations(), rewards.tolist(),
            [self._game_over] * self.n_agents, self._info(),
        )

    def _info(self):
        width = self._station_member_count()
        total = len(self._task_plan)
        missed = self._open_seen - self._completed_tasks
        completion = self._completed_tasks / total if total else 0.0
        service_rate = self._completed_tasks / self._open_seen if self._open_seen else 0.0
        hold_rate = self._correct_holds / self._blocked_seen if self._blocked_seen else 0.0
        both_classes = self._open_seen > 0 and self._blocked_seen > 0
        balanced = 0.5 * (service_rate + hold_rate) if both_classes else 0.0
        safe_episode = bool(
            self._game_over and self._completed_tasks == total
            and not self._unsafe_decisions
        )



        teacher_actions = [0] * self.n_agents
        for outcome in self._last_decision.get("stations", []):
            for member in outcome["members"]:
                teacher_actions[member] = int(outcome["eligible"])
        support = self._communication_incidence.T @ self._communication_incidence
        directed_links = int(
            np.count_nonzero(support) - np.count_nonzero(np.diag(support))
        )
        return dict(
            scenario_id=self.type, protocol_version=self.protocol_version,
            scenario_registry_sha256=self.registry_sha256,
            num_collisions=0, initial_food=total, remaining_food=total-self._completed_tasks,
            workload_total=total, workload_arrived=self._released_tasks,
            workload_completed=self._completed_tasks,
            workload_active=sum(t["eligible"] for t in self._active_task_by_position.values()),
            workload_backlog=0, workload_unreleased=total-self._released_tasks,
            capacity_workload_total=width*total,
            capacity_workload_completed=width*self._completed_tasks,
            task_completion_ratio=completion,
            episode_success=float(self._game_over and self._completed_tasks == total and total > 0),
            probe_locked_agents=[], probe_locked_agent_count=0,
            probe1_exact_service_pairs=0, probe1_exact_service_pair_coverage=0.0,
            probe2_premature_load_attempts=self._unsafe_decisions,
            probe2_ready_load_participations=self._ready_participations,
            probe2_incomplete_ready_load_attempts=self._incomplete,
            probe2_safe_completions=self._completed_tasks,
            probe2_safe_completion_ratio=completion,
            probe2_safe_episode_success=float(safe_episode),
            probe2_unsafe_station_count=len(self._unsafe_stations),
            probe2_missed_service_windows=missed, probe2_expired_tasks=missed,
            probe2_ready_to_success_delay_sum=self._completed_tasks,
            probe2_ready_to_success_delay_count=self._completed_tasks,
            probe2_mean_ready_to_success_delay=float(self._completed_tasks > 0),
            probe2_station_ready_steps={}, probe2_sensor_values={
                str(p.id): self._probe2_sensor_value(p) for p in self.players
            },
            probe_load_cooldown_until={}, arrival_schedule=[
                {key: task[key] for key in ("task_id", "requirement", "release_after_step")}
                for task in self._task_plan
            ],
            probe2_actuator_load_attempts=self._load_attempts,
            probe2_service_opportunities=self._open_seen,
            probe2_blocked_opportunities=self._blocked_seen,
            probe2_correct_safety_holds=self._correct_holds,
            probe2_unsafe_individual_loads=self._unsafe_individual,
            probe2_correct_individual_decisions=self._correct_individual,
            probe2_hop_correct_counts=self._hop_correct.tolist(),
            probe2_hop_decision_counts=self._hop_decisions.tolist(),
            probe2_balanced_safe_coordination=balanced,
            probe2_both_classes_observed=int(both_classes),
            probe2_support_directed_links=directed_links,
            probe2_communication_scaffold_directed_links=directed_links,
            probe2_required_sensor_to_actuator_hops=int(
                self.probe_rules["max_propagation_hops"]
            ),


            probe2_fixed_incidence=self._communication_incidence.tolist(),
            probe2_force_physical_links=True,
            probe2_evaluation_mode=int(self._evaluation_mode),
            probe2_teacher_actions=teacher_actions,
            probe2_decision_audit=self._last_decision,
        )


def evaluation_columns(raw, info):

    if not getattr(raw, "is_probe2", False):
        return {}
    columns = {
        "Probe2_Service_Opportunities": info["probe2_service_opportunities"],
        "Probe2_Blocked_Opportunities": info["probe2_blocked_opportunities"],
        "Probe2_Correct_Safety_Holds": info["probe2_correct_safety_holds"],
        "Probe2_Unsafe_Individual_Loads": info["probe2_unsafe_individual_loads"],
        "Probe2_Correct_Individual_Decisions": info["probe2_correct_individual_decisions"],
        PRIMARY_METRIC: info["probe2_balanced_safe_coordination"],
        "Probe2_Both_Classes_Observed": info["probe2_both_classes_observed"],
        "Probe2_False_Activation_Rate": (
            info["probe2_premature_load_attempts"] / info["probe2_blocked_opportunities"]
            if info["probe2_blocked_opportunities"] else 0.0
        ),
        "Probe2_Native_Communication": 1,
        "Probe2_Forced_Physical_Links": 1,
        "Probe2_Learned_Router_Enabled": 0,
        "Probe2_Action_Count": 2,
        "Probe2_Evaluation_Mode": info["probe2_evaluation_mode"],
    }
    for hop, (correct, total) in enumerate(zip(
        info["probe2_hop_correct_counts"], info["probe2_hop_decision_counts"]
    )):
        columns[f"Probe2_Hop_{hop}_Correct"] = int(correct)
        columns[f"Probe2_Hop_{hop}_Decisions"] = int(total)
        columns[f"Probe2_Hop_{hop}_Accuracy"] = float(correct / total if total else 0.0)
    return columns
