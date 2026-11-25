from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


REGISTRY_PATH = Path(__file__).with_name("scenario_registry.json")


def normalize_scenario_id(value: str) -> str:

    scenario_id = str(value)
    if scenario_id.startswith("Foraging-"):
        scenario_id = scenario_id[len("Foraging-") :]
    if scenario_id.startswith("lbf_"):
        scenario_id = scenario_id[len("lbf_") :]
    if scenario_id.endswith("-v0"):
        scenario_id = scenario_id[: -len("-v0")]
    return scenario_id


def canonical_registry_bytes(registry: Dict[str, Any]) -> bytes:
    return json.dumps(
        registry, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def registry_sha256(registry: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_registry_bytes(registry)).hexdigest()


def validate_registry(registry: Dict[str, Any]) -> None:
    required_root = {
        "schema_version",
        "protocol_version",
        "status",
        "formal_seeds",
        "formal_scenario_ids",
        "supplementary_scenario_ids",
        "reviewer_probe_ids",
        "shared_protocol",
        "scenarios",
        "excluded_from_formal_suite",
    }
    missing_root = required_root.difference(registry)
    if missing_root:
        raise ValueError(f"Scenario registry is missing root fields: {sorted(missing_root)}")

    if registry["status"] != "FROZEN":
        raise ValueError("Formal runs require a FROZEN scenario registry.")
    if registry["formal_seeds"] != [20, 23, 56, 89, 2025]:
        raise ValueError("Formal seeds must be exactly 20, 23, 56, 89, 2025.")

    formal_ids = registry["formal_scenario_ids"]
    if len(formal_ids) != 6:
        raise ValueError("The formal suite must contain exactly 6 scenarios.")
    if len(formal_ids) != len(set(formal_ids)):
        raise ValueError("Formal scenario IDs must be unique.")
    probe_ids = registry["reviewer_probe_ids"]
    supplementary_ids = registry["supplementary_scenario_ids"]
    if supplementary_ids:
        raise ValueError(
            "The frozen revision contains only six main scenarios and two Reviewer probes."
        )
    if probe_ids != ["probe_complementary", "probe_interaction"]:
        raise ValueError("Reviewer probes must be Probe 1 and Probe 2 in order.")
    if set(formal_ids).intersection(probe_ids + supplementary_ids):
        raise ValueError("Reviewer probes must not be counted among the six comparison scenarios.")
    if set(supplementary_ids).intersection(probe_ids):
        raise ValueError("Supplementary controls and Reviewer probes must be disjoint.")
    experiment_ids = formal_ids + supplementary_ids + probe_ids
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("Comparison and probe scenario IDs must be unique.")
    if set(experiment_ids) != set(registry["scenarios"]):
        raise ValueError("Comparison and reviewer probe IDs must match scenario keys.")

    required_scenario = {
        "name",
        "family",
        "suite",
        "real_world_meaning",
        "industrial_challenges",
        "innovation_targets",
        "baseline_theoretical_difficulty",
        "map",
        "agents",
        "tasks",
        "load_synchronization",
        "collision",
        "horizon",
        "success_definition",
        "termination",
        "observation",
        "action",
        "reward",
        "communication_setting",
        "uses",
        "reviewer_comments",
        "core_claims",
    }
    for scenario_id in experiment_ids:
        spec = registry["scenarios"][scenario_id]
        missing = required_scenario.difference(spec)
        if missing:
            raise ValueError(f"{scenario_id} is missing fields: {sorted(missing)}")

        capabilities = spec["agents"]["capabilities"]
        if len(capabilities) != spec["agents"]["count"]:
            raise ValueError(f"{scenario_id}: agent count does not match capabilities.")

        requirements = spec["tasks"]["requirements"]
        workload = spec["tasks"]["workload"]
        if len(requirements) != workload["task_count"]:
            raise ValueError(f"{scenario_id}: task count does not match requirements.")
        if sum(requirements) != workload["capacity_units"]:
            raise ValueError(f"{scenario_id}: capacity workload is inconsistent.")
        if min(requirements) <= max(capabilities):
            raise ValueError(
                f"{scenario_id}: every formal task must exceed the strongest "
                "individual capability."
            )

        arrival = spec["tasks"]["arrival_process"]
        declared = list(arrival["initial_requirements"])
        for event in arrival["events"]:
            declared.extend(event["requirements"])
            jitter_steps = int(event.get("jitter_steps", 0))
            nominal_step = int(event["after_step"])
            if jitter_steps < 0:
                raise ValueError(f"{scenario_id}: arrival jitter must be nonnegative.")
            if nominal_step - jitter_steps <= 0:
                raise ValueError(
                    f"{scenario_id}: a jittered event may overlap reset release."
                )
            if nominal_step + jitter_steps >= int(spec["horizon"]):
                raise ValueError(
                    f"{scenario_id}: a jittered event may occur outside the horizon."
                )
        if declared != requirements:
            raise ValueError(
                f"{scenario_id}: arrival declaration must reproduce requirements in order."
            )

        rows, cols = spec["map"]["shape"]
        if rows < 3 or cols < 3 or spec["horizon"] <= 0:
            raise ValueError(f"{scenario_id}: invalid map or horizon.")

    for scenario_id in formal_ids:
        if registry["scenarios"][scenario_id]["suite"] != "comparison":
            raise ValueError(f"{scenario_id}: formal comparison scenario has wrong suite.")
    for scenario_id in supplementary_ids:
        if registry["scenarios"][scenario_id]["suite"] != "supplementary":
            raise ValueError(f"{scenario_id}: supplementary scenario has wrong suite.")
    for scenario_id in probe_ids:
        if registry["scenarios"][scenario_id]["suite"] != "reviewer_probe":
            raise ValueError(f"{scenario_id}: reviewer probe has wrong suite.")

    probe2_spec = registry["scenarios"]["probe_interaction"]
    rules = probe2_spec["probe_rules"]
    if rules.get("probe_version") != "V1":
        raise ValueError("Probe 2 must use the disclosed safety-broadcast V1 protocol.")
    width = int(rules["station_member_count"])
    positions, stations = rules["agent_positions"], rules["task_positions"]
    if width != 5 or len(stations) != 4 or len(positions) != width * len(stations):
        raise ValueError("Probe 2 V1 requires four stations with five controllers each.")
    if len({tuple(p) for p in positions + stations}) != len(positions) + len(stations):
        raise ValueError("Probe 2 positions and station cells must be distinct.")
    if int(rules["max_propagation_hops"]) != width - 1:
        raise ValueError("Probe 2 V1 path diameter must equal four hops.")
    if probe2_spec["agents"]["capabilities"] != [1] * len(positions):
        raise ValueError("Probe 2 V1 uses homogeneous capability to avoid a capability confound.")
    if probe2_spec["tasks"]["requirements"] != [width] * (
        len(stations) * probe2_spec["horizon"]
    ):
        raise ValueError("Probe 2 V1 must register every threshold-five service slot.")
    if rules["station_ready_probability"] != 0.5:
        raise ValueError("Probe 2 station permits use a balanced distribution.")
    if any(key in rules for key in ("fixed_incidence_edges", "station_chains")):
        raise ValueError("Probe 2 derives its physical path from station membership.")
    if rules["randomize_agent_start_assignment"] is not True:
        raise ValueError("Probe 2 requires seeded station-membership permutation.")
    if not 0 < rules["training_entropy_coefficient"] <= 0.05:
        raise ValueError("Invalid Probe 2 exploration coefficient.")
    if not 0 < rules.get("training_lr_decay", 0) <= 1:
        raise ValueError("Invalid Probe 2 learning-rate decay.")
    if rules.get("training_gradient_clip_norm", 0) <= 0:
        raise ValueError("Probe 2 requires positive shared gradient clipping.")
    if not (rules["sensor_not_ready_level"] + rules["sensor_noise_half_width"]
            < rules["sensor_decision_threshold"]
            < rules["sensor_ready_level"] - rules["sensor_noise_half_width"]):
        raise ValueError("The hop-zero permit must be observable without sensor ambiguity.")
    if rules.get("relay_neutral_level") != rules["sensor_decision_threshold"]:
        raise ValueError("Non-sensor hops must receive the neutral sensor value.")
    if not (rules["decision_incorrect_reward"] < 0 < rules["decision_correct_reward"]
            and rules["individual_shaping_weight"] >= 0):
        raise ValueError("Probe 2 rewards must favor the joint decision and shaping.")
    if set(rules["development_eval_seeds"]) & set(rules["formal_eval_seeds"]):
        raise ValueError("Development and final evaluation seeds must be disjoint.")
    if len(rules["formal_eval_seeds"]) != 10:
        raise ValueError("Probe 2 V1 requires ten held-out evaluation seeds.")
    if rules.get("training_support") != rules.get("evaluation_support"):
        raise ValueError("Probe 2 V1 must use the same path in training and evaluation.")
    if rules.get("force_physical_links") is not True:
        raise ValueError("Probe 2 V1 must enable every legal physical link for both methods.")
    if rules.get("evaluation_action_mode") != "greedy":
        raise ValueError("Probe 2 V1 uses deterministic greedy frozen evaluation.")


def load_registry(path: Path | str | None = None) -> Dict[str, Any]:
    registry_path = Path(path) if path is not None else REGISTRY_PATH
    with registry_path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    validate_registry(registry)
    return registry


def scenario_spec(value: str, registry: Dict[str, Any] | None = None) -> Dict[str, Any]:
    loaded = load_registry() if registry is None else registry
    scenario_id = normalize_scenario_id(value)
    try:
        return loaded["scenarios"][scenario_id]
    except KeyError as exc:
        allowed = ", ".join(
            loaded["formal_scenario_ids"]
            + loaded["supplementary_scenario_ids"]
            + loaded["reviewer_probe_ids"]
        )
        raise ValueError(f"Unknown experiment scenario '{scenario_id}'. Allowed: {allowed}") from exc


if __name__ == "__main__":
    loaded_registry = load_registry()
    print(registry_sha256(loaded_registry))
