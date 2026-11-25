"""Validated zero-shot configuration shifts for Reviewer 2.

The six formal scenario definitions remain untouched.  This module overlays a
small, preregistered configuration change on an already constructed source
environment and rejects any change that would alter the policy interface.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any
import math


PROJECT = Path(__file__).resolve().parent
REGISTRY_PATH = PROJECT / "cross_configuration_registry.json"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _deep_update(target: dict, update: dict) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def _declared_requirements(spec: dict) -> list[int]:
    arrival = spec["tasks"]["arrival_process"]
    values = list(arrival["initial_requirements"])
    for event in arrival["events"]:
        values.extend(event["requirements"])
    return [int(value) for value in values]


def _observation_width(spec: dict) -> int:
    sight = int(spec["observation"]["sight"])
    crop_cells = (1 + 2 * sight) ** 2
    rows, cols = (int(value) for value in spec["map"]["shape"])
    agents = int(spec["agents"]["count"])
    return 2 * crop_cells + rows + cols + agents + 1


def load_design(
    scenario_registry: dict,
    path: Path | str = REGISTRY_PATH,
) -> dict:
    design_path = Path(path)
    design = json.loads(design_path.read_text(encoding="utf-8"))
    source_id = design.get("source_scenario")
    if source_id not in scenario_registry["formal_scenario_ids"]:
        raise ValueError(f"Invalid generalization source scenario: {source_id!r}")
    if design.get("training_seeds") != scenario_registry["formal_seeds"]:
        raise ValueError("Generalization training seeds must match the formal seeds.")
    eval_seeds = design.get("evaluation_seeds", [])
    if len(eval_seeds) != 10 or len(eval_seeds) != len(set(eval_seeds)):
        raise ValueError("The generalization design requires ten unique evaluation seeds.")
    if set(eval_seeds) & set(design["training_seeds"]):
        raise ValueError("Training and generalization-evaluation seeds must be disjoint.")
    if int(design.get("episodes_per_evaluation_seed", 0)) < 1:
        raise ValueError("Generalization evaluation episode count must be positive.")
    if set(design.get("configurations", {})) != {
        "source_configuration", "demand_shift", "fleet_shift"
    }:
        raise ValueError("The registered cross-configuration design is incomplete.")
    for configuration_id in design["configurations"]:
        target_spec(design, scenario_registry, configuration_id)
    design["semantic_sha256"] = canonical_sha256(
        {key: value for key, value in design.items() if key != "semantic_sha256"}
    )
    return design


def target_spec(
    design: dict,
    scenario_registry: dict,
    configuration_id: str,
) -> dict:
    try:
        definition = design["configurations"][configuration_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown target configuration {configuration_id!r}; allowed="
            f"{sorted(design['configurations'])}"
        ) from exc
    source = deepcopy(scenario_registry["scenarios"][design["source_scenario"]])
    original = deepcopy(source)
    overrides = definition.get("overrides", {})
    allowed_top_level = {"agents", "tasks"}
    if set(overrides) - allowed_top_level:
        raise ValueError(
            "Cross-configuration overrides may change only fleet capabilities "
            "or the registered task stream."
        )
    if "agents" in overrides and set(overrides["agents"]) != {"capabilities"}:
        raise ValueError("Fleet shift may change only the capability roster.")
    if "tasks" in overrides and set(overrides["tasks"]) - {
        "requirements", "arrival_process"
    }:
        raise ValueError("Demand shift may change only requirements and arrivals.")
    _deep_update(source, overrides)

    invariant_paths = [
        ("map", "shape"),
        ("agents", "count"),
        ("observation", "sight"),
    ]
    for parent, key in invariant_paths:
        if source[parent][key] != original[parent][key]:
            raise ValueError(f"Interface invariant changed: {parent}.{key}")
    if int(source["horizon"]) != int(original["horizon"]):
        raise ValueError("Cross-configuration horizon must remain unchanged.")
    if _observation_width(source) != _observation_width(original):
        raise ValueError("Cross-configuration observation width changed.")

    capabilities = [int(value) for value in source["agents"]["capabilities"]]
    original_capabilities = [
        int(value) for value in original["agents"]["capabilities"]
    ]
    if len(capabilities) != int(source["agents"]["count"]):
        raise ValueError("Target capability roster length is invalid.")
    if sum(capabilities) != sum(original_capabilities):
        raise ValueError("Fleet shift must preserve total fleet capability.")
    if max(capabilities) > max(original_capabilities):
        raise ValueError("Fleet shift contains an unseen capability embedding value.")
    if not set(capabilities).issubset(set(original_capabilities)):
        raise ValueError("Fleet shift contains a capability level unseen in training.")

    requirements = [int(value) for value in source["tasks"]["requirements"]]
    original_requirements = [
        int(value) for value in original["tasks"]["requirements"]
    ]
    if sorted(requirements) != sorted(original_requirements):
        raise ValueError("Demand shift must preserve the complete task multiset.")
    if _declared_requirements(source) != requirements:
        raise ValueError("Target arrival stream does not reproduce task requirements.")
    if len(requirements) != int(source["tasks"]["workload"]["task_count"]):
        raise ValueError("Target task count disagrees with workload declaration.")
    if sum(requirements) != int(source["tasks"]["workload"]["capacity_units"]):
        raise ValueError("Target capacity workload is inconsistent.")
    if min(requirements) <= max(capabilities):
        raise ValueError("Every target task must still require collaboration.")
    for event in source["tasks"]["arrival_process"]["events"]:
        if not 0 < int(event["after_step"]) < int(source["horizon"]):
            raise ValueError("Target arrival lies outside the episode horizon.")

    source["cross_configuration_id"] = configuration_id
    source["cross_configuration_description"] = definition["description"]
    source["cross_configuration_sha256"] = canonical_sha256(source)
    return source


def apply_to_environment(raw_environment, spec: dict) -> None:
    """Apply a validated, interface-compatible target before evaluation reset."""
    if tuple(raw_environment.field.shape) != tuple(spec["map"]["shape"]):
        raise ValueError("Target map shape is incompatible with the source environment.")
    if raw_environment.n_agents != int(spec["agents"]["count"]):
        raise ValueError("Target agent count is incompatible with the source policy.")
    if raw_environment.sight != int(spec["observation"]["sight"]):
        raise ValueError("Target sight radius is incompatible with the source policy.")
    before_width = raw_environment.observation_space[0].shape[0]
    raw_environment.spec = deepcopy(spec)
    raw_environment.fixed_levels = [
        int(value) for value in spec["agents"]["capabilities"]
    ]
    raw_environment.max_player_level = max(raw_environment.fixed_levels)
    raw_environment._max_episode_steps = int(spec["horizon"])
    raw_environment.max_food = int(spec["tasks"]["workload"]["task_count"])
    raw_environment.max_food_level = max(spec["tasks"]["requirements"])
    raw_environment.food_level_dist = [
        int(value) for value in spec["tasks"]["requirements"]
    ]
    raw_environment.generalization_configuration_id = spec[
        "cross_configuration_id"
    ]
    raw_environment.generalization_configuration_sha256 = spec[
        "cross_configuration_sha256"
    ]
    raw_environment.observation_space = type(raw_environment.observation_space)(
        tuple(raw_environment._get_observation_space() for _ in raw_environment.players)
    )
    after_width = raw_environment.observation_space[0].shape[0]
    if after_width != before_width:
        raise ValueError(
            f"Target observation width changed from {before_width} to {after_width}."
        )


def validate_evaluation_rows(rows: list[dict], spec: dict) -> None:
    if not rows:
        raise ValueError("Cross-configuration evaluation produced no rows.")
    expected_total = int(spec["tasks"]["workload"]["task_count"])
    expected_capacity = int(spec["tasks"]["workload"]["capacity_units"])
    expected_steps = int(spec["horizon"])
    for row in rows:
        completed = int(row["Workload_Completed"])
        capacity_completed = float(row["Capacity_Workload_Completed"])
        if int(row["Workload_Total"]) != expected_total:
            raise ValueError("Cross-configuration task ledger mismatch.")
        if int(row["Capacity_Workload_Total"]) != expected_capacity:
            raise ValueError("Cross-configuration capacity ledger mismatch.")
        if int(row["Episode_Steps"]) != expected_steps:
            raise ValueError("Cross-configuration episode did not use the fixed horizon.")
        if not math.isclose(
            float(row["Task_Completion_Ratio"]),
            completed / expected_total,
            abs_tol=1e-12,
        ):
            raise ValueError("Cross-configuration TCR is inconsistent.")
        if not math.isclose(
            float(row["Capacity_Completion_Ratio"]),
            capacity_completed / expected_capacity,
            abs_tol=1e-12,
        ):
            raise ValueError("Cross-configuration CCR is inconsistent.")
        success = float(completed == expected_total)
        if not math.isclose(float(row["Episode_Success_Rate"]), success):
            raise ValueError("Cross-configuration ESR is inconsistent.")
        if not all(
            math.isfinite(float(value))
            for value in row.values()
            if isinstance(value, (int, float))
        ):
            raise ValueError("Cross-configuration metrics contain a non-finite value.")
