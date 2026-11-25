"""Shared validation and statistics for the Graph Intelligence experiments."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from paper_statistics import holm, paired_sign_flip
from summarize_test_results import ci95_half_width, mean, sample_sd


PROJECT = Path(__file__).resolve().parent
FORMAL_SEEDS = [20, 23, 56, 89, 2025]
FORMAL_EVAL_SEEDS = list(range(1001, 1011))
FORMAL_EVAL_EPISODES = 20

TEST_METRICS = {
    "Task_Completion_Ratio": "higher",
    "Capacity_Completion_Ratio": "higher",
    "Episode_Success_Rate": "higher",
    "Collision_Rate": "lower",
    "Synchronization_Success_Rate": "higher",
    "Missing_Member_Rate": "lower",
    "Mean_Active_Link_Ratio": "lower",
    "Mean_Active_Links_Per_Step": "lower",
    "Estimated_Payload_Bytes_Per_Step": "lower",
}
COMMUNICATION_METRICS = {
    "Logical_Transfers_Per_Step": "lower",
    "Logical_Feature_Scalars_Per_Step": "lower",
    "Incidence_Traversals_Per_Step": "lower",
    "Logical_Feature_Bytes_FP32_Per_Step": "lower",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def locate_single_run(root: Path, method: str, scenario: str, seed: int) -> Path:
    seed_root = root / method / f"lbf_{scenario}-v0" / f"seed_{seed}"
    runs = sorted(path for path in seed_root.glob("run*") if path.is_dir())
    if len(runs) != 1:
        raise RuntimeError(
            f"Expected one immutable run below {seed_root}; found {len(runs)}."
        )
    return runs[0]


def validate_result_run(
    run: Path,
    *,
    method: str,
    scenario: str,
    seed: int,
    eval_seeds: list[int],
    eval_episodes: int,
    require_checkpoint: bool,
) -> dict:
    required = {
        "config.json",
        "training_protocol.json",
        "evaluation_protocol.json",
        "test_metrics.csv",
        "comparison_metrics.csv",
        "communication_costs.csv",
    }
    if require_checkpoint:
        required.update({"model.pt", "training_metrics.csv"})
    missing = sorted(name for name in required if not (run / name).is_file())
    if missing:
        raise RuntimeError(f"Missing experiment artifacts {missing}: {run}")

    config = read_json(run / "config.json")
    training = read_json(run / "training_protocol.json")
    evaluation = read_json(run / "evaluation_protocol.json")
    test_rows = read_csv(run / "test_metrics.csv")
    communication_rows = read_csv(run / "communication_costs.csv")
    expected_scenario = f"lbf_{scenario}-v0"
    expected_cases = {
        (eval_seed, episode)
        for eval_seed in eval_seeds
        for episode in range(1, eval_episodes + 1)
    }
    checks = {
        "config_scenario": config.get("env_name") == expected_scenario,
        "config_seed": int(config.get("seed", -1)) == seed,
        "training_method": training.get("method") == method,
        "training_seed": int(training.get("seed", -1)) == seed,
        "evaluation_seeds": evaluation.get("eval_seeds") == eval_seeds,
        "evaluation_episodes": int(evaluation.get("episodes_per_seed", -1))
        == eval_episodes,
        "checkpoint_loaded": bool(evaluation.get("checkpoint_load_verified")),
        "parameters_unchanged": bool(evaluation.get("parameters_unchanged")),
        "test_row_count": len(test_rows) == len(expected_cases),
        "communication_row_count": len(communication_rows) == len(expected_cases),
        "test_cases": {
            (int(row["Eval_Seed"]), int(row["Episode"])) for row in test_rows
        }
        == expected_cases,
        "communication_cases": {
            (int(row["Eval_Seed"]), int(row["Episode"]))
            for row in communication_rows
        }
        == expected_cases,
    }
    if require_checkpoint:
        checks.update(
            {
                "backward_verified": bool(training.get("backward_verified")),
                "optimizer_step_verified": bool(
                    training.get("optimizer_step_verified")
                ),
                "parameters_finite": bool(training.get("parameters_finite")),
            }
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Experiment artifact validation failed {failed}: {run}")

    for row in test_rows:
        if (
            row.get("Method") != method
            or row.get("Scenario") != expected_scenario
            or int(row.get("Train_Seed", -1)) != seed
        ):
            raise RuntimeError(f"Test identity mismatch: {run}")
        for metric in TEST_METRICS:
            if metric not in row or not math.isfinite(float(row[metric])):
                raise RuntimeError(f"Invalid test metric {metric}: {run}")
    for row in communication_rows:
        if (
            row.get("Method") != method
            or row.get("Scenario") != expected_scenario
            or int(row.get("Train_Seed", -1)) != seed
        ):
            raise RuntimeError(f"Communication identity mismatch: {run}")
        for metric in COMMUNICATION_METRICS:
            if metric not in row or not math.isfinite(float(row[metric])):
                raise RuntimeError(f"Invalid communication metric {metric}: {run}")
    return {
        "run_dir": str(run.resolve().relative_to(PROJECT)),
        "test_rows": len(test_rows),
        "communication_rows": len(communication_rows),
        "status": "PASS",
    }


def aggregate_experiment(
    records: list[dict],
    *,
    output_dir: Path,
    prefix: str,
    reference_method: str,
    include_dropout_metrics: bool = False,
) -> list[Path]:
    """Aggregate episodes within seed, then summarize and pair five seeds."""
    metric_directions = {**TEST_METRICS, **COMMUNICATION_METRICS}
    by_seed = []
    for record in records:
        run = Path(record["run_dir"])
        if run.is_absolute():
            raise ValueError(f"Experiment run paths must be relative: {run}")
        run = (PROJECT / run).resolve()
        try:
            run.relative_to(PROJECT)
        except ValueError as error:
            raise ValueError(
                f"Experiment run paths must remain inside CAHC: {record['run_dir']}"
            ) from error
        test_rows = read_csv(run / "test_metrics.csv")
        communication_rows = read_csv(run / "communication_costs.csv")
        row = {
            "Method": record["method"],
            "Scenario": f"lbf_{record['scenario']}-v0",
            "Train_Seed": record["seed"],
            "N_Test_Episodes": len(test_rows),
        }
        for metric in TEST_METRICS:
            row[metric] = f"{mean([float(item[metric]) for item in test_rows]):.9f}"
        for metric in COMMUNICATION_METRICS:
            row[metric] = f"{mean([float(item[metric]) for item in communication_rows]):.9f}"
        if include_dropout_metrics:
            missing_dropout_metrics = sorted(
                name
                for name in (
                    "Requested_Link_Dropout_Rate",
                    "Realized_Active_Link_Dropout_Rate",
                )
                if name not in test_rows[0]
            )
            if missing_dropout_metrics:
                raise RuntimeError(
                    f"Missing dropout metrics {missing_dropout_metrics}: {run}"
                )
            row["Requested_Link_Dropout_Rate"] = f"{mean([float(item['Requested_Link_Dropout_Rate']) for item in test_rows]):.9f}"
            row["Realized_Active_Link_Dropout_Rate"] = f"{mean([float(item['Realized_Active_Link_Dropout_Rate']) for item in test_rows]):.9f}"
        by_seed.append(row)

    identities = [
        (row["Method"], row["Scenario"], int(row["Train_Seed"]))
        for row in by_seed
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("Duplicate method/scenario/training-seed result group.")

    by_seed_path = output_dir / f"{prefix}_by_seed.csv"
    write_csv(by_seed_path, by_seed)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in by_seed:
        grouped[(row["Method"], row["Scenario"])].append(row)
    summary = []
    for (method, scenario), rows in sorted(grouped.items()):
        seeds = sorted(int(row["Train_Seed"]) for row in rows)
        if seeds != FORMAL_SEEDS:
            raise RuntimeError(
                f"Expected formal seeds {FORMAL_SEEDS}; got {seeds} for "
                f"{method}/{scenario}."
            )
        optional = [
            metric
            for metric in (
                "Requested_Link_Dropout_Rate",
                "Realized_Active_Link_Dropout_Rate",
            )
            if metric in rows[0]
        ]
        for metric in [*metric_directions, *optional]:
            values = [float(row[metric]) for row in rows]
            summary.append(
                {
                    "Method": method,
                    "Scenario": scenario,
                    "Metric": metric,
                    "Mean": f"{mean(values):.9f}",
                    "Sample_SD_Across_Train_Seeds": f"{sample_sd(values):.9f}",
                    "CI95_Half_Width": f"{ci95_half_width(values):.9f}",
                    "N_Train_Seeds": len(values),
                    "Train_Seeds": " ".join(str(seed) for seed in seeds),
                }
            )
    summary_path = output_dir / f"{prefix}_summary.csv"
    write_csv(summary_path, summary)

    indexed = {
        (row["Method"], row["Scenario"], int(row["Train_Seed"]), metric):
        float(row[metric])
        for row in by_seed
        for metric in metric_directions
    }
    methods = sorted({row["Method"] for row in by_seed})
    scenarios = sorted({row["Scenario"] for row in by_seed})
    paired = []
    for method in methods:
        if method == reference_method:
            continue
        for scenario in scenarios:
            for metric, direction in metric_directions.items():
                differences = []
                advantages = []
                for seed in FORMAL_SEEDS:
                    reference_key = (reference_method, scenario, seed, metric)
                    method_key = (method, scenario, seed, metric)
                    if reference_key not in indexed or method_key not in indexed:
                        continue
                    reference = indexed[reference_key]
                    comparator = indexed[method_key]
                    difference = reference - comparator
                    differences.append(difference)
                    advantages.append(
                        difference if direction == "higher" else -difference
                    )
                if not differences:
                    continue
                if len(differences) != len(FORMAL_SEEDS):
                    raise RuntimeError(
                        f"Incomplete paired comparison: {method}/{scenario}/{metric}."
                    )
                paired.append(
                    {
                        "Reference": reference_method,
                        "Comparator": method,
                        "Scenario": scenario,
                        "Metric": metric,
                        "Preferred_Direction": direction,
                        "Mean_Reference_Minus_Comparator": f"{mean(differences):.9f}",
                        "Mean_Reference_Advantage": f"{mean(advantages):.9f}",
                        "Sample_SD_of_Advantage": f"{sample_sd(advantages):.9f}",
                        "CI95_Half_Width_of_Advantage": f"{ci95_half_width(advantages):.9f}",
                        "Exact_P_Two_Sided": paired_sign_flip(advantages),
                        "N_Paired_Train_Seeds": len(advantages),
                        "Train_Seeds": " ".join(str(seed) for seed in FORMAL_SEEDS),
                    }
                )
    adjusted = holm([row["Exact_P_Two_Sided"] for row in paired])
    for row, value in zip(paired, adjusted):
        row["Holm_Adjusted_P"] = value
        row["Significant_At_0_05"] = value < 0.05
    paired_path = output_dir / f"{prefix}_paired_comparisons.csv"
    write_csv(paired_path, paired)
    return [by_seed_path, summary_path, paired_path]
