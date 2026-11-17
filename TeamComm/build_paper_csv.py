"""Build publication tables from immutable TeamComm formal-run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scenario_registry import load_registry


IDENTITY_COLUMNS = {
    "Method", "Scenario", "Train_Seed", "Eval_Seed", "Episode",
}
T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows available for {path.name}")
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_number(value: str) -> float | None:
    if value == "" or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def expected_run_dirs(
    results_root: Path,
    scenario_ids: list[str] | None = None,
    baseline_ids: list[str] | None = None,
    seed_ids: list[int] | None = None,
):
    registry = load_registry()
    all_baselines = json.loads(
        Path(__file__).with_name("formal_baselines.json").read_text(encoding="utf-8")
    )["baselines"]
    scenarios = scenario_ids or registry["formal_scenario_ids"]
    baselines = baseline_ids or all_baselines
    seeds = seed_ids or registry["formal_seeds"]
    allowed_scenarios = (
        registry["formal_scenario_ids"] + registry["supplementary_scenario_ids"]
    )
    unknown_scenarios = [item for item in scenarios if item not in allowed_scenarios]
    unknown_baselines = [item for item in baselines if item not in all_baselines]
    unknown_seeds = [item for item in seeds if item not in registry["formal_seeds"]]
    if unknown_scenarios or unknown_baselines or unknown_seeds:
        raise ValueError(
            "Unknown paper-table selection: "
            f"scenarios={unknown_scenarios}, baselines={unknown_baselines}, "
            f"seeds={unknown_seeds}"
        )
    for scenario in scenarios:
        for method in baselines:
            for seed in seeds:
                seed_root = results_root / method / f"lbf_{scenario}-v0" / f"seed_{seed}"
                runs = sorted(path for path in seed_root.glob("run*") if path.is_dir())
                yield scenario, method, seed, runs


def build_tables(
    results_root: Path | str,
    output_dir: Path | str | None = None,
    require_complete: bool = True,
    scenario_ids: list[str] | None = None,
    baseline_ids: list[str] | None = None,
    seed_ids: list[int] | None = None,
) -> dict:
    results_root = Path(results_root).resolve()
    output_dir = (
        Path(output_dir).resolve() if output_dir is not None
        else results_root / "paper_tables"
    )
    evaluation_rows: list[dict] = []
    communication_rows: list[dict] = []
    training_rows: list[dict] = []
    missing: list[str] = []
    completed_jobs = 0

    expected_jobs = (
        len(scenario_ids or load_registry()["formal_scenario_ids"])
        * len(
            baseline_ids
            or json.loads(
                Path(__file__).with_name("formal_baselines.json").read_text(
                    encoding="utf-8"
                )
            )["baselines"]
        )
        * len(seed_ids or load_registry()["formal_seeds"])
    )
    for scenario, method, seed, runs in expected_run_dirs(
        results_root, scenario_ids, baseline_ids, seed_ids
    ):
        job_id = f"{scenario}/{method}/seed_{seed}"
        if len(runs) != 1:
            missing.append(f"{job_id}: expected one run directory, found {len(runs)}")
            continue
        run_dir = runs[0]
        required = [run_dir / "comparison_metrics.csv", run_dir / "communication_costs.csv"]
        training_path = (
            run_dir / "metrics.csv" if method == "qmix"
            else run_dir / f"{method}_Foraging-{scenario}_metrics.csv"
        )
        required.append(training_path)
        absent = [path.name for path in required if not path.is_file()]
        if absent:
            missing.append(f"{job_id}: missing {', '.join(absent)}")
            continue
        comparison = read_rows(required[0])
        communication = read_rows(required[1])
        training = read_rows(training_path)
        if not comparison or not communication or not training:
            missing.append(f"{job_id}: one or more CSV files are empty")
            continue
        required_metrics = {
            "Agent_Count", "Capacity_Workload_Total",
            "Capacity_Workload_Completed", "Capacity_Completion_Ratio",
            "Episode_Success", "Episode_Success_Rate", "Collisions",
            "Collision_Rate", "Episode_Steps",
        }
        absent_metrics = sorted(required_metrics.difference(comparison[0]))
        if absent_metrics:
            missing.append(
                f"{job_id}: common metrics missing ({', '.join(absent_metrics)})"
            )
            continue
        evaluation_rows.extend(comparison)
        communication_rows.extend(communication)
        for row in training:
            training_rows.append(
                {"Method": method, "Scenario": f"lbf_{scenario}-v0", "Train_Seed": seed, **row}
            )
        completed_jobs += 1

    if require_complete and missing:
        sample = "; ".join(missing[:5])
        raise ValueError(
            f"Formal result set is incomplete: {completed_jobs}/{expected_jobs} "
            f"jobs available. {sample}"
        )
    if not evaluation_rows:
        raise ValueError(f"No completed formal runs found under {results_root}")

    seed_groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in evaluation_rows:
        seed_groups[(row["Method"], row["Scenario"], int(row["Train_Seed"]))].append(row)
    numeric_metrics = [
        key for key in evaluation_rows[0]
        if key not in IDENTITY_COLUMNS
        and all(finite_number(row.get(key, "")) is not None for row in evaluation_rows)
    ]
    seed_summary = []
    for (method, scenario, seed), rows in sorted(seed_groups.items()):
        summary = {
            "Method": method,
            "Scenario": scenario,
            "Train_Seed": seed,
            "Evaluation_Episodes": len(rows),
        }
        for metric in numeric_metrics:
            summary[metric] = statistics.fmean(float(row[metric]) for row in rows)
        seed_summary.append(summary)

    scenario_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in seed_summary:
        scenario_groups[(row["Method"], row["Scenario"])].append(row)
    method_scenario_summary = []
    for (method, scenario), rows in sorted(scenario_groups.items()):
        n = len(rows)
        for metric in numeric_metrics:
            values = [float(row[metric]) for row in rows]
            mean = statistics.fmean(values)
            sd = statistics.stdev(values) if n > 1 else 0.0
            critical = T_CRITICAL_95.get(n)
            half_width = critical * sd / math.sqrt(n) if critical is not None else ""
            method_scenario_summary.append(
                {
                    "Method": method,
                    "Scenario": scenario,
                    "Metric": metric,
                    "N_Training_Seeds": n,
                    "Mean": mean,
                    "SD": sd,
                    "CI95_Low": mean - half_width if half_width != "" else "",
                    "CI95_High": mean + half_width if half_width != "" else "",
                }
            )

    write_rows(output_dir / "paper_episode_metrics.csv", evaluation_rows)
    write_rows(output_dir / "paper_communication_metrics.csv", communication_rows)
    write_rows(output_dir / "paper_training_curves.csv", training_rows)
    write_rows(output_dir / "paper_seed_summary.csv", seed_summary)
    write_rows(output_dir / "paper_method_scenario_summary.csv", method_scenario_summary)
    manifest = {
        "status": "COMPLETE" if not missing else "PARTIAL_NONREPORTABLE",
        "results_root": str(results_root),
        "scenario_ids": scenario_ids or load_registry()["formal_scenario_ids"],
        "baseline_ids": baseline_ids or json.loads(
            Path(__file__).with_name("formal_baselines.json").read_text(
                encoding="utf-8"
            )
        )["baselines"],
        "seed_ids": seed_ids or load_registry()["formal_seeds"],
        "completed_jobs": completed_jobs,
        "expected_jobs": expected_jobs,
        "missing_jobs": missing,
        "evaluation_rows": len(evaluation_rows),
        "communication_rows": len(communication_rows),
        "training_rows": len(training_rows),
        "seed_summary_rows": len(seed_summary),
        "method_scenario_summary_rows": len(method_scenario_summary),
        "confidence_interval": "two-sided 95% Student-t interval over five training-seed means",
        "files": sorted(path.name for path in output_dir.glob("*.csv")),
    }
    (output_dir / "paper_tables_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="write auditable PARTIAL_NONREPORTABLE tables for progress checks only",
    )
    args = parser.parse_args()
    manifest = build_tables(
        args.results_root,
        output_dir=args.output_dir,
        require_complete=not args.allow_incomplete,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
