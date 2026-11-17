"""Run and summarize Reviewer-2 zero-shot cross-configuration evaluation.

This launcher performs no training.  It evaluates the five existing
``hetero_strong`` checkpoints for HyperWave-X and all eight formal baselines
on the registered source, demand-shift, and fleet-shift configurations.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import shutil


PROJECT = Path(__file__).resolve().parent
WORKSPACE = PROJECT.parent
HYPERCOMM = WORKSPACE / "HyperComm"
TEAMCOMM = WORKSPACE / "TeamComm"
DESIGN_PATH = PROJECT / "cross_configuration_registry.json"
TEAMCOMM_METHODS = {
    "commnet", "ic3net", "magic", "qmix", "tarmac", "teamcomm", "tiecomm"
}
METHOD_ORDER = [
    "hyperwave_x", "hypercomm", "commnet", "ic3net", "magic",
    "qmix", "tarmac", "teamcomm", "tiecomm",
]
METHOD_PROJECT = {
    "hyperwave_x": PROJECT,
    "hypercomm": HYPERCOMM,
    **{method: TEAMCOMM for method in TEAMCOMM_METHODS},
}
METHOD_EVALUATOR = {
    "hyperwave_x": Path("evaluate_hyperwavex_generalization.py"),
    "hypercomm": Path("evaluate_hypercomm_generalization.py"),
    **{method: Path("evaluate_teamcomm_generalization.py") for method in TEAMCOMM_METHODS},
}
DEFAULT_SOURCE_ROOT = {
    "hyperwave_x": Path("results/FORMAL_6SCENARIOS_ORGANIZED_V1"),
    "hypercomm": Path("../HyperComm/results/FORMAL_6SCENARIOS"),
    "teamcomm": Path("../TeamComm/results"),
}
DEFAULT_RESULTS_ROOT = {
    "hyperwave_x": Path("results/CROSS_CONFIGURATION_HETERO_STRONG_ALL_BASELINES"),
    "hypercomm": Path("../HyperComm/results/CROSS_CONFIGURATION_HETERO_STRONG_ALL_BASELINES"),
    "teamcomm": Path("../TeamComm/results/CROSS_CONFIGURATION_HETERO_STRONG_ALL_BASELINES"),
}
METRICS = [
    "Task_Completion_Ratio",
    "Capacity_Completion_Ratio",
    "Episode_Success_Rate",
    "Collision_Rate",
    "Missing_Member_Rate",
    "Synchronization_Success_Rate",
    "Mean_Active_Link_Ratio",
    "Logical_Transfers_Per_Step",
]
LOWER_IS_BETTER = {
    "Collision_Rate", "Missing_Member_Rate", "Mean_Active_Link_Ratio",
    "Logical_Transfers_Per_Step",
}
T_CRITICAL_95_DF4 = 2.7764451051977987


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command frozen cross-configuration evaluation for Reviewer 2."
    )
    parser.add_argument(
        "--methods", nargs="+", choices=sorted(METHOD_PROJECT),
        default=METHOD_ORDER,
    )
    parser.add_argument(
        "--configurations", nargs="+",
        choices=["source_configuration", "demand_shift", "fleet_shift"],
        default=["source_configuration", "demand_shift", "fleet_shift"],
    )
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--cpu-threads-per-job", type=int, default=2)
    parser.add_argument(
        "--hyperwavex-source-root", type=Path,
        default=DEFAULT_SOURCE_ROOT["hyperwave_x"],
    )
    parser.add_argument(
        "--hypercomm-source-root", type=Path,
        default=DEFAULT_SOURCE_ROOT["hypercomm"],
    )
    parser.add_argument(
        "--teamcomm-source-root", type=Path,
        default=DEFAULT_SOURCE_ROOT["teamcomm"],
    )
    parser.add_argument(
        "--hyperwavex-results-root", type=Path,
        default=DEFAULT_RESULTS_ROOT["hyperwave_x"],
    )
    parser.add_argument(
        "--hypercomm-results-root", type=Path,
        default=DEFAULT_RESULTS_ROOT["hypercomm"],
    )
    parser.add_argument(
        "--teamcomm-results-root", type=Path,
        default=DEFAULT_RESULTS_ROOT["teamcomm"],
    )
    parser.add_argument(
        "--teamcomm-python", type=str, default=None,
        help="Python executable from the TeamComm conda environment; required when TeamComm baselines are selected.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_root(args: argparse.Namespace, method: str) -> Path:
    if method == "hyperwave_x":
        return args.hyperwavex_source_root
    if method == "hypercomm":
        return args.hypercomm_source_root
    return args.teamcomm_source_root


def results_root(args: argparse.Namespace, method: str) -> Path:
    if method == "hyperwave_x":
        return args.hyperwavex_results_root
    if method == "hypercomm":
        return args.hypercomm_results_root
    return args.teamcomm_results_root


def discover_source(args: argparse.Namespace, method: str, scenario: str, seed: int) -> Path:
    root = source_root(args, method) / method / f"lbf_{scenario}-v0" / f"seed_{seed}"
    runs = sorted(path for path in root.glob("run*") if path.is_dir())
    if len(runs) != 1:
        raise FileNotFoundError(
            f"Expected one source run for {method}/{scenario}/seed_{seed}; "
            f"found {len(runs)} under {root}."
        )
    required = {"model.pt", "config.json", "training_protocol.json", "evaluation_protocol.json"}
    missing = sorted(name for name in required if not (runs[0] / name).is_file())
    if missing:
        raise FileNotFoundError(f"Source run {runs[0]} is missing {missing}.")
    return runs[0]


def output_dir(
    args: argparse.Namespace, method: str, scenario: str,
    configuration: str, seed: int,
) -> Path:
    return (
        results_root(args, method) / method
        / f"{scenario}_to_{configuration}" / f"seed_{seed}" / "run1"
    )


def reusable(
    directory: Path, method: str, configuration: str, seed: int,
    eval_seeds: list[int], eval_episodes: int,
) -> tuple[bool, str]:
    required = {
        "test_metrics.csv", "comparison_metrics.csv", "communication_costs.csv",
        "evaluation_protocol.json", "target_configuration.json",
    }
    if not directory.is_dir():
        return False, "result directory is absent"
    missing = sorted(name for name in required if not (directory / name).is_file())
    if missing:
        return False, f"missing {missing}"
    try:
        rows = read_csv(directory / "test_metrics.csv")
        protocol = read_json(directory / "evaluation_protocol.json")
    except Exception as error:
        return False, f"unreadable artifacts: {error}"
    if len(rows) != len(eval_seeds) * eval_episodes:
        return False, "episode row count mismatch"
    if protocol.get("method") != method or int(protocol.get("training_seed", -1)) != seed:
        return False, "method or training seed mismatch"
    if protocol.get("target_configuration") != configuration:
        return False, "target configuration mismatch"
    if protocol.get("evaluation_seeds") != eval_seeds:
        return False, "evaluation seed mismatch"
    if int(protocol.get("episodes_per_seed", -1)) != eval_episodes:
        return False, "evaluation episode budget mismatch"
    if not protocol.get("parameters_unchanged") or protocol.get("optimizer_updates") != 0:
        return False, "evaluation was not certified frozen"
    identities = {
        (int(row["Eval_Seed"]), int(row["Episode"])) for row in rows
    }
    expected = {
        (eval_seed, episode)
        for eval_seed in eval_seeds
        for episode in range(1, eval_episodes + 1)
    }
    if identities != expected:
        return False, "evaluation seed/episode coverage mismatch"
    if any(
        row.get("Method") != method
        or int(row.get("Train_Seed", -1)) != seed
        or row.get("Target_Configuration") != configuration
        for row in rows
    ):
        return False, "test-row identity mismatch"
    return True, "complete"


def run_job(job: dict, threads: int) -> tuple[int, Path]:
    environment = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
    }
    completed = subprocess.run(
        job["command"], cwd=METHOD_PROJECT[job["method"]],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=environment,
    )
    job["log"].parent.mkdir(parents=True, exist_ok=True)
    job["log"].write_text(completed.stdout, encoding="utf-8")
    return completed.returncode, job["log"]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize(args: argparse.Namespace, design: dict) -> Path:
    source = design["source_scenario"]
    seed_rows: list[dict] = []
    for method in args.methods:
        for configuration in args.configurations:
            for train_seed in design["training_seeds"]:
                directory = output_dir(args, method, source, configuration, train_seed)
                rows = read_csv(directory / "test_metrics.csv")
                costs = read_csv(directory / "communication_costs.csv")
                cost_index = {
                    (int(row["Eval_Seed"]), int(row["Episode"])): row
                    for row in costs
                }
                merged: dict[str, list[float]] = {metric: [] for metric in METRICS}
                for row in rows:
                    identity = (int(row["Eval_Seed"]), int(row["Episode"]))
                    for metric in METRICS:
                        source_row = cost_index[identity] if metric == "Logical_Transfers_Per_Step" else row
                        if metric == "Mean_Active_Link_Ratio":
                            value = source_row.get(
                                "Mean_Active_Link_Ratio",
                                source_row.get("Active_Link_Ratio", ""),
                            )
                        else:
                            value = source_row.get(metric, "")
                        if value != "":
                            merged[metric].append(float(value))
                seed_row = {
                    "Method": method,
                    "Source_Configuration": source,
                    "Target_Configuration": configuration,
                    "Train_Seed": train_seed,
                    "Evaluation_Episodes": len(rows),
                }
                seed_row.update({
                    metric: (mean(values) if values else "")
                    for metric, values in merged.items()
                })
                seed_rows.append(seed_row)

    common_root = args.hyperwavex_results_root
    summary_root = common_root / "paper_summary"
    write_csv(summary_root / "cross_configuration_seed_summary.csv", seed_rows)

    summary_rows: list[dict] = []
    for method in args.methods:
        for configuration in args.configurations:
            subset = [
                row for row in seed_rows
                if row["Method"] == method
                and row["Target_Configuration"] == configuration
            ]
            for metric in METRICS:
                values = [float(row[metric]) for row in subset if row[metric] != ""]
                if not values:
                    continue
                sample_sd = statistics.stdev(values) if len(values) > 1 else 0.0
                half = (
                    T_CRITICAL_95_DF4 * sample_sd / math.sqrt(len(values))
                    if len(values) == 5 else 0.0
                )
                summary_rows.append({
                    "Method": method,
                    "Source_Configuration": source,
                    "Target_Configuration": configuration,
                    "Metric": metric,
                    "Preferred_Direction": "lower" if metric in LOWER_IS_BETTER else "higher",
                    "Train_Seed_Count": len(values),
                    "Mean": mean(values),
                    "Sample_SD": sample_sd,
                    "CI95_Lower": mean(values) - half,
                    "CI95_Upper": mean(values) + half,
                })
    write_csv(summary_root / "cross_configuration_summary.csv", summary_rows)

    seed_index = {
        (row["Method"], row["Target_Configuration"], int(row["Train_Seed"])): row
        for row in seed_rows
    }
    gap_rows: list[dict] = []
    for method in args.methods:
        for configuration in args.configurations:
            if configuration == "source_configuration":
                continue
            for metric in METRICS:
                differences = []
                retentions = []
                for train_seed in design["training_seeds"]:
                    base = seed_index[(method, "source_configuration", train_seed)]
                    target = seed_index[(method, configuration, train_seed)]
                    if base[metric] == "" or target[metric] == "":
                        continue
                    base_value, target_value = float(base[metric]), float(target[metric])
                    differences.append(target_value - base_value)
                    if metric in {
                        "Task_Completion_Ratio", "Capacity_Completion_Ratio",
                        "Episode_Success_Rate",
                    } and base_value != 0:
                        retentions.append(target_value / base_value)
                if not differences:
                    continue
                sample_sd = statistics.stdev(differences) if len(differences) > 1 else 0.0
                half = (
                    T_CRITICAL_95_DF4 * sample_sd / math.sqrt(len(differences))
                    if len(differences) == 5 else 0.0
                )
                gap_rows.append({
                    "Method": method,
                    "Source_Configuration": source,
                    "Target_Configuration": configuration,
                    "Metric": metric,
                    "Target_Minus_Source_Mean": mean(differences),
                    "Target_Minus_Source_SD": sample_sd,
                    "CI95_Lower": mean(differences) - half,
                    "CI95_Upper": mean(differences) + half,
                    "Mean_Retention_Ratio": mean(retentions) if retentions else "",
                })
    write_csv(summary_root / "generalization_gap.csv", gap_rows)

    if "hyperwave_x" in args.methods and len(args.methods) > 1:
        comparison_rows: list[dict] = []
        for comparator in args.methods:
            if comparator == "hyperwave_x":
                continue
            for configuration in args.configurations:
                for metric in METRICS:
                    differences = []
                    for train_seed in design["training_seeds"]:
                        proposed = seed_index[("hyperwave_x", configuration, train_seed)]
                        baseline = seed_index[(comparator, configuration, train_seed)]
                        if proposed[metric] != "" and baseline[metric] != "":
                            differences.append(
                                float(proposed[metric]) - float(baseline[metric])
                            )
                    if not differences:
                        continue
                    sample_sd = (
                        statistics.stdev(differences) if len(differences) > 1 else 0.0
                    )
                    half = (
                        T_CRITICAL_95_DF4 * sample_sd / math.sqrt(len(differences))
                        if len(differences) == 5 else 0.0
                    )
                    comparison_rows.append({
                        "Target_Configuration": configuration,
                        "Comparator": comparator,
                        "Metric": metric,
                        "Preferred_Direction": (
                            "lower" if metric in LOWER_IS_BETTER else "higher"
                        ),
                        "HyperWaveX_Minus_Comparator_Mean": mean(differences),
                        "Sample_SD": sample_sd,
                        "CI95_Lower": mean(differences) - half,
                        "CI95_Upper": mean(differences) + half,
                    })
        write_csv(summary_root / "paired_method_comparison.csv", comparison_rows)
    return summary_root


def main() -> None:
    args = parse_args()
    if len(args.methods) != len(set(args.methods)):
        raise ValueError("Methods must be unique.")
    if not 1 <= args.workers <= 5:
        raise ValueError("--workers must be between 1 and 5.")
    if args.cpu_threads_per_job < 1:
        raise ValueError("--cpu-threads-per-job must be positive.")
    if args.gpu >= 0 and args.workers > 2:
        raise ValueError("A single GPU permits at most two concurrent evaluation jobs.")
    if args.gpu >= 0 and not 0 < args.gpu_memory_fraction <= 0.5:
        raise ValueError("GPU memory fraction must be in (0, 0.5].")
    if args.gpu >= 0 and set(args.methods) & TEAMCOMM_METHODS:
        raise ValueError(
            "The unified all-baseline evaluation is CPU-only; use --gpu -1."
        )
    if len(args.configurations) != len(set(args.configurations)):
        raise ValueError("Target configurations must be unique.")
    if "source_configuration" not in args.configurations:
        raise ValueError("The source_configuration control is required for retention analysis.")
    if not HYPERCOMM.is_dir():
        raise FileNotFoundError(f"Sibling HyperComm project not found: {HYPERCOMM}")
    if not TEAMCOMM.is_dir():
        raise FileNotFoundError(f"Sibling TeamComm project not found: {TEAMCOMM}")
    design_copies = [
        DESIGN_PATH,
        HYPERCOMM / DESIGN_PATH.name,
        TEAMCOMM / DESIGN_PATH.name,
    ]
    if any(path.read_bytes() != DESIGN_PATH.read_bytes() for path in design_copies[1:]):
        raise RuntimeError(
            "HyperWave-X, HyperComm, and TeamComm generalization designs differ."
        )

    teamcomm_python = None
    if set(args.methods) & TEAMCOMM_METHODS:
        if not args.teamcomm_python:
            raise ValueError(
                "--teamcomm-python is required when a TeamComm-owned baseline is selected."
            )
        candidate = Path(args.teamcomm_python).expanduser()
        resolved = (
            str(candidate.resolve()) if candidate.is_file()
            else shutil.which(args.teamcomm_python)
        )
        if not resolved:
            raise FileNotFoundError(
                f"TeamComm Python executable not found: {args.teamcomm_python}"
            )
        teamcomm_python = resolved

    design = read_json(DESIGN_PATH)
    source = design["source_scenario"]
    eval_seeds = design["evaluation_seeds"]
    eval_episodes = int(design["episodes_per_evaluation_seed"])
    missing_methods = sorted(set(args.methods) - set(design["methods"]))
    if missing_methods:
        raise ValueError(f"Methods absent from the registered design: {missing_methods}")
    jobs = []
    reused = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for method in args.methods:
        for configuration in args.configurations:
            for seed in design["training_seeds"]:
                source_run = discover_source(args, method, source, seed)
                destination = output_dir(args, method, source, configuration, seed)
                reusable_result, reason = reusable(
                    destination, method, configuration, seed, eval_seeds, eval_episodes
                )
                if args.resume and reusable_result:
                    reused.append(str(destination))
                    continue
                if destination.exists():
                    raise RuntimeError(
                        f"Existing result cannot be reused ({reason}): {destination}. "
                        "Move that directory aside before rerunning."
                    )
                command = [
                    teamcomm_python if method in TEAMCOMM_METHODS else sys.executable,
                    str(METHOD_EVALUATOR[method]),
                    *(["--method", method] if method in TEAMCOMM_METHODS else []),
                    "--source-run-dir", str(source_run),
                    "--target-configuration", configuration,
                    "--eval-seeds", *[str(value) for value in eval_seeds],
                    "--eval-episodes", str(eval_episodes),
                    "--gpu", str(args.gpu),
                    "--gpu-memory-fraction", str(
                        args.gpu_memory_fraction if args.gpu >= 0 else 0.0
                    ),
                    "--results-root", str(results_root(args, method)),
                ]
                log = (
                    results_root(args, method) / "launcher_logs"
                    / f"{method}__{source}_to_{configuration}__seed_{seed}.log"
                )
                jobs.append({
                    "method": method,
                    "configuration": configuration,
                    "seed": seed,
                    "command": command,
                    "log": log,
                })

    manifest_path = args.hyperwavex_results_root / "cross_configuration_launcher_manifest.json"
    manifest = {
        "experiment_id": design["experiment_id"],
        "created_at_utc": timestamp,
        "source_scenario": source,
        "methods": args.methods,
        "target_configurations": args.configurations,
        "training_seeds": design["training_seeds"],
        "evaluation_seeds": eval_seeds,
        "episodes_per_evaluation_seed": eval_episodes,
        "teamcomm_python": teamcomm_python,
        "training_jobs": 0,
        "evaluation_jobs_requested": len(args.methods) * len(args.configurations)
        * len(design["training_seeds"]),
        "evaluation_jobs_reused": len(reused),
        "evaluation_jobs_to_run": len(jobs),
        "reused_results": reused,
        "jobs": [
            {
                "method": job["method"],
                "configuration": job["configuration"],
                "seed": job["seed"],
                "command": job["command"],
                "log": str(job["log"]),
            }
            for job in jobs
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.dry_run:
        print(manifest_path)
        print(f"DRY RUN: {len(jobs)} evaluation jobs; {len(reused)} reusable jobs")
        return

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(run_job, job, args.cpu_threads_per_job): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            returncode, log = future.result()
            status = "PASS" if returncode == 0 else "FAIL"
            print(
                f"{status} method={job['method']} target={job['configuration']} "
                f"seed={job['seed']} log={log}", flush=True
            )
            if returncode:
                failures.append({**job, "returncode": returncode})
    if failures:
        manifest["failures"] = [
            {
                "method": job["method"], "configuration": job["configuration"],
                "seed": job["seed"], "returncode": job["returncode"],
                "log": str(job["log"]),
            }
            for job in failures
        ]
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise RuntimeError(
            "Cross-configuration evaluation failed: "
            + ", ".join(
                f"{job['method']}/{job['configuration']}/seed_{job['seed']}"
                for job in failures
            )
        )

    # Revalidate every requested artifact before producing paper-facing tables.
    for method in args.methods:
        for configuration in args.configurations:
            for seed in design["training_seeds"]:
                directory = output_dir(args, method, source, configuration, seed)
                valid, reason = reusable(
                    directory, method, configuration, seed, eval_seeds, eval_episodes
                )
                if not valid:
                    raise RuntimeError(f"Final artifact validation failed ({reason}): {directory}")
    summary = summarize(args, design)
    manifest["status"] = "complete"
    manifest["summary_directory"] = str(summary)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"COMPLETE summary={summary}")


if __name__ == "__main__":
    main()
