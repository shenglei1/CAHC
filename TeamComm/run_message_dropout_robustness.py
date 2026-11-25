"""Run frozen IC3Net link-dropout evaluation on existing checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


PROJECT = Path(__file__).resolve().parent
FORMAL_SEEDS = [20, 23, 56, 89, 2025]
EVAL_SEEDS = list(range(1001, 1011))
SCENARIOS = ["fog_sight1_level", "scale_20"]
RATES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
METRICS = [
    "Task_Completion_Ratio", "Capacity_Completion_Ratio",
    "Episode_Success_Rate", "Collision_Rate",
    "Synchronization_Success_Rate", "Missing_Member_Rate",
    "Mean_Active_Link_Ratio", "Requested_Link_Dropout_Rate",
    "Realized_Active_Link_Dropout_Rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate existing IC3Net checkpoints under link dropout."
    )
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--dropout-rates", nargs="+", type=float, default=RATES)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=EVAL_SEEDS)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--cpu-threads-per-job", type=int, default=2)
    parser.add_argument(
        "--source-results-root", type=Path,
        default=Path("results"),
    )
    parser.add_argument(
        "--results-root", type=Path,
        default=Path("results/graph_intelligence/message_dropout"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    if path.is_absolute():
        raise ValueError(f"Experiment paths must be relative: {path}")
    resolved = (PROJECT / path).resolve()
    try:
        resolved.relative_to(PROJECT)
    except ValueError as error:
        raise ValueError(f"Experiment path leaves the TeamComm directory: {path}") from error
    return resolved


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT))


def label(rate: float) -> str:
    percentage = 100.0 * rate
    if not math.isclose(percentage, round(percentage), abs_tol=1e-9):
        raise ValueError("Dropout rates must be integer percentages.")
    return f"ic3net_dropout_{int(round(percentage))}pct"


def locate(root: Path, method: str, scenario: str, seed: int) -> Path:
    base = root / method / f"lbf_{scenario}-v0" / f"seed_{seed}"
    runs = sorted(path for path in base.glob("run*") if path.is_dir())
    if len(runs) != 1:
        raise RuntimeError(f"Expected one run below {relative(base)}; found {len(runs)}.")
    return runs[0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def validate_output(run: Path, method: str, scenario: str, seed: int,
                    eval_seeds: list[int], episodes: int) -> dict:
    required = {
        "config.json", "training_protocol.json", "evaluation_protocol.json",
        "test_metrics.csv", "comparison_metrics.csv", "communication_costs.csv",
    }
    missing = sorted(name for name in required if not (run / name).is_file())
    if missing:
        raise RuntimeError(f"Missing {missing}: {relative(run)}")
    rows = read_csv(run / "test_metrics.csv")
    expected = {(s, e) for s in eval_seeds for e in range(1, episodes + 1)}
    observed = {(int(row["Eval_Seed"]), int(row["Episode"])) for row in rows}
    if len(rows) != len(expected) or observed != expected:
        raise RuntimeError(f"Incomplete evaluation coverage: {relative(run)}")
    for row in rows:
        if row["Method"] != method or row["Scenario"] != f"lbf_{scenario}-v0":
            raise RuntimeError(f"Evaluation identity mismatch: {relative(run)}")
        if int(row["Train_Seed"]) != seed:
            raise RuntimeError(f"Training seed mismatch: {relative(run)}")
        for metric in METRICS:
            if metric not in row or not math.isfinite(float(row[metric])):
                raise RuntimeError(f"Invalid {metric}: {relative(run)}")
    protocol = json.loads((run / "evaluation_protocol.json").read_text(encoding="utf-8"))
    if not protocol.get("parameters_unchanged") or not protocol.get("checkpoint_load_verified"):
        raise RuntimeError(f"Frozen evaluation audit failed: {relative(run)}")
    return {"run_dir": relative(run), "rows": len(rows), "status": "PASS"}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(jobs: list[dict], root: Path) -> list[Path]:
    seed_rows = []
    for job in jobs:
        run = PROJECT / job["run_dir"]
        rows = read_csv(run / "test_metrics.csv")
        item = {
            "Method": job["method"],
            "Scenario": f"lbf_{job['scenario']}-v0",
            "Train_Seed": job["seed"],
            "N_Test_Episodes": len(rows),
        }
        for metric in METRICS:
            item[metric] = sum(float(row[metric]) for row in rows) / len(rows)
        seed_rows.append(item)
    by_seed = root / "summaries" / "message_dropout_by_seed.csv"
    write_csv(by_seed, seed_rows)

    summary = []
    for method in sorted({row["Method"] for row in seed_rows}):
        for scenario in sorted({row["Scenario"] for row in seed_rows}):
            group = [row for row in seed_rows if row["Method"] == method and row["Scenario"] == scenario]
            if not group:
                continue
            if sorted(int(row["Train_Seed"]) for row in group) != FORMAL_SEEDS:
                raise RuntimeError(f"Incomplete five-seed group: {method}/{scenario}")
            for metric in METRICS:
                values = [float(row[metric]) for row in group]
                sd = statistics.stdev(values)
                summary.append({
                    "Method": method, "Scenario": scenario, "Metric": metric,
                    "Mean": statistics.mean(values),
                    "Sample_SD_Across_Train_Seeds": sd,
                    "CI95_Half_Width": 2.776445105 * sd / math.sqrt(5),
                    "N_Train_Seeds": 5,
                })
    summary_path = root / "summaries" / "message_dropout_summary.csv"
    write_csv(summary_path, summary)
    return [by_seed, summary_path]


def main() -> None:
    args = parse_args()
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError(
            f"The active interpreter cannot import PyTorch: {sys.executable}"
        )
    if args.eval_seeds != EVAL_SEEDS or args.eval_episodes != 20:
        raise ValueError("Formal evaluation requires seeds 1001--1010 x 20 episodes.")
    if len(args.scenarios) != len(set(args.scenarios)) or not set(args.scenarios) <= set(SCENARIOS):
        raise ValueError(f"--scenarios must be unique members of {SCENARIOS}.")
    if args.dropout_rates != sorted(set(args.dropout_rates)) or 0.0 not in args.dropout_rates:
        raise ValueError("Rates must be unique, ascending, and include zero.")
    if any(not 0.0 <= rate <= 1.0 for rate in args.dropout_rates):
        raise ValueError("Every dropout rate must be in [0, 1].")
    if not 1 <= args.workers <= 5 or args.cpu_threads_per_job < 1:
        raise ValueError("Use 1--5 workers and at least one CPU thread per job.")

    source_root = project_path(args.source_results_root)
    results_root = project_path(args.results_root)
    manifest_path = results_root / "ic3net_message_dropout_manifest.json"
    if manifest_path.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(f"Manifest exists; use --resume: {relative(manifest_path)}")

    jobs = []
    for scenario in args.scenarios:
        for rate in args.dropout_rates:
            method = label(rate)
            for seed in FORMAL_SEEDS:
                source = locate(source_root, "ic3net", scenario, seed)
                for name in ("model.pt", "config.json", "training_protocol.json", "evaluation_protocol.json"):
                    if not (source / name).is_file():
                        raise FileNotFoundError(f"Missing source {name}: {relative(source)}")
                command = [
                    sys.executable, "evaluate_ic3net_link_dropout.py",
                    "--source-run-dir", relative(source),
                    "--scenario", scenario,
                    "--seed", str(seed),
                    "--link-dropout-rate", str(rate),
                    "--method-label", method,
                    "--eval-seeds", *[str(value) for value in args.eval_seeds],
                    "--eval-episodes", str(args.eval_episodes),
                    "--gpu", str(args.gpu),
                    "--results-root", relative(results_root),
                ]
                jobs.append({
                    "scenario": scenario, "dropout_rate": rate,
                    "method": method, "seed": seed,
                    "source_run_dir": relative(source), "command": command,
                    "status": "PENDING",
                })

    manifest = {
        "experiment": "ic3net_zero_shot_execution_link_dropout",
        "scenarios": args.scenarios, "dropout_rates": args.dropout_rates,
        "formal_seeds": FORMAL_SEEDS, "eval_seeds": args.eval_seeds,
        "eval_episodes_per_seed": args.eval_episodes,
        "source_results_root": relative(source_root),
        "results_root": relative(results_root),
        "new_training_jobs": 0, "frozen_evaluation_jobs": len(jobs),
        "started_utc": datetime.now(timezone.utc).isoformat(), "jobs": jobs,
        "status": "DRY_RUN" if args.dry_run else "RUNNING",
    }
    results_root.mkdir(parents=True, exist_ok=True)

    if args.resume:
        for job in jobs:
            base = results_root / job["method"] / f"lbf_{job['scenario']}-v0" / f"seed_{job['seed']}"
            runs = sorted(path for path in base.glob("run*") if path.is_dir())
            if len(runs) == 1:
                try:
                    validation = validate_output(
                        runs[0], job["method"], job["scenario"], job["seed"],
                        args.eval_seeds, args.eval_episodes,
                    )
                    job.update(validation)
                    job["status"] = "PASS"
                except Exception:
                    pass

    portable = json.loads(json.dumps(manifest))
    for job in portable["jobs"]:
        job["command"][0] = "python"
    manifest_path.write_text(json.dumps(portable, indent=2), encoding="utf-8")
    if args.dry_run:
        print(f"DRY RUN: {len(jobs)} frozen jobs; 0 training jobs")
        return

    environment = {
        **os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": str(args.cpu_threads_per_job),
        "MKL_NUM_THREADS": str(args.cpu_threads_per_job),
        "OPENBLAS_NUM_THREADS": str(args.cpu_threads_per_job),
        "NUMEXPR_NUM_THREADS": str(args.cpu_threads_per_job),
    }
    logs = results_root / "launcher_logs"
    logs.mkdir(exist_ok=True)
    for scenario in args.scenarios:
        for rate in args.dropout_rates:
            batch = [job for job in jobs if job["scenario"] == scenario and job["dropout_rate"] == rate and job["status"] == "PENDING"]
            if not batch:
                continue
            print(f"START {scenario}/dropout={rate:.0%}: seeds={FORMAL_SEEDS}", flush=True)
            failures = []
            with ThreadPoolExecutor(max_workers=min(args.workers, len(batch))) as pool:
                futures = {
                    pool.submit(subprocess.run, job["command"], cwd=PROJECT,
                                text=True, encoding="utf-8", errors="replace",
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=environment): job
                    for job in batch
                }
                for future in as_completed(futures):
                    job = futures[future]
                    completed = future.result()
                    log_path = logs / f"{job['scenario']}__{job['method']}__seed_{job['seed']}.log"
                    log_path.write_text(completed.stdout, encoding="utf-8")
                    try:
                        run = locate(results_root, job["method"], job["scenario"], job["seed"])
                        validation = validate_output(run, job["method"], job["scenario"], job["seed"], args.eval_seeds, args.eval_episodes)
                    except Exception as error:
                        validation = {"status": "FAIL", "error": repr(error)}
                    if completed.returncode == 0 and validation["status"] == "PASS":
                        job.update(validation)
                        job["status"] = "PASS"
                        log_path.unlink(missing_ok=True)
                    else:
                        job.update(status="FAIL", returncode=completed.returncode,
                                   validation=validation, log=relative(log_path))
                        failures.append(job)
            portable = json.loads(json.dumps(manifest))
            for job in portable["jobs"]:
                job["command"][0] = "python"
            manifest_path.write_text(json.dumps(portable, indent=2), encoding="utf-8")
            if failures:
                raise SystemExit(f"Stopped after {len(failures)} failed jobs.")
            print(f"PASS {scenario}/dropout={rate:.0%}", flush=True)

    outputs = summarize(jobs, results_root)
    manifest["status"] = "PASS"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["summary_files"] = [relative(path) for path in outputs]
    portable = json.loads(json.dumps(manifest))
    for job in portable["jobs"]:
        job["command"][0] = "python"
    manifest_path.write_text(json.dumps(portable, indent=2), encoding="utf-8")
    if logs.is_dir() and not any(logs.iterdir()):
        logs.rmdir()
    print(f"PASS: IC3Net message-dropout evaluation complete: {relative(manifest_path)}")


if __name__ == "__main__":
    main()
