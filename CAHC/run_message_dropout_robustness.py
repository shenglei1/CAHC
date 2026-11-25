"""Evaluate existing CAHC checkpoints under execution-link dropout."""

from __future__ import annotations

import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from graph_experiment_utils import (
    FORMAL_EVAL_EPISODES,
    FORMAL_EVAL_SEEDS,
    FORMAL_SEEDS,
    aggregate_experiment,
    locate_single_run,
    validate_result_run,
)


PROJECT = Path(__file__).resolve().parent
PLAN = PROJECT / "cahc_experiment_plan.json"
SCENARIOS = ["fog_sight1_level", "scale_20"]
DEFAULT_RATES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rate_label(rate: float) -> str:
    percentage = 100.0 * rate
    if not math.isclose(percentage, round(percentage), abs_tol=1e-9):
        raise ValueError("Dropout rates must be expressible as an integer percentage.")
    return f"cahc_dropout_{int(round(percentage))}pct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run zero-shot deterministic message-link dropout evaluation on "
            "existing full CAHC checkpoints."
        )
    )
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--dropout-rates", nargs="+", type=float, default=DEFAULT_RATES)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=FORMAL_EVAL_SEEDS)
    parser.add_argument("--eval-episodes", type=int, default=FORMAL_EVAL_EPISODES)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--cpu-threads-per-job", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument(
        "--source-results-root",
        type=Path,
        default=Path("results/FORMAL_6SCENARIOS_ORGANIZED_V1"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/graph_intelligence/message_dropout"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    if path.is_absolute():
        raise ValueError(
            f"Experiment paths must be relative to the CAHC directory: {path}"
        )
    candidate = (PROJECT / path).resolve()
    try:
        candidate.relative_to(PROJECT)
    except ValueError as error:
        raise ValueError(
            f"Experiment paths must remain inside the CAHC directory: {path}"
        ) from error
    return candidate


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT))


def write_manifest(path: Path, payload: dict) -> None:
    portable = deepcopy(payload)
    for job in portable.get("jobs", []):
        if job.get("command"):
            job["command"][0] = "python"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(portable, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError(
            "The active Python interpreter cannot import PyTorch. Activate the "
            "CAHC environment before launching this frozen evaluation "
            f"(current interpreter: {sys.executable})."
        )
    if args.eval_seeds != FORMAL_EVAL_SEEDS or args.eval_episodes != 20:
        raise ValueError("Formal robustness evaluation requires seeds 1001--1010 x 20 episodes.")
    if len(args.scenarios) != len(set(args.scenarios)) or not set(args.scenarios) <= set(SCENARIOS):
        raise ValueError(f"--scenarios must be unique members of {SCENARIOS}.")
    if len(args.dropout_rates) != len(set(args.dropout_rates)):
        raise ValueError("--dropout-rates must not contain duplicates.")
    if sorted(args.dropout_rates) != args.dropout_rates:
        raise ValueError("--dropout-rates must be listed in ascending order.")
    if 0.0 not in args.dropout_rates:
        raise ValueError("--dropout-rates must include 0.0 as the paired reference.")
    if any(not 0.0 <= rate <= 1.0 for rate in args.dropout_rates):
        raise ValueError("Every dropout rate must lie in [0, 1].")
    labels = [rate_label(rate) for rate in args.dropout_rates]
    if not 1 <= args.workers <= 5:
        raise ValueError("--workers must be between 1 and 5.")
    if args.cpu_threads_per_job < 1:
        raise ValueError("--cpu-threads-per-job must be positive.")
    if args.gpu >= 0 and args.workers > 2:
        raise ValueError("A single GPU permits at most two concurrent evaluation jobs.")
    if args.gpu >= 0 and args.workers == 2 and not 0 < args.gpu_memory_fraction <= 0.5:
        raise ValueError("Two GPU workers require --gpu-memory-fraction in (0, 0.5].")

    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    protocol = plan["graph_intelligence_protocol"]
    registered = protocol["message_dropout"]
    if (
        protocol["scenarios"] != SCENARIOS
        or registered["rates"] != DEFAULT_RATES
        or int(registered["frozen_evaluation_jobs"]) != 80
        or int(registered["new_training_jobs"]) != 0
    ):
        raise RuntimeError("The experiment plan disagrees with the fixed dropout study.")

    source_root = project_path(args.source_results_root)
    results_root = project_path(args.results_root)
    manifest_path = results_root / "cahc_message_dropout_manifest.json"
    if manifest_path.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(
            f"Experiment manifest already exists: {relative(manifest_path)}; "
            "use --resume or a new result root."
        )

    source_runs = {}
    for scenario in args.scenarios:
        for seed in FORMAL_SEEDS:
            run = locate_single_run(source_root, "cahc", scenario, seed)
            validate_result_run(
                run,
                method="cahc",
                scenario=scenario,
                seed=seed,
                eval_seeds=args.eval_seeds,
                eval_episodes=args.eval_episodes,
                require_checkpoint=True,
            )
            source_runs[(scenario, seed)] = run

    jobs = []
    for scenario in args.scenarios:
        for rate, label in zip(args.dropout_rates, labels):
            for seed in FORMAL_SEEDS:
                source = source_runs[(scenario, seed)]
                command = [
                    sys.executable,
                    "evaluate_cahc_link_dropout.py",
                    "--source-run-dir", relative(source),
                    "--scenario", scenario,
                    "--seed", str(seed),
                    "--link-dropout-rate", str(rate),
                    "--method-label", label,
                    "--eval-seeds", *[str(value) for value in args.eval_seeds],
                    "--eval-episodes", str(args.eval_episodes),
                    "--gpu", str(args.gpu),
                    "--gpu-memory-fraction", str(
                        args.gpu_memory_fraction
                        if args.gpu >= 0 and args.workers == 2 else 0.0
                    ),
                    "--results-root", relative(results_root),
                ]
                jobs.append(
                    {
                        "scenario": scenario,
                        "dropout_rate": rate,
                        "method": label,
                        "seed": seed,
                        "source_run_dir": relative(source),
                        "command": command,
                        "status": "PENDING",
                    }
                )

    manifest = {
        "experiment": "zero_shot_execution_link_dropout",
        "design": {
            "intervention": (
                "independently remove decoded non-self directed links after "
                "topology decoding and before GAT aggregation"
            ),
            "mask_pairing": (
                "deterministic nested masks use a local RNG and do not alter the "
                "environment or policy RNG sequence"
            ),
            "training": "reuse existing full checkpoints; no parameter updates",
        },
        "scenarios": args.scenarios,
        "dropout_rates": args.dropout_rates,
        "method_labels": labels,
        "formal_seeds": FORMAL_SEEDS,
        "eval_seeds": args.eval_seeds,
        "eval_episodes_per_seed": args.eval_episodes,
        "source_results_root": relative(source_root),
        "results_root": relative(results_root),
        "workers": args.workers,
        "cpu_threads_per_job": args.cpu_threads_per_job,
        "gpu": args.gpu,
        "job_count": len(jobs),
        "new_training_jobs": 0,
        "frozen_evaluation_jobs": len(jobs),
        "source_sha256": {
            name: sha256(PROJECT / name)
            for name in (
                "cahc.py",
                "formal_cahc.py",
                "evaluate_cahc_link_dropout.py",
                "run_message_dropout_robustness.py",
                "graph_experiment_utils.py",
                "cahc_experiment_plan.json",
            )
        },
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "resume": args.resume,
        "jobs": jobs,
        "status": "DRY_RUN" if args.dry_run else "RUNNING",
    }
    results_root.mkdir(parents=True, exist_ok=True)
    if args.resume:
        for job in jobs:
            seed_root = (
                results_root
                / job["method"]
                / f"lbf_{job['scenario']}-v0"
                / f"seed_{job['seed']}"
            )
            runs = sorted(path for path in seed_root.glob("run*") if path.is_dir())
            if not runs:
                continue
            if len(runs) != 1:
                raise RuntimeError(f"Resume found multiple runs below {relative(seed_root)}.")
            validation = validate_result_run(
                runs[0],
                method=job["method"],
                scenario=job["scenario"],
                seed=job["seed"],
                eval_seeds=args.eval_seeds,
                eval_episodes=args.eval_episodes,
                require_checkpoint=False,
            )
            job.update(
                status="PASS",
                completion_source="validated_existing_artifacts",
                artifact_validation=validation,
            )
    write_manifest(manifest_path, manifest)
    if args.dry_run:
        print(
            f"DRY RUN: {len(jobs)} frozen evaluation jobs; 0 training jobs; "
            f"manifest={relative(manifest_path)}"
        )
        return

    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUTF8": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": str(args.cpu_threads_per_job),
        "MKL_NUM_THREADS": str(args.cpu_threads_per_job),
        "OPENBLAS_NUM_THREADS": str(args.cpu_threads_per_job),
        "NUMEXPR_NUM_THREADS": str(args.cpu_threads_per_job),
    }
    logs = results_root / "launcher_logs"
    logs.mkdir(exist_ok=True)
    for scenario in args.scenarios:
        for rate, label in zip(args.dropout_rates, labels):
            batch = [
                job
                for job in jobs
                if job["scenario"] == scenario
                and job["method"] == label
                and job["status"] == "PENDING"
            ]
            if not batch:
                continue
            print(f"START {scenario}/dropout={rate:.0%}: seeds={FORMAL_SEEDS}", flush=True)
            failures = []
            with ThreadPoolExecutor(max_workers=min(args.workers, len(batch))) as executor:
                futures = {
                    executor.submit(
                        subprocess.run,
                        job["command"],
                        cwd=PROJECT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=environment,
                    ): job
                    for job in batch
                }
                for future in as_completed(futures):
                    job = futures[future]
                    completed = future.result()
                    log_path = logs / (
                        f"{job['scenario']}__{job['method']}__seed_{job['seed']}.log"
                    )
                    log_path.write_text(completed.stdout, encoding="utf-8")
                    try:
                        run = locate_single_run(
                            results_root,
                            job["method"],
                            job["scenario"],
                            job["seed"],
                        )
                        validation = validate_result_run(
                            run,
                            method=job["method"],
                            scenario=job["scenario"],
                            seed=job["seed"],
                            eval_seeds=args.eval_seeds,
                            eval_episodes=args.eval_episodes,
                            require_checkpoint=False,
                        )
                    except Exception as error:
                        validation = {"status": "FAIL", "error": repr(error)}
                    passed = completed.returncode == 0 and validation["status"] == "PASS"
                    job.update(
                        status="PASS" if passed else "FAIL",
                        returncode=completed.returncode,
                        artifact_validation=validation,
                        log=relative(log_path),
                    )
                    if passed:
                        log_path.unlink(missing_ok=True)
                        job["log"] = None
                    else:
                        failures.append(job)
                    write_manifest(manifest_path, manifest)
            if failures:
                manifest["status"] = "FAIL"
                manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
                write_manifest(manifest_path, manifest)
                raise SystemExit(
                    f"Stopped after {len(failures)} failed {scenario}/{rate:.0%} jobs."
                )
            print(f"PASS {scenario}/dropout={rate:.0%}", flush=True)

    records = []
    for job in jobs:
        if job["status"] != "PASS":
            raise RuntimeError(f"Cannot summarize a non-PASS job: {job}")
        records.append(
            {
                "method": job["method"],
                "scenario": job["scenario"],
                "seed": job["seed"],
                "run_dir": job["artifact_validation"]["run_dir"],
            }
        )
    summary_dir = results_root / "summaries"
    outputs = aggregate_experiment(
        records,
        output_dir=summary_dir,
        prefix="message_dropout",
        reference_method=rate_label(0.0),
        include_dropout_metrics=True,
    )
    manifest["status"] = "PASS"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["summary_files"] = [relative(path) for path in outputs]
    write_manifest(manifest_path, manifest)
    if logs.is_dir() and not any(logs.iterdir()):
        logs.rmdir()
    print(f"PASS: message-dropout evaluation complete; manifest={relative(manifest_path)}")
    for path in outputs:
        print(relative(path))


if __name__ == "__main__":
    main()
