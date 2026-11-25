"""Train the controlled graph-refinement comparison for Graph Intelligence."""

from __future__ import annotations

import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
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
CONTROLS = ["no_wave", "plain_diffusion", "repeated_gat"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare no refinement, ordinary normalized hypergraph diffusion, "
            "and repeated GAT propagation with full CAHC."
        )
    )
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--variants", nargs="+", default=CONTROLS)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--episodes-per-epoch", type=int, default=64)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=FORMAL_EVAL_SEEDS)
    parser.add_argument("--eval-episodes", type=int, default=FORMAL_EVAL_EPISODES)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--cpu-threads-per-job", type=int, default=2)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument(
        "--full-results-root",
        type=Path,
        default=Path("results/FORMAL_6SCENARIOS_ORGANIZED_V1"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/graph_intelligence/refinement_controls"),
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


def output_scope(scenarios: list[str]) -> str:
    """Give distributed single-scenario runs collision-free metadata names."""
    return "all_scenarios" if scenarios == SCENARIOS else "__".join(scenarios)


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
            "CAHC environment before launching this experiment "
            f"(current interpreter: {sys.executable})."
        )
    if args.epochs != 700 or args.episodes_per_epoch != 64:
        raise ValueError("Formal refinement controls require 700 epochs x 64 episodes.")
    if args.eval_seeds != FORMAL_EVAL_SEEDS or args.eval_episodes != 20:
        raise ValueError("Formal controls require evaluation seeds 1001--1010 x 20 episodes.")
    if len(args.scenarios) != len(set(args.scenarios)) or not set(args.scenarios) <= set(SCENARIOS):
        raise ValueError(f"--scenarios must be unique members of {SCENARIOS}.")
    if len(args.variants) != len(set(args.variants)) or not set(args.variants) <= set(CONTROLS):
        raise ValueError(f"--variants must be unique members of {CONTROLS}.")
    if not 1 <= args.workers <= 5:
        raise ValueError("--workers must be between 1 and 5.")
    if args.cpu_threads_per_job < 1:
        raise ValueError("--cpu-threads-per-job must be positive.")
    if args.gpu >= 0 and args.workers > 2:
        raise ValueError("A single GPU permits at most two concurrent training jobs.")
    if args.gpu >= 0 and args.workers == 2 and not 0 < args.gpu_memory_fraction <= 0.5:
        raise ValueError("Two GPU workers require --gpu-memory-fraction in (0, 0.5].")

    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    registered = plan["graph_intelligence_protocol"]["refinement_controls"]
    if (
        plan["graph_intelligence_protocol"]["scenarios"] != SCENARIOS
        or registered["variants"] != CONTROLS
        or int(registered["iterations"]) != 2
        or int(registered["new_training_jobs"]) != 30
        or int(registered["frozen_full_evaluation_jobs"]) != 10
    ):
        raise RuntimeError("The experiment plan disagrees with the fixed controls.")

    full_root = project_path(args.full_results_root)
    results_root = project_path(args.results_root)
    scope = output_scope(args.scenarios)
    manifest_path = results_root / (
        f"cahc_refinement_controls__{scope}_manifest.json"
    )
    if manifest_path.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(
            f"Experiment manifest already exists: {relative(manifest_path)}; "
            "use --resume or a new result root."
        )

    source_runs = {}
    for scenario in args.scenarios:
        for seed in FORMAL_SEEDS:
            run = locate_single_run(full_root, "cahc", scenario, seed)
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
        for seed in FORMAL_SEEDS:
            source = source_runs[(scenario, seed)]
            command = [
                sys.executable,
                "evaluate_cahc_link_dropout.py",
                "--source-run-dir", relative(source),
                "--scenario", scenario,
                "--seed", str(seed),
                "--link-dropout-rate", "0",
                "--method-label", "cahc",
                "--evaluation-role", "refinement_reference",
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
                    "job_type": "frozen_reference",
                    "scenario": scenario,
                    "variant": "cahc",
                    "method": "cahc",
                    "seed": seed,
                    "source_run_dir": relative(source),
                    "command": command,
                    "status": "PENDING",
                }
            )
        for variant in args.variants:
            for seed in FORMAL_SEEDS:
                command = [
                    sys.executable,
                    "formal_cahc.py",
                    "--scenario", scenario,
                    "--seed", str(seed),
                    "--epochs", str(args.epochs),
                    "--episodes-per-epoch", str(args.episodes_per_epoch),
                    "--eval-seeds", *[str(value) for value in args.eval_seeds],
                    "--eval-episodes", str(args.eval_episodes),
                    "--gpu", str(args.gpu),
                    "--gpu-memory-fraction", str(
                        args.gpu_memory_fraction
                        if args.gpu >= 0 and args.workers == 2 else 0.0
                    ),
                    "--variant", variant,
                    "--results-root", relative(results_root),
                ]
                jobs.append(
                    {
                        "job_type": "new_training",
                        "scenario": scenario,
                        "variant": variant,
                        "method": variant,
                        "seed": seed,
                        "command": command,
                        "status": "PENDING",
                    }
                )

    manifest = {
        "experiment": "controlled_hypergraph_refinement_comparison",
        "design": {
            "reference": "anchored_wave",
            "controls": {
                "no_wave": "no post-aggregation refinement",
                "plain_diffusion": (
                    "same normalized hypergraph operator and iteration count; "
                    "no input anchor or damping"
                ),
                "repeated_gat": (
                    "same hypergraph-derived support and iteration count; shared "
                    "two-head graph-attention propagation"
                ),
            },
            "held_constant": [
                "scenario",
                "task-conditioned hypergraph constructor",
                "topology decoder",
                "execution GAT",
                "policy/value backbone",
                "training budget",
                "training and evaluation seeds",
            ],
        },
        "scenarios": args.scenarios,
        "variants": args.variants,
        "formal_seeds": FORMAL_SEEDS,
        "epochs": args.epochs,
        "episodes_per_epoch": args.episodes_per_epoch,
        "eval_seeds": args.eval_seeds,
        "eval_episodes_per_seed": args.eval_episodes,
        "full_results_root": relative(full_root),
        "results_root": relative(results_root),
        "workers": args.workers,
        "cpu_threads_per_job": args.cpu_threads_per_job,
        "gpu": args.gpu,
        "job_count": len(jobs),
        "new_training_jobs": sum(
            job["job_type"] == "new_training" for job in jobs
        ),
        "frozen_full_evaluation_jobs": sum(
            job["job_type"] == "frozen_reference" for job in jobs
        ),
        "reused_full_checkpoints": len(source_runs),
        "source_sha256": {
            name: sha256(PROJECT / name)
            for name in (
                "gnn_layers.py",
                "cahc.py",
                "formal_cahc.py",
                "evaluate_cahc_link_dropout.py",
                "run_graph_refinement_controls.py",
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
                require_checkpoint=job["job_type"] == "new_training",
            )
            job.update(
                status="PASS",
                completion_source="validated_existing_artifacts",
                artifact_validation=validation,
            )
    write_manifest(manifest_path, manifest)
    if args.dry_run:
        print(
            f"DRY RUN: {manifest['new_training_jobs']} new training jobs; "
            f"{manifest['frozen_full_evaluation_jobs']} frozen reference evaluations; "
            f"{len(source_runs)} source checkpoints validated; "
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
        for variant in ["cahc", *args.variants]:
            batch = [
                job
                for job in jobs
                if job["scenario"] == scenario
                and job["variant"] == variant
                and job["status"] == "PENDING"
            ]
            if not batch:
                continue
            print(f"START {scenario}/{variant}: seeds={FORMAL_SEEDS}", flush=True)
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
                        f"{job['scenario']}__{job['variant']}__seed_{job['seed']}.log"
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
                            require_checkpoint=job["job_type"] == "new_training",
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
                    f"Stopped after {len(failures)} failed {scenario}/{variant} jobs."
                )
            print(f"PASS {scenario}/{variant}", flush=True)

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
        prefix=f"refinement_controls__{scope}",
        reference_method="cahc",
    )
    manifest["status"] = "PASS"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["summary_files"] = [relative(path) for path in outputs]
    write_manifest(manifest_path, manifest)
    if logs.is_dir() and not any(logs.iterdir()):
        logs.rmdir()
    print(f"PASS: refinement controls complete; manifest={relative(manifest_path)}")
    for path in outputs:
        print(relative(path))


if __name__ == "__main__":
    main()
