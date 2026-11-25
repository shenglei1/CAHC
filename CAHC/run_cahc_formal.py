from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from runtime_support import runtime_metadata


PROJECT = Path(__file__).resolve().parent
REGISTRY_PATH = PROJECT / "scenario_registry.json"
PLAN_PATH = PROJECT / "cahc_experiment_plan.json"
TRACE_SCENARIOS = {"probe_complementary", "probe_interaction"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal scenarios or one isolated Reviewer-probe shard."
    )
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--episodes-per-epoch", type=int, default=64)
    parser.add_argument(
        "--eval-seeds", nargs="+", type=int, default=list(range(1001, 1011))
    )
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--train-seeds", nargs=5, type=int, default=None,
        help=(
            "Five unique training seeds. Defaults to the project-wide formal "
            "seeds; Probe2 does not require particular numeric seed values."
        ),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help=(
            "Optional registry scenario IDs for a reproducible machine shard. "
            "Probe IDs also require --include-probes."
        ),
    )
    parser.add_argument("--cpu-threads-per-job", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--include-probes", action="store_true")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT / "results/FORMAL_6SCENARIOS_ORGANIZED_V1",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Validate and reuse complete five-seed runs in the requested result root.",
    )
    parser.add_argument("--variant", choices=("full", "no_wave"), default="full",
                        help="no_wave is the mandatory Probe2 mechanism control.")
    return parser.parse_args()


def command_for(
    scenario: str, seed: int, args, probe_ids: set[str], trace_seed: int
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT / "formal_cahc.py"),
        "--scenario",
        scenario,
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--episodes-per-epoch",
        str(args.episodes_per_epoch),
        "--eval-seeds",
        *[str(value) for value in args.eval_seeds],
        "--eval-episodes",
        str(args.eval_episodes),
        "--gpu",
        str(args.gpu),
        "--gpu-memory-fraction",
        str(args.gpu_memory_fraction if args.gpu >= 0 and args.workers == 2 else 0.0),
        "--variant",
        args.variant,
        "--results-root",
        str(args.results_root),
    ]
    if scenario in probe_ids:
        command.append("--allow-probe")
    if scenario in TRACE_SCENARIOS and seed == trace_seed:
        command.append("--record-trace")
    return command


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_paper_tables(
    args: argparse.Namespace, train_seeds: list[int], scenarios: list[str]
) -> str:
    summary_dir = args.results_root / "summaries" / "__".join(scenarios)
    command = [
        sys.executable,
        str(PROJECT / "summarize_test_results.py"),
        str(args.results_root),
        "--expected-train-seeds",
        str(len(train_seeds)),
        "--train-seeds",
        *[str(value) for value in train_seeds],
        "--expected-eval-seeds",
        *[str(value) for value in args.eval_seeds],
        "--episodes-per-eval-seed",
        str(args.eval_episodes),
        "--scenarios",
        *[f"lbf_{scenario}-v0" for scenario in scenarios],
        "--methods",
        "cahc" if args.variant == "full" else args.variant,
        "--output-dir",
        str(summary_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        raise RuntimeError(
            "Paper-table aggregation failed:\n" + completed.stdout
        )
    return completed.stdout.strip()


def reusable_run(job: dict, args: argparse.Namespace) -> tuple[bool, str]:
    seed_root = (
        args.results_root / job["method"] / f"lbf_{job['scenario']}-v0"
        / f"seed_{job['seed']}"
    )
    runs = sorted(path for path in seed_root.glob("run*") if path.is_dir())
    if len(runs) != 1:
        return False, f"expected one run, found {len(runs)}"
    run = runs[0]
    required = [
        "model.pt", "training_protocol.json", "evaluation_protocol.json",
        "test_metrics.csv", "comparison_metrics.csv", "communication_costs.csv",
    ]
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        return False, "missing " + ", ".join(missing)
    try:
        training = json.loads((run / "training_protocol.json").read_text(encoding="utf-8"))
        evaluation = json.loads((run / "evaluation_protocol.json").read_text(encoding="utf-8"))
        with (run / "test_metrics.csv").open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    except Exception as error:
        return False, f"unreadable artifacts: {error}"
    if (
        training.get("method") != job["method"]
        or int(training.get("seed", -1)) != job["seed"]
        or int(training.get("epochs", -1)) != args.epochs
        or int(training.get("episodes_per_epoch", -1)) != args.episodes_per_epoch
    ):
        return False, "training protocol mismatch"
    if (
        evaluation.get("eval_seeds") != args.eval_seeds
        or int(evaluation.get("episodes_per_seed", -1)) != args.eval_episodes
    ):
        return False, "evaluation protocol mismatch"
    if len(rows) != len(args.eval_seeds) * args.eval_episodes:
        return False, "frozen-test row count mismatch"
    if any(
        row.get("Scenario") != f"lbf_{job['scenario']}-v0"
        or int(row.get("Train_Seed", -1)) != job["seed"]
        for row in rows
    ):
        return False, "test identity mismatch"
    return True, str(run)


def run_job(job: dict, log_root: Path, threads: int):
    log_path = log_root / f"{job['scenario']}__seed_{job['seed']}.log"
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
    }
    completed = subprocess.run(
        job["command"],
        cwd=PROJECT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    return completed.returncode, log_path


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 5:
        raise ValueError("--workers must be between 1 and 5.")
    if args.gpu >= 0 and args.workers > 2:
        raise ValueError(
            "CAHC permits at most two concurrent training processes on one GPU."
        )
    if args.cpu_threads_per_job < 1:
        raise ValueError("--cpu-threads-per-job must be positive.")
    if args.gpu >= 0 and args.workers == 2 and not 0 < args.gpu_memory_fraction <= 0.5:
        raise ValueError(
            "Two GPU workers require a per-process memory fraction in (0, 0.5]."
        )
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    all_comparisons = registry["formal_scenario_ids"]
    all_supplementary = registry["supplementary_scenario_ids"]
    all_probes = registry["reviewer_probe_ids"]
    allowed = set(all_comparisons)
    if args.include_probes:
        allowed.update(all_probes)
    scenarios = args.scenarios or [
        *all_comparisons,
        *(all_probes if args.include_probes else []),
    ]
    unknown = [scenario for scenario in scenarios if scenario not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown or disabled --scenarios values: {unknown}. "
            "Use --include-probes when selecting a Reviewer probe."
        )
    if len(scenarios) != len(set(scenarios)):
        raise ValueError("--scenarios must not contain duplicate IDs.")
    if args.variant != "full" and scenarios != ["probe_interaction"]:
        raise ValueError("This launcher's no_wave control is restricted to Probe2.")
    comparisons = [scenario for scenario in scenarios if scenario in all_comparisons]
    supplementary = [scenario for scenario in scenarios if scenario in all_supplementary]
    probes = [scenario for scenario in scenarios if scenario in all_probes]
    main_seeds = registry["formal_seeds"]
    probe_ids = set(registry["reviewer_probe_ids"])
    if (len(all_comparisons) != 6 or all_supplementary
            or main_seeds != [20, 23, 56, 89, 2025]):
        raise ValueError("The frozen registry must contain six scenarios and five seeds.")
    if len(all_probes) != 2:
        raise ValueError("Exactly two independent Reviewer probes are required.")
    if "probe_interaction" in scenarios:
        if scenarios != ["probe_interaction"]:
            raise ValueError("Probe2 V1 must run alone in its fresh-seed result root.")
        rules = registry["scenarios"]["probe_interaction"]["probe_rules"]
        seeds = args.train_seeds or main_seeds
        if args.eval_seeds != rules["formal_eval_seeds"]:
            raise ValueError(
                f"Probe2 V1 --eval-seeds must be {rules['formal_eval_seeds']}."
            )
    else:
        seeds = args.train_seeds or main_seeds
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("--train-seeds must contain exactly five unique integers.")
    if set(seeds) & set(args.eval_seeds):
        raise ValueError("Training and evaluation seeds must be disjoint.")
    trace_seed = seeds[0]

    args.results_root = args.results_root.resolve()
    manifest_path = args.results_root / "cahc_launcher_manifest.json"
    if "probe_interaction" in scenarios and not args.resume and any(
        args.results_root.glob("*/lbf_probe_interaction-v0/seed_*/run*/test_metrics.csv")
    ):
        raise ValueError("Probe2 runs already exist; use --resume or an empty result root.")
    log_root = args.results_root / "launcher_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for scenario_index, scenario in enumerate(scenarios, start=1):
        for seed in seeds:
            jobs.append(
                {
                    "scenario_index": scenario_index,
                    "scenario": scenario,
                    "suite": (
                        "reviewer_probe" if scenario in probe_ids
                        else "supplementary" if scenario in all_supplementary
                        else "comparison"
                    ),
                    "method": "cahc" if args.variant == "full" else args.variant,
                    "seed": seed,
                    "gpu": args.gpu,
                    "status": "PENDING",
                    "command": command_for(
                        scenario, seed, args, probe_ids, trace_seed
                    ),
                }
            )
    manifest = {
        "protocol": registry["protocol_version"],
        "registry_sha256": sha256(REGISTRY_PATH),
        "experiment_plan_sha256": sha256(PLAN_PATH),
        "launcher_sha256": sha256(Path(__file__)),
        "formal_entrypoint_sha256": sha256(PROJECT / "formal_cahc.py"),
        "runtime": runtime_metadata(),
        "schedule": (
            "scenario first; five seeds dispatched within the current scenario; "
            "the next scenario waits for all five"
        ),
        "comparison_scenarios": comparisons,
        "supplementary_scenarios": supplementary,
        "probe_scenarios": probes,
        "trace_scenarios": sorted(TRACE_SCENARIOS),
        "trace_training_seed": trace_seed,
        "seeds": seeds,
        "epochs": args.epochs,
        "variant": args.variant,
        "episodes_per_epoch": args.episodes_per_epoch,
        "eval_seeds": args.eval_seeds,
        "eval_episodes_per_seed": args.eval_episodes,
        "gpu": args.gpu,
        "workers": args.workers,
        "gpu_worker_limit": 2,
        "cpu_threads_per_job": args.cpu_threads_per_job,
        "include_probes": args.include_probes,
        "job_count": len(jobs),
        "dry_run": args.dry_run,
        "resume": args.resume,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": jobs,
    }
    if args.resume:
        for job in jobs:
            reusable, detail = reusable_run(job, args)
            if reusable:
                job.update(status="PASS", completion_source="validated_existing_artifacts", run_dir=detail)
            else:
                job["resume_validation"] = detail
    if args.dry_run:
        for job in jobs:
            if job["status"] != "PASS":
                job["status"] = "DRY_RUN"
        manifest["status"] = "DRY_RUN"
        write_manifest(manifest_path, manifest)
        reused = sum(job["status"] == "PASS" for job in jobs)
        print(f"DRY RUN: {len(jobs)} jobs; reused={reused}; remaining={len(jobs)-reused}; manifest={manifest_path}")
        if log_root.is_dir() and not any(log_root.iterdir()):
            log_root.rmdir()
        return

    manifest["status"] = "RUNNING"
    write_manifest(manifest_path, manifest)
    for scenario in scenarios:
        batch = [
            job for job in jobs
            if job["scenario"] == scenario and job["status"] != "PASS"
        ]
        if not batch:
            print(f"RESUME scenario={scenario}: all five seeds validated", flush=True)
            continue
        print(f"START scenario={scenario} seeds={seeds}", flush=True)
        with ThreadPoolExecutor(max_workers=min(args.workers, len(batch))) as executor:
            futures = {
                executor.submit(
                    run_job, job, log_root, args.cpu_threads_per_job
                ): job
                for job in batch
            }
            failed = []
            for future in as_completed(futures):
                job = futures[future]
                returncode, log_path = future.result()
                job.update(
                    returncode=returncode,
                    log=str(log_path),
                    status="PASS" if returncode == 0 else "FAIL",
                )
                print(
                    f"{job['status']} scenario={scenario} seed={job['seed']}",
                    flush=True,
                )
                if returncode:
                    failed.append(job)
                else:
                    log_path.unlink(missing_ok=True)
                    job["log"] = None
                write_manifest(manifest_path, manifest)
        if failed:
            manifest["status"] = "FAIL"
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_manifest(manifest_path, manifest)
            raise SystemExit(
                f"Stopped after {len(failed)} failed job(s); inspect {log_root}."
            )
    manifest["status"] = "PASS"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_manifest(manifest_path, manifest)
    try:
        summary_message = build_paper_tables(args, seeds, scenarios)
    except Exception as error:
        manifest["status"] = "FAIL_SUMMARY"
        manifest["summary_error"] = f"{type(error).__name__}: {error}"
        write_manifest(manifest_path, manifest)
        raise
    manifest["paper_tables"] = ["test_seed_summary.csv", "test_summary.csv"]
    write_manifest(manifest_path, manifest)
    if log_root.is_dir() and not any(log_root.iterdir()):
        log_root.rmdir()
    print(f"PASS: all {len(jobs)} CAHC jobs completed; {manifest_path}")
    print(summary_message)


if __name__ == "__main__":
    main()
