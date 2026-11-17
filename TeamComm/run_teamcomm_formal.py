"""Scenario-first launcher for every formal TeamComm baseline and seed.

One invocation covers 6 scenarios x 7 baselines x 5 training seeds.  Within
each baseline, the five seeds can run concurrently; the next scenario starts
only after all baselines and seeds for the current scenario have succeeded.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


TEAMCOMM = Path(__file__).resolve().parent
WORKSPACE = TEAMCOMM.parent
ARTIFACTS = TEAMCOMM / "experiment_artifacts"
REGISTRY_PATH = TEAMCOMM / "scenario_registry.json"
BASELINES_PATH = TEAMCOMM / "formal_baselines.json"

# Historical reuse is allowed only for scenarios whose executable specification
# is unchanged. hetero_mixed, hetero_strong, dynamic_overload, and scale_20
# were deliberately redesigned on 2026-10-19 and therefore have no
# legacy-registry exception.
EXECUTION_EQUIVALENT_REGISTRY_HASHES = {
    **{
        (scenario, "8d68ef6f26c040fa56089304a2f9fcfafa28f794cd04e3977d5601a004946742"):
        "same final-V1 executable scenario; registry-only cleanup removed excluded legacy names"
        for scenario in ("hetero_mixed", "hetero_strong", "scale_20")
    },
    ("dynamic_overload", "717e78f3bc8326b82e3253531b2bc325a84a044be59692440bcf0de694fddada"):
        "same final-V1 executable scenario; registry-only cleanup removed excluded legacy names",
    **{
        (scenario, "fa38dc42420b48697af78886a782afb90ce0df117734b03c3d0fdc53ba6cf9b4"):
        "comparison scenario executable specification unchanged by the noisy-interlock Probe-2-only revision"
        for scenario in (
            "hetero_mixed",
            "hetero_strong",
            "consensus_wide",
            "fog_sight1_level",
            "dynamic_overload",
            "scale_20",
        )
    },
    **{
        (scenario, "7572945a5fe2fbc2a8a30de6e037cb9bb66e3b036c79464a6f1786e88bdad571"):
        "scenario executable specification unchanged by the consensus_wide-only redesign"
        for scenario in (
            "fog_sight1_level",
        )
    },
    **{
        (scenario, "35f62642a3562f9ad4e4c70020b5be56097ca7866d76a78f358bb84bfec37b94"):
        "six-scenario executable specification unchanged by the final-V1 namespace and Probe-2-only revision"
        for scenario in (
            "consensus_wide",
            "fog_sight1_level",
        )
    },
}

# This historical TeamComm environment hash differs from final V1 only in the
# two reviewer-probe branches and probe-only observation/status fields.  It is
# accepted solely for the six comparison scenarios and solely together with an
# approved scenario-registry equivalence above.
PROBE_ONLY_SOURCE_EQUIVALENCE = {
    "shared_lbf_env.py": {
        "0ae92c6b1ae69e60660027b1d05f24327d4bd41bc1ae0bb416c2f054e8d2d124",
        "f3e57f3d383dbe56f0a4eb7b56f393de624c41658b2a6dbd43f10a4ef1ecca67",
        "285e2e8d5362e0e062e5967f2ca3a591a9881c1493f2dae88271d8fd321ef87f",
        "3567f5206d889be6224478418016c96ade98691d74ee47e41ee7a765fb41487c",
    },
}

# These exact recorded digests differ from the retained sources only through
# paper-facing CSV schema cleanup, removal of host-dependent timing, or deleted
# audit helpers. They cannot change observations, actions, rewards, model
# outputs, losses, optimizer updates, or checkpoints. Keeping this whitelist
# explicit prevents unrelated future source edits from being silently reused.
REPORTING_ONLY_SOURCE_EQUIVALENCE = {
    "qmix_main.py": {
        "9d432201482a865049c021dafd2f3f8905a9d4e52fc74c12c29b84fee63de254",
        "c4077d8c4a9377da8c39a9fe7bda5fb4c6e315846683eb176588236391c337c5",
    },
    "operational_metrics.py": {
        "3c82ce12614fac161e515232fbe3c425063d3a2a1f55d0ef211c5ef5ea7d74fc",
        "bb0b029e610267bc0e3c2d097d1d72ac08fe6820a7ab60dc879270c455ea09f1",
    },
    "benchmark_fair_latency.py": {
        "00029855de7ab9d731c3764599f3a56d15c86c694e60dd9245e56911059fc610",
    },
    "main.py": {
        "224ec10aa76d129cfa8770dfe5eba2d0d351ba3fca73e081e8d2727984f781d6",
        "4ba520fefd02078cee5e847fb4c8c1c7c20bc468453ab45d2b74df1a435a8075",
    },
    "validate_reviewer_probes.py": {
        "62447ecabc0d8a41368c1696ac28e0efde1f37efcd29b19cd1d5fdf3d1ed1213",
        "949174404df2d1c2633aa316ccd9bc12f92c5c3b05f2fcd97bcf6f0a5b7309a5",
        "a3fc2e05cae77806bf5c1a926e78bb3bc25ee4f9dab80dd6bed9882ae2b3af7d",
        "ac5a0331f103af38ddabafea3a5c2e9111dfc55c7bb38c62b4a6383958e8b8a2",
    },
    "paper_evaluation.py": {
        "002578805b26eea259d97954d60e3f439bf20f853df3c26697d959fad496a327",
        "4801b5441c23a3d579c5388b93d86fa65d1b77526206d98da1e1a24c47263535",
    },
    "fair_costs.py": {
        "73241b736297ad6e53511f0337328fd2abfa28ed30fdb5ddcb179af57c83e30b",
    },
    "scenario_registry.py": {
        "2a1ce9fe6aa2cbe80330371ad274d5fd1d7475d424494fcd544fe83b82e290bd",
        "a2b7939d1dd549614c95167a37ee4c0c3ae9d7e8cbd339a6d7def0cab05d20c7",
    },
    "runner/runner_ic3net.py": {
        "5d1a69e0991ed805b93e244661af490b36d2f146787a66674803b767ee3aaba8",
    },
    "runner/runner_teamcomm.py": {
        "9addb3b80f36d6082a91d506932526c31f5e7f4c9a587f8030b2036ea3816d75",
    },
    "runner/runner_magic.py": {
        "2f13e53983ff5e670b61114634d65dc2f6943350503a6ae35a92159d6bb642da",
    },
    "runner/runner_tiecomm.py": {
        "cbc0bd90bf9f2b1c343c85ce671ea1d3432aab22a3dc10012a24f6f0ce606d7b",
    },
    "runner/runner_baselines.py": {
        "521322cb8fcb80a957e167b97c4f3788c9e013d34c47f1687d34bcdcfde4953d",
    },
}

# Orchestration and offline audit/table utilities cannot affect a trained
# policy, environment transition, loss, or metric. They may be repaired after
# a run without invalidating the scientific artifacts produced by that run.
NON_SCIENTIFIC_SOURCE_FILES = {
    "run_teamcomm_formal.py",
    "build_paper_csv.py",
    "validate_formal_baselines.py",
    "tests/test_paper_protocol.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_completed_artifacts(job, args) -> tuple[bool, dict]:
    """Validate one immutable formal run before accepting or resuming it.

    This also distinguishes a late native-runtime teardown signal from a
    failed/incomplete scientific run.  A nonzero process exit is never enough
    to pass: every training, checkpoint, frozen-test, and cost artifact must be
    complete and internally consistent.
    """
    scenario_name = f"lbf_{job['scenario']}-v0"
    seed_root = args.results_root / job["baseline"] / scenario_name / f"seed_{job['seed']}"
    runs = sorted(path for path in seed_root.glob("run*") if path.is_dir())
    details = {"seed_root": str(seed_root), "run_count": len(runs)}
    if len(runs) != 1:
        details["reason"] = "expected exactly one immutable run directory"
        return False, details
    run_dir = runs[0]
    details["run_dir"] = str(run_dir)
    required = [
        "model.pt", "config.json", "training_protocol.json",
        "evaluation_protocol.json", "source_manifest.json", "test_metrics.csv",
        "comparison_metrics.csv", "communication_costs.csv",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        details.update(reason="missing required artifacts", missing=missing)
        return False, details
    training_name = (
        "metrics.csv" if job["baseline"] == "qmix"
        else f"{job['baseline']}_Foraging-{job['scenario']}_metrics.csv"
    )
    training_path = run_dir / training_name
    if not training_path.is_file():
        details.update(reason="missing training metrics", missing=[training_name])
        return False, details
    try:
        training = read_csv(training_path)
        test_rows = read_csv(run_dir / "test_metrics.csv")
        comparison = read_csv(run_dir / "comparison_metrics.csv")
        costs = read_csv(run_dir / "communication_costs.csv")
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        training_protocol = json.loads(
            (run_dir / "training_protocol.json").read_text(encoding="utf-8")
        )
        evaluation = json.loads(
            (run_dir / "evaluation_protocol.json").read_text(encoding="utf-8")
        )
        source_manifest = json.loads(
            (run_dir / "source_manifest.json").read_text(encoding="utf-8")
        )
    except Exception as error:
        details.update(reason="unreadable artifact", error=repr(error))
        return False, details

    expected_test_rows = len(args.eval_seeds) * args.eval_episodes
    if len(training) != args.epochs:
        details.update(reason="training row count mismatch", observed=len(training))
        return False, details
    epoch_key = "Epoch" if job["baseline"] == "qmix" else "epoch"
    if int(training[-1][epoch_key]) != args.epochs:
        details.update(reason="last epoch mismatch", observed=training[-1][epoch_key])
        return False, details
    if not all(len(rows) == expected_test_rows for rows in (test_rows, comparison, costs)):
        details.update(
            reason="frozen-test row count mismatch",
            test=len(test_rows), comparison=len(comparison), costs=len(costs),
            expected=expected_test_rows,
        )
        return False, details

    required_common_metrics = {
        "Agent_Count", "Capacity_Workload_Total",
        "Capacity_Workload_Completed", "Capacity_Completion_Ratio",
        "Episode_Success", "Episode_Success_Rate", "Collisions",
        "Collision_Rate", "Episode_Steps",
    }
    for row in comparison:
        missing_metrics = required_common_metrics.difference(row)
        if missing_metrics:
            details.update(
                reason="comparison CSV predates required common metric schema",
                missing=sorted(missing_metrics),
            )
            return False, details
        capacity_total = float(row["Capacity_Workload_Total"])
        expected_ccr = (
            float(row["Capacity_Workload_Completed"]) / capacity_total
            if capacity_total else 0.0
        )
        expected_cr = (
            float(row["Collisions"])
            / (float(row["Episode_Steps"]) * float(row["Agent_Count"]))
            if float(row["Episode_Steps"]) and float(row["Agent_Count"])
            else 0.0
        )
        if not math.isclose(
            float(row["Capacity_Completion_Ratio"]), expected_ccr, abs_tol=1e-12
        ):
            details["reason"] = "capacity-completion ratio mismatch"
            return False, details
        if not math.isclose(
            float(row["Episode_Success_Rate"]),
            float(row["Episode_Success"]),
            abs_tol=1e-12,
        ):
            details["reason"] = "episode-level ESR mismatch"
            return False, details
        if not math.isclose(
            float(row["Collision_Rate"]), expected_cr, abs_tol=1e-12
        ):
            details["reason"] = "collision-rate mismatch"
            return False, details

    observed_pairs = set()
    for row in test_rows:
        if (
            row["Method"] != job["baseline"]
            or row["Scenario"] != scenario_name
            or int(row["Train_Seed"]) != job["seed"]
        ):
            details["reason"] = "method/scenario/training-seed mismatch"
            return False, details
        eval_seed = int(row["Eval_Seed"])
        episode = int(row["Episode"])
        observed_pairs.add((eval_seed, episode))
        for key, value in row.items():
            if key in {"Method", "Scenario"}:
                continue
            if value != "" and not math.isfinite(float(value)):
                details.update(reason="non-finite frozen-test metric", column=key)
                return False, details
    expected_pairs = {
        (seed, episode)
        for seed in args.eval_seeds
        for episode in range(1, args.eval_episodes + 1)
    }
    if observed_pairs != expected_pairs:
        details["reason"] = "evaluation seed/episode coverage mismatch"
        return False, details
    checkpoint_digest = sha256(run_dir / "model.pt")
    if evaluation.get("checkpoint_sha256") != checkpoint_digest:
        details["reason"] = "evaluation protocol checkpoint mismatch"
        return False, details
    if not evaluation.get("checkpoint_load_verified"):
        details["reason"] = "checkpoint reload was not verified"
        return False, details
    if evaluation.get("eval_seeds") != args.eval_seeds:
        details["reason"] = "evaluation seed list mismatch"
        return False, details
    if int(evaluation.get("episodes_per_seed", -1)) != args.eval_episodes:
        details["reason"] = "evaluation episode budget mismatch"
        return False, details
    if (
        training_protocol.get("method") != job["baseline"]
        or training_protocol.get("scenario") != job["scenario"]
        or int(training_protocol.get("seed", -1)) != job["seed"]
        or not training_protocol.get("backward_verified")
        or not training_protocol.get("optimizer_step_verified")
        or not training_protocol.get("parameters_finite")
    ):
        details["reason"] = "training protocol verification mismatch"
        return False, details
    training_registry = training_protocol.get("scenario_registry_sha256")
    evaluation_registry = evaluation.get("scenario_registry_sha256")
    registry_matches = (
        training_registry == args.registry_semantic_sha256
        and evaluation_registry == args.registry_semantic_sha256
    )
    if not registry_matches:
        equivalence_reason = EXECUTION_EQUIVALENT_REGISTRY_HASHES.get(
            (job["scenario"], training_registry)
        )
        if training_registry != evaluation_registry or equivalence_reason is None:
            details.update(
                reason="scenario-registry semantic hash mismatch",
                training_registry_sha256=training_registry,
                evaluation_registry_sha256=evaluation_registry,
                required_registry_sha256=args.registry_semantic_sha256,
            )
            return False, details
        source_mismatches = {}
        for relative, recorded_digest in source_manifest.items():
            # The launcher is the orchestration code being repaired here; it
            # does not define the environment, model, loss, or evaluator.
            if relative in NON_SCIENTIFIC_SOURCE_FILES:
                continue
            current = TEAMCOMM / relative
            reporting_only = REPORTING_ONLY_SOURCE_EQUIVALENCE.get(relative, set())
            if recorded_digest in reporting_only:
                continue
            if not current.is_file():
                source_mismatches[relative] = "missing"
            elif sha256(current) != recorded_digest:
                approved_hashes = PROBE_ONLY_SOURCE_EQUIVALENCE.get(relative, set())
                if recorded_digest not in approved_hashes:
                    source_mismatches[relative] = "hash_changed"
        if source_mismatches:
            details.update(
                reason="legacy registry exception rejected by source mismatch",
                source_mismatches=source_mismatches,
            )
            return False, details
        details.update(
            registry_equivalence="APPROVED_SCENARIO_SPECIFIC",
            historical_registry_sha256=training_registry,
            required_registry_sha256=args.registry_semantic_sha256,
            registry_equivalence_reason=equivalence_reason,
            scientific_source_hashes_verified=sum(
                relative not in NON_SCIENTIFIC_SOURCE_FILES
                for relative in source_manifest
            ),
        )
    configured_epochs = config.get("epochs", config.get("total_epoches"))
    if configured_epochs is not None and int(configured_epochs) != args.epochs:
        details["reason"] = "configured epoch budget mismatch"
        return False, details
    details.update(
        status="PASS", training_rows=len(training), frozen_test_rows=len(test_rows),
        checkpoint_sha256=checkpoint_digest,
    )
    return True, details


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all TeamComm-owned formal baselines under the frozen CAIE protocol."
    )
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--episodes-per-epoch", type=int, default=64)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=list(range(1001, 1011)))
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--cpu-threads-per-job", type=int, default=2)
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Optional main scenario IDs; omission runs all six main scenarios.",
    )
    parser.add_argument(
        "--baselines", nargs="+", default=None,
        help="Optional TeamComm-owned baseline IDs; omission runs all baselines.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=TEAMCOMM / "results/FORMAL_6SCENARIOS_V1",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="continue this immutable results root after validating and reusing complete jobs",
    )
    parser.add_argument(
        "--archive-incompatible", action="store_true",
        help="with --resume, move incompatible/incomplete prior seed directories into a non-reportable audit archive before rerunning them",
    )
    return parser.parse_args()


def command_for(method, scenario, seed, horizon, args):
    common_eval = [str(value) for value in args.eval_seeds]
    if method == "qmix":
        return [
            sys.executable,
            str(TEAMCOMM / "qmix_main.py"),
            "--scenario", scenario,
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--episodes-per-epoch", str(args.episodes_per_epoch),
            "--eval-seeds", *common_eval,
            "--eval-episodes", str(args.eval_episodes),
            "--results-root", str(args.results_root),
        ]
    return [
        sys.executable,
        str(TEAMCOMM / "main.py"),
        "--paper_protocol",
        "--env", "lbf",
        "--agent", method,
        "--map", f"Foraging-{scenario}",
        "--seed", str(seed),
        "--total_epoches", str(args.epochs),
        # The original TeamComm runner defines batch_size in transitions.
        # Fixed-horizon episodes therefore use horizon * episode budget so the
        # original collection loop remains unchanged.
        "--batch_size", str(args.episodes_per_epoch * horizon),
        "--eval_seeds", *common_eval,
        "--eval_episodes_per_seed", str(args.eval_episodes),
        "--paper_results_root", str(args.results_root),
    ]


def run_job(job, log_root, cpu_threads_per_job):
    log_path = log_root / f"{job['scenario']}__{job['baseline']}__seed_{job['seed']}.log"
    environment = dict(os.environ)
    for variable in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = str(cpu_threads_per_job)
    # Keep the child output outside results_root until the process exits.  A
    # results organiser or cleanup must not make the launcher crash merely by
    # moving/removing launcher_logs while a long formal job is running.
    environment.setdefault("MPLCONFIGDIR", str(TEAMCOMM / ".matplotlib"))
    log_root.mkdir(parents=True, exist_ok=True)
    prefix = f"teamcomm_{job['scenario']}_{job['baseline']}_{job['seed']}_"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=prefix, suffix=".log", delete=False
    ) as spool:
        spool_path = Path(spool.name)
        try:
            completed = subprocess.run(
                job["command"], cwd=TEAMCOMM, text=True, stdout=spool,
                stderr=subprocess.STDOUT, env=environment,
            )
            returncode = completed.returncode
        except BaseException as error:
            spool.write(
                "\nLAUNCHER_EXCEPTION: "
                f"{type(error).__name__}: {error}\n"
            )
            returncode = -1

    # Recreate the directory immediately before publishing the log.  shutil.move
    # also works when the temporary spool and workspace are different mounts.
    for attempt in range(3):
        try:
            log_root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(spool_path), str(log_path))
            break
        except FileNotFoundError:
            if attempt == 2:
                raise
    return returncode, log_path


def write_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    if args.archive_incompatible and not args.resume:
        raise ValueError("--archive-incompatible requires --resume")
    if args.workers < 1 or args.workers > 5:
        raise ValueError("--workers must be between 1 and 5 (one worker per formal seed).")
    if args.cpu_threads_per_job < 1:
        raise ValueError("--cpu-threads-per-job must be at least 1")
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    from scenario_registry import registry_sha256 as canonical_registry_sha256
    registry_semantic_sha256 = canonical_registry_sha256(registry)
    args.registry_semantic_sha256 = registry_semantic_sha256
    baseline_spec = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))
    all_scenarios = registry["formal_scenario_ids"]
    supplementary_scenarios = registry["supplementary_scenario_ids"]
    seeds = registry["formal_seeds"]
    all_methods = baseline_spec["baselines"]
    if (len(all_scenarios) != 6 or supplementary_scenarios
            or seeds != [20, 23, 56, 89, 2025]):
        raise ValueError("Frozen registry must contain exactly six scenarios and the five formal seeds.")
    if "hypercomm" in all_methods or "hyperwave_x" in all_methods:
        raise ValueError("TeamComm launcher cannot own HyperComm or HyperWave-X jobs.")
    allowed_scenarios = all_scenarios
    scenarios = args.scenarios or [
        *all_scenarios,
    ]
    methods = args.baselines or list(all_methods)
    unknown_scenarios = [value for value in scenarios if value not in allowed_scenarios]
    unknown_methods = [value for value in methods if value not in all_methods]
    if unknown_scenarios:
        raise ValueError(f"Unknown formal scenario IDs: {unknown_scenarios}")
    if unknown_methods:
        raise ValueError(f"Unknown TeamComm baseline IDs: {unknown_methods}")
    if len(scenarios) != len(set(scenarios)):
        raise ValueError("--scenarios must not contain duplicates")
    if len(methods) != len(set(methods)):
        raise ValueError("--baselines must not contain duplicates")

    args.results_root = args.results_root.resolve()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    formal_manifest_path = ARTIFACTS / "teamcomm_launcher_manifest.json"
    manifest_path = (
        ARTIFACTS / "teamcomm_resume_dryrun_manifest.json"
        if args.resume and args.dry_run else formal_manifest_path
    )
    if formal_manifest_path.exists() and not args.resume and not args.dry_run:
        raise ValueError(
            f"Results root already contains a formal manifest: {formal_manifest_path}. "
            "Use --resume to validate completed jobs, or choose a new results root."
        )
    log_root = ARTIFACTS / "launcher_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    previous_manifest = None
    if args.resume:
        expected_resume_fields = {
            "protocol": registry["protocol_version"],
            "baseline_registry_sha256": sha256(BASELINES_PATH),
            "scenarios": scenarios,
            "baselines": methods,
            "seeds": seeds,
            "epochs": args.epochs,
            "episodes_per_epoch": args.episodes_per_epoch,
            "eval_seeds": args.eval_seeds,
            "eval_episodes_per_seed": args.eval_episodes,
        }
        previous_manifest = (
            json.loads(formal_manifest_path.read_text(encoding="utf-8"))
            if formal_manifest_path.is_file()
            else {**expected_resume_fields, "jobs": []}
        )
        mismatches = {}
        for key, value in expected_resume_fields.items():
            previous_value = previous_manifest.get(key)
            if previous_value == value:
                continue
            # A stopped scenario-first run may be extended by appending new
            # scenarios.  Previously declared scenarios must remain in the
            # same order; run-level validation below still decides which
            # completed jobs are scientifically reusable.
            if (
                key == "scenarios"
                and isinstance(previous_value, list)
                and value[:len(previous_value)] == previous_value
            ):
                continue
            mismatches[key] = {"previous": previous_value, "requested": value}
        if mismatches:
            raise ValueError(
                "Refusing to resume with a changed formal protocol: "
                + json.dumps(mismatches, ensure_ascii=False)
            )
    jobs = []
    for scenario_index, scenario in enumerate(scenarios, start=1):
        horizon = int(registry["scenarios"][scenario]["horizon"])
        for method_index, method in enumerate(methods, start=1):
            for seed in seeds:
                jobs.append({
                    "scenario_index": scenario_index,
                    "baseline_index": method_index,
                    "scenario": scenario,
                    "baseline": method,
                    "seed": seed,
                    "status": "PENDING",
                    "command": command_for(method, scenario, seed, horizon, args),
                })
    manifest = {
        "protocol": registry["protocol_version"],
        "registry_sha256": sha256(REGISTRY_PATH),
        "registry_semantic_sha256": registry_semantic_sha256,
        "baseline_registry_sha256": sha256(BASELINES_PATH),
        "python": sys.executable,
        "schedule": "scenario -> baseline -> five training seeds in parallel",
        "scenario_count": len(scenarios),
        "main_scenario_count": sum(item in all_scenarios for item in scenarios),
        "supplementary_scenario_count": 0,
        "baseline_count": len(methods),
        "seed_count": len(seeds),
        "job_count": len(jobs),
        "scenarios": scenarios,
        "baselines": methods,
        "seeds": seeds,
        "epochs": args.epochs,
        "episodes_per_epoch": args.episodes_per_epoch,
        "actor_critic_batch_rule": "batch_size = episodes_per_epoch * fixed scenario horizon",
        "eval_seeds": args.eval_seeds,
        "eval_episodes_per_seed": args.eval_episodes,
        "workers": args.workers,
        "cpu_threads_per_job": args.cpu_threads_per_job,
        "resume": args.resume,
        "dry_run": args.dry_run,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": jobs,
    }
    if previous_manifest is not None:
        previous_scenarios = previous_manifest.get("scenarios", [])
        manifest["resume_previous_scenarios"] = previous_scenarios
        manifest["resume_added_scenarios"] = [
            scenario for scenario in scenarios
            if scenario not in previous_scenarios
        ]
        previous_jobs = {
            (job["scenario"], job["baseline"], job["seed"]): job
            for job in previous_manifest["jobs"]
        }
        reused = 0
        for job in jobs:
            key = (job["scenario"], job["baseline"], job["seed"])
            previous_job = previous_jobs.get(key, {})
            job["previous_status"] = previous_job.get("status")
            job["previous_log"] = previous_job.get("log")
            job["previous_process_returncode"] = previous_job.get(
                "process_returncode", previous_job.get("returncode")
            )
            valid, validation = validate_completed_artifacts(job, args)
            job["resume_validation"] = validation
            if valid:
                job["status"] = "PASS"
                job["completion_source"] = "validated_existing_artifacts"
                job["process_returncode"] = previous_job.get(
                    "process_returncode", previous_job.get("returncode")
                )
                job["log"] = previous_job.get("log")
                reused += 1
        manifest["resume_from_started_utc"] = previous_manifest.get("started_utc")
        manifest["resume_reused_complete_jobs"] = reused
        manifest["resume_previous_registry_sha256"] = previous_manifest.get(
            "registry_sha256"
        )
    if args.dry_run:
        for job in jobs:
            job["status"] = (
                "REUSE_VALIDATED" if job["status"] == "PASS" else "DRY_RUN"
            )
        manifest["status"] = "RESUME_DRY_RUN" if args.resume else "DRY_RUN"
        write_manifest(manifest_path, manifest)
        reused = sum(job["status"] == "REUSE_VALIDATED" for job in jobs)
        print(
            f"DRY RUN: {len(jobs)} jobs; reused={reused}; "
            f"remaining={len(jobs)-reused}; manifest={manifest_path}"
        )
        if log_root.is_dir() and not any(log_root.iterdir()):
            log_root.rmdir()
        return

    incompatible_existing = [
        job for job in jobs
        if job["status"] != "PASS"
        and job.get("resume_validation", {}).get("run_count", 0) > 0
    ]
    if incompatible_existing and not args.archive_incompatible:
        examples = [
            f"{job['scenario']}/{job['baseline']}/seed_{job['seed']}: "
            f"{job['resume_validation'].get('reason')}"
            for job in incompatible_existing[:5]
        ]
        raise ValueError(
            "Resume found existing runs that cannot enter the frozen result set. "
            "Inspect the resume dry-run manifest, then rerun with "
            "--archive-incompatible to preserve and replace them. Examples: "
            + "; ".join(examples)
        )
    if incompatible_existing:
        archive_root = ARTIFACTS / "_nonreportable_resume_archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        archived_manifest = archive_root / "previous_launcher_manifest.json"
        if not archived_manifest.exists():
            archived_manifest.write_text(
                json.dumps(previous_manifest, indent=2), encoding="utf-8"
            )
        for job in incompatible_existing:
            seed_root = Path(job["resume_validation"]["seed_root"])
            registry_tag = str(
                job["resume_validation"].get("training_registry_sha256", "unknown")
            )[:12]
            destination = (
                archive_root / registry_tag / job["baseline"]
                / f"lbf_{job['scenario']}-v0" / f"seed_{job['seed']}"
            )
            if destination.exists():
                raise ValueError(f"Archive destination already exists: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(seed_root), str(destination))
            previous_log = job.get("previous_log")
            if previous_log and Path(previous_log).is_file():
                shutil.copy2(previous_log, destination / "launcher.log")
            job["archived_incompatible_run"] = str(destination)
            job["resume_validation"]["run_count"] = 0

    manifest["status"] = "RUNNING"
    write_manifest(manifest_path, manifest)
    for scenario in scenarios:
        for method in methods:
            batch = [
                job for job in jobs
                if job["scenario"] == scenario and job["baseline"] == method
                and job["status"] != "PASS"
            ]
            if not batch:
                print(
                    f"RESUME scenario={scenario} baseline={method}: "
                    "all five seeds passed artifact validation",
                    flush=True,
                )
                continue
            print(f"START scenario={scenario} baseline={method} seeds={seeds}", flush=True)
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
                    job["process_returncode"] = returncode
                    job["log"] = str(log_path)
                    valid, validation = validate_completed_artifacts(job, args)
                    job["artifact_validation"] = validation
                    try:
                        finished_marker = log_path.read_text(
                            encoding="utf-8", errors="replace"
                        ).rstrip().endswith("=====Done!!!=====")
                    except OSError as error:
                        # Artifact validation, rather than a missing auxiliary
                        # launcher log, determines whether a scientific run is
                        # complete. Preserve the diagnostic in the manifest.
                        finished_marker = False
                        validation["launcher_log_error"] = repr(error)
                    if valid and (returncode == 0 or finished_marker):
                        job["status"] = "PASS"
                        job["completion_source"] = (
                            "process_exit_zero" if returncode == 0
                            else "artifact_validation_after_native_teardown_signal"
                        )
                    else:
                        job["status"] = "FAIL"
                    print(
                        f"{job['status']} scenario={scenario} baseline={method} "
                        f"seed={job['seed']} process_returncode={returncode}", flush=True,
                    )
                    if job["status"] != "PASS":
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
    from build_paper_csv import build_tables
    manifest["paper_tables"] = build_tables(
        args.results_root,
        output_dir=ARTIFACTS / "paper_tables",
        require_complete=True,
        scenario_ids=scenarios,
        baseline_ids=methods,
        seed_ids=seeds,
    )
    manifest["status"] = "PASS"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_manifest(manifest_path, manifest)
    if log_root.is_dir() and not any(log_root.iterdir()):
        log_root.rmdir()
    print(
        f"PASS: all {len(jobs)} formal TeamComm jobs completed; "
        f"paper CSVs={ARTIFACTS / 'paper_tables'}; manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
