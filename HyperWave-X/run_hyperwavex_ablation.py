"""Run the preregistered cross-regime HyperWave-X ablation matrix."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from paper_statistics import holm, paired_sign_flip
from summarize_test_results import ci95_half_width, mean, sample_sd


PROJECT = Path(__file__).resolve().parent
PLAN = PROJECT / "hyperwavex_experiment_plan.json"
REGISTRY = PROJECT / "scenario_registry.json"

# Reviewer-aligned ablation: every component control runs on the same two
# retained scenarios. This complete crossing avoids selecting a different
# favourable scenario for each component.
FINAL_ABLATION_MATRIX = {
    "dynamic_overload": ["no_capability", "no_wave", "complete_topology"],
    "scale_20": ["no_capability", "no_wave", "complete_topology"],
}
EXPECTED_SEEDS = [20, 23, 56, 89, 2025]
EXPECTED_EPOCHS = 700
EXPECTED_EPISODES_PER_EPOCH = 64
EXPECTED_EVAL_SEEDS = list(range(1001, 1011))
EXPECTED_EVAL_EPISODES = 20
EXPECTED_VARIANT_FLAGS = {
    "no_capability": {
        "capability_aware": False, "use_wave": True, "complete_topology": False,
    },
    "no_wave": {
        "capability_aware": True, "use_wave": False, "complete_topology": False,
    },
    "complete_topology": {
        "capability_aware": True, "use_wave": True, "complete_topology": True,
    },
}
REQUIRED_TEST_METRICS = (
    "Episode_Return",
    "Task_Completion_Ratio",
    "Capacity_Completion_Ratio",
    "Episode_Success_Rate",
    "Synchronization_Success_Rate",
    "Missing_Member_Rate",
    "Mean_Active_Link_Ratio",
    "Mean_Active_Links_Per_Step",
    "Estimated_Payload_Bytes_Per_Step",
)
REQUIRED_COMMUNICATION_METRICS = (
    "Logical_Transfers_Per_Step",
    "Logical_Feature_Scalars_Per_Step",
    "Incidence_Traversals_Per_Step",
    "Logical_Feature_Bytes_FP32_Per_Step",
)
# These files determine the learned policy and environment semantics.  The
# formal driver is validated separately through its frozen budget/config
# artifacts; excluding it permits reporting-only diagnostic fixes (such as a
# null capability field in the no_capability trace) without falsely rejecting
# otherwise identical full-control checkpoints.
CORE_COMPARABILITY_FILES = (
    "hyperwavex.py",
    "gnn_layers.py",
    "trainer.py",
    "data.py",
    "env_wrappers.py",
    "shared_lbf_env.py",
    "operational_metrics.py",
    "fair_costs.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--episodes-per-epoch", type=int, default=64)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=list(range(1001, 1011)))
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu-threads-per-job", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Run only these preregistered ablation scenarios.",
    )
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help="Run only these preregistered ablation variants.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Reuse only fully validated existing runs and execute missing jobs. "
            "An incomplete existing run is rejected instead of overwritten."
        ),
    )
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument(
        "--full-results-roots", nargs="+", type=Path, default=None,
        help=(
            "One or more PASS formal-launcher roots containing the matching "
            "five-seed hyperwave_x full controls. Required for a formal run."
        ),
    )
    reference_group.add_argument(
        "--auto-discover-full-results", action="store_true",
        help=(
            "Find PASS formal-launcher roots below --full-results-search-root "
            "that contain the required full controls. Duplicate controls fail."
        ),
    )
    parser.add_argument(
        "--full-results-search-root", type=Path, default=PROJECT / "results",
        help="Search location used by --auto-discover-full-results.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT / "results" / "ablation_results",
        help=(
            "Result root. The default layout is results/ablation_results/<variant>/"
            "lbf_<scenario>-v0/seed_<seed>/runN."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def current_core_hashes() -> dict[str, str]:
    """Hashes that must match between full controls and ablation variants."""
    return {name: sha256(PROJECT / name) for name in CORE_COMPARABILITY_FILES}


def accepted_result_registry_hashes() -> set[str]:
    """Registry hashes whose retained scenario specifications are identical."""
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    current = hashlib.sha256(
        json.dumps(
            registry, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        current,
        *plan.get("accepted_completed_result_registry_sha256", []),
    }


def validate_completed_run(
    results_root: Path,
    scenario: str,
    variant: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[bool, dict]:
    seed_root = (
        results_root / variant / f"lbf_{scenario}-v0" / f"seed_{seed}"
    )
    runs = sorted(path for path in seed_root.glob("run*") if path.is_dir())
    details = {"seed_root": str(seed_root), "run_count": len(runs)}
    if len(runs) != 1:
        details["reason"] = "expected exactly one immutable run directory"
        return False, details
    run = runs[0]
    details["run_dir"] = str(run)
    required = [
        "model.pt", "config.json", "training_metrics.csv",
        "training_protocol.json", "evaluation_protocol.json",
        "test_metrics.csv", "comparison_metrics.csv",
        "communication_costs.csv",
    ]
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        details.update(reason="missing required artifacts", missing=missing)
        return False, details
    try:
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        training_protocol = json.loads(
            (run / "training_protocol.json").read_text(encoding="utf-8")
        )
        evaluation = json.loads(
            (run / "evaluation_protocol.json").read_text(encoding="utf-8")
        )
        training = read_csv(run / "training_metrics.csv")
        test = read_csv(run / "test_metrics.csv")
        comparison = read_csv(run / "comparison_metrics.csv")
        communication = read_csv(run / "communication_costs.csv")
    except Exception as error:
        details.update(reason="unreadable artifact", error=repr(error))
        return False, details
    expected_test_rows = len(args.eval_seeds) * args.eval_episodes
    expected_flags = EXPECTED_VARIANT_FLAGS[variant]
    accepted_registry_hashes = accepted_result_registry_hashes()
    checks = {
        "scenario": config.get("env_name") == f"lbf_{scenario}-v0",
        "variant": config.get("variant") == variant,
        "method_label": config.get("method_label") == variant,
        **{
            f"variant_flag_{name}": config.get(name) is value
            for name, value in expected_flags.items()
        },
        "seed": int(config.get("seed", -1)) == seed,
        "training_rows": len(training) == args.epochs,
        "final_epoch": bool(training)
        and int(float(training[-1].get("Epoch", -1))) == args.epochs,
        "test_rows": len(test) == expected_test_rows,
        "comparison_rows": len(comparison) == expected_test_rows,
        "communication_rows": len(communication) == expected_test_rows,
        "backward": bool(training_protocol.get("backward_verified")),
        "optimizer": bool(training_protocol.get("optimizer_step_verified")),
        "finite_parameters": bool(training_protocol.get("parameters_finite")),
        "zero_eval_updates": evaluation.get("optimizer_updates") == 0,
        "checkpoint_loaded": bool(evaluation.get("checkpoint_load_verified")),
        "parameters_unchanged": bool(evaluation.get("parameters_unchanged")),
        "eval_seeds": evaluation.get("eval_seeds") == args.eval_seeds,
        "eval_episodes": int(evaluation.get("episodes_per_seed", -1))
        == args.eval_episodes,
        "paper_metric_columns": bool(test),
        "training_epochs": int(training_protocol.get("epochs", -1)) == args.epochs,
        "episodes_per_epoch": int(
            training_protocol.get("episodes_per_epoch", -1)
        ) == args.episodes_per_epoch,
        "scenario_registry_semantics": (
            training_protocol.get("scenario_registry_sha256")
            in accepted_registry_hashes
            and config.get("scenario_registry_sha256")
            in accepted_registry_hashes
            and evaluation.get("scenario_registry_sha256")
            in accepted_registry_hashes
        ),
        "source_hashes_recorded": bool(training_protocol.get("source_sha256")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        details.update(reason="formal artifact check failed", failed_checks=failed)
        return False, details
    test_columns = set(test[0]) if test else set()
    communication_columns = set(communication[0]) if communication else set()
    missing_test_columns = sorted(set(REQUIRED_TEST_METRICS) - test_columns)
    missing_communication_columns = sorted(
        set(REQUIRED_COMMUNICATION_METRICS) - communication_columns
    )
    if missing_test_columns or missing_communication_columns:
        details.update(
            reason="Reviewer-required ablation CSV columns are missing",
            missing_test_columns=missing_test_columns,
            missing_communication_columns=missing_communication_columns,
        )
        return False, details
    expected_pairs = {
        (eval_seed, episode)
        for eval_seed in args.eval_seeds
        for episode in range(1, args.eval_episodes + 1)
    }
    observed_pairs = {
        (int(row["Eval_Seed"]), int(row["Episode"])) for row in test
    }
    if observed_pairs != expected_pairs:
        details["reason"] = "evaluation seed/episode coverage mismatch"
        return False, details
    checkpoint_hash = sha256(run / "model.pt")
    csv_hashes = {
        row.get("Checkpoint_SHA256") for row in test
        if row.get("Checkpoint_SHA256")
    }
    if (evaluation.get("checkpoint_sha256") != checkpoint_hash
            or (csv_hashes and csv_hashes != {checkpoint_hash})):
        details["reason"] = "checkpoint hash mismatch"
        return False, details
    metadata = {
        "Method", "Scenario", "Evaluation_Phase", "Action_Mode",
        "Checkpoint_SHA256",
    }
    for row in test:
        if (
            row.get("Method") != variant
            or row.get("Scenario") != f"lbf_{scenario}-v0"
            or int(row.get("Train_Seed", -1)) != seed
            or row.get("Evaluation_Phase") not in (None, "", "frozen_test")
        ):
            details["reason"] = "frozen-test identity mismatch"
            return False, details
        for key, value in row.items():
            if key not in metadata and value != "" and not math.isfinite(float(value)):
                details.update(reason="non-finite metric", metric=key)
                return False, details
    for row in communication:
        if (
            row.get("Method") != variant
            or row.get("Scenario") != f"lbf_{scenario}-v0"
            or int(row.get("Train_Seed", -1)) != seed
        ):
            details["reason"] = "communication-cost identity mismatch"
            return False, details
    communication_pairs = {
        (int(row["Eval_Seed"]), int(row["Episode"])) for row in communication
    }
    if communication_pairs != expected_pairs:
        details["reason"] = "communication seed/episode coverage mismatch"
        return False, details
    communication_hashes = {
        row.get("Checkpoint_SHA256") for row in communication
        if row.get("Checkpoint_SHA256")
    }
    if communication_hashes and communication_hashes != {checkpoint_hash}:
        details["reason"] = "communication checkpoint hash mismatch"
        return False, details
    for row in communication:
        for metric in REQUIRED_COMMUNICATION_METRICS:
            if not math.isfinite(float(row[metric])):
                details.update(reason="non-finite communication metric", metric=metric)
                return False, details
    details.update(
        status="PASS",
        run_dir=str(run),
        checkpoint_sha256=checkpoint_hash,
        training_rows=len(training),
        frozen_test_rows=len(test),
    )
    return True, details


def build_paper_tables(args: argparse.Namespace) -> str:
    # A canonical organized directory may contain both the ablation variants
    # and their Full references.  Do not pass that same directory twice to the
    # summarizer, otherwise every episode would be discovered twice.
    result_root = args.results_root.resolve()
    additional_roots = []
    for path in args.full_results_roots:
        resolved = path.resolve()
        if resolved != result_root and resolved not in additional_roots:
            additional_roots.append(resolved)
    command = [
        sys.executable,
        str(PROJECT / "summarize_test_results.py"),
        str(args.results_root),
        "--expected-train-seeds", "5",
        "--expected-eval-seeds", *[str(value) for value in args.eval_seeds],
        "--episodes-per-eval-seed", str(args.eval_episodes),
        "--paired-reference", "hyperwave_x",
    ]
    if additional_roots:
        command[3:3] = [
            "--additional-roots", *[str(path) for path in additional_roots],
        ]
    completed = subprocess.run(
        command, cwd=PROJECT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        raise RuntimeError("Ablation summary failed:\n" + completed.stdout)
    return completed.stdout.strip()


def discover_full_results_roots(
    search_root: Path, scenarios: list[str],
) -> list[Path]:
    """Find PASS formal or canonical-organized roots with required controls."""
    search_root = search_root.resolve()
    if not search_root.is_dir():
        raise FileNotFoundError(f"Full-results search root does not exist: {search_root}")
    required = set(scenarios)
    roots = []
    manifests = sorted(search_root.rglob("hyperwavex_launcher_manifest.json"))
    manifests += sorted(search_root.rglob("organization_manifest.json"))
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as error:
            raise RuntimeError(
                f"Unreadable formal manifest during discovery: {manifest_path}"
            ) from error
        if manifest.get("status") != "PASS":
            continue
        if manifest_path.name == "organization_manifest.json":
            if manifest.get("method") != "hyperwave_x":
                continue
            available = set(manifest.get("scenario_order", []))
        else:
            available = {
                job.get("scenario")
                for job in manifest.get("jobs", [])
                if job.get("method") == "hyperwave_x" and job.get("status") == "PASS"
            }
        if required & available:
            roots.append(manifest_path.parent.resolve())
    roots = sorted(set(roots))
    if not roots:
        raise RuntimeError(
            "No PASS formal HyperWave-X roots containing any of "
            f"{sorted(required)} were found below {search_root}."
        )
    return roots


def find_full_reference_runs(
    roots: list[Path], scenarios: list[str], args: argparse.Namespace,
) -> dict[tuple[str, str, int], Path]:
    """Resolve one immutable full-method run for every required paired control."""
    indexed: dict[tuple[str, str, int], list[Path]] = defaultdict(list)
    for root in roots:
        launcher_manifest = root / "hyperwavex_launcher_manifest.json"
        organized_manifest = root / "organization_manifest.json"
        if launcher_manifest.is_file():
            manifest_path = launcher_manifest
            manifest_kind = "formal_launcher"
        elif organized_manifest.is_file():
            manifest_path = organized_manifest
            manifest_kind = "canonical_organization"
        else:
            raise FileNotFoundError(
                "Missing HyperWave-X formal or canonical organization manifest "
                f"below full-reference root: {root}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS":
            raise RuntimeError(
                f"Full-reference manifest must be PASS: {manifest_path}; "
                f"got {manifest.get('status')}"
            )
        if manifest_kind == "canonical_organization":
            if manifest.get("method") != "hyperwave_x":
                raise RuntimeError(
                    f"Organized full-reference method must be hyperwave_x: {manifest_path}"
                )
            if manifest.get("formal_seeds") != EXPECTED_SEEDS:
                raise RuntimeError(
                    "Organized full-reference seeds disagree with the frozen "
                    f"protocol: {manifest_path}"
                )
            missing_scenarios = sorted(
                set(scenarios) - set(manifest.get("scenario_order", []))
            )
            if missing_scenarios:
                raise RuntimeError(
                    f"Organized full-reference root lacks {missing_scenarios}: "
                    f"{manifest_path}"
                )
            validation = manifest.get("validation", {})
            failed_checks = sorted(
                name for name, passed in validation.items() if not bool(passed)
            )
            if failed_checks:
                raise RuntimeError(
                    "Organized full-reference manifest has failed validation "
                    f"checks {failed_checks}: {manifest_path}"
                )
        for scenario in scenarios:
            for seed in EXPECTED_SEEDS:
                seed_root = root / "hyperwave_x" / f"lbf_{scenario}-v0" / f"seed_{seed}"
                for run in seed_root.glob("run*"):
                    if run.is_dir():
                        indexed[("hyperwave_x", scenario, seed)].append(run)

    resolved = {}
    expected_rows = len(args.eval_seeds) * args.eval_episodes
    required_files = {
        "model.pt", "config.json", "training_protocol.json",
        "evaluation_protocol.json",
        "test_metrics.csv", "communication_costs.csv",
    }
    expected_core_hashes = current_core_hashes()
    accepted_registry_hashes = accepted_result_registry_hashes()
    for scenario in scenarios:
        for seed in EXPECTED_SEEDS:
            key = ("hyperwave_x", scenario, seed)
            runs = indexed.get(key, [])
            if len(runs) != 1:
                raise RuntimeError(
                    f"Expected exactly one full reference for {scenario}/seed_{seed}; "
                    f"found {len(runs)} across {roots}."
                )
            run = runs[0]
            missing = sorted(name for name in required_files if not (run / name).is_file())
            if missing:
                raise RuntimeError(f"Full reference {run} is missing {missing}.")
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            training_protocol = json.loads(
                (run / "training_protocol.json").read_text(encoding="utf-8")
            )
            evaluation = json.loads(
                (run / "evaluation_protocol.json").read_text(encoding="utf-8")
            )
            test = read_csv(run / "test_metrics.csv")
            communication = read_csv(run / "communication_costs.csv")
            checkpoint_hash = sha256(run / "model.pt")
            checks = {
                "variant": config.get("variant") == "full",
                "method_label": config.get("method_label") == "hyperwave_x",
                "capability_aware": config.get("capability_aware") is True,
                "use_wave": config.get("use_wave") is True,
                "complete_topology": config.get("complete_topology") is False,
                "scenario": config.get("env_name") == f"lbf_{scenario}-v0",
                "seed": int(config.get("seed", -1)) == seed,
                "training_method": training_protocol.get("method") == "hyperwave_x",
                "training_epochs": int(training_protocol.get("epochs", -1))
                == args.epochs,
                "episodes_per_epoch": int(
                    training_protocol.get("episodes_per_epoch", -1)
                ) == args.episodes_per_epoch,
                "source_hashes_recorded": bool(training_protocol.get("source_sha256")),
                "scenario_registry_semantics": (
                    training_protocol.get("scenario_registry_sha256")
                    in accepted_registry_hashes
                    and config.get("scenario_registry_sha256")
                    in accepted_registry_hashes
                    and evaluation.get("scenario_registry_sha256")
                    in accepted_registry_hashes
                ),
                "test_rows": len(test) == expected_rows,
                "communication_rows": len(communication) == expected_rows,
                "eval_seeds": evaluation.get("eval_seeds") == args.eval_seeds,
                "eval_episodes": int(evaluation.get("episodes_per_seed", -1))
                == args.eval_episodes,
                "parameters_unchanged": bool(evaluation.get("parameters_unchanged")),
                "checkpoint_loaded": bool(evaluation.get("checkpoint_load_verified")),
                "checkpoint_hash": evaluation.get("checkpoint_sha256")
                == checkpoint_hash,
                "paper_metric_columns": bool(test),
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise RuntimeError(
                    f"Full reference {run} failed preflight checks: {failed}."
                )
            if not test or not set(REQUIRED_TEST_METRICS).issubset(test[0]):
                raise RuntimeError(f"Full reference test CSV lacks required metrics: {run}")
            if not communication or not set(REQUIRED_COMMUNICATION_METRICS).issubset(
                communication[0]
            ):
                raise RuntimeError(
                    f"Full reference communication CSV lacks required metrics: {run}"
                )
            expected_pairs = {
                (eval_seed, episode)
                for eval_seed in args.eval_seeds
                for episode in range(1, args.eval_episodes + 1)
            }
            test_pairs = {
                (int(row["Eval_Seed"]), int(row["Episode"])) for row in test
            }
            communication_pairs = {
                (int(row["Eval_Seed"]), int(row["Episode"]))
                for row in communication
            }
            if test_pairs != expected_pairs or communication_pairs != expected_pairs:
                raise RuntimeError(
                    f"Full reference seed/episode coverage mismatch: {run}"
                )
            for row in test:
                if (
                    row.get("Method") != "hyperwave_x"
                    or row.get("Scenario") != f"lbf_{scenario}-v0"
                    or int(row.get("Train_Seed", -1)) != seed
                    or row.get("Evaluation_Phase") not in (None, "", "frozen_test")
                    or row.get("Checkpoint_SHA256") not in (None, "", checkpoint_hash)
                ):
                    raise RuntimeError(f"Full reference test identity mismatch: {run}")
                for metric in REQUIRED_TEST_METRICS:
                    if not math.isfinite(float(row[metric])):
                        raise RuntimeError(
                            f"Non-finite full-reference metric {metric}: {run}"
                        )
            for row in communication:
                if (
                    row.get("Method") != "hyperwave_x"
                    or row.get("Scenario") != f"lbf_{scenario}-v0"
                    or int(row.get("Train_Seed", -1)) != seed
                    or row.get("Checkpoint_SHA256") not in (None, "", checkpoint_hash)
                ):
                    raise RuntimeError(
                        f"Full reference communication identity mismatch: {run}"
                    )
                for metric in REQUIRED_COMMUNICATION_METRICS:
                    if not math.isfinite(float(row[metric])):
                        raise RuntimeError(
                            f"Non-finite full-reference communication metric "
                            f"{metric}: {run}"
                        )
            resolved[key] = run
    return resolved


def write_reviewer_ablation_tables(
    args: argparse.Namespace,
    scenarios: list[str],
    variants: list[str],
    run_index: dict[tuple[str, str, int], Path],
) -> list[str]:
    """Write auditable performance and communication ablation tables."""
    methods = ["hyperwave_x", *variants]
    by_seed = []
    for scenario in scenarios:
        formal_scenario = f"lbf_{scenario}-v0"
        for method in methods:
            for seed in EXPECTED_SEEDS:
                identity = (method, formal_scenario, seed)
                run_dir = run_index[(method, scenario, seed)]
                test = read_csv(run_dir / "test_metrics.csv")
                communication = read_csv(run_dir / "communication_costs.csv")
                expected_rows = len(args.eval_seeds) * args.eval_episodes
                if len(test) != expected_rows:
                    raise RuntimeError(
                        f"Incomplete frozen-test rows for {identity}: "
                        f"{len(test)} of {expected_rows}."
                    )
                row = {
                    "Method": method,
                    "Scenario": formal_scenario,
                    "Train_Seed": seed,
                    "N_Frozen_Test_Episodes": len(test),
                    "N_Eval_Seeds": len(args.eval_seeds),
                }
                for metric in REQUIRED_TEST_METRICS:
                    values = [float(item[metric]) for item in test]
                    if not all(math.isfinite(value) for value in values):
                        raise RuntimeError(f"Non-finite frozen-test metric: {identity}/{metric}")
                    row[metric] = f"{mean(values):.9f}"
                for metric in REQUIRED_COMMUNICATION_METRICS:
                    values = [float(item[metric]) for item in communication]
                    if len(values) != len(args.eval_seeds) * args.eval_episodes:
                        raise RuntimeError(
                            f"Incomplete communication rows for {identity}/{metric}."
                        )
                    row[metric] = f"{mean(values):.9f}"
                by_seed.append(row)

    by_seed_path = args.results_root / "ablation_reviewer_metrics_by_seed.csv"
    with by_seed_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(by_seed[0]))
        writer.writeheader()
        writer.writerows(by_seed)

    metric_names = [*REQUIRED_TEST_METRICS, *REQUIRED_COMMUNICATION_METRICS]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in by_seed:
        grouped[(row["Method"], row["Scenario"])].append(row)
    summary_rows = []
    for (method, scenario), rows in sorted(grouped.items()):
        seeds = sorted(int(row["Train_Seed"]) for row in rows)
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(
                f"Reviewer table requires seeds {EXPECTED_SEEDS}: "
                f"{method}/{scenario} has {seeds}."
            )
        for metric in metric_names:
            values = [float(row[metric]) for row in rows]
            summary_rows.append({
                "Method": method,
                "Scenario": scenario,
                "Metric": metric,
                "Mean": f"{mean(values):.9f}",
                "Sample_SD_Across_Train_Seeds": f"{sample_sd(values):.9f}",
                "CI95_Half_Width": f"{ci95_half_width(values):.9f}",
                "N_Train_Seeds": len(values),
                "Train_Seeds": " ".join(str(seed) for seed in EXPECTED_SEEDS),
            })
    summary_path = args.results_root / "ablation_reviewer_metrics_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    values_by_key = {
        (row["Method"], row["Scenario"], int(row["Train_Seed"]), metric):
        float(row[metric])
        for row in by_seed
        for metric in metric_names
    }
    paired_rows = []
    for scenario in scenarios:
        formal_scenario = f"lbf_{scenario}-v0"
        for variant in variants:
            for metric in metric_names:
                differences = [
                    values_by_key[("hyperwave_x", formal_scenario, seed, metric)]
                    - values_by_key[(variant, formal_scenario, seed, metric)]
                    for seed in EXPECTED_SEEDS
                ]
                paired_rows.append({
                    "Reference": "hyperwave_x",
                    "Variant": variant,
                    "Scenario": formal_scenario,
                    "Metric": metric,
                    "Mean_Full_Minus_Variant": f"{mean(differences):.9f}",
                    "Sample_SD_of_Differences": f"{sample_sd(differences):.9f}",
                    "CI95_Half_Width": f"{ci95_half_width(differences):.9f}",
                    "Exact_P_Two_Sided": paired_sign_flip(differences),
                    "N_Paired_Train_Seeds": len(differences),
                    "Train_Seeds": " ".join(str(seed) for seed in EXPECTED_SEEDS),
                    "Multiplicity_Family": "same scenario and endpoint",
                })
    families: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(paired_rows):
        families[(row["Scenario"], row["Metric"])].append(index)
    for indices in families.values():
        adjusted = holm([paired_rows[index]["Exact_P_Two_Sided"] for index in indices])
        for index, adjusted_p in zip(indices, adjusted):
            paired_rows[index]["Holm_Adjusted_P"] = adjusted_p
            paired_rows[index]["Significant_At_0_05"] = adjusted_p < 0.05
    paired_path = args.results_root / "ablation_reviewer_paired_comparisons.csv"
    with paired_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    return [path.name for path in (by_seed_path, summary_path, paired_path)]


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 5:
        raise ValueError("--workers must be between 1 and 5.")
    if args.gpu >= 0 and args.workers > 2:
        raise ValueError("A single GPU may run at most two HyperWave-X processes.")
    if args.cpu_threads_per_job < 1:
        raise ValueError("--cpu-threads-per-job must be positive.")
    if args.gpu >= 0 and args.workers == 2 and not 0 < args.gpu_memory_fraction <= 0.5:
        raise ValueError("Two GPU workers require memory fraction in (0, 0.5].")
    formal_protocol = {
        "epochs": (args.epochs, EXPECTED_EPOCHS),
        "episodes_per_epoch": (
            args.episodes_per_epoch, EXPECTED_EPISODES_PER_EPOCH
        ),
        "eval_seeds": (args.eval_seeds, EXPECTED_EVAL_SEEDS),
        "eval_episodes": (args.eval_episodes, EXPECTED_EVAL_EPISODES),
    }
    mismatched_protocol = {
        name: {"received": received, "required": required}
        for name, (received, required) in formal_protocol.items()
        if received != required
    }
    if mismatched_protocol:
        raise ValueError(
            "Reviewer ablation protocol is fixed; mismatches: "
            f"{mismatched_protocol}"
        )
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    seeds = registry["formal_seeds"]
    if seeds != EXPECTED_SEEDS:
        raise ValueError(
            f"Reviewer ablation requires seeds {EXPECTED_SEEDS}; found {seeds}."
        )
    main_matrix = plan["ablation_matrix"]
    if main_matrix != FINAL_ABLATION_MATRIX:
        raise ValueError(
            "The experiment plan disagrees with the Reviewer-aligned final "
            "2-scenario x 3-variant ablation matrix. Expected "
            f"{FINAL_ABLATION_MATRIX}, found {main_matrix}."
        )
    matrix = main_matrix
    known_scenarios = set(matrix)
    known_variants = {
        variant for variants in matrix.values() for variant in variants
    }
    # The no-argument command runs the frozen Reviewer-complete 30-job grid.
    selected_scenarios = args.scenarios or list(FINAL_ABLATION_MATRIX)
    if len(selected_scenarios) != len(set(selected_scenarios)):
        raise ValueError(f"Duplicate --scenarios are not allowed: {selected_scenarios}")
    selected_variants = set(args.variants or known_variants)
    unknown_scenarios = sorted(set(selected_scenarios) - known_scenarios)
    unknown_variants = sorted(selected_variants - known_variants)
    if unknown_scenarios or unknown_variants:
        raise ValueError(
            "Unknown preregistered ablation selection: "
            f"scenarios={unknown_scenarios}, variants={unknown_variants}"
        )
    args.full_results_search_root = args.full_results_search_root.resolve()
    if args.full_results_roots:
        args.full_results_roots = [path.resolve() for path in args.full_results_roots]
        full_reference_selection = "explicit"
    elif args.auto_discover_full_results:
        args.full_results_roots = discover_full_results_roots(
            args.full_results_search_root, selected_scenarios
        )
        full_reference_selection = "auto_discovered"
    else:
        args.full_results_roots = []
        full_reference_selection = "omitted_for_dry_run"
    if not args.dry_run and not args.full_results_roots:
        raise ValueError(
            "A formal Reviewer-compliant run requires --full-results-roots or "
            "--auto-discover-full-results so full-vs-ablation paired statistics "
            "can be generated."
        )
    selected_variant_order = []
    for scenario in selected_scenarios:
        for variant in matrix[scenario]:
            if variant in selected_variants and variant not in selected_variant_order:
                selected_variant_order.append(variant)
    full_reference_runs = (
        {}
        if args.dry_run
        else find_full_reference_runs(args.full_results_roots, selected_scenarios, args)
    )
    if not args.dry_run:
        print(
            f"FULL REFERENCE PREFLIGHT PASS: {len(full_reference_runs)} "
            f"scenario-seed controls from {len(args.full_results_roots)} root(s)",
            flush=True,
        )
    jobs = []
    suites = [("main", matrix, False)]
    for suite, matrix, allow_probe in suites:
        for scenario, variants in matrix.items():
            if scenario not in selected_scenarios:
                continue
            for variant in variants:
                if variant not in selected_variants:
                    continue
                for seed in seeds:
                    command = [
                        sys.executable,
                        str(PROJECT / "formal_hyperwavex.py"),
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
                        "--results-root", str(args.results_root.resolve()),
                    ]
                    if allow_probe:
                        command.append("--allow-probe")
                    jobs.append((suite, scenario, variant, seed, command))
    if not jobs:
        raise ValueError(
            "The selected scenarios and variants contain no preregistered pair."
        )
    reviewer_complete_grid = (
        set(selected_scenarios) == set(FINAL_ABLATION_MATRIX)
        and selected_variants
        == {variant for values in FINAL_ABLATION_MATRIX.values() for variant in values}
    )
    if reviewer_complete_grid and len(jobs) != 30:
        raise RuntimeError(
            f"Reviewer-complete ablation must schedule 30 jobs; found {len(jobs)}."
        )
    args.results_root = args.results_root.resolve()
    manifest = {
        "source_plan": str(PLAN),
        "source_sha256": {
            "hyperwavex_experiment_plan.json": sha256(PLAN),
            "scenario_registry.json": sha256(REGISTRY),
            "run_hyperwavex_ablation.py": sha256(Path(__file__)),
            "formal_hyperwavex.py": sha256(PROJECT / "formal_hyperwavex.py"),
        },
        "schedule": "scenario -> variant -> five seeds in parallel; strict variant barrier",
        "seeds": seeds,
        "job_count": len(jobs),
        "main_job_count": sum(job[0] == "main" for job in jobs),
        "probe_job_count": sum(job[0] == "probe" for job in jobs),
        "epochs": args.epochs,
        "episodes_per_epoch": args.episodes_per_epoch,
        "eval_seeds": args.eval_seeds,
        "eval_episodes_per_seed": args.eval_episodes,
        "gpu": args.gpu,
        "cpu_threads_per_job": args.cpu_threads_per_job,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "workers": args.workers,
        "gpu_worker_limit": 2,
        "results_root": str(args.results_root),
        "protocol": registry["protocol_version"],
        "selected_scenarios": selected_scenarios,
        "selected_variants": selected_variant_order,
        "reviewer_complete_grid": reviewer_complete_grid,
        "full_results_roots": [str(path) for path in args.full_results_roots],
        "full_results_root_selection": full_reference_selection,
        "full_results_search_root": str(args.full_results_search_root),
        "paired_reference": "hyperwave_x",
        "required_test_metrics": list(REQUIRED_TEST_METRICS),
        "required_communication_metrics": list(REQUIRED_COMMUNICATION_METRICS),
        "resume": args.resume,
        "main_full_reference": plan["main_ablation_references"],
        "main_ablation_rationale": plan["main_ablation_rationale"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": [
            {
                "suite": suite,
                "scenario": scenario,
                "variant": variant,
                "seed": seed,
                "command": command,
                "status": "DRY_RUN" if args.dry_run else "PENDING",
            }
            for suite, scenario, variant, seed, command in jobs
        ],
        "status": "DRY_RUN" if args.dry_run else "RUNNING",
    }
    args.results_root.mkdir(parents=True, exist_ok=True)
    formal_manifest_path = args.results_root / "hyperwavex_ablation_manifest.json"
    manifest_path = (
        args.results_root / "hyperwavex_ablation_resume_dryrun_manifest.json"
        if args.resume and args.dry_run else formal_manifest_path
    )
    if formal_manifest_path.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(
            f"A formal manifest already exists at {formal_manifest_path}. "
            "Use --resume to validate and reuse complete runs, or select a new "
            "results root."
        )
    if args.resume:
        reused = 0
        for record in manifest["jobs"]:
            valid, details = validate_completed_run(
                args.results_root,
                record["scenario"],
                record["variant"],
                record["seed"],
                args,
            )
            record["resume_validation"] = details
            if valid:
                record["status"] = "PASS"
                record["completion_source"] = "validated_existing_artifacts"
                reused += 1
            elif details.get("run_count", 0):
                raise RuntimeError(
                    "Resume found an incomplete or incompatible existing run: "
                    f"{record['scenario']}/{record['variant']}/seed_{record['seed']}: "
                    f"{details.get('reason')}"
                )
        manifest["resume_reused_complete_jobs"] = reused
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.dry_run:
        manifest["status"] = "RESUME_DRY_RUN" if args.resume else "DRY_RUN"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            f"DRY RUN: {len(jobs)} jobs; "
            f"reused={manifest.get('resume_reused_complete_jobs', 0)}; "
            f"manifest={manifest_path}"
        )
        return

    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": str(args.cpu_threads_per_job),
        "MKL_NUM_THREADS": str(args.cpu_threads_per_job),
        "OPENBLAS_NUM_THREADS": str(args.cpu_threads_per_job),
        "NUMEXPR_NUM_THREADS": str(args.cpu_threads_per_job),
    }
    logs = args.results_root / "launcher_logs"
    logs.mkdir(exist_ok=True)
    scenario_order = []
    for suite, scenario, _variant, _seed, _command in jobs:
        if (suite, scenario) not in scenario_order:
            scenario_order.append((suite, scenario))
    for suite, scenario in scenario_order:
        for variant in matrix[scenario]:
            variant_jobs = [
                record for record in manifest["jobs"]
                if record["suite"] == suite
                and record["scenario"] == scenario
                and record["variant"] == variant
                and record["status"] == "PENDING"
            ]
            if not variant_jobs:
                continue
            if len(variant_jobs) != len(EXPECTED_SEEDS):
                raise RuntimeError(
                    f"{scenario}/{variant} must contain exactly five seed jobs."
                )
            print(
                f"START {scenario}/{variant}: parallel seeds {EXPECTED_SEEDS}",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        subprocess.run,
                        record["command"],
                        cwd=PROJECT,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=environment,
                    ): record
                    for record in variant_jobs
                }
                failed = []
                for future in as_completed(futures):
                    record = futures[future]
                    seed = record["seed"]
                    completed = future.result()
                    log_path = logs / f"{suite}__{scenario}__{variant}__seed_{seed}.log"
                    log_path.write_text(completed.stdout, encoding="utf-8")
                    artifacts_valid, artifact_details = validate_completed_run(
                        args.results_root, scenario, variant, seed, args
                    )
                    record.update(
                        status="PASS" if artifacts_valid else "FAIL",
                        returncode=completed.returncode,
                        log=str(log_path),
                        artifact_validation=artifact_details,
                        completion_source=(
                            "process_exit_zero_and_validated_artifacts"
                            if completed.returncode == 0 and artifacts_valid
                            else "validated_artifacts_after_nonzero_native_teardown"
                            if artifacts_valid
                            else "failed_or_incomplete_artifacts"
                        ),
                    )
                    if not artifacts_valid:
                        failed.append((variant, seed))
                    else:
                        log_path.unlink(missing_ok=True)
                        record["log"] = None
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2), encoding="utf-8"
                    )
                if failed:
                    manifest["status"] = "FAIL"
                    manifest["failed"] = failed
                    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2), encoding="utf-8"
                    )
                    raise SystemExit(
                        f"Ablation failed in {suite}/{scenario}/{variant}: {failed}"
                    )
            print(f"PASS {scenario}/{variant}", flush=True)
    manifest["status"] = "PASS"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        summary_message = build_paper_tables(args)
        run_index = dict(full_reference_runs)
        for record in manifest["jobs"]:
            details = record.get("artifact_validation") or record.get(
                "resume_validation", {}
            )
            run_dir = details.get("run_dir")
            if record.get("status") != "PASS" or not run_dir:
                raise RuntimeError(
                    "Cannot build Reviewer tables from an unvalidated job: "
                    f"{record['scenario']}/{record['variant']}/seed_{record['seed']}"
                )
            key = (record["variant"], record["scenario"], record["seed"])
            if key in run_index:
                raise RuntimeError(f"Duplicate ablation result identity: {key}")
            run_index[key] = Path(run_dir)
        expected_run_count = (
            len(selected_scenarios)
            * (1 + len(selected_variant_order))
            * len(EXPECTED_SEEDS)
        )
        if len(run_index) != expected_run_count:
            raise RuntimeError(
                f"Expected {expected_run_count} full-plus-ablation run groups; "
                f"found {len(run_index)}."
            )
        reviewer_tables = write_reviewer_ablation_tables(
            args,
            selected_scenarios,
            selected_variant_order,
            run_index,
        )
    except Exception as error:
        manifest["status"] = "FAIL_ANALYSIS"
        manifest["analysis_error"] = f"{type(error).__name__}: {error}"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise
    manifest["paper_tables"] = [
        "test_seed_summary.csv",
        "test_summary.csv",
        "test_paired_comparisons.csv",
        *reviewer_tables,
    ]
    manifest["analysis_finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if logs.is_dir() and not any(logs.iterdir()):
        logs.rmdir()
    print(
        f"PASS: {len(jobs)} Reviewer-aligned ablation jobs plus paired full "
        f"controls; manifest={manifest_path}"
    )
    print(summary_message)
    print(
        "Wrote Reviewer ablation tables: "
        + ", ".join(str(args.results_root / name) for name in reviewer_tables)
    )


if __name__ == "__main__":
    main()
