"""Run the Reviewer-aligned, five-seed CAHC sensitivity protocol."""

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
PLAN_PATH = PROJECT / "cahc_experiment_plan.json"
REGISTRY_PATH = PROJECT / "scenario_registry.json"
CLI_NAMES = {
    "capability_weight": "--capability-weight",
    "hypergraph_temperature": "--hypergraph-temperature",
    "wave_iterations": "--wave-iterations",
    "wave_tau": "--wave-tau",
    "wave_damping": "--wave-damping",
    "topology_temperature": "--topology-temperature",
}

# Deliberately duplicated from the JSON plan: the launcher refuses to start if
# an editable plan silently adds unnecessary sweeps or drops a requested factor.
REVIEWER_ONE_FACTOR_DESIGN = {
    ("consensus_wide", "wave_iterations"): [1, 2, 3],
    ("consensus_wide", "wave_damping"): [0.5, 0.8, 0.95],
}
EXPECTED_FORMAL_SEEDS = [20, 23, 56, 89, 2025]
EXPECTED_UNIQUE_SETTINGS = 5
EXPECTED_FULL_JOBS = EXPECTED_UNIQUE_SETTINGS * len(EXPECTED_FORMAL_SEEDS)
EXPECTED_NEW_TRAINING_JOBS = 4 * len(EXPECTED_FORMAL_SEEDS)
EXPECTED_REUSED_REFERENCE_EVALUATIONS = len(EXPECTED_FORMAL_SEEDS)
CORE_COMPARABILITY_FILES = (
    "cahc.py",
    "gnn_layers.py",
    "trainer.py",
    "data.py",
    "env_wrappers.py",
    "scenario_registry.py",
    "scenario_registry.json",
    "shared_lbf_env.py",
    "formal_cahc.py",
    "operational_metrics.py",
    "fair_costs.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def completed_job_ledger(
    results_root: Path,
    job: dict,
    validation_seeds: list[int],
    eval_episodes: int,
) -> Path | None:
    """Return the unique complete validation ledger for one job, if present."""
    seed_root = (
        results_root
        / f"sensitivity_{job['scenario']}_{job['label']}"
        / f"lbf_{job['scenario']}-v0"
        / f"seed_{job['seed']}"
    )
    expected_rows = len(validation_seeds) * eval_episodes
    complete = []
    for path in sorted(seed_root.glob("run*/test_metrics.csv")):
        rows = read_csv(path)
        observed_seeds = sorted({int(row["Eval_Seed"]) for row in rows})
        if len(rows) == expected_rows and observed_seeds == validation_seeds:
            complete.append(path)
    if len(complete) > 1:
        raise RuntimeError(
            "Resume found duplicate complete ledgers for "
            f"{job['scenario']}/{job['label']}/seed_{job['seed']}: {complete}"
        )
    return complete[0] if complete else None


def validate_resume_manifest(
    previous: dict,
    args: argparse.Namespace,
    jobs: list[dict],
    source_hashes: dict[str, str],
) -> None:
    """Reject a resume when the frozen scientific protocol has changed."""
    scalar_checks = {
        "epochs": args.epochs,
        "episodes_per_epoch": args.episodes_per_epoch,
        "eval_episodes_per_seed": args.eval_episodes,
        "training_seeds": EXPECTED_FORMAL_SEEDS,
    }
    mismatches = {
        name: (previous.get(name), expected)
        for name, expected in scalar_checks.items()
        if previous.get(name) != expected
    }
    previous_keys = {
        (job.get("scenario"), job.get("label"), int(job.get("seed", -1)))
        for job in previous.get("jobs", [])
    }
    current_keys = {
        (job["scenario"], job["label"], int(job["seed"])) for job in jobs
    }
    if previous_keys != current_keys:
        mismatches["job_grid"] = (sorted(previous_keys), sorted(current_keys))
    # Source hashes remain recorded in the old and new manifests.  Resume
    # eligibility is determined by the immutable job grid, budgets and the
    # complete per-job validation ledgers, allowing reporting-only schema and
    # relocation changes without retraining completed policies.
    if mismatches:
        raise ValueError(
            "Refusing to resume because the frozen sensitivity protocol changed: "
            f"{mismatches}"
        )


def current_core_hashes() -> dict[str, str]:
    return {name: sha256(PROJECT / name) for name in CORE_COMPARABILITY_FILES}


def resolve_resume_root(requested: Path) -> Path:
    """Resolve a moved/mistyped Windows resume root without guessing ambiguously."""
    requested = requested.resolve()
    manifest_name = "cahc_sensitivity_manifest.json"
    if (requested / manifest_name).is_file():
        return requested
    candidates = sorted({
        path.parent.resolve() for path in PROJECT.rglob(manifest_name)
        if path.is_file()
    })
    if len(candidates) == 1:
        recovered = candidates[0]
        print(
            "RESUME ROOT AUTO-RECOVERED: requested root has no manifest; "
            f"using the unique project manifest at {recovered}",
            flush=True,
        )
        return recovered
    details = "\n".join(f"  - {path}" for path in candidates) or "  - none"
    raise FileNotFoundError(
        "--resume requires an existing sensitivity manifest. "
        f"Requested: {requested / manifest_name}\n"
        "Project candidates:\n"
        f"{details}\n"
        "Pass the exact parent directory of the intended manifest. On Windows, "
        "do not copy Markdown escapes such as \\_ into the path."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal Reviewer-aligned CAHC OFAT sensitivity: K_w and "
            "delta_w on one scenario, with five training seeds per setting."
        )
    )
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--episodes-per-epoch", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu-threads-per-job", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Run only Reviewer-registered sweeps for these scenarios.",
    )
    parser.add_argument(
        "--parameters", nargs="+", default=None, choices=sorted(CLI_NAMES),
        help="Run only the selected Reviewer-registered one-factor sweeps.",
    )
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument(
        "--default-results-roots", nargs="+", type=Path, default=None,
        help=(
            "PASS formal-launcher root(s) containing the five default "
            "consensus_wide checkpoints to reuse without retraining."
        ),
    )
    reference_group.add_argument(
        "--auto-discover-default-results", action="store_true",
        help=(
            "Discover PASS formal roots containing consensus_wide below "
            "--default-results-search-root. Duplicate checkpoints are rejected."
        ),
    )
    parser.add_argument(
        "--default-results-search-root", type=Path, default=PROJECT / "results",
        help="Search location used by --auto-discover-default-results.",
    )
    parser.add_argument(
        "--results-root", type=Path,
        default=PROJECT / "results" / "sensitivity-results",
        help=(
            "Result root. The default setting layout is sensitivity-results/"
            "sensitivity_<scenario>_<parameter>_<value>."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume an interrupted result root after validating its protocol "
            "and completed per-job validation ledgers."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def token(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def default_values(plan: dict) -> dict:
    return {name: plan["default_hyperparameters"][name] for name in CLI_NAMES}


def validate_reviewer_protocol(plan: dict, registry: dict) -> None:
    protocol = plan.get("sensitivity_protocol", {})
    observed = {
        (entry.get("scenario"), entry.get("parameter")): entry.get("values")
        for entry in protocol.get("one_factor_design", [])
    }
    if observed != REVIEWER_ONE_FACTOR_DESIGN:
        raise ValueError(
            "Sensitivity plan is not the minimal Reviewer-aligned three-wave-factor protocol. "
            f"Expected {REVIEWER_ONE_FACTOR_DESIGN}; found {observed}."
        )
    if registry.get("formal_seeds") != EXPECTED_FORMAL_SEEDS:
        raise ValueError(
            "Reviewer protocol requires the five independent registered seeds "
            f"{EXPECTED_FORMAL_SEEDS}; found {registry.get('formal_seeds')}."
        )
    if protocol.get("validation_eval_seeds") != [901, 902, 903, 904, 905]:
        raise ValueError("Sensitivity analysis must use validation seeds 901--905.")
    if protocol.get("final_test_eval_seeds") != list(range(1001, 1011)):
        raise ValueError("Final test seeds 1001--1010 must remain untouched.")
    if protocol.get("validation_selection_allowed") is not True:
        raise ValueError("Reviewer protocol must permit selection from validation data only.")
    if protocol.get("direct_reuse_in_current_main_results_allowed") is not False:
        raise ValueError(
            "A validation-selected tuple must receive a fresh five-seed final run."
        )
    if protocol.get("expected_unique_setting_count") != EXPECTED_UNIQUE_SETTINGS:
        raise ValueError("Plan must declare exactly five unique sensitivity settings.")
    if protocol.get("expected_job_count") != EXPECTED_FULL_JOBS:
        raise ValueError("Plan must declare exactly 25 complete analysis jobs.")
    if protocol.get("expected_new_training_job_count") != EXPECTED_NEW_TRAINING_JOBS:
        raise ValueError("Plan must declare exactly 20 new sensitivity training jobs.")
    if (
        protocol.get("expected_reused_default_evaluation_count")
        != EXPECTED_REUSED_REFERENCE_EVALUATIONS
    ):
        raise ValueError(
            "Plan must declare five reused-default frozen validation evaluations."
        )
    if not protocol.get("default_checkpoint_reuse_rule"):
        raise ValueError("Plan must define the audited default-checkpoint reuse rule.")

    defaults = default_values(plan)
    for (_scenario, parameter), values in REVIEWER_ONE_FACTOR_DESIGN.items():
        if defaults[parameter] not in values:
            raise ValueError(
                f"The default {parameter}={defaults[parameter]} is absent from its grid."
            )
    if not all(isinstance(value, int) and value >= 1 for value in observed[("consensus_wide", "wave_iterations")]):
        raise ValueError("wave_iterations must be positive integers and include K_w=1.")
    if not all(0.0 < value < 1.0 for value in observed[("consensus_wide", "wave_damping")]):
        raise ValueError("wave_damping must stay in the contraction interval (0,1).")


def discover_default_results_roots(search_root: Path) -> list[Path]:
    """Find PASS formal shards that contain a consensus_wide full run."""
    search_root = search_root.resolve()
    if not search_root.is_dir():
        raise FileNotFoundError(
            f"Default-results search root does not exist: {search_root}"
        )
    roots = []
    manifests = sorted(search_root.rglob("cahc_launcher_manifest.json"))
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
            contains_reference = (
                manifest.get("method") == "cahc"
                and "consensus_wide" in manifest.get("scenario_order", [])
            )
        else:
            contains_reference = any(
                job.get("method") == "cahc"
                and job.get("scenario") == "consensus_wide"
                and job.get("status") == "PASS"
                for job in manifest.get("jobs", [])
            )
        if contains_reference:
            roots.append(manifest_path.parent.resolve())
    roots = sorted(set(roots))
    if not roots:
        raise RuntimeError(
            "No PASS formal CAHC root containing consensus_wide was "
            f"found below {search_root}."
        )
    return roots


def resolve_default_reference_runs(
    roots: list[Path],
    plan: dict,
    args: argparse.Namespace,
) -> dict[int, dict]:
    """Audit exactly one preregistered default checkpoint for every seed."""
    indexed: dict[int, list[Path]] = defaultdict(list)
    for root in roots:
        launcher_manifest = root / "cahc_launcher_manifest.json"
        organized_manifest = root / "organization_manifest.json"
        manifest_path = (
            launcher_manifest if launcher_manifest.is_file() else organized_manifest
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing formal manifest below: {root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS":
            raise RuntimeError(
                f"Default-reference manifest must be PASS: {manifest_path}; "
                f"got {manifest.get('status')}"
            )
        if manifest_path.name == "organization_manifest.json":
            if (manifest.get("method") != "cahc"
                    or "consensus_wide" not in manifest.get("scenario_order", [])):
                raise RuntimeError(f"Invalid organized full-result manifest: {manifest_path}")
            passed_reference_seeds = set(manifest.get("formal_seeds", []))
        else:
            passed_reference_seeds = {
                int(job["seed"])
                for job in manifest.get("jobs", [])
                if job.get("method") == "cahc"
                and job.get("scenario") == "consensus_wide"
                and job.get("status") == "PASS"
                and "seed" in job
            }
        for seed in sorted(passed_reference_seeds & set(EXPECTED_FORMAL_SEEDS)):
            seed_root = (
                root
                / "cahc"
                / "lbf_consensus_wide-v0"
                / f"seed_{seed}"
            )
            indexed[seed].extend(
                path for path in seed_root.glob("run*") if path.is_dir()
            )

    expected_hashes = current_core_hashes()
    defaults = default_values(plan)
    required_files = {
        "model.pt",
        "config.json",
        "training_metrics.csv",
        "training_protocol.json",
        "evaluation_protocol.json",
        "test_metrics.csv",
    }
    resolved = {}
    for seed in EXPECTED_FORMAL_SEEDS:
        runs = indexed.get(seed, [])
        if len(runs) != 1:
            raise RuntimeError(
                "Expected exactly one default consensus_wide checkpoint for "
                f"seed_{seed}; found {len(runs)} across {roots}."
            )
        run = runs[0]
        missing = sorted(name for name in required_files if not (run / name).is_file())
        if missing:
            raise RuntimeError(f"Default checkpoint {run} is missing {missing}.")
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        training_protocol = json.loads(
            (run / "training_protocol.json").read_text(encoding="utf-8")
        )
        evaluation_protocol = json.loads(
            (run / "evaluation_protocol.json").read_text(encoding="utf-8")
        )
        training_rows = read_csv(run / "training_metrics.csv")
        test_rows = read_csv(run / "test_metrics.csv")
        checkpoint_hash = sha256(run / "model.pt")
        expected_parameters = {
            **{name: defaults[name] for name in CLI_NAMES},
            # Zero activates the formal task-conditioned adaptive cardinality
            # rule and must not be replaced by a diagnostic fixed size.
            "hyperedge_size": 0,
            "hid_size": plan["default_hyperparameters"]["hidden_size"],
            "qk_hid_size": plan["default_hyperparameters"]["query_key_size"],
            "gat_num_heads": plan["default_hyperparameters"]["gat_heads"],
            "gat_hid_size": plan["default_hyperparameters"]["gat_head_width"],
            "lrate": plan["default_hyperparameters"]["learning_rate"],
            "value_coeff": plan["default_hyperparameters"][
                "value_loss_coefficient"
            ],
            "entr": plan["default_hyperparameters"]["entropy_coefficient"],
        }
        parameter_checks = {
            name: (
                isinstance(config.get(name), (int, float))
                and math.isclose(
                    float(config[name]), float(value), rel_tol=0.0, abs_tol=1e-12
                )
                if isinstance(value, (int, float))
                else config.get(name) == value
            )
            for name, value in expected_parameters.items()
        }
        checks = {
            "scenario": config.get("env_name") == "lbf_consensus_wide-v0",
            "seed": int(config.get("seed", -1)) == seed,
            "variant": config.get("variant") == "full",
            "method": config.get("method_label") == "cahc",
            "capability_aware": config.get("capability_aware") is True,
            "use_wave": config.get("use_wave") is True,
            "complete_topology": config.get("complete_topology") is False,
            "training_method": training_protocol.get("method") == "cahc",
            "epochs": int(training_protocol.get("epochs", -1)) == args.epochs,
            "episodes_per_epoch": int(
                training_protocol.get("episodes_per_epoch", -1)
            ) == args.episodes_per_epoch,
            "backward_verified": bool(
                training_protocol.get("backward_verified")
            ),
            "optimizer_step_verified": bool(
                training_protocol.get("optimizer_step_verified")
            ),
            "parameters_finite": bool(training_protocol.get("parameters_finite")),
            # Source hashes are retained in JSON provenance. Relocated result
            # bundles are accepted here; model/config compatibility is checked
            # again when the checkpoint is loaded for frozen evaluation.
            "source_hashes_recorded": bool(training_protocol.get("source_sha256")),
            "complete_training_rows": [
                int(float(row.get("Epoch", -1))) for row in training_rows
            ] == list(range(1, args.epochs + 1)),
            "complete_episode_budget": bool(training_rows)
            and all(
                int(float(row.get("Episodes", -1)))
                == args.episodes_per_epoch
                for row in training_rows
            ),
            "checkpoint_record": evaluation_protocol.get("checkpoint_sha256")
            == checkpoint_hash,
            "checkpoint_loaded": bool(
                evaluation_protocol.get("checkpoint_load_verified")
            ),
            "source_evaluation_frozen": bool(
                evaluation_protocol.get("parameters_unchanged")
            ),
            "test_metrics_present": bool(test_rows),
            "test_checkpoint": bool(test_rows) and (
                not {
                    row.get("Checkpoint_SHA256") for row in test_rows
                    if row.get("Checkpoint_SHA256")
                }
                or {
                    row.get("Checkpoint_SHA256") for row in test_rows
                    if row.get("Checkpoint_SHA256")
                } == {checkpoint_hash}
            ),
            **{f"parameter_{name}": passed for name, passed in parameter_checks.items()},
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                f"Default checkpoint preflight failed at {run}: {failed}"
            )
        resolved[seed] = {
            "run_dir": run.resolve(),
            "checkpoint_sha256": checkpoint_hash,
            "training_protocol_sha256": sha256(run / "training_protocol.json"),
        }
    return resolved


def validate_k1_degeneracy_identity() -> dict:
    """Check the Reviewer-requested K_w=1 identity on the actual implementation."""
    from types import SimpleNamespace

    import torch

    from gnn_layers import HyperGraph

    common = {
        "nagents": 3,
        "capability_aware": False,
        "capability_weight": 0.1,
        "hypergraph_temperature": 1.0,
        "hyperedge_size": 2,
        "task_observation_width": 3,
        "use_wave": True,
        "obs_size": 5,
        "hid_size": 4,
        "qk_hid_size": 3,
        "max_player_level": 6,
    }
    incidence = torch.tensor(
        [[[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )
    features = torch.tensor(
        [[[0.2, -0.4], [1.1, 0.3], [-0.7, 0.8]]],
        dtype=torch.float64,
    )

    def refine(iterations: int, tau: float, damping: float):
        module = HyperGraph(SimpleNamespace(
            **common,
            wave_iterations=iterations,
            wave_tau=tau,
            wave_damping=damping,
        ))
        return module.refine_wave(features, incidence)

    # Both pairs have delta*tau=0.4. They must be indistinguishable for one
    # step, while two anchored recurrence steps separate tau from delta.
    k1_left = refine(1, tau=0.5, damping=0.8)
    k1_right = refine(1, tau=0.8, damping=0.5)
    k2_left = refine(2, tau=0.5, damping=0.8)
    k2_right = refine(2, tau=0.8, damping=0.5)
    k1_difference = float((k1_left - k1_right).abs().max())
    k2_difference = float((k2_left - k2_right).abs().max())
    if k1_difference > 1e-12:
        raise AssertionError(
            f"K_w=1 same-product degeneracy failed: max difference={k1_difference}"
        )
    if k2_difference <= 1e-8:
        raise AssertionError(
            "K_w=2 should separate tau and damping in the anchored recurrence."
        )
    return {
        "status": "PASS",
        "same_product_pairs": [
            {"wave_tau": 0.5, "wave_damping": 0.8},
            {"wave_tau": 0.8, "wave_damping": 0.5},
        ],
        "delta_times_tau": 0.4,
        "k1_max_abs_difference": k1_difference,
        "k2_max_abs_difference": k2_difference,
    }


def effective_settings(
    plan: dict,
    selected_scenarios: set[str] | None = None,
    selected_parameters: set[str] | None = None,
) -> list[dict]:
    defaults = default_values(plan)
    unique: dict[tuple, dict] = {}
    for sweep in plan["sensitivity_protocol"]["one_factor_design"]:
        parameter = sweep["parameter"]
        if selected_scenarios is not None and sweep["scenario"] not in selected_scenarios:
            continue
        if selected_parameters is not None and parameter not in selected_parameters:
            continue
        if parameter not in CLI_NAMES:
            raise ValueError(f"Unsupported sensitivity parameter: {parameter}")
        for value in sweep["values"]:
            values = {**defaults, parameter: value}
            key = (sweep["scenario"], *[values[name] for name in CLI_NAMES])
            label = f"{parameter}_{token(value)}"
            if key not in unique:
                unique[key] = {
                    "scenario": sweep["scenario"],
                    "values": values,
                    "label": label,
                    "aliases": [],
                    "sweep_parameters": [],
                    "reviewer_roles": [],
                    "is_default": values == defaults,
                }
            unique[key]["aliases"].append(label)
            unique[key]["sweep_parameters"].append(parameter)
            unique[key]["reviewer_roles"].append(sweep["reviewer_role"])
    return list(unique.values())


def changed_factor(setting: dict, defaults: dict) -> tuple[str, object, object]:
    changed = [name for name in CLI_NAMES if setting["values"][name] != defaults[name]]
    if not changed:
        return "default", "", ""
    if len(changed) != 1:
        raise ValueError(f"Setting is not one-factor-at-a-time: {setting}")
    parameter = changed[0]
    return parameter, setting["values"][parameter], defaults[parameter]


def write_design_table(path: Path, settings: list[dict], plan: dict) -> None:
    defaults = default_values(plan)
    rows = []
    for index, setting in enumerate(settings, start=1):
        parameter, value, default = changed_factor(setting, defaults)
        row = {
            "Setting_Index": index,
            "Scenario": setting["scenario"],
            "Run_Label": f"sensitivity_{setting['scenario']}_{setting['label']}",
            "Primary_Label": setting["label"],
            "Aliases": " | ".join(setting["aliases"]),
            "Is_Default": int(setting["is_default"]),
            "Varied_Parameter": parameter,
            "Setting_Value": value,
            "Default_Value": default,
            "Reviewer_Role": " | ".join(setting["reviewer_roles"]),
            "Train_Seeds": " ".join(str(seed) for seed in EXPECTED_FORMAL_SEEDS),
            "Execution_Mode": (
                "reuse_main_checkpoint_frozen_validation"
                if setting["is_default"]
                else "new_training_and_frozen_validation"
            ),
            "New_Training_Required": int(not setting["is_default"]),
        }
        row.update({f"Effective_{name}": setting["values"][name] for name in CLI_NAMES})
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_job(job: dict, log_root: Path, environment: dict) -> tuple[int, Path]:
    completed = subprocess.run(
        job["command"], cwd=PROJECT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment,
    )
    log_path = log_root / f"{job['scenario']}__{job['label']}__seed_{job['seed']}.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    return completed.returncode, log_path


def write_default_checkpoint_provenance(
    path: Path,
    references: dict[int, dict],
    validation_seeds: list[int],
    eval_episodes: int,
) -> None:
    """Write one root-level audit record for all reused default checkpoints."""
    payload = {
        "scenario": "consensus_wide",
        "execution_mode": "reuse_main_checkpoint_frozen_validation",
        "optimizer_updates": 0,
        "training_seeds": EXPECTED_FORMAL_SEEDS,
        "validation_eval_seeds": validation_seeds,
        "evaluation_episodes_per_seed": eval_episodes,
        "source_checkpoints": {
            str(seed): {
                "source_run_dir": str(reference["run_dir"]),
                "checkpoint_sha256": reference["checkpoint_sha256"],
                "training_protocol_sha256": reference[
                    "training_protocol_sha256"
                ],
            }
            for seed, reference in sorted(references.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_complete_episode_ledger(
    results_root: Path,
    expected_jobs: int,
    validation_seeds: list[int],
    eval_episodes: int,
) -> dict:
    """Require every analysis job to contribute its complete validation ledger."""
    metric_files = sorted(results_root.rglob("test_metrics.csv"))
    if len(metric_files) != expected_jobs:
        raise RuntimeError(
            "Sensitivity output must contain one test_metrics.csv per analysis "
            f"job: expected {expected_jobs}, found {len(metric_files)}."
        )
    expected_rows = len(validation_seeds) * eval_episodes
    total_rows = 0
    for path in metric_files:
        rows = read_csv(path)
        if len(rows) != expected_rows:
            raise RuntimeError(
                f"Incomplete sensitivity validation ledger at {path}: "
                f"expected {expected_rows}, found {len(rows)}."
            )
        observed_seeds = sorted({int(row["Eval_Seed"]) for row in rows})
        if observed_seeds != validation_seeds:
            raise RuntimeError(
                f"Validation-seed mismatch at {path}: {observed_seeds}."
            )
        total_rows += len(rows)
    return {
        "test_metric_file_count": len(metric_files),
        "validation_episodes_per_job": expected_rows,
        "total_validation_episode_rows": total_rows,
        "status": "PASS",
    }


def build_seed_and_summary_tables(args: argparse.Namespace, validation_seeds: list[int]) -> str:
    command = [
        sys.executable, str(PROJECT / "summarize_test_results.py"),
        str(args.results_root),
        "--expected-train-seeds", str(len(EXPECTED_FORMAL_SEEDS)),
        "--expected-eval-seeds", *[str(value) for value in validation_seeds],
        "--episodes-per-eval-seed", str(args.eval_episodes),
    ]
    completed = subprocess.run(
        command, cwd=PROJECT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        raise RuntimeError("Sensitivity summary failed:\n" + completed.stdout)
    return completed.stdout.strip()


def write_paired_sensitivity_table(
    output: Path, seed_summary: Path, settings: list[dict], plan: dict,
) -> None:
    with seed_summary.open(newline="", encoding="utf-8") as stream:
        seed_rows = list(csv.DictReader(stream))
    values_by_key = {
        (row["Method"], row["Scenario"], int(row["Train_Seed"]), row["Metric"]):
        float(row["Test_Mean"])
        for row in seed_rows
    }
    defaults = default_values(plan)
    settings_by_scenario: dict[str, list[dict]] = defaultdict(list)
    for setting in settings:
        settings_by_scenario[setting["scenario"]].append(setting)

    paired_rows = []

    def stored_method_label(short_scenario: str, setting: dict) -> str:
        """Accept both the canonical folder-era label and immutable legacy CSV label."""
        candidates = (
            f"sensitivity_{short_scenario}_{setting['label']}",
            f"sensitivity__{short_scenario}__{setting['label']}",
        )
        for candidate in candidates:
            if any(row["Method"] == candidate for row in seed_rows):
                return candidate
        return candidates[0]

    for short_scenario, scenario_settings in settings_by_scenario.items():
        default_settings = [setting for setting in scenario_settings if setting["is_default"]]
        if len(default_settings) != 1:
            raise RuntimeError(
                f"Expected one deduplicated default in {short_scenario}; found {len(default_settings)}."
            )
        default_setting = default_settings[0]
        default_method = stored_method_label(short_scenario, default_setting)
        formal_scenario = f"lbf_{short_scenario}-v0"
        metrics = sorted({
            row["Metric"] for row in seed_rows
            if row["Method"] == default_method and row["Scenario"] == formal_scenario
        })
        if not metrics:
            raise RuntimeError(f"No default sensitivity rows found for {short_scenario}.")
        for setting in scenario_settings:
            if setting["is_default"]:
                continue
            parameter, value, default_value = changed_factor(setting, defaults)
            variant_method = stored_method_label(short_scenario, setting)
            for metric in metrics:
                paired = []
                for seed in EXPECTED_FORMAL_SEEDS:
                    default_key = (default_method, formal_scenario, seed, metric)
                    variant_key = (variant_method, formal_scenario, seed, metric)
                    if default_key not in values_by_key or variant_key not in values_by_key:
                        raise RuntimeError(
                            "Missing paired sensitivity result: "
                            f"{short_scenario}/{parameter}={value}/seed={seed}/{metric}"
                        )
                    paired.append(values_by_key[default_key] - values_by_key[variant_key])
                paired_rows.append({
                    "Scenario": formal_scenario,
                    "Parameter": parameter,
                    "Default_Value": default_value,
                    "Setting_Value": value,
                    "Metric": metric,
                    "Default_Method_Label": default_method,
                    "Setting_Method_Label": variant_method,
                    "Mean_Default_Minus_Setting": f"{mean(paired):.6f}",
                    "Sample_SD_of_Paired_Differences": f"{sample_sd(paired):.6f}",
                    "CI95_Half_Width": f"{ci95_half_width(paired):.6f}",
                    "Exact_P_Two_Sided": paired_sign_flip(paired),
                    "N_Paired_Train_Seeds": len(paired),
                    "Train_Seeds": " ".join(str(seed) for seed in EXPECTED_FORMAL_SEEDS),
                    "Multiplicity_Family": "all settings within the same scenario and metric",
                })
    if not paired_rows:
        raise RuntimeError("No non-default sensitivity comparison was generated.")

    families: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(paired_rows):
        families[(row["Scenario"], row["Metric"])].append(index)
    for indices in families.values():
        adjusted = holm([paired_rows[index]["Exact_P_Two_Sided"] for index in indices])
        for index, adjusted_p in zip(indices, adjusted):
            paired_rows[index]["Holm_Adjusted_P"] = adjusted_p
            paired_rows[index]["Significant_At_0_05"] = adjusted_p < 0.05

    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)


def main() -> None:
    args = parse_args()
    if args.resume and args.dry_run:
        raise ValueError("--resume and --dry-run cannot be used together.")
    args.results_root = (
        resolve_resume_root(args.results_root)
        if args.resume else args.results_root.resolve()
    )
    if not 1 <= args.workers <= 5:
        raise ValueError("--workers must be between 1 and 5.")
    if args.gpu >= 0 and args.workers > 2:
        raise ValueError(
            "One GPU may run at most two processes. Use --gpu -1 --workers 5 "
            "for the requested five-seed parallel CPU protocol."
        )
    if args.cpu_threads_per_job < 1:
        raise ValueError("--cpu-threads-per-job must be positive.")
    if args.gpu >= 0 and args.workers == 2 and not 0 < args.gpu_memory_fraction <= 0.5:
        raise ValueError("Two GPU workers require memory fraction in (0, 0.5].")
    if args.epochs < 1 or args.episodes_per_epoch < 1 or args.eval_episodes < 1:
        raise ValueError("Epoch and episode counts must be positive.")

    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    validate_reviewer_protocol(plan, registry)
    k1_validation = validate_k1_degeneracy_identity()
    registered_scenarios = {
        sweep["scenario"] for sweep in plan["sensitivity_protocol"]["one_factor_design"]
    }
    registered_parameters = {
        sweep["parameter"] for sweep in plan["sensitivity_protocol"]["one_factor_design"]
    }
    selected_scenarios = set(args.scenarios) if args.scenarios else None
    selected_parameters = set(args.parameters) if args.parameters else None
    unknown_scenarios = sorted((selected_scenarios or set()) - registered_scenarios)
    unknown_parameters = sorted((selected_parameters or set()) - registered_parameters)
    if unknown_scenarios or unknown_parameters:
        raise ValueError(
            "Unknown Reviewer-registered sensitivity selection: "
            f"scenarios={unknown_scenarios}, parameters={unknown_parameters}"
        )
    settings = effective_settings(plan, selected_scenarios, selected_parameters)
    if not settings:
        raise ValueError("The selected scenarios and parameters contain no sensitivity setting.")
    full_protocol = selected_scenarios is None and selected_parameters is None
    if full_protocol and len(settings) != EXPECTED_UNIQUE_SETTINGS:
        raise ValueError(
            f"Full Reviewer protocol must contain {EXPECTED_UNIQUE_SETTINGS} unique "
            f"settings after default deduplication; found {len(settings)}."
        )

    validation_seeds = plan["sensitivity_protocol"]["validation_eval_seeds"]
    needs_default_reference = any(setting["is_default"] for setting in settings)
    args.default_results_search_root = args.default_results_search_root.resolve()
    if args.default_results_roots:
        args.default_results_roots = [
            path.resolve() for path in args.default_results_roots
        ]
        reference_root_selection = "explicit"
    elif args.auto_discover_default_results:
        args.default_results_roots = discover_default_results_roots(
            args.default_results_search_root
        )
        reference_root_selection = "auto_discovered"
    else:
        args.default_results_roots = []
        reference_root_selection = "unresolved_dry_run"
    if needs_default_reference and not args.default_results_roots and not args.dry_run:
        raise ValueError(
            "The default setting must reuse the existing five formal checkpoints. "
            "Provide --default-results-roots or --auto-discover-default-results."
        )
    default_references = (
        resolve_default_reference_runs(args.default_results_roots, plan, args)
        if args.default_results_roots
        else {}
    )
    if default_references:
        print(
            "DEFAULT REFERENCE PREFLIGHT PASS: 5 consensus_wide checkpoints; "
            "new default training jobs=0",
            flush=True,
        )

    jobs = []
    for setting in settings:
        run_label = f"sensitivity_{setting['scenario']}_{setting['label']}"
        for seed in registry["formal_seeds"]:
            if setting["is_default"]:
                reference = default_references.get(seed, {
                    "run_dir": Path(f"<required-default-seed-{seed}>"),
                    "checkpoint_sha256": "<required>",
                    "training_protocol_sha256": "<required>",
                })
                command = [
                    sys.executable,
                    str(PROJECT / "evaluate_cahc_checkpoint.py"),
                    "--source-run-dir", str(reference["run_dir"]),
                    "--expected-checkpoint-sha256",
                    reference["checkpoint_sha256"],
                    "--expected-training-protocol-sha256",
                    reference["training_protocol_sha256"],
                    "--scenario", setting["scenario"],
                    "--seed", str(seed),
                    "--run-label", run_label,
                    "--expected-epochs", str(args.epochs),
                    "--expected-episodes-per-epoch", str(args.episodes_per_epoch),
                    "--eval-seeds", *[str(value) for value in validation_seeds],
                    "--eval-episodes", str(args.eval_episodes),
                    "--gpu", str(args.gpu),
                    "--gpu-memory-fraction", str(
                        args.gpu_memory_fraction
                        if args.gpu >= 0 and args.workers == 2
                        else 0.0
                    ),
                    "--results-root", str(args.results_root.resolve()),
                ]
                job_type = "reused_default_checkpoint_frozen_validation"
                source_run_dir = str(reference["run_dir"])
                source_checkpoint_sha256 = reference["checkpoint_sha256"]
            else:
                command = [
                    sys.executable, str(PROJECT / "formal_cahc.py"),
                    "--scenario", setting["scenario"], "--seed", str(seed),
                    "--epochs", str(args.epochs),
                    "--episodes-per-epoch", str(args.episodes_per_epoch),
                    "--eval-seeds", *[str(value) for value in validation_seeds],
                    "--eval-episodes", str(args.eval_episodes),
                    "--gpu", str(args.gpu),
                    "--gpu-memory-fraction", str(
                        args.gpu_memory_fraction
                        if args.gpu >= 0 and args.workers == 2
                        else 0.0
                    ),
                    "--run-label", run_label,
                    "--results-root", str(args.results_root.resolve()),
                ]
                for name, cli_name in CLI_NAMES.items():
                    command.extend([cli_name, str(setting["values"][name])])
                job_type = "new_training_and_frozen_validation"
                source_run_dir = None
                source_checkpoint_sha256 = None
            jobs.append({
                "scenario": setting["scenario"], "label": setting["label"],
                "aliases": setting["aliases"], "reviewer_roles": setting["reviewer_roles"],
                "is_default": setting["is_default"],
                "effective_values": setting["values"], "seed": seed,
                "job_type": job_type,
                "source_run_dir": source_run_dir,
                "source_checkpoint_sha256": source_checkpoint_sha256,
                "command": command, "status": "PENDING",
            })
    training_job_count = sum(
        job["job_type"] == "new_training_and_frozen_validation" for job in jobs
    )
    reused_reference_evaluation_count = sum(
        job["job_type"] == "reused_default_checkpoint_frozen_validation"
        for job in jobs
    )
    if full_protocol and (
        len(jobs) != EXPECTED_FULL_JOBS
        or training_job_count != EXPECTED_NEW_TRAINING_JOBS
        or reused_reference_evaluation_count
        != EXPECTED_REUSED_REFERENCE_EVALUATIONS
    ):
        raise ValueError(
            "Full Reviewer protocol must contain 25 analysis jobs: 20 new "
            "training jobs plus 5 reused-checkpoint frozen evaluations; found "
            f"total={len(jobs)}, training={training_job_count}, "
            f"reuse_eval={reused_reference_evaluation_count}."
        )

    args.results_root = args.results_root.resolve()
    args.results_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.results_root / "cahc_sensitivity_manifest.json"
    previous_manifest = None
    if manifest_path.exists() and not args.dry_run:
        if not args.resume:
            raise FileExistsError(
                f"A sensitivity manifest already exists at {manifest_path}. "
                "Use --resume to validate and continue the interrupted grid, "
                "or use a new --results-root for a separate experiment."
            )
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif args.resume:
        raise FileNotFoundError(
            f"--resume requires an existing sensitivity manifest: {manifest_path}"
        )
    log_root = args.results_root / "launcher_logs"
    log_root.mkdir(exist_ok=True)
    design_path = args.results_root / "sensitivity_design.csv"
    write_design_table(design_path, settings, plan)
    provenance_path = args.results_root / "source_checkpoint_provenance.json"
    if default_references:
        write_default_checkpoint_provenance(
            provenance_path,
            default_references,
            validation_seeds,
            args.eval_episodes,
        )
    source_hashes = {
        "cahc_experiment_plan.json": sha256(PLAN_PATH),
        "scenario_registry.json": sha256(REGISTRY_PATH),
        "run_cahc_sensitivity.py": sha256(Path(__file__)),
        "formal_cahc.py": sha256(PROJECT / "formal_cahc.py"),
        "evaluate_cahc_checkpoint.py": sha256(
            PROJECT / "evaluate_cahc_checkpoint.py"
        ),
    }
    manifest = {
        "status": "DRY_RUN" if args.dry_run else "RUNNING",
        "protocol_name": "CAIE_D_26_04732_Reviewer_minimal_CAHC_wave_hyperparameters",
        "reviewer_complete_grid": full_protocol,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": plan["sensitivity_protocol"]["selection_rule"],
        "k1_degeneracy_rule": plan["sensitivity_protocol"]["k1_degeneracy_rule"],
        "k1_degeneracy_implementation_check": k1_validation,
        "selected_scenarios": sorted(selected_scenarios or registered_scenarios),
        "selected_parameters": sorted(selected_parameters or registered_parameters),
        "source_sha256": source_hashes,
        "training_seeds": registry["formal_seeds"],
        "validation_eval_seeds": validation_seeds,
        "final_test_eval_seeds_reserved": plan["sensitivity_protocol"]["final_test_eval_seeds"],
        "schedule": (
            "scenario -> setting -> five seeds in parallel; strict setting "
            "barrier; default performs frozen checkpoint reuse evaluation"
        ),
        "gpu_worker_limit": 2, "workers": args.workers,
        "epochs": args.epochs, "episodes_per_epoch": args.episodes_per_epoch,
        "eval_episodes_per_seed": args.eval_episodes, "gpu": args.gpu,
        "cpu_threads_per_job": args.cpu_threads_per_job,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "unique_setting_count": len(settings), "job_count": len(jobs),
        "new_training_job_count": training_job_count,
        "reused_default_evaluation_count": reused_reference_evaluation_count,
        "default_results_roots": [
            str(path) for path in args.default_results_roots
        ],
        "default_reference_root_selection": reference_root_selection,
        "default_results_search_root": str(args.default_results_search_root),
        "default_reference_runs": {
            str(seed): {
                key: str(value) if isinstance(value, Path) else value
                for key, value in reference.items()
            }
            for seed, reference in default_references.items()
        },
        "source_checkpoint_provenance": (
            str(provenance_path) if default_references else None
        ),
        "expected_validation_episodes_per_job": (
            len(validation_seeds) * args.eval_episodes
        ),
        "expected_total_validation_episode_rows": (
            len(jobs) * len(validation_seeds) * args.eval_episodes
        ),
        "design_table": str(design_path), "jobs": jobs,
    }
    if previous_manifest is not None:
        validate_resume_manifest(previous_manifest, args, jobs, source_hashes)
        previous_by_key = {
            (job["scenario"], job["label"], int(job["seed"])): job
            for job in previous_manifest["jobs"]
        }
        skipped = 0
        for job in jobs:
            key = (job["scenario"], job["label"], int(job["seed"]))
            old = previous_by_key[key]
            ledger = completed_job_ledger(
                args.results_root, job, validation_seeds, args.eval_episodes
            )
            if old.get("status") == "PASS" and ledger is None:
                raise RuntimeError(
                    "A previously PASS job has no complete validation ledger: "
                    f"{key}"
                )
            if ledger is not None:
                job.update(
                    status="PASS",
                    returncode=old.get("returncode", 0),
                    log=old.get("log"),
                    resume_validated_ledger=str(ledger),
                )
                skipped += 1
        manifest.update({
            "started_utc": previous_manifest.get("started_utc"),
            "resumed_utc": datetime.now(timezone.utc).isoformat(),
            "resume_from_status": previous_manifest.get("status"),
            "resume_skipped_complete_jobs": skipped,
            "resume_jobs_to_execute": len(jobs) - skipped,
        })
        print(
            f"RESUME PREFLIGHT PASS: skipped {skipped} complete jobs; "
            f"remaining jobs={len(jobs) - skipped}",
            flush=True,
        )
    if args.dry_run:
        for job in jobs:
            job["status"] = "DRY_RUN"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            f"DRY RUN PASS: {len(settings)} unique settings, "
            f"{training_job_count} new training jobs + "
            f"{reused_reference_evaluation_count} reused-checkpoint evaluations "
            f"= {len(jobs)} analysis jobs; manifest={manifest_path}"
        )
        return

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    environment = {
        **os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": str(args.cpu_threads_per_job),
        "MKL_NUM_THREADS": str(args.cpu_threads_per_job),
        "OPENBLAS_NUM_THREADS": str(args.cpu_threads_per_job),
        "NUMEXPR_NUM_THREADS": str(args.cpu_threads_per_job),
    }
    completed_settings = 0
    for scenario in dict.fromkeys(setting["scenario"] for setting in settings):
        labels = [setting["label"] for setting in settings if setting["scenario"] == scenario]
        for label in labels:
            batch = [
                job for job in jobs if job["scenario"] == scenario and job["label"] == label
            ]
            if len(batch) != len(EXPECTED_FORMAL_SEEDS):
                raise RuntimeError(
                    f"{scenario}/{label} does not contain exactly five seed jobs."
                )
            modes = {job["job_type"] for job in batch}
            if len(modes) != 1:
                raise RuntimeError(f"Mixed execution modes in {scenario}/{label}: {modes}")
            mode = next(iter(modes))
            pending_batch = [job for job in batch if job["status"] != "PASS"]
            if not pending_batch:
                completed_settings += 1
                print(
                    f"SKIP  [{completed_settings}/{len(settings)}] {scenario}/{label}: "
                    "all five seed ledgers already complete",
                    flush=True,
                )
                continue
            print(
                f"START [{completed_settings + 1}/{len(settings)}] {scenario}/{label}: "
                f"mode={mode}; pending seeds {[job['seed'] for job in pending_batch]}",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(run_job, job, log_root, environment): job
                    for job in pending_batch
                }
                failed = []
                for future in as_completed(futures):
                    job = futures[future]
                    code, log_path = future.result()
                    job.update(
                        status="PASS" if code == 0 else "FAIL",
                        returncode=code, log=str(log_path),
                    )
                    if code:
                        failed.append(job)
                    else:
                        log_path.unlink(missing_ok=True)
                        job["log"] = None
                    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            if failed:
                manifest["status"] = "FAIL"
                manifest["failed"] = [
                    {"scenario": job["scenario"], "label": job["label"], "seed": job["seed"]}
                    for job in failed
                ]
                manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                raise SystemExit(
                    f"Sensitivity failed in {scenario}/{label}: {len(failed)} seed jobs"
                )
            completed_settings += 1
            print(f"PASS  [{completed_settings}/{len(settings)}] {scenario}/{label}", flush=True)

    manifest["status"] = "PASS"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        manifest["validation_episode_ledger"] = validate_complete_episode_ledger(
            args.results_root,
            len(jobs),
            validation_seeds,
            args.eval_episodes,
        )
        summary_message = build_seed_and_summary_tables(args, validation_seeds)
        write_paired_sensitivity_table(
            args.results_root / "sensitivity_paired_comparisons.csv",
            args.results_root / "test_seed_summary.csv", settings, plan,
        )
    except Exception as error:
        manifest["status"] = "FAIL_SUMMARY"
        manifest["summary_error"] = f"{type(error).__name__}: {error}"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise
    manifest["analysis_outputs"] = [
        "sensitivity_design.csv", "test_seed_summary.csv", "test_summary.csv",
        "sensitivity_paired_comparisons.csv",
        "source_checkpoint_provenance.json",
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if log_root.is_dir() and not any(log_root.iterdir()):
        log_root.rmdir()
    print(
        f"PASS: {training_job_count} new training jobs + "
        f"{reused_reference_evaluation_count} reused-checkpoint evaluations; "
        f"{len(jobs)} complete Reviewer analysis jobs; manifest={manifest_path}"
    )
    print(summary_message)


if __name__ == "__main__":
    main()
