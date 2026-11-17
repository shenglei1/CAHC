from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


TEST_SEEDS = list(range(31001, 31011))
PRIMARY = "Probe2_Balanced_Safe_Coordination"
VERSION = "V1"
WIDTH = 5
SUPPORT_LINKS_PER_STATION = 8
REQUIRED_HOPS = 4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path, method: str, development: bool = False):
    manifests = list(root.glob("*_launcher_manifest.json"))
    require(len(manifests) == 1, f"{root}: exactly one launcher manifest required")
    manifest = read_json(manifests[0])
    require(manifest["status"] == "PASS", f"{root}: launcher not PASS")
    seeds = [int(seed) for seed in manifest["seeds"]]
    require(len(seeds) == 5 and len(set(seeds)) == 5,
            f"{root}: exactly five unique training seeds required")
    jobs = manifest["jobs"]
    require(len(jobs) == 5 and {job["seed"] for job in jobs} == set(seeds),
            f"{root}: incomplete seed jobs")
    matching_jobs = [job for job in jobs if job.get("method") == method]
    if matching_jobs:
        require(all(job["status"] == "PASS" and
                    job["scenario"] == "probe_interaction" for job in matching_jobs),
                f"{root}: mixed or failed jobs")
    else:
        require(method == "no_wave", f"{root}: manifest does not describe {method}")
    eval_seeds = manifest["eval_seeds"]
    episodes = int(manifest["eval_episodes_per_seed"])
    if not development:
        require(manifest["epochs"] == 700 and manifest["episodes_per_epoch"] == 64,
                "Formal budget must be 700 x 64")
        require(eval_seeds == TEST_SEEDS and episodes == 20,
                "Formal held-out evaluation protocol mismatch")
    expected_cases = {(seed, episode) for seed in eval_seeds
                      for episode in range(1, episodes + 1)}
    files = list(root.glob(
        f"{method}/lbf_probe_interaction-v0/seed_*/run*/test_metrics.csv"
    ))
    require(len(files) == 5,
            "Exactly five run directories required; retries cannot be selected away")

    records, inputs, common_sources = {}, {}, {}
    for path in files:
        rows = read_rows(path)
        require(bool(rows), f"Empty test CSV: {path}")
        train_seed = int(rows[0]["Train_Seed"])
        require(train_seed in seeds and train_seed not in records,
                "Unexpected or duplicate training seed")
        folder = path.parent
        training = read_json(folder / "training_protocol.json")
        evaluation = read_json(folder / "evaluation_protocol.json")
        config = read_json(folder / "config.json")
        require(training["method"] == method and training["seed"] == train_seed,
                "Training identity mismatch")
        require(training["probe_version"] == evaluation["probe_version"] == VERSION,
                "Old or mixed Probe2 version")
        require(training["probe2_training_support"] == "five_node_path" and
                training["probe2_evaluation_support"] == "five_node_path",
                "V1 requires the same path in training and evaluation")
        require(training.get("probe2_force_physical_links") is True,
                "V1 requires all legal physical links")
        require(training["teacher_only_training"] is True and
                math.isclose(float(training["teacher_loss_coefficient"]), 1.0),
                "Shared teacher objective mismatch")
        require(config["num_actions"] == [2] and config["gamma"] == 0.0,
                "Action or discount mismatch")
        require(math.isclose(float(config["lrate"]), 0.001) and
                math.isclose(float(config["lr_decay"]), 0.995) and
                math.isclose(float(config["max_grad_norm"]), 5.0),
                "Shared optimizer-stability settings mismatch")
        require(math.isclose(float(training["learning_rate_decay"]), 0.995) and
                math.isclose(float(training["gradient_clip_norm"]), 5.0),
                "Training protocol omitted optimizer-stability settings")
        require(all(training[key] for key in
                    ("backward_verified", "optimizer_step_verified", "parameters_finite")),
                "Optimizer verification failed")
        if not development:
            require(not training.get("development_probe", False),
                    "Development output cannot support the formal claim")
        require(evaluation["optimizer_updates"] == 0 and
                evaluation["parameters_unchanged"] and
                evaluation["checkpoint_load_verified"] and
                evaluation["action_mode"] == "greedy",
                "Frozen greedy evaluation was not verified")
        require(evaluation["eval_seeds"] == eval_seeds and
                int(evaluation["episodes_per_seed"]) == episodes,
                "Evaluation metadata mismatch")
        checkpoint = digest(folder / "model.pt")
        require(checkpoint == evaluation["checkpoint_sha256"],
                "Checkpoint checksum mismatch")

        n_agents = int(config["nagents"])
        require(n_agents % WIDTH == 0, "Agent count is not a multiple of five")
        stations = n_agents // WIDTH
        station_decisions = stations * int(config["max_steps"])
        support_links = stations * SUPPORT_LINKS_PER_STATION
        seen, fingerprints = set(), {}
        for row in rows:
            case = (int(row["Eval_Seed"]), int(row["Episode"]))
            require(case not in seen, "Duplicate evaluation episode")
            seen.add(case)
            require(row["Method"] == method and int(row["Train_Seed"]) == train_seed and
                    row["Scenario"] == "lbf_probe_interaction-v0",
                    "Mixed CSV identity")
            require(row.get("Evaluation_Phase", "frozen_test") == "frozen_test" and
                    row.get("Action_Mode", "greedy") == "greedy", "Wrong action mode")
            opened = float(row["Probe2_Service_Opportunities"])
            blocked = float(row["Probe2_Blocked_Opportunities"])
            completed = float(row["Workload_Completed"])
            holds = float(row["Probe2_Correct_Safety_Holds"])
            unsafe = float(row["Probe2_Premature_Load_Attempts"])
            score = float(row[PRIMARY])
            require(all(math.isfinite(value) for value in
                        (opened, blocked, completed, holds, unsafe, score)),
                    "Non-finite outcome")
            require(opened + blocked == station_decisions and
                    opened == float(row["Workload_Total"]), "Wrong denominators")
            require(0 <= completed <= opened and 0 <= holds <= blocked and
                    holds + unsafe == blocked, "Inconsistent safety ledger")
            require(math.isclose(score,
                                 0.5 * (completed / opened + holds / blocked),
                                 abs_tol=1e-10), "Balanced score mismatch")
            require(float(row["Probe2_Forced_Physical_Links"]) == 1 and
                    float(row["Probe2_Learned_Router_Enabled"]) == 0,
                    "Router control mismatch")
            require(float(row["Probe2_Required_Sensor_To_Actuator_Hops"]) == REQUIRED_HOPS and
                    float(row["Probe2_Communication_Scaffold_Directed_Links"]) == support_links,
                    "Path audit mismatch")
            for hop in range(WIDTH):
                correct = float(row[f"Probe2_Hop_{hop}_Correct"])
                decisions = float(row[f"Probe2_Hop_{hop}_Decisions"])
                accuracy = float(row[f"Probe2_Hop_{hop}_Accuracy"])
                require(decisions == station_decisions and 0 <= correct <= decisions and
                        math.isclose(accuracy, correct / decisions, abs_tol=1e-10),
                        f"Hop-{hop} ledger mismatch")
            fingerprints[case] = (opened, blocked)
        require(seen == expected_cases, "Missing or extra evaluation episodes")

        costs = read_rows(folder / "communication_costs.csv")
        require(len(costs) == len(rows) and
                {(int(row["Eval_Seed"]), int(row["Episode"])) for row in costs}
                == expected_cases, "Communication costs do not cover test cases")
        opened = sum(float(row["Probe2_Service_Opportunities"]) for row in rows)
        blocked = sum(float(row["Probe2_Blocked_Opportunities"]) for row in rows)
        completed = sum(float(row["Workload_Completed"]) for row in rows)
        holds = sum(float(row["Probe2_Correct_Safety_Holds"]) for row in rows)
        record = {
            PRIMARY: 0.5 * (completed / opened + holds / blocked),
            "Eligible_Service_Completion_Ratio": completed / opened,
            "Safe_Hold_Ratio": holds / blocked,
            "False_Activation_Rate": 1.0 - holds / blocked,
            "Mean_Active_Link_Ratio": statistics.mean(
                float(row["Mean_Active_Link_Ratio"]) for row in rows),
            "Logical_Feature_Bytes_FP32_Per_Step": statistics.mean(
                float(row["Logical_Feature_Bytes_FP32_Per_Step"]) for row in costs),
        }
        for hop in range(WIDTH):
            correct = sum(float(row[f"Probe2_Hop_{hop}_Correct"]) for row in rows)
            decisions = sum(float(row[f"Probe2_Hop_{hop}_Decisions"]) for row in rows)
            record[f"Hop_{hop}_Accuracy"] = correct / decisions
        records[train_seed] = record
        inputs[train_seed] = fingerprints
        common_sources[train_seed] = {
            key: training["source_sha256"][key] for key in (
                "probe2_interlock.py", "scenario_registry.json", "scenario_registry.py",
                "shared_lbf_env.py", "operational_metrics.py", "env_wrappers.py",
            )
        }
        require(training["scenario_registry_sha256"] ==
                evaluation["scenario_registry_sha256"], "Registry mismatch within run")
    require(set(records) == set(seeds), "Missing training seeds")
    require(len({json.dumps(source, sort_keys=True)
                 for source in common_sources.values()}) == 1,
            "Shared sources changed between seeds")
    return manifest, records, inputs, common_sources, seeds


def mean_sd_ci(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = 2.7764451052 * sd / math.sqrt(5)
    return mean, sd, mean - half, mean + half


def exact_sign_flip_p(differences):
    target = abs(statistics.mean(differences))
    extreme = sum(
        abs(statistics.mean(sign * value for sign, value in zip(signs, differences)))
        >= target - 1e-12
        for signs in itertools.product((-1, 1), repeat=5)
    )
    return extreme / 32.0


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Audit paired Probe2 V1 runs and evaluate the pre-registered claim gate.")
    parser.add_argument("--hypercomm-root", type=Path, required=True)
    parser.add_argument("--hyperwavex-root", type=Path, required=True)
    parser.add_argument("--no-wave-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Relative or absolute output directory. Defaults to "
            "<hyperwavex-root>/summaries/probe_interaction/comparison."
        ),
    )
    parser.add_argument("--allow-development", action="store_true")
    args = parser.parse_args()
    roots = {
        "hypercomm": args.hypercomm_root.resolve(),
        "hyperwave_x": args.hyperwavex_root.resolve(),
        "no_wave": args.no_wave_root.resolve(),
    }
    audits = {
        method: audit(root, method, args.allow_development)
        for method, root in roots.items()
    }
    reference = audits["hyperwave_x"]
    seeds = reference[4]
    for method, current in audits.items():
        for field in ("epochs", "episodes_per_epoch", "eval_seeds",
                      "eval_episodes_per_seed"):
            require(current[0][field] == reference[0][field],
                    f"{method}: unequal {field}")
        require(current[2] == reference[2], f"{method}: evaluation case IDs or opportunity counts differ")
        require(current[3] == reference[3], f"{method}: shared source hashes differ")
        require(current[4] == seeds, f"{method}: training seeds are not paired")

    summary, paired, comparisons = [], [], {}
    for method, (_, records, _, _, _) in audits.items():
        for metric in records[seeds[0]]:
            values = [records[seed][metric] for seed in seeds]
            mean, sd, lower, upper = mean_sd_ci(values)
            summary.append(dict(Method=method, Metric=metric, Mean=mean,
                                Sample_SD=sd, CI95_Lower=lower,
                                CI95_Upper=upper, N_Train_Seeds=5))

    full = reference[1]
    full_service = statistics.mean(
        full[seed]["Eligible_Service_Completion_Ratio"] for seed in seeds)
    full_hold = statistics.mean(full[seed]["Safe_Hold_Ratio"] for seed in seeds)
    full_hop4 = statistics.mean(full[seed]["Hop_4_Accuracy"] for seed in seeds)
    for comparator in ("hypercomm", "no_wave"):
        differences = []
        for seed in seeds:
            left = full[seed][PRIMARY]
            right = audits[comparator][1][seed][PRIMARY]
            differences.append(left - right)
            paired.append(dict(Comparator=comparator, Train_Seed=seed,
                               HyperWaveX=left, Comparator_Score=right,
                               Difference=left - right))
        mean, sd, lower, upper = mean_sd_ci(differences)
        wins = sum(value > 0 for value in differences)
        comparisons[comparator] = dict(
            mean_difference=mean,
            sample_sd=sd,
            descriptive_ci95=[lower, upper],
            exact_two_sided_sign_flip_p=exact_sign_flip_p(differences),
            positive_seed_differences=wins,
            hyperwavex_service_completion=full_service,
            hyperwavex_safe_hold=full_hold,
            hyperwavex_hop4_accuracy=full_hop4,
            intro_claim_gate_passed=bool(
                mean > 0 and wins >= 4 and full_service >= 0.90
                and full_hold >= 0.90 and full_hop4 >= 0.90
            ),
        )
    overall_gate = all(
        comparisons[name]["intro_claim_gate_passed"]
        for name in ("hypercomm", "no_wave")
    )
    report = {
        "audit_status": "PASS",
        "formal_test": not args.allow_development,
        "primary_metric": PRIMARY,
        "aggregation": "pool test counts within each of five independent training seeds",
        "comparisons": comparisons,
        "overall_intro_claim_gate_passed": overall_gate,
        "interpretation": (
            "Development pipeline only; not paper evidence."
            if args.allow_development else
            "Use only the narrow four-hop propagation claim when the overall gate passes. "
            "Otherwise report the probe as negative or inconclusive."
        ),
        "latency_note": "Parallel timing is non-isolated; logical bytes are not measured traffic.",
    }
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else roots["hyperwave_x"] / "summaries" / "probe_interaction" / "comparison"
    )
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "probe2_summary.csv", summary)
    write_csv(output / "probe2_paired_seeds.csv", paired)
    (output / "probe2_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
