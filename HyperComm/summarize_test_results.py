import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from paper_statistics import paired_sign_flip, holm


T_975 = {
    1: 12.7062,
    2: 4.3027,
    3: 3.1824,
    4: 2.7764,
    5: 2.5706,
    6: 2.4469,
    7: 2.3646,
    8: 2.3060,
    9: 2.2622,
    10: 2.2281,
}

METADATA_COLUMNS = {
    "Method",
    "Scenario",
    "Train_Seed",
    "Eval_Seed",
    "Episode",
    "Evaluation_Phase",
    "Action_Mode",
    "Latency_Isolated",
    "Checkpoint_SHA256",
    "Metrics_Schema_Version",
}


def mean(values):
    return sum(values) / len(values)


def sample_sd(values):
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(
        sum((value - center) ** 2 for value in values) / (len(values) - 1)
    )


def ci95_half_width(values):
    if len(values) < 2:
        return 0.0
    degrees_of_freedom = len(values) - 1
    critical_value = T_975.get(degrees_of_freedom, 1.96)
    return critical_value * sample_sd(values) / math.sqrt(len(values))


def run_number(path):
    try:
        return int(path.parent.name.removeprefix("run"))
    except ValueError:
        return -1


def newest_test_files(root, scenarios=None, methods=None):
    grouped = defaultdict(list)
    for path in root.glob("*/*/seed_*/run*/test_metrics.csv"):
        method, scenario, seed_dir = path.parts[-5:-2]
        if scenarios and scenario not in scenarios:
            continue
        if methods and method not in methods:
            continue
        grouped[(method, scenario, seed_dir)].append(path)


    return [path for paths in grouped.values() for path in sorted(paths, key=run_number)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--additional-roots", nargs="*", type=Path, default=[])
    parser.add_argument("--expected-train-seeds", type=int, default=5)
    parser.add_argument(
        "--train-seeds",
        nargs="+",
        type=int,
        default=[20, 23, 56, 89, 2025],
        help="Exact training-seed set expected in this result shard.",
    )
    parser.add_argument("--expected-eval-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--episodes-per-eval-seed", type=int, required=True)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--paired-reference",
        help="write paired reference-minus-variant comparisons for this method label",
    )
    args = parser.parse_args()

    expected_train_seeds = set(args.train_seeds)
    if len(expected_train_seeds) != len(args.train_seeds):
        raise SystemExit("Expected training seeds must be unique")
    if len(expected_train_seeds) != args.expected_train_seeds:
        raise SystemExit(
            "--expected-train-seeds must equal the number supplied to --train-seeds"
        )
    expected_eval_seeds = sorted(set(args.expected_eval_seeds))
    if len(expected_eval_seeds) != len(args.expected_eval_seeds):
        raise SystemExit("Expected evaluation seeds must be unique")
    expected_episodes = len(expected_eval_seeds) * args.episodes_per_eval_seed

    seen_runs = set()
    seed_rows = []
    incomplete = []
    roots = [args.root, *args.additional_roots]
    expected_job_count = 0
    for root in roots:
        manifests = sorted(
            [
                *root.glob("cahc_*_manifest.json"),
                *root.glob("hypercomm_launcher_manifest.json"),
            ]
        )
        manifests = [path for path in manifests if "dryrun" not in path.name]
        if len(manifests) != 1:
            incomplete.append(
                f"{root}: expected exactly one experiment manifest; found {len(manifests)}"
            )
            continue
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS":
            incomplete.append(
                f"{manifests[0]}: status must be PASS, got {manifest.get('status')}"
            )
        jobs = manifest.get("jobs", [])
        manifest_job_count = manifest.get("job_count", len(jobs))
        expected_job_count += manifest_job_count
        if len(jobs) != manifest_job_count:
            incomplete.append(f"{manifests[0]}: job count does not match manifest")
        nonpassing = [job for job in jobs if job.get("status") != "PASS"]
        if nonpassing:
            incomplete.append(
                f"{manifests[0]}: {len(nonpassing)} jobs are not PASS"
            )
    test_files = [
        path for root in roots for path in newest_test_files(
            root, set(args.scenarios or []), set(args.methods or [])
        )
    ]
    for test_file in test_files:
        with test_file.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            incomplete.append(f"{test_file}: empty test file")
            continue
        method = rows[0]["Method"]
        scenario = rows[0]["Scenario"]
        train_seed = int(rows[0]["Train_Seed"])
        run_key = (method, scenario, train_seed)
        if run_key in seen_runs:
            incomplete.append(f'Duplicate training run: {run_key}')
        seen_runs.add(run_key)
        if train_seed not in expected_train_seeds:
            incomplete.append(f'Unexpected training seed: {train_seed}')
        episode_keys = [(int(row['Eval_Seed']), int(row['Episode'])) for row in rows]
        if len(set(episode_keys)) != len(episode_keys):
            incomplete.append(f'{test_file}: duplicate evaluation episodes')
        for es in expected_eval_seeds:
            if sum(key[0] == es for key in episode_keys) != args.episodes_per_eval_seed:
                incomplete.append(f'{test_file}: incorrect episode count for eval seed {es}')
        observed_eval_seeds = sorted({int(row["Eval_Seed"]) for row in rows})

        if len(rows) != expected_episodes:
            incomplete.append(
                f"{method}/{scenario}/train_seed_{train_seed}: "
                f"{len(rows)} of {expected_episodes} test episodes"
            )
        if observed_eval_seeds != expected_eval_seeds:
            incomplete.append(
                f"{method}/{scenario}/train_seed_{train_seed}: evaluation seed mismatch"
            )
        metric_names = [
            name for name in rows[0] if name not in METADATA_COLUMNS
        ]
        if scenario != "lbf_probe_interaction-v0":
            metric_names = [name for name in metric_names if not name.startswith("Probe2_")]
        if scenario != "lbf_probe_complementary-v0":
            metric_names = [name for name in metric_names if not name.startswith("Probe1_")]
        for metric in metric_names:
            if metric.endswith('_ms') or metric == 'Avg_Inference_Delay':
                continue
            values = [float(row[metric]) for row in rows if row[metric] != ""]
            if len(values) != len(rows) or not all(math.isfinite(v) for v in values):
                incomplete.append(
                    f"{method}/{scenario}/train_seed_{train_seed}/{metric}: missing values"
                )
                continue
            seed_rows.append(
                {
                    "Method": method,
                    "Scenario": scenario,
                    "Train_Seed": train_seed,
                    "Metric": metric,
                    "Test_Mean": f"{mean(values):.6f}",
                    "N_Test_Episodes": len(values),
                    "N_Eval_Seeds": len(observed_eval_seeds),
                    "Eval_Seeds": " ".join(str(seed) for seed in observed_eval_seeds),
                }
            )

    if not seed_rows:
        raise SystemExit(f"No frozen test_metrics.csv files found below {args.root}")
    if len(seen_runs) != expected_job_count:
        incomplete.append(
            f"{args.root}: found {len(seen_runs)} unique result groups for "
            f"{expected_job_count} manifest jobs"
        )


    preflight_groups = defaultdict(set)
    for row in seed_rows:
        preflight_groups[(row['Method'], row['Scenario'], row['Metric'])].add(row['Train_Seed'])
    for group, seeds in preflight_groups.items():
        if seeds != expected_train_seeds:
            incomplete.append(
                f'{group}: expected exactly {sorted(expected_train_seeds)}; '
                f'got {sorted(seeds)}'
            )
    if incomplete:
        raise SystemExit('Invalid frozen-test data:\n' + '\n'.join(incomplete))

    output_dir = args.output_dir or args.root
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_output = output_dir / "test_seed_summary.csv"
    with seed_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_rows[0].keys())
        writer.writeheader()
        writer.writerows(seed_rows)

    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(row["Method"], row["Scenario"], row["Metric"])].append(row)

    summary_rows = []
    for (method, scenario, metric), rows in sorted(grouped.items()):
        values = [float(row["Test_Mean"]) for row in rows]
        train_seeds = sorted({int(row["Train_Seed"]) for row in rows})
        if len(train_seeds) != args.expected_train_seeds:
            incomplete.append(
                f"{method}/{scenario}/{metric}: "
                f"{len(train_seeds)} of {args.expected_train_seeds} training seeds"
            )
        summary_rows.append(
            {
                "Method": method,
                "Scenario": scenario,
                "Metric": metric,
                "Frozen_Test_Mean": f"{mean(values):.6f}",
                "Sample_SD_Across_Train_Seeds": f"{sample_sd(values):.6f}",
                "CI95_Half_Width": f"{ci95_half_width(values):.6f}",
                "N_Train_Seeds": len(train_seeds),
                "Train_Seeds": " ".join(str(seed) for seed in train_seeds),
                "Test_Episodes_Per_Train_Seed": expected_episodes,
                "N_Eval_Seeds_Per_Train_Seed": len(expected_eval_seeds),
            }
        )

    summary_output = output_dir / "test_summary.csv"
    with summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    outputs = [seed_output, summary_output]
    if args.paired_reference:
        by_key = {
            (row["Method"], row["Scenario"], row["Train_Seed"], row["Metric"]):
            float(row["Test_Mean"])
            for row in seed_rows
        }
        methods = sorted({row["Method"] for row in seed_rows})
        scenarios = sorted({row["Scenario"] for row in seed_rows})
        metrics = sorted({row["Metric"] for row in seed_rows})
        train_seeds = sorted({int(row["Train_Seed"]) for row in seed_rows})
        paired_rows = []
        for variant in methods:
            if variant == args.paired_reference:
                continue
            for scenario in scenarios:
                for metric in metrics:
                    paired = []
                    for train_seed in train_seeds:
                        reference_key = (
                            args.paired_reference, scenario, train_seed, metric
                        )
                        variant_key = (variant, scenario, train_seed, metric)
                        if reference_key in by_key and variant_key in by_key:
                            paired.append(
                                (train_seed, by_key[reference_key] - by_key[variant_key])
                            )
                    if not paired:
                        continue
                    differences = [difference for _, difference in paired]
                    paired_rows.append(
                        {
                            "Reference": args.paired_reference,
                            "Variant": variant,
                            "Scenario": scenario,
                            "Metric": metric,
                            "Mean_Reference_Minus_Variant": f"{mean(differences):.6f}",
                            "Sample_SD_of_Differences": f"{sample_sd(differences):.6f}",
                            "CI95_Half_Width": f"{ci95_half_width(differences):.6f}",
                            "Exact_P_Two_Sided": paired_sign_flip(differences),
                            "N_Paired_Train_Seeds": len(paired),
                            "Train_Seeds": " ".join(str(seed) for seed, _ in paired),
                        }
                    )
                    if len(paired) != args.expected_train_seeds:
                        incomplete.append(
                            f"paired {args.paired_reference}-{variant}/"
                            f"{scenario}/{metric}: {len(paired)} training seeds"
                        )
        if not paired_rows:
            raise SystemExit(
                f"No frozen-test paired comparisons found for "
                f"reference {args.paired_reference!r}"
            )

        adjusted = holm([row['Exact_P_Two_Sided'] for row in paired_rows])
        for row, p in zip(paired_rows, adjusted):
            row['Holm_Adjusted_P'] = p
            row['Significant_At_0_05'] = p < 0.05
        paired_output = output_dir / "test_paired_comparisons.csv"
        with paired_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=paired_rows[0].keys())
            writer.writeheader()
            writer.writerows(paired_rows)
        outputs.append(paired_output)

    if incomplete:
        raise SystemExit("Incomplete frozen-test groups:\n" + "\n".join(incomplete))
    print("Wrote " + ", ".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
