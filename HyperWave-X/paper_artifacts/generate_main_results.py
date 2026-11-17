"""Generate publication figures and tables from the frozen six-scenario results.

All input and output paths are relative to this script.  The script never
modifies source CSV files.  It treats the five training seeds as the
independent replication units; the 200 frozen-test episodes belonging to each
trained model are averaged before across-seed uncertainty is calculated.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import t as student_t


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_DIR = SCRIPT_DIR / "tables_en"
from figure_support import prepare, save
LANGUAGE = "en"

TEAMCOMM_ROOT = REPO_ROOT / "TeamComm" / "results"
HYPERCOMM_ROOT = REPO_ROOT / "HyperComm" / "results" / "FORMAL_6SCENARIOS"
HYPERWAVEX_ROOT = (
    REPO_ROOT / "HyperWave-X" / "results" / "FORMAL_6SCENARIOS_ORGANIZED_V1"
)
SCENARIO_REGISTRY = REPO_ROOT / "HyperWave-X" / "scenario_registry.json"

METHOD_ORDER = [
    "commnet", "ic3net", "magic", "qmix", "tarmac", "teamcomm", "tiecomm",
    "hypercomm", "hyperwave_x",
]
METHOD_LABEL = {
    "commnet": "CommNet",
    "ic3net": "IC3Net",
    "magic": "MAGIC",
    "qmix": "QMIX",
    "tarmac": "TarMAC",
    "teamcomm": "TeamComm",
    "tiecomm": "TieComm",
    "hypercomm": "HyperComm",
    "hyperwave_x": "HyperWave-X",
}
METHOD_COLORS = {
    # Color-blind-conscious palette.  Markers provide a second visual channel
    # so the figure remains readable in grayscale and for color-vision
    # deficiencies.
    "commnet": "#009E73",
    "ic3net": "#E69F00",
    "magic": "#CC79A7",
    "qmix": "#000000",
    "tarmac": "#56B4E9",
    "teamcomm": "#D55E00",
    "tiecomm": "#7F7F7F",
    "hypercomm": "#0072B2",
    "hyperwave_x": "#C00000",
}
METHOD_MARKERS = {
    "commnet": "^",
    "ic3net": "v",
    "magic": "D",
    "qmix": "P",
    "tarmac": "X",
    "teamcomm": "<",
    "tiecomm": ">",
    "hypercomm": "s",
    "hyperwave_x": "o",
}
SCENARIO_ORDER = [
    "lbf_hetero_mixed-v0",
    "lbf_hetero_strong-v0",
    "lbf_consensus_wide-v0",
    "lbf_fog_sight1_level-v0",
    "lbf_dynamic_overload-v0",
    "lbf_scale_20-v0",
]
SCENARIO_LABEL = {
    "lbf_hetero_mixed-v0": "hetero_mixed",
    "lbf_hetero_strong-v0": "hetero_strong",
    "lbf_consensus_wide-v0": "consensus_wide",
    "lbf_fog_sight1_level-v0": "fog_sight1_level",
    "lbf_dynamic_overload-v0": "dynamic_overload",
    "lbf_scale_20-v0": "scale_20",
}
FORMAL_SEEDS = [20, 23, 56, 89, 2025]
EXPECTED_EVAL_SEEDS = list(range(1001, 1011))
EXPECTED_EPISODES_PER_EVAL_SEED = 20
EXPECTED_EPISODES_PER_TRAIN_SEED = 200
T95_DF4 = float(student_t.ppf(0.975, 4))

PRIMARY_METRICS = [
    "Task_Completion_Ratio",
    "Episode_Return",
    "Episode_Success_Rate",
    "Collision_Rate",
    "Capacity_Completion_Ratio",
    "Missing_Member_Rate",
    "Synchronization_Success_Rate",
    "Active_Link_Ratio",
]
COMMUNICATION_METRICS = [
    "Logical_Transfers_Per_Step",
    "Logical_Feature_Bytes_FP32_Per_Step",
    "Incidence_Traversals_Per_Step",
]


@dataclass(frozen=True)
class SourceSpec:
    method: str
    root: Path
    method_dir: str


SOURCE_SPECS = [
    *[
        SourceSpec(method, TEAMCOMM_ROOT, method)
        for method in METHOD_ORDER[:7]
    ],
    SourceSpec("hypercomm", HYPERCOMM_ROOT, "hypercomm"),
    SourceSpec("hyperwave_x", HYPERWAVEX_ROOT, "hyperwave_x"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"Empty source CSV: {path}")
    return rows, fields


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return math.nan
    return float(value)


def mean_or_nan(values: Iterable[float]) -> float:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    return float(array.mean()) if array.size else math.nan


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if array.size == 0:
        return {"n": 0, "mean": math.nan, "sd": math.nan, "ci95": math.nan}
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if array.size > 1 else math.nan
    ci = T95_DF4 * sd / math.sqrt(array.size) if array.size == 5 else math.nan
    return {"n": int(array.size), "mean": mean, "sd": sd, "ci95": ci}


def scenario_id_from_env(env_name: str) -> str:
    if not env_name.startswith("lbf_") or not env_name.endswith("-v0"):
        raise ValueError(f"Unexpected scenario name: {env_name}")
    return env_name[len("lbf_"):-len("-v0")]


def validate_episode_keys(rows: list[dict[str, str]], path: Path, train_seed: int) -> None:
    if len(rows) != EXPECTED_EPISODES_PER_TRAIN_SEED:
        raise ValueError(f"{path}: expected 200 rows, found {len(rows)}")
    if {int(row["Train_Seed"]) for row in rows} != {train_seed}:
        raise ValueError(f"{path}: training seed mismatch")
    eval_counts = Counter(int(row["Eval_Seed"]) for row in rows)
    expected = Counter({seed: EXPECTED_EPISODES_PER_EVAL_SEED for seed in EXPECTED_EVAL_SEEDS})
    if eval_counts != expected:
        raise ValueError(f"{path}: evaluation seed/episode mismatch: {eval_counts}")
    phases = {row.get("Evaluation_Phase", "frozen_test") for row in rows}
    if phases != {"frozen_test"}:
        raise ValueError(f"{path}: expected frozen_test rows, found {phases}")


def locate_run(spec: SourceSpec, scenario: str, seed: int) -> Path:
    seed_root = spec.root / spec.method_dir / scenario / f"seed_{seed}"
    candidates = sorted(
        path for path in seed_root.glob("run*")
        if path.is_dir()
        and (path / "comparison_metrics.csv").is_file()
        and (path / "test_metrics.csv").is_file()
        and (path / "communication_costs.csv").is_file()
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one complete run under {seed_root}, found {candidates}"
        )
    return candidates[0]


def capacity_by_episode(test_rows: list[dict[str, str]]) -> dict[tuple[int, int], float]:
    output: dict[tuple[int, int], float] = {}
    for row in test_rows:
        key = (int(row["Eval_Seed"]), int(row["Episode"]))
        direct = as_float(row, "Capacity_Completion_Ratio")
        if math.isfinite(direct):
            output[key] = direct
            continue
        total = as_float(row, "Capacity_Workload_Total")
        completed = as_float(row, "Capacity_Workload_Completed")
        if math.isfinite(total) and total > 0 and math.isfinite(completed):
            output[key] = completed / total
    return output


def load_seed_records() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    registry = json.loads(SCENARIO_REGISTRY.read_text(encoding="utf-8"))
    agent_count = {
        f"lbf_{scenario_id}-v0": int(spec["agents"]["count"])
        for scenario_id, spec in registry["scenarios"].items()
        if f"lbf_{scenario_id}-v0" in SCENARIO_ORDER
    }
    records: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []

    for spec in SOURCE_SPECS:
        for scenario in SCENARIO_ORDER:
            for train_seed in FORMAL_SEEDS:
                run = locate_run(spec, scenario, train_seed)
                comparison_path = run / "comparison_metrics.csv"
                test_path = run / "test_metrics.csv"
                communication_path = run / "communication_costs.csv"
                capacity_path = run / "capacity_metrics.csv"
                comparison, _ = read_csv(comparison_path)
                test, _ = read_csv(test_path)
                communication, _ = read_csv(communication_path)
                validate_episode_keys(comparison, comparison_path, train_seed)
                validate_episode_keys(test, test_path, train_seed)
                if len(communication) != EXPECTED_EPISODES_PER_TRAIN_SEED:
                    raise ValueError(
                        f"{communication_path}: expected 200 rows, found {len(communication)}"
                    )

                # Some legacy HyperComm checkpoints were re-evaluated solely to
                # recover the exact task-threshold ledger.  Keep that independent
                # evaluation in capacity_metrics.csv rather than combining its
                # trajectory-level values with the original test episode.
                capacity_rows = test
                if capacity_path.is_file():
                    capacity_rows, _ = read_csv(capacity_path)
                    validate_episode_keys(capacity_rows, capacity_path, train_seed)
                test_capacity = capacity_by_episode(capacity_rows)
                test_by_episode = {
                    (int(row["Eval_Seed"]), int(row["Episode"])): row
                    for row in test
                }
                per_episode: list[dict[str, float]] = []
                for row in comparison:
                    key = (int(row["Eval_Seed"]), int(row["Episode"]))
                    test_row = test_by_episode[key]
                    steps = as_float(row, "Episode_Steps")
                    collisions = as_float(row, "Collisions")
                    per_episode.append(
                        {
                            "Task_Completion_Ratio": as_float(row, "Task_Completion_Ratio"),
                            "Episode_Return": as_float(row, "Episode_Return"),
                            "Episode_Success_Rate": as_float(row, "Episode_Success"),
                            "Collision_Rate": collisions / (steps * agent_count[scenario]),
                            "Capacity_Completion_Ratio": test_capacity.get(key, math.nan),
                            "Missing_Member_Rate": as_float(row, "Missing_Member_Rate"),
                            "Synchronization_Success_Rate": as_float(
                                row, "Synchronization_Success_Rate"
                            ),
                            "Active_Link_Ratio": as_float(
                                test_row, "Mean_Active_Link_Ratio"
                            ),
                            "Throughput": as_float(row, "Throughput"),
                        }
                    )

                seed_record: dict[str, object] = {
                    "Method": spec.method,
                    "Scenario": scenario,
                    "Train_Seed": train_seed,
                    "Eval_Episodes": len(per_episode),
                }
                for metric in PRIMARY_METRICS + ["Throughput"]:
                    seed_record[metric] = mean_or_nan(row[metric] for row in per_episode)
                for metric in COMMUNICATION_METRICS:
                    seed_record[metric] = mean_or_nan(
                        as_float(row, metric) for row in communication
                    )
                records.append(seed_record)
                for path, role in [
                    (comparison_path, "comparison_metrics"),
                    (test_path, "test_metrics"),
                    (communication_path, "communication_costs"),
                ]:
                    inventory.append(source_inventory_row(path, role))
    return records, inventory


def source_inventory_row(path: Path, role: str) -> dict[str, object]:
    rows, _ = read_csv(path)
    return {
        "Role": role,
        "Relative_Path": str(path.relative_to(REPO_ROOT)),
        "Rows": len(rows),
        "Bytes": path.stat().st_size,
        "SHA256": sha256(path),
    }


def group_summary(seed_records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in seed_records:
        grouped[(str(record["Method"]), str(record["Scenario"]))].append(record)

    output: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        for scenario in SCENARIO_ORDER:
            rows = grouped[(method, scenario)]
            seeds = sorted(int(row["Train_Seed"]) for row in rows)
            if seeds != sorted(FORMAL_SEEDS):
                raise ValueError(f"{method}/{scenario}: training seeds {seeds}")
            item: dict[str, object] = {
                "Method": method,
                "Method_Label": METHOD_LABEL[method],
                "Scenario": scenario,
                "Scenario_Label": SCENARIO_LABEL[scenario],
                "Training_Seeds": ";".join(map(str, seeds)),
                "N_Training_Seeds": 5,
                "Evaluation_Episodes_Per_Training_Seed": 200,
            }
            for metric in PRIMARY_METRICS + ["Throughput"] + COMMUNICATION_METRICS:
                stats = summarize(float(row[metric]) for row in rows)
                item[f"{metric}_N"] = stats["n"]
                item[f"{metric}_Mean"] = stats["mean"]
                item[f"{metric}_SD"] = stats["sd"]
                item[f"{metric}_CI95_HalfWidth"] = stats["ci95"]
            output.append(item)
    return output


def exact_sign_flip_p(differences: list[float]) -> float:
    values = np.asarray(differences, dtype=float)
    if values.size != 5 or not np.all(np.isfinite(values)):
        return math.nan
    observed = abs(float(values.mean()))
    null = [
        abs(float(np.mean(values * np.asarray(signs, dtype=float))))
        for signs in itertools.product([-1.0, 1.0], repeat=values.size)
    ]
    return float(sum(value >= observed - 1e-15 for value in null) / len(null))


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [math.nan] * len(p_values)
    finite = [(index, value) for index, value in enumerate(p_values) if math.isfinite(value)]
    finite.sort(key=lambda item: item[1])
    running = 0.0
    total = len(finite)
    for rank, (index, value) in enumerate(finite):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_tests(seed_records: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = {
        (str(row["Method"]), str(row["Scenario"]), int(row["Train_Seed"])): row
        for row in seed_records
    }
    rows: list[dict[str, object]] = []
    metrics = [
        "Task_Completion_Ratio", "Episode_Return", "Episode_Success_Rate",
        "Collision_Rate", "Missing_Member_Rate", "Synchronization_Success_Rate",
        "Logical_Feature_Bytes_FP32_Per_Step",
    ]
    lower_is_better = {
        "Collision_Rate", "Missing_Member_Rate", "Logical_Feature_Bytes_FP32_Per_Step"
    }
    for metric in metrics:
        metric_start = len(rows)
        for scenario in SCENARIO_ORDER:
            for baseline in METHOD_ORDER[:-1]:
                differences = [
                    float(lookup[("hyperwave_x", scenario, seed)][metric])
                    - float(lookup[(baseline, scenario, seed)][metric])
                    for seed in FORMAL_SEEDS
                ]
                stats = summarize(differences)
                mean_difference = float(stats["mean"])
                favors = (
                    mean_difference < 0 if metric in lower_is_better else mean_difference > 0
                )
                rows.append(
                    {
                        "Metric": metric,
                        "Scenario": scenario,
                        "Scenario_Label": SCENARIO_LABEL[scenario],
                        "Baseline": baseline,
                        "Baseline_Label": METHOD_LABEL[baseline],
                        "Difference_Definition": "HyperWave-X minus baseline",
                        "Paired_Seed_Count": 5,
                        "Mean_Paired_Difference": mean_difference,
                        "SD_Paired_Difference": stats["sd"],
                        "CI95_HalfWidth": stats["ci95"],
                        "Exact_Two_Sided_SignFlip_P": exact_sign_flip_p(differences),
                        "Holm_Adjusted_P_Within_Metric": math.nan,
                        "Effect_Favors_HyperWaveX": int(favors),
                    }
                )
        metric_rows = rows[metric_start:]
        adjusted = holm_adjust(
            [float(row["Exact_Two_Sided_SignFlip_P"]) for row in metric_rows]
        )
        for row, value in zip(metric_rows, adjusted):
            row["Holm_Adjusted_P_Within_Metric"] = value
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ("NA" if isinstance(value, float) and not math.isfinite(value) else value)
                    for key, value in row.items()
                }
            )


def format_value(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def latex_escape(text: object) -> str:
    return (
        str(text).replace("\\", "\\textbackslash{}")
        .replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
    )


def write_latex_longtable(
    path: Path, rows: list[dict[str, object]], columns: list[tuple[str, str]], caption: str,
) -> None:
    align = "l" * len(columns)
    lines = [
        "% Generated from frozen CSV results; do not edit numerical entries manually.",
        f"\\begin{{longtable}}{{{align}}}",
        f"\\caption{{{latex_escape(caption)}}}\\\\",
        "\\toprule",
        " & ".join(latex_escape(label) for _, label in columns) + " \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        " & ".join(latex_escape(label) for _, label in columns) + " \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for row in rows:
        lines.append(
            " & ".join(latex_escape(format_value(row[key])) for key, _ in columns) + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{longtable}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def summary_lookup(summary_rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    return {(str(row["Method"]), str(row["Scenario"])): row for row in summary_rows}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.5,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    save(fig, stem, LANGUAGE)


def dot_whisker_figure(
    summary_rows: list[dict[str, object]], metric: str, ylabel: str, stem: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    lookup = summary_lookup(summary_rows)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharey=ylim is not None)
    x = np.arange(len(METHOD_ORDER))
    for ax, scenario in zip(axes.flat, SCENARIO_ORDER):
        means = [float(lookup[(method, scenario)][f"{metric}_Mean"]) for method in METHOD_ORDER]
        cis = [float(lookup[(method, scenario)][f"{metric}_CI95_HalfWidth"]) for method in METHOD_ORDER]
        for index, method in enumerate(METHOD_ORDER):
            ax.errorbar(
                index, means[index], yerr=cis[index], fmt=METHOD_MARKERS[method], ms=4.2,
                color=METHOD_COLORS[method], ecolor=METHOD_COLORS[method],
                elinewidth=1.0, capsize=2.2, capthick=0.9, zorder=3,
            )
        ax.set_title(SCENARIO_LABEL[scenario])
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER], rotation=42, ha="right")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.axvline(6.5, color="#D5D5D5", lw=0.7, zorder=0)
    fig.legend(
        handles=[
            Line2D([0], [0], marker=METHOD_MARKERS[m], color="none", markerfacecolor=METHOD_COLORS[m],
                   markeredgecolor=METHOD_COLORS[m], label=METHOD_LABEL[m], markersize=4)
            for m in METHOD_ORDER
        ],
        loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, stem)


def two_axis_tradeoff_figure(
    summary_rows: list[dict[str, object]], x_metric: str, x_label: str,
    y_metric: str, y_label: str, stem: str,
) -> None:
    lookup = summary_lookup(summary_rows)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    for ax, scenario in zip(axes.flat, SCENARIO_ORDER):
        for method in METHOD_ORDER:
            row = lookup[(method, scenario)]
            ax.errorbar(
                float(row[f"{x_metric}_Mean"]),
                float(row[f"{y_metric}_Mean"]),
                xerr=float(row[f"{x_metric}_CI95_HalfWidth"]),
                yerr=float(row[f"{y_metric}_CI95_HalfWidth"]),
                fmt=METHOD_MARKERS[method], ms=4.4,
                color=METHOD_COLORS[method], ecolor=METHOD_COLORS[method],
                elinewidth=0.8, capsize=1.8, alpha=0.95, zorder=3,
            )
        ax.set_title(SCENARIO_LABEL[scenario])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(left=-0.01)
        ax.set_ylim(bottom=-0.01)
        ax.text(
            0.03, 0.95, "preferred", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.5, color="#555555",
        )
    fig.legend(
        handles=[
            Line2D([0], [0], marker=METHOD_MARKERS[m], color="none",
                   markerfacecolor=METHOD_COLORS[m], markeredgecolor=METHOD_COLORS[m],
                   label=METHOD_LABEL[m], markersize=4)
            for m in METHOD_ORDER
        ],
        loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, stem)


def pareto_figure(summary_rows: list[dict[str, object]]) -> None:
    lookup = summary_lookup(summary_rows)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
    compared_methods = [method for method in METHOD_ORDER if method != "qmix"]
    for ax, scenario in zip(axes.flat, SCENARIO_ORDER):
        for method in compared_methods:
            row = lookup[(method, scenario)]
            x = float(row["Logical_Transfers_Per_Step_Mean"])
            xerr = float(row["Logical_Transfers_Per_Step_CI95_HalfWidth"])
            y = float(row["Task_Completion_Ratio_Mean"])
            yerr = float(row["Task_Completion_Ratio_CI95_HalfWidth"])
            ax.errorbar(
                x, y, xerr=xerr, yerr=yerr, fmt=METHOD_MARKERS[method], ms=4.4,
                color=METHOD_COLORS[method], ecolor=METHOD_COLORS[method],
                elinewidth=0.8, capsize=1.8, alpha=0.95,
            )
        ax.set_title(SCENARIO_LABEL[scenario])
        ax.set_xlabel("Logical transfers per step")
        ax.set_ylabel("Task completion ratio")
        ax.set_xlim(left=0)
        ax.set_ylim(-0.03, 1.03)
    fig.legend(
        handles=[
            Line2D([0], [0], marker=METHOD_MARKERS[m], color="none", markerfacecolor=METHOD_COLORS[m],
                   markeredgecolor=METHOD_COLORS[m], label=METHOD_LABEL[m], markersize=4)
            for m in compared_methods
        ],
        loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, "Figure_4_Communication_Performance_Pareto")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    prepare(LANGUAGE)
    seed_records, inventory = load_seed_records()
    summary_rows = group_summary(seed_records)
    test_rows = paired_tests(seed_records)
    missing_hypercomm_ccr = [
        str(row["Scenario"])
        for row in summary_rows
        if row["Method"] == "hypercomm"
        and int(row["Capacity_Completion_Ratio_N"]) != 5
    ]

    write_csv(OUTPUT_DIR / "Table_1_Main_Performance.csv", summary_rows)
    write_csv(OUTPUT_DIR / "Table_2_Paired_Tests_vs_HyperWaveX.csv", test_rows)
    macro_rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        method_rows = [row for row in summary_rows if row["Method"] == method]
        ccr_values = [
            float(row["Capacity_Completion_Ratio_Mean"])
            for row in method_rows
            if math.isfinite(float(row["Capacity_Completion_Ratio_Mean"]))
        ]
        macro_rows.append(
            {
                "Method": method,
                "Method_Label": METHOD_LABEL[method],
                "Scenarios": 6,
                "TCR_Macro_Mean": mean_or_nan(
                    float(row["Task_Completion_Ratio_Mean"]) for row in method_rows
                ),
                "CCR_Macro_Mean": mean_or_nan(ccr_values),
                "CCR_Scenarios": len(ccr_values),
                "CR_Macro_Mean": mean_or_nan(
                    float(row["Collision_Rate_Mean"]) for row in method_rows
                ),
                "MMR_Macro_Mean": mean_or_nan(
                    float(row["Missing_Member_Rate_Mean"]) for row in method_rows
                ),
                "Logical_Transfers_Per_Step_Macro_Mean": (
                    math.nan if method == "qmix" else mean_or_nan(
                        float(row["Logical_Transfers_Per_Step_Mean"])
                        for row in method_rows
                    )
                ),
            }
        )
    write_csv(OUTPUT_DIR / "Table_4_Macro_Balance.csv", macro_rows)
    communication_rows = [
        {
            key: row[key]
            for key in [
                "Method", "Method_Label", "Scenario", "Scenario_Label",
                "N_Training_Seeds",
                "Task_Completion_Ratio_Mean", "Task_Completion_Ratio_SD",
                "Logical_Transfers_Per_Step_Mean", "Logical_Transfers_Per_Step_SD",
                "Logical_Feature_Bytes_FP32_Per_Step_Mean",
                "Logical_Feature_Bytes_FP32_Per_Step_SD",
                "Incidence_Traversals_Per_Step_Mean",
                "Incidence_Traversals_Per_Step_SD",
            ]
        }
        for row in summary_rows
    ]
    write_csv(OUTPUT_DIR / "Table_3_Communication_Cost.csv", communication_rows)
    write_csv(OUTPUT_DIR / "Source_Data_Inventory.csv", inventory)

    write_latex_longtable(
        OUTPUT_DIR / "Table_1_Main_Performance.tex",
        summary_rows,
        [
            ("Method_Label", "Method"), ("Scenario_Label", "Scenario"),
            ("Task_Completion_Ratio_Mean", "TCR mean"),
            ("Task_Completion_Ratio_SD", "TCR SD"),
            ("Task_Completion_Ratio_CI95_HalfWidth", "TCR 95% half-width"),
            ("Episode_Return_Mean", "Return mean"),
            ("Episode_Success_Rate_Mean", "ESR mean"),
            ("Capacity_Completion_Ratio_Mean", "CCR mean"),
            ("Collision_Rate_Mean", "CR mean"),
        ],
        "Frozen-test performance across five independent training seeds.",
    )
    write_latex_longtable(
        OUTPUT_DIR / "Table_2_Paired_Tests_vs_HyperWaveX.tex",
        test_rows,
        [
            ("Metric", "Metric"), ("Scenario_Label", "Scenario"),
            ("Baseline_Label", "Baseline"),
            ("Mean_Paired_Difference", "Mean paired difference"),
            ("CI95_HalfWidth", "95% half-width"),
            ("Exact_Two_Sided_SignFlip_P", "Exact p"),
            ("Holm_Adjusted_P_Within_Metric", "Holm p"),
        ],
        "Paired seed-level comparisons of HyperWave-X against each baseline.",
    )
    write_latex_longtable(
        OUTPUT_DIR / "Table_3_Communication_Cost.tex",
        communication_rows,
        [
            ("Method_Label", "Method"), ("Scenario_Label", "Scenario"),
            ("Logical_Transfers_Per_Step_Mean", "Transfers/step"),
            ("Logical_Feature_Bytes_FP32_Per_Step_Mean", "FP32 bytes/step"),
            ("Incidence_Traversals_Per_Step_Mean", "Incidence traversals/step"),
            ("Task_Completion_Ratio_Mean", "TCR"),
        ],
        "Logical communication accounting and task completion.",
    )

    dot_whisker_figure(
        summary_rows, "Task_Completion_Ratio", "Task completion ratio",
        "Figure_1_Task_Completion", (-0.03, 1.03),
    )
    two_axis_tradeoff_figure(
        summary_rows,
        "Collision_Rate", "Collision rate",
        "Episode_Success_Rate", "Episode success rate",
        "Figure_3_Reliability_and_Safety",
    )
    pareto_figure(summary_rows)
    two_axis_tradeoff_figure(
        summary_rows,
        "Missing_Member_Rate", "Missing-member rate",
        "Task_Completion_Ratio", "Task completion ratio",
        "Figure_5_Synchronization_Process",
    )

    missing_data = [
        "No cross-method isolated latency figure is produced because the stored "
        "latency boundaries are not uniformly certified across all methods."
    ]
    if missing_hypercomm_ccr:
        missing_data.insert(
            0,
            "HyperComm CCR is unavailable for: "
            + ", ".join(missing_hypercomm_ccr)
            + "; these cells remain NA and are not estimated.",
        )

    captions = """Main-results figure captions

Figure 1. Frozen-test task completion ratio across the six main scenarios. Points show means over five independent training seeds; whiskers show two-sided 95% t intervals computed from seed-level means. Each trained policy is evaluated for 200 episodes (10 evaluation seeds x 20 episodes).

Figure 3. Reliability--safety trade-off. Each panel places episode success rate against collision rate; the preferred direction is upward and leftward. Episode success requires completion of all tasks within the scenario horizon. Points are five-seed means and whiskers are 95% t intervals on both axes.

Figure 4. Task completion versus logical feature transfers per environment step for the eight methods that communicate during decentralized execution. QMIX is omitted because its execution transfer count is zero by design. Moving upward improves completion and moving left reduces counted transfers. Both axes show five-seed means with 95% t intervals.

Figure 5. Completion--coalition-readiness trade-off. Task completion is plotted against missing-member rate; the preferred direction is upward and leftward. Points show five-seed means with 95% t intervals on both axes.

The accompanying tables report sample standard deviations, confidence intervals, source availability, and exact paired tests.
"""
    (OUTPUT_DIR / "Figure_Captions.txt").write_text(captions, encoding="utf-8")

    manifest = {
        "status": "PASS",
        "path_policy": "all paths in this script and manifest are repository-relative",
        "source_csv_files": len(inventory),
        "seed_level_records": len(seed_records),
        "method_scenario_summaries": len(summary_rows),
        "methods": METHOD_ORDER,
        "scenarios": SCENARIO_ORDER,
        "training_seeds": FORMAL_SEEDS,
        "evaluation_seeds": EXPECTED_EVAL_SEEDS,
        "evaluation_episodes_per_seed": EXPECTED_EPISODES_PER_EVAL_SEED,
        "independent_unit": "training seed",
        "ci_method": "two-sided t interval, df=4, seed-level means",
        "paired_test": "exact two-sided sign-flip permutation test on five paired seeds",
        "multiplicity": "Holm correction within metric across 48 scenario-baseline tests",
        "known_missing_data": missing_data,
        "outputs": sorted(path.name for path in OUTPUT_DIR.iterdir()),
    }
    (OUTPUT_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
