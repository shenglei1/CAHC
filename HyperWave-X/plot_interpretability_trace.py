from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot coalition formation, routing, wave energy, and LOAD execution."
    )
    parser.add_argument("trace", type=Path, help="Path to episode_trace.jsonl")
    parser.add_argument("--line", type=int, default=1, help="One-based JSONL episode line")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path; the extension selects PNG, PDF, or SVG",
    )
    return parser.parse_args()


def load_episode(path: Path, line_number: int) -> dict:
    if line_number < 1:
        raise ValueError("--line must be at least 1")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if line_number > len(lines):
        raise ValueError(f"Trace has {len(lines)} episode lines, not {line_number}.")
    episode = json.loads(lines[line_number - 1])
    if not episode.get("steps"):
        raise ValueError("Selected trace episode contains no steps.")
    return episode


def representative_step(steps: list[dict]) -> int:
    completed = [index for index, step in enumerate(steps) if step["task_completion_delta"] > 0]
    if completed:
        return completed[0]
    return max(
        range(len(steps)),
        key=lambda index: (
            steps[index]["probe1_exact_pair_coverage"],
            steps[index]["candidate_feasible_fraction"],
        ),
    )


def write_timeseries(path: Path, episode: dict) -> None:
    fields = [
        "eval_seed",
        "episode",
        "step",
        "candidate_feasible_fraction",
        "mean_candidate_capacity_gap",
        "probe1_exact_pair_coverage",
        "active_directed_links",
        "active_link_ratio",
        "load_action_count",
        "false_load_attempt",
        "task_completion_delta",
        "task_completion_ratio",
        "wave_energy_ratio",
        "wave_lambda_max",
        "wave_contraction_bound",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in episode["steps"]:
            writer.writerow(
                {
                    "eval_seed": episode["eval_seed"],
                    "episode": episode["episode"],
                    "step": step["step"],
                    "candidate_feasible_fraction": step["candidate_feasible_fraction"],
                    "mean_candidate_capacity_gap": step["mean_candidate_capacity_gap"],
                    "probe1_exact_pair_coverage": step["probe1_exact_pair_coverage"],
                    "active_directed_links": step["active_directed_links"],
                    "active_link_ratio": step["active_link_ratio"],
                    "load_action_count": step["load_action_count"],
                    "false_load_attempt": step["false_load_attempt"],
                    "task_completion_delta": step["task_completion_delta"],
                    "task_completion_ratio": step["task_completion_ratio"],
                    "wave_energy_ratio": step["wave"]["energy_ratio"],
                    "wave_lambda_max": step["wave"]["lambda_max"],
                    "wave_contraction_bound": step["wave"]["contraction_bound"],
                }
            )


def render(episode: dict, output: Path) -> None:
    steps = episode["steps"]
    x = np.asarray([step["step"] for step in steps])
    selected = representative_step(steps)
    snapshot = steps[selected]
    agent_count = len(snapshot["agent_positions_before_action"])

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(x, [s["candidate_feasible_fraction"] for s in steps], label="threshold-feasible candidates")
    ax.plot(x, [s["probe1_exact_pair_coverage"] for s in steps], label="exact complementary-pair coverage")
    ax.set(title="Coalition construction", xlabel="decision step", ylabel="fraction", ylim=(-0.03, 1.03))
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, [s["active_link_ratio"] for s in steps], color="tab:purple", label="active-link ratio")
    ax.set(title="Sparse routing over time", xlabel="decision step", ylabel="directed-link ratio", ylim=(-0.03, 1.03))
    ax.axvline(snapshot["step"], color="0.45", linestyle="--", linewidth=1)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.step(x, [s["task_completion_ratio"] for s in steps], where="post", color="tab:green", label="workload completed")
    load_steps = [s["step"] for s in steps if s["load_action_count"] > 0]
    false_steps = [s["step"] for s in steps if s["false_load_attempt"] > 0]
    if load_steps:
        ax.scatter(load_steps, [0.04] * len(load_steps), marker="|", s=100, color="tab:blue", label="LOAD attempted")
    if false_steps:
        ax.scatter(false_steps, [0.10] * len(false_steps), marker="x", s=30, color="tab:red", label="unsuccessful LOAD step")
    ax.set(title="Synchronized task execution", xlabel="decision step", ylabel="completion ratio", ylim=(-0.03, 1.03))
    ax.legend(frameon=False, fontsize=8)

    matrix = np.full((agent_count, agent_count), np.nan)
    for link in snapshot["active_routing_links"]:
        matrix[link["receiver"], link["sender"]] = float(np.mean(link["attention_by_head"]))
    ax = axes[1, 1]
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", interpolation="nearest")
    ax.set(
        title=f"Active routing at step {snapshot['step']}",
        xlabel="sender agent",
        ylabel="receiver agent",
        xticks=range(agent_count),
        yticks=range(agent_count),
    )
    figure.colorbar(image, ax=ax, fraction=0.046, label="mean attention over heads")

    figure.suptitle(
        f"Representative trace: evaluation seed {episode['eval_seed']}, episode {episode['episode']}",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    episode = load_episode(args.trace.resolve(), args.line)
    output = (args.output or args.trace.with_name("interpretability_trace.png")).resolve()
    render(episode, output)
    csv_path = output.with_name(f"{output.stem}_timeseries.csv")
    write_timeseries(csv_path, episode)
    print(output)
    print(csv_path)


if __name__ == "__main__":
    main()
