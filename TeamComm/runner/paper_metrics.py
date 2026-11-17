"""Single source of truth for CAIE workload and success metrics."""

from __future__ import annotations

import math


def apply_paper_metrics(log, env_info, horizon, episode_return, num_steps):
    total = int(env_info["workload_total"])
    completed = int(env_info["workload_completed"])
    if not 0 <= completed <= total:
        raise ValueError(f"Invalid workload ledger: completed={completed}, total={total}")
    ratio = float(completed / total if total else 0.0)
    throughput = float(completed / horizon if horizon else 0.0)
    episode_success = float(env_info["episode_success"])
    if episode_success and completed != total:
        raise ValueError("episode_success=1 with incomplete workload")
    log.update(
        episode_return=float(episode_return),
        episode_steps=[int(num_steps)],
        success_food=completed,
        success_rate=ratio,
        task_completion_ratio=ratio,
        episode_success=episode_success,
        throughput=throughput,
        workload_total=total,
        workload_completed=completed,
        workload_arrived=int(env_info["workload_arrived"]),
        workload_backlog=int(env_info["workload_backlog"]),
        workload_unreleased=int(env_info["workload_unreleased"]),
        capacity_workload_total=int(env_info["capacity_workload_total"]),
        capacity_workload_completed=int(env_info["capacity_workload_completed"]),
    )
    if not math.isclose(ratio * total, completed, abs_tol=1e-8):
        raise ValueError("Task completion ratio is inconsistent with workload ledger")
    if not math.isclose(throughput * horizon, completed, abs_tol=1e-8):
        raise ValueError("Throughput is inconsistent with fixed horizon")
    return log
