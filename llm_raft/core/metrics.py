from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeStats:
    task_completion_time_s: float
    average_speed: float
    collision: bool
    success: bool
    rule_violation: bool


def summarize(stats: list[EpisodeStats]) -> dict[str, float]:
    if not stats:
        return {
            "task_completion_time_s": 0.0,
            "average_speed": 0.0,
            "collision_rate": 0.0,
            "rule_violation_rate": 0.0,
            "success_rate": 0.0,
        }

    n = len(stats)
    return {
        "task_completion_time_s": sum(s.task_completion_time_s for s in stats) / n,
        "average_speed": sum(s.average_speed for s in stats) / n,
        "collision_rate": 100.0 * sum(1 for s in stats if s.collision) / n,
        "rule_violation_rate": 100.0 * sum(1 for s in stats if s.rule_violation) / n,
        "success_rate": 100.0 * sum(1 for s in stats if s.success) / n,
    }
