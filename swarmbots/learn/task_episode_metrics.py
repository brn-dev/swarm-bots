from collections.abc import Iterable
from typing import Any

from swarmbots.learn.summary_statistics import SummaryStatistics, compute_summary_statistics


def episode_success_metrics(episode_infos: list[dict[str, Any]], *, prefix: str = "") -> dict[str, float | int]:
    outcomes = [float(episode["success"]) for episode in episode_infos if "success" in episode]
    if not outcomes:
        return {}
    key = f"{prefix}ep_success_rate"
    return {key: 100.0 * sum(outcomes) / len(outcomes), f"{key}__n": len(outcomes)}


def task_episode_metric_names(keys: Iterable[str]) -> dict[str, str]:
    task_keys = sorted({key for key in keys if key.startswith("task/")})
    return {key: f"ep_{key.replace('/', '_')}" for key in task_keys}


def summarize_task_episode_metrics(
    episode_infos: list[dict[str, Any]], *, prefix: str = "",
) -> dict[str, SummaryStatistics | None]:
    metric_names = task_episode_metric_names(key for episode in episode_infos for key in episode)
    return {
        f"{prefix}{name}": compute_summary_statistics(
            [episode[key] for episode in episode_infos if key in episode], find_min=True, find_max=True,
        )
        for key, name in metric_names.items()
    }
