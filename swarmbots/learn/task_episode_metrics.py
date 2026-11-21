from typing import Any

from swarmbots.learn.summary_statistics import SummaryStatistics, compute_summary_statistics


def summarize_task_episode_metrics(
    episode_infos: list[dict[str, Any]], *, prefix: str = "",
) -> dict[str, SummaryStatistics | None]:
    keys = sorted({key for episode in episode_infos for key in episode if key.startswith("task/")})
    return {
        f"{prefix}ep_{key.replace('/', '_')}": compute_summary_statistics(
            [episode[key] for episode in episode_infos if key in episode], find_min=True, find_max=True,
        )
        for key in keys
    }
