from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import torch

TensorNpArrayOrList = torch.Tensor | np.ndarray | list

SMALL_DATA_THRESHOLD = 0

@dataclass
class SummaryStatistics:
    n: int
    mean: float
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    data: Optional[list] = None

@dataclass
class SummaryStatisticsFormat:
    n: Optional[str] = None
    mean: Optional[str] = None
    std: Optional[str] = None
    min_value: Optional[str] = None
    max_value: Optional[str] = None


def is_summary_statistics(obj: Any):
    return isinstance(obj, SummaryStatistics)


def format_summary_statistics(
        x: TensorNpArrayOrList | SummaryStatistics | None,
        stats_format: SummaryStatisticsFormat | None = None
):
    if stats_format is None:
        stats_format = SummaryStatisticsFormat(
            n=None,
            mean='.3f',
            std='.3f',
            min_value=None,
            max_value=None,
        )

    summary_statistics = maybe_compute_summary_statistics(
        x,
        find_min=stats_format.min_value is not None,
        find_max=stats_format.max_value is not None,
    )

    if summary_statistics is None:
        return 'N/A'

    n = summary_statistics.n
    mean = summary_statistics.mean
    std = summary_statistics.std
    min_value = summary_statistics.min_value
    max_value = summary_statistics.max_value

    representation = ''

    if stats_format.mean:
        representation += format(mean, stats_format.mean)

    if std is not None and stats_format.std:
        representation += f' ± {format(std, stats_format.std)}'

    min_val_available = min_value is not None and stats_format.min_value
    max_val_available = max_value is not None and stats_format.max_value

    if min_val_available and max_val_available:
        representation += (f' [{format(min_value, stats_format.min_value)}, '
                           f'{format(max_value, stats_format.max_value)}]')
    elif min_val_available:
        representation += f' ≥ {format(min_value, stats_format.min_value)}'
    elif max_val_available:
        representation += f' ≤ {format(max_value, stats_format.max_value)}'

    if stats_format.n:
        representation += f' (n={format(n, stats_format.n)})'

    return representation


def compute_summary_statistics(
        arr: TensorNpArrayOrList,
        find_min: bool = False,
        find_max: bool = False,
        small_data_threshold: int = SMALL_DATA_THRESHOLD,
) -> Optional[SummaryStatistics]:
    if isinstance(arr, list):
        arr = np.array(arr)
    if isinstance(arr, np.ndarray):
        n = arr.size
    else:
        n = arr.numel()

    if n == 0:
        return None

    mean = arr.ravel().mean().item()

    if n == 1:
        summary_stats = SummaryStatistics(
            n=n,
            mean=mean,
        )
        return summary_stats

    summary_stats: SummaryStatistics = SummaryStatistics(
        n=n,
        mean=mean,
        std=arr.ravel().std().item(),
    )
    if find_min:
        summary_stats.min_value = arr.min().item()
    if find_max:
        summary_stats.max_value = arr.max().item()

    if n <= small_data_threshold:
        summary_stats.data = arr.tolist()

    return summary_stats

def maybe_compute_summary_statistics(
        x: TensorNpArrayOrList | SummaryStatistics | None,
        find_min: bool = False,
        find_max: bool = False,
        small_data_threshold: int = SMALL_DATA_THRESHOLD,
):
    if is_summary_statistics(x):
        return x
    if x is None:
        return None
    return compute_summary_statistics(
        x,
        find_min=find_min,
        find_max=find_max,
        small_data_threshold=small_data_threshold,
    )