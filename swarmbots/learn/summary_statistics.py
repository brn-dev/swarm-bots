from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import torch

TensorNpArrayOrList = torch.Tensor | np.ndarray | list

HISTOGRAM_DEFAULT_BINS = 10
HISTOGRAM_BLOCKS = " ▁▂▃▄▅▆▇█"

@dataclass
class Histogram:
    bin_frequencies: list[float]  # (n_bins)
    bin_edges: list[float]  # (n_bins + 1)

@dataclass
class SummaryStatistics:
    n: int
    mean: float
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    data: Optional[np.ndarray] = None
    histogram: Optional[Histogram] = None

@dataclass
class SummaryStatisticsFormat:
    n: Optional[str] = None
    mean: Optional[str] = None
    std: Optional[str] = None
    min_value: Optional[str] = None
    max_value: Optional[str] = None
    histogram: bool | int = False


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
            histogram=False,
        )

    summary_statistics = maybe_compute_summary_statistics(
        x,
        find_min=stats_format.min_value is not None,
        find_max=stats_format.max_value is not None,
        make_histogram=stats_format.histogram,
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
    
    if stats_format.histogram and summary_statistics.histogram is not None:
        if representation:
            representation += ' '
        representation += _format_histogram_blocks(summary_statistics.histogram.bin_frequencies)

    return representation


def compute_summary_statistics(
        arr: TensorNpArrayOrList,
        find_min: bool = False,
        find_max: bool = False,
        make_histogram: bool | int = False,
        keep_data: bool = True,
) -> Optional[SummaryStatistics]:
    values: np.ndarray
    if isinstance(arr, list):
        values = np.array(arr).ravel()
    elif isinstance(arr, np.ndarray):
        values = arr.ravel()
    elif isinstance(arr, torch.Tensor):
        values = arr.detach().ravel().cpu().numpy()
    else:
        raise ValueError(arr)

    n = values.size

    if n == 0:
        return None

    mean = values.mean().item()

    if n == 1:
        summary_stats = SummaryStatistics(
            n=n,
            mean=mean,
        )
        return summary_stats

    summary_stats: SummaryStatistics = SummaryStatistics(
        n=n,
        mean=mean,
        std=values.std().item(),
    )
    find_min = find_min or make_histogram
    find_max = find_max or make_histogram
    if find_min:
        summary_stats.min_value = values.min().item()
    if find_max:
        summary_stats.max_value = values.max().item()

    if keep_data:
        summary_stats.data = values

    if make_histogram:
        summary_stats.histogram = _compute_histogram(
            values,
            min_val=summary_stats.min_value, 
            max_val=summary_stats.max_value,
            n_bins=HISTOGRAM_DEFAULT_BINS if isinstance(make_histogram, bool) else make_histogram,
        )

    return summary_stats

def maybe_compute_summary_statistics(
        x: TensorNpArrayOrList | SummaryStatistics | None,
        find_min: bool = False,
        find_max: bool = False,
        make_histogram: bool | int = False,
        keep_data: bool = True,
):
    if is_summary_statistics(x):
        if make_histogram and x.histogram is None and x.data is not None:
            compute_histogram(x, n_bins=make_histogram if isinstance(make_histogram, int) else HISTOGRAM_DEFAULT_BINS)
        return x
    if x is None:
        return None
    return compute_summary_statistics(
        x,
        find_min=find_min,
        find_max=find_max,
        make_histogram=make_histogram,
        keep_data=keep_data,
    )
    
def compute_histogram(stats: SummaryStatistics, n_bins: int = HISTOGRAM_DEFAULT_BINS):
    assert stats.data is not None
    values = stats.data
    stats.min_value = values.min().item()
    stats.max_value = values.max().item()
    stats.histogram = _compute_histogram(values, stats.min_value, stats.max_value, n_bins)


def _compute_histogram(
    values: np.ndarray,
    min_val: float, 
    max_val: float, 
    n_bins: int = HISTOGRAM_DEFAULT_BINS
) -> Histogram:
    if values.size == 0:
        return Histogram(bin_frequencies=[0.0], bin_edges=[0.0, 1.0])

    if min_val == max_val:
        width = 1.0 if min_val == 0.0 else abs(min_val) * 0.01
        low = min_val - width
        high = max_val + width
        return Histogram(bin_frequencies=[1.0], bin_edges=[float(low), float(high)])

    bin_count = int(max(1, n_bins))
    counts, edges = np.histogram(values, bins=bin_count, range=(min_val, max_val))
    total = float(counts.sum())
    frequencies = (counts.astype(np.float64) / total) if total > 0.0 else np.zeros_like(counts, dtype=np.float64)

    return Histogram(
        bin_frequencies=[float(x) for x in frequencies.tolist()],
        bin_edges=[float(x) for x in edges.tolist()],
    )

def _format_histogram_blocks(freqs: list[float]) -> str:
    if not freqs:
        return "[]"

    max_freq = max(freqs)
    if not np.isfinite(max_freq) or max_freq <= 0.0:
        return "[" + (" " * len(freqs)) + "]"

    levels = len(HISTOGRAM_BLOCKS) - 1
    chars: list[str] = []
    for f in freqs:
        if not np.isfinite(f) or f <= 0.0:
            idx = 0
        else:
            idx = int(round(levels * (f / max_freq)))
            idx = max(0, min(levels, idx))
        chars.append(HISTOGRAM_BLOCKS[idx])
    return "[" + "".join(chars) + "]"
