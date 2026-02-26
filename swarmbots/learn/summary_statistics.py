from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import torch

class NoData:
    __slots__ = ()

    def __repr__(self) -> str:
        return "NO_DATA"

NO_DATA: NoData = NoData()


HISTOGRAM_DEFAULT_BINS = 10
HISTOGRAM_BLOCKS = " ▁▂▃▄▅▆▇█"

@dataclass
class Histogram:
    bin_frequencies: list[float]  # (n_bins)
    bin_edges: list[float]  # (n_bins + 1)

@dataclass
class SummaryStatistics:
    n: int
    mean: float | NoData
    std: float | NoData
    skewness: Optional[float | NoData] = None
    kurtosis: Optional[float | NoData] = None
    min_value: Optional[float | NoData] = None
    max_value: Optional[float | NoData] = None
    data: Optional[np.ndarray] = None
    histogram: Optional[Histogram | NoData] = None

@dataclass
class SummaryStatisticsFormat:
    n: Optional[str] = None
    mean: Optional[str] = None
    std: Optional[str] = None
    skewness: Optional[str] = None
    kurtosis: Optional[str] = None
    min_value: Optional[str] = None
    max_value: Optional[str] = None
    histogram: bool | int = False

SummaryStatisticsInput = list[SummaryStatistics] | torch.Tensor | np.ndarray | list

def is_summary_statistics(obj: Any) -> bool:
    return isinstance(obj, SummaryStatistics)


def format_summary_statistics(
        x: SummaryStatisticsInput | SummaryStatistics | None,
        stats_format: SummaryStatisticsFormat | None = None
) -> str:
    if stats_format is None:
        stats_format = SummaryStatisticsFormat(
            n=None,
            mean='.3f',
            std='.3f',
            skewness=None,
            kurtosis=None,
            min_value=None,
            max_value=None,
            histogram=False,
        )

    summary_statistics = maybe_compute_summary_statistics(
        x,
        find_min=stats_format.min_value is not None,
        find_max=stats_format.max_value is not None,
        make_histogram=stats_format.histogram,
        compute_skewness=stats_format.skewness is not None,
        compute_kurtosis=stats_format.kurtosis is not None,
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
        representation += _fmt_no_data(mean, stats_format.mean)

    if std is not None and stats_format.std:
        representation += f' ± {_fmt_no_data(std, stats_format.std)}'

    if stats_format.skewness and summary_statistics.skewness is not None:
        if representation:
            representation += ' '
        representation += f'skew={_fmt_no_data(summary_statistics.skewness, stats_format.skewness)}'

    if stats_format.kurtosis and summary_statistics.kurtosis is not None:
        if representation:
            representation += ' '
        representation += f'kurt={_fmt_no_data(summary_statistics.kurtosis, stats_format.kurtosis)}'

    min_val_available = min_value is not None and stats_format.min_value
    max_val_available = max_value is not None and stats_format.max_value

    if min_val_available and max_val_available:
        representation += (f' [{_fmt_no_data(min_value, stats_format.min_value)}, '
                           f'{_fmt_no_data(max_value, stats_format.max_value)}]')
    elif min_val_available:
        representation += f' ≥ {_fmt_no_data(min_value, stats_format.min_value)}'
    elif max_val_available:
        representation += f' ≤ {_fmt_no_data(max_value, stats_format.max_value)}'

    if stats_format.n:
        representation += f' (n={format(n, stats_format.n)})'
    
    if stats_format.histogram and summary_statistics.histogram not in (None, NO_DATA):
        if representation:
            representation += ' '
        display_bin_count: Optional[int]
        if isinstance(stats_format.histogram, bool):
            display_bin_count = None
        else:
            display_bin_count = stats_format.histogram
        representation += _format_histogram_blocks(
            summary_statistics.histogram.bin_frequencies,
            n_bins=display_bin_count,
        )

    return representation

def _fmt_no_data(x: float | NoData, fmt: str) -> str:
    if x is NO_DATA:
        return 'n/a'
    return format(x, fmt)


@torch.no_grad()
def compute_summary_statistics(
        arr: SummaryStatisticsInput,
        find_min: bool = False,
        find_max: bool = False,
        make_histogram: bool | int = False,
        compute_skewness: bool = False,
        compute_kurtosis: bool = False,
        keep_data: bool = True,
) -> Optional[SummaryStatistics]:
    values: np.ndarray | torch.Tensor
    if isinstance(arr, list):
        if len(arr) == 0:
            return _no_data_stats(
                find_min=find_min,
                find_max=find_max,
                make_histogram=make_histogram,
                compute_skewness=compute_skewness,
                compute_kurtosis=compute_kurtosis,
            )
        if isinstance(arr[0], SummaryStatistics):
            return combine_summary_statistics(arr, combine_data=False, combine_histograms=False)
        else:
            values = np.array(arr).ravel()
    elif isinstance(arr, np.ndarray):
        values = arr.ravel()
    elif isinstance(arr, torch.Tensor):
        values = arr.ravel()
    else:
        raise ValueError(arr)

    if isinstance(values, np.ndarray):
        n = values.size
    else:
        n = values.numel()

    if n == 0:
        return _no_data_stats(
            find_min=find_min,
            find_max=find_max,
            make_histogram=make_histogram,
            compute_skewness=compute_skewness,
            compute_kurtosis=compute_kurtosis,
        )

    mean = values.mean().item()

    if n == 1:
        summary_stats = SummaryStatistics(
            n=n,
            mean=mean,
            std=0.0,
            skewness=0.0 if compute_skewness else None,
            kurtosis=0.0 if compute_kurtosis else None,
            min_value=mean if find_min else None,
            max_value=mean if find_max else None,
        )
        if make_histogram:
            summary_stats.histogram = _compute_histogram(
                values,
                min_val=summary_stats.min_value,
                max_val=summary_stats.max_value,
                n_bins=HISTOGRAM_DEFAULT_BINS if isinstance(make_histogram, bool) else make_histogram,
            )
        return summary_stats

    summary_stats: SummaryStatistics = SummaryStatistics(
        n=n,
        mean=mean,
        std=values.std().item(),
    )
    if compute_skewness or compute_kurtosis:
        skewness, kurtosis = _compute_standardized_moments(
            values,
            mean,
            summary_stats.std,
            compute_skewness=compute_skewness,
            compute_kurtosis=compute_kurtosis,
        )
        if compute_skewness:
            summary_stats.skewness = skewness if skewness is not None else NO_DATA
        if compute_kurtosis:
            summary_stats.kurtosis = kurtosis if kurtosis is not None else NO_DATA
    find_min = find_min or make_histogram
    find_max = find_max or make_histogram
    if find_min:
        summary_stats.min_value = values.min().item()
    if find_max:
        summary_stats.max_value = values.max().item()

    if keep_data:
        if isinstance(values, np.ndarray):
            summary_stats.data = values
        else:
            summary_stats.data = values.detach().cpu().numpy()

    if make_histogram:
        if summary_stats.std > 1e-6:
            summary_stats.histogram = _compute_histogram(
                values if isinstance(values, np.ndarray) else values.detach().cpu().numpy(),
                min_val=summary_stats.min_value,
                max_val=summary_stats.max_value,
                n_bins=HISTOGRAM_DEFAULT_BINS if isinstance(make_histogram, bool) else make_histogram,
            )
        else:
            summary_stats.histogram = _compute_histogram(
                values if isinstance(values, np.ndarray) else values.detach().cpu().numpy(),
                min_val=summary_stats.min_value,
                max_val=summary_stats.max_value,
                n_bins=1,
            )

    return summary_stats

def maybe_compute_summary_statistics(
        x: SummaryStatisticsInput | SummaryStatistics | None,
        find_min: bool = False,
        find_max: bool = False,
        make_histogram: bool | int = False,
        compute_skewness: bool = False,
        compute_kurtosis: bool = False,
        keep_data: bool = True,
) -> Optional[SummaryStatistics]:
    if is_summary_statistics(x):
        if make_histogram and x.histogram in (None, NO_DATA) and x.data is not None:
            compute_histogram(x, n_bins=make_histogram if isinstance(make_histogram, int) else HISTOGRAM_DEFAULT_BINS)
        if compute_skewness or compute_kurtosis:
            needs_skewness = compute_skewness and x.skewness in (None, NO_DATA)
            needs_kurtosis = compute_kurtosis and x.kurtosis in (None, NO_DATA)
            if x.data is None:
                if needs_skewness and x.skewness is None:
                    x.skewness = NO_DATA
                if needs_kurtosis and x.kurtosis is None:
                    x.kurtosis = NO_DATA
            elif needs_skewness or needs_kurtosis:
                skewness, kurtosis = _compute_standardized_moments(
                    x.data,
                    x.mean,
                    x.std,
                    compute_skewness=needs_skewness,
                    compute_kurtosis=needs_kurtosis,
                )
                if needs_skewness:
                    x.skewness = skewness if skewness is not None else NO_DATA
                if needs_kurtosis:
                    x.kurtosis = kurtosis if kurtosis is not None else NO_DATA
        return x
    if x is None:
        return _no_data_stats(
            find_min=find_min,
            find_max=find_max,
            make_histogram=make_histogram,
            compute_skewness=compute_skewness,
            compute_kurtosis=compute_kurtosis,
        )
    return compute_summary_statistics(
        x,
        find_min=find_min,
        find_max=find_max,
        make_histogram=make_histogram,
        compute_skewness=compute_skewness,
        compute_kurtosis=compute_kurtosis,
        keep_data=keep_data,
    )
    
def compute_histogram(stats: SummaryStatistics, n_bins: int = HISTOGRAM_DEFAULT_BINS) -> None:
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

def _compute_standardized_moments(
    values: np.ndarray | torch.Tensor,
    mean: float,
    std: Optional[float],
    compute_skewness: bool = True,
    compute_kurtosis: bool = True,
) -> tuple[Optional[float], Optional[float]]:
    if std is None or not np.isfinite(std) or std <= 0.0:
        return None, None

    if isinstance(values, np.ndarray):
        centered = values - mean
        skewness = None
        kurtosis = None
        if compute_skewness:
            m3 = np.mean(centered ** 3)
            skewness = float(m3 / (std ** 3))
        if compute_kurtosis:
            m4 = np.mean(centered ** 4)
            kurtosis = float(m4 / (std ** 4))
    else:
        centered = values - mean
        skewness = None
        kurtosis = None
        if compute_skewness:
            m3 = (centered ** 3).mean()
            skewness = (m3 / (std ** 3)).item()
        if compute_kurtosis:
            m4 = (centered ** 4).mean()
            kurtosis = (m4 / (std ** 4)).item()

    if skewness is not None and not np.isfinite(skewness):
        skewness = None
    if kurtosis is not None and not np.isfinite(kurtosis):
        kurtosis = None

    return skewness, kurtosis

def _format_histogram_blocks(freqs: list[float], n_bins: Optional[int] = None) -> str:
    if not freqs:
        return "[]"

    frequencies = np.asarray(freqs, dtype=np.float64)
    if n_bins is not None:
        target_bins = max(1, int(n_bins))
        if target_bins < frequencies.size:
            source_edges = np.arange(frequencies.size + 1, dtype=np.float64)
            source_cdf = np.concatenate(([0.0], np.cumsum(frequencies)))
            target_edges = np.linspace(0.0, float(frequencies.size), target_bins + 1)
            target_cdf = np.interp(target_edges, source_edges, source_cdf)
            frequencies = np.diff(target_cdf)

    max_freq = float(np.max(frequencies))
    if not np.isfinite(max_freq) or max_freq <= 0.0:
        return "[" + (" " * int(frequencies.size)) + "]"

    levels = len(HISTOGRAM_BLOCKS) - 1
    chars: list[str] = []
    for f in frequencies:
        if not np.isfinite(f) or f <= 0.0:
            idx = 0
        else:
            idx = int(round(levels * (f / max_freq)))
            idx = max(0, min(levels, idx))
        chars.append(HISTOGRAM_BLOCKS[idx])
    return "[" + "".join(chars) + "]"

def combine_summary_statistics(
    stats_list: list[SummaryStatistics],
    combine_data: bool = False,
    combine_histograms: bool = False,
) -> SummaryStatistics:
    if not stats_list:
        raise ValueError("stats_list must not be empty.")

    stats_with_data = [s for s in stats_list if int(s.n) > 0]
    total_n = sum(int(s.n) for s in stats_with_data)
    if total_n <= 0:
        raise ValueError(f"Total n must be > 0, got {total_n}.")

    invalid_mean = [s for s in stats_with_data if s.mean is NO_DATA]
    if invalid_mean:
        raise ValueError("SummaryStatistics with n>0 must have mean.")
    combined_mean = sum(float(s.mean) * int(s.n) for s in stats_with_data) / total_n

    if total_n == 1:
        combined_std: Optional[float] = 0.0
    else:
        sum_weighted_second_moment = 0.0
        for s in stats_with_data:
            n_i = int(s.n)
            if n_i > 1 and s.std in (None, NO_DATA):
                raise ValueError("SummaryStatistics with n>1 must have std.")
            var_i = float(s.std) ** 2 if (s.std is not None and s.std is not NO_DATA and n_i > 1) else 0.0
            mean_delta = (float(s.mean) - combined_mean) if s.mean is not NO_DATA else 0.0
            sum_weighted_second_moment += n_i * (var_i + (mean_delta * mean_delta))
        combined_var = sum_weighted_second_moment / total_n
        combined_std = np.sqrt(combined_var)

    combined = SummaryStatistics(
        n=total_n,
        mean=combined_mean,
        std=combined_std,
    )

    min_values = [s.min_value for s in stats_with_data if s.min_value is not None and s.min_value is not NO_DATA]
    if stats_with_data and len(min_values) == len(stats_with_data):
        combined.min_value = min(min_values)

    max_values = [s.max_value for s in stats_with_data if s.max_value is not None and s.max_value is not NO_DATA]
    if stats_with_data and len(max_values) == len(stats_with_data):
        combined.max_value = max(max_values)

    if combine_data and stats_with_data and all(s.data is not None for s in stats_with_data):
        combined.data = np.concatenate([s.data for s in stats_with_data]).ravel()
        skewness, kurtosis = _compute_standardized_moments(combined.data, combined.mean, combined.std)
        combined.skewness = skewness if skewness is not None else NO_DATA
        combined.kurtosis = kurtosis if kurtosis is not None else NO_DATA

    if combine_histograms:
        raise NotImplementedError("Combining histograms is not implemented yet.")

    return combined

def _no_data_stats(
        find_min: bool = False,
        find_max: bool = False,
        make_histogram: bool | int = False,
        compute_skewness: bool = False,
        compute_kurtosis: bool = False,
) -> SummaryStatistics:
    return SummaryStatistics(
        n=0,
        mean=NO_DATA,
        std=NO_DATA,
        skewness=NO_DATA if compute_skewness else None,
        kurtosis=NO_DATA if compute_kurtosis else None,
        min_value=NO_DATA if find_min else None,
        max_value=NO_DATA if find_max else None,
        histogram=NO_DATA if make_histogram else None
    )
