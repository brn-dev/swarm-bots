"""Matplotlib figures for individual logs and comparisons across seeds."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from swarmbots.plotting.logs import (
    GroupSources,
    LogSources,
    RunLog,
    _source_paths,
    group_statistics,
    load_groups,
    load_log,
    log_columns,
    smooth_values,
)

PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000")
METRIC_LABELS = {"ep_rew_ema": "Episode return EMA", "ep_success_rate_ema": "Success rate EMA (%)"}


def _pyplot() -> ModuleType:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        if error.name == "matplotlib":
            raise ModuleNotFoundError(
                "Plotting requires Matplotlib; install with pip install 'swarmbots[plot]'"
            ) from error
        raise
    return plt


@dataclass(slots=True)
class ExperimentPlotResult:
    groups: dict[str, list[RunLog]]
    output_paths: list[Path]


def _selected_columns(runs: Sequence[RunLog], columns: Sequence[str] | None) -> tuple[str, ...]:
    selected = (
        tuple(columns) if columns is not None else tuple(dict.fromkeys(column for run in runs for column in run.series))
    )
    selected = tuple(
        column
        for column in selected
        if any(column in run.series and np.isfinite(run.series[column]).any() for run in runs)
    )
    if not selected:
        raise ValueError("No finite metric values to plot")
    return selected


def _make_figure(
    runs: Sequence[RunLog],
    columns: Sequence[str],
    *,
    title: str | None,
    ylabel_overrides: Mapping[str, str] | None,
    reference_values: Mapping[str, float] | None,
    font_size: float,
    figsize: tuple[float, float] | None,
) -> tuple[Figure, np.ndarray]:
    plt = _pyplot()
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter

    if len({(run.x_column, run.x_is_datetime) for run in runs}) > 1:
        raise ValueError("All runs must use the same X column and value type")
    figure, axes = plt.subplots(
        len(columns),
        1,
        sharex=True,
        squeeze=False,
        figsize=figsize or (10, 3.5 * len(columns)),
    )
    for axis, column in zip(axes[:, 0], columns, strict=True):
        axis.set_ylabel((ylabel_overrides or {}).get(column, METRIC_LABELS.get(column, column)), fontsize=font_size)
        axis.tick_params(labelsize=font_size)
        axis.grid(alpha=0.25)
        if reference_values and column in reference_values:
            axis.axhline(reference_values[column], color="#444444", linestyle="--", label="Reference")
    first_run = runs[0]
    if first_run.x_is_datetime:
        locator = mdates.AutoDateLocator()
        axes[-1, 0].xaxis.set_major_locator(locator)
        axes[-1, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        xlabel = first_run.x_column
    elif first_run.x_column == "timesteps":
        axes[-1, 0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:g}"))
        xlabel = "Environment steps (millions)"
    else:
        xlabel = first_run.x_column
    axes[-1, 0].set_xlabel(xlabel, fontsize=font_size)
    if title:
        figure.suptitle(title, fontsize=font_size + 2)
    return figure, axes[:, 0]


def _finish_figure(figure: Figure, *, font_size: float) -> Figure:
    for axis in figure.axes:
        axis.legend(fontsize=font_size)
    figure.tight_layout()
    return figure


def _save_figure(figure: Figure, output_path: str | Path, *, dpi: int) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_logs(
    sources: LogSources,
    *,
    columns: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    x_column: str = "timesteps",
    delimiter: str = ";",
    std_columns: Mapping[str, str] | None = None,
    smooth: float | None = None,
    max_steps: float | None = None,
    title: str | None = None,
    ylabel_overrides: Mapping[str, str] | None = None,
    reference_values: Mapping[str, float] | None = None,
    font_size: float = 11,
    figsize: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
    dpi: int = 200,
) -> Figure:
    """Plot one panel per scalar metric from files or run directories.

    Defaults to available return/success EMAs. ``std_columns`` maps metrics to
    logged standard deviations for shaded bands; these are distinct from the
    across-run standard deviations in grouped plots. ``smooth`` is an optional
    EMA alpha. The returned figure remains open for customization or plt.show().
    """
    plt = _pyplot()
    paths = _source_paths(sources)
    if labels is not None and len(labels) != len(paths):
        raise ValueError("labels must have one entry per resolved log file")
    runs: list[RunLog] = []
    for index, path in enumerate(paths):
        selected = columns
        if std_columns:
            available = log_columns(path, delimiter=delimiter)
            selected = (
                tuple(columns)
                if columns is not None
                else tuple(column for column in METRIC_LABELS if column in available)
            )
            selected = tuple(dict.fromkeys((*selected, *std_columns.values())))
        runs.append(
            load_log(
                path,
                columns=selected,
                x_column=x_column,
                delimiter=delimiter,
                max_steps=max_steps,
                label=labels[index] if labels is not None else None,
            )
        )
    if labels is None and len({run.label for run in runs}) != len(runs):
        for run in runs:
            run.label = run.path.as_posix()
    requested = columns
    if requested is None and std_columns:
        requested = tuple(column for column in METRIC_LABELS if any(column in run.series for run in runs))
    selected_columns = _selected_columns(runs, requested)
    figure, axes = _make_figure(
        runs,
        selected_columns,
        title=title,
        ylabel_overrides=ylabel_overrides,
        reference_values=reference_values,
        font_size=font_size,
        figsize=figsize,
    )
    try:
        for axis, column in zip(axes, selected_columns, strict=True):
            for run in runs:
                if column not in run.series or not np.isfinite(run.series[column]).any():
                    continue
                values = smooth_values(run.series[column], smooth)
                (line,) = axis.plot(run.x_values, values, label=run.label, marker="." if len(values) == 1 else None)
                if std_columns and column in std_columns:
                    std = smooth_values(run.series[std_columns[column]], smooth)
                    axis.fill_between(run.x_values, values - std, values + std, color=line.get_color(), alpha=0.2)
        _finish_figure(figure, font_size=font_size)
        if output_path is not None:
            _save_figure(figure, output_path, dpi=dpi)
    except BaseException:
        plt.close(figure)
        raise
    return figure


def plot_groups(
    groups: Mapping[str, Sequence[RunLog]],
    *,
    columns: Sequence[str] | None = None,
    individual_runs: bool = False,
    smooth: float | None = None,
    title: str | None = None,
    colors: Mapping[str, str] | None = None,
    linestyles: Mapping[str, str] | None = None,
    ylabel_overrides: Mapping[str, str] | None = None,
    reference_values: Mapping[str, float] | None = None,
    font_size: float = 11,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot loaded groups as mean ± population std, or as individual run curves.

    Groups keep their mapping order. Legends report the number of runs with
    finite values for each metric; group_statistics() exposes counts at each X.
    """
    plt = _pyplot()
    runs = [run for group_runs in groups.values() for run in group_runs]
    selected_columns = _selected_columns(runs, columns)
    figure, axes = _make_figure(
        runs,
        selected_columns,
        title=title,
        ylabel_overrides=ylabel_overrides,
        reference_values=reference_values,
        font_size=font_size,
        figsize=figsize,
    )
    try:
        for axis, column in zip(axes, selected_columns, strict=True):
            for index, (name, group_runs) in enumerate(groups.items()):
                metric_runs = [
                    run for run in group_runs if column in run.series and np.isfinite(run.series[column]).any()
                ]
                if not metric_runs:
                    continue
                color = (colors or {}).get(name, PALETTE[index % len(PALETTE)])
                style = (linestyles or {}).get(name, "-")
                label = f"{name} (n={len(metric_runs)})"
                if individual_runs:
                    for run_index, run in enumerate(metric_runs):
                        axis.plot(
                            run.x_values,
                            smooth_values(run.series[column], smooth),
                            color=color,
                            linestyle=style,
                            alpha=0.7,
                            linewidth=1,
                            marker="." if len(run.x_values) == 1 else None,
                            label=label if run_index == 0 else "_nolegend_",
                        )
                else:
                    stats = group_statistics(metric_runs, column, smooth=smooth)
                    axis.plot(
                        stats.x_values,
                        stats.mean,
                        color=color,
                        linestyle=style,
                        label=label,
                        marker="." if len(stats.x_values) == 1 else None,
                    )
                    if len(metric_runs) > 1:
                        axis.fill_between(
                            stats.x_values,
                            stats.mean - stats.std,
                            stats.mean + stats.std,
                            where=stats.counts > 1,
                            color=color,
                            alpha=0.15,
                            linewidth=0,
                        )
        return _finish_figure(figure, font_size=font_size)
    except BaseException:
        plt.close(figure)
        raise


def plot_experiment_results(
    sources: GroupSources,
    output_dir: str | Path,
    *,
    columns: Sequence[str] | None = None,
    x_column: str = "timesteps",
    delimiter: str = ";",
    max_steps: float | None = None,
    smooth: float | None = None,
    title: str | None = None,
    colors: Mapping[str, str] | None = None,
    linestyles: Mapping[str, str] | None = None,
    ylabel_overrides: Mapping[str, str] | None = None,
    reference_values: Mapping[str, float] | None = None,
    font_size: float = 11,
    dpi: int = 200,
    formats: Sequence[str] = ("png",),
) -> ExperimentPlotResult:
    """Save grouped and individual-run figures for each available metric.

    ``sources`` is either an experiment root or a mapping of display names to
    files/run directories. Figures are closed after saving. There is no implicit
    training-length cutoff. Use load_groups()/plot_groups() to customize figures.
    """
    plt = _pyplot()
    if not formats or any(fmt not in {"png", "pdf", "svg"} for fmt in formats):
        raise ValueError("formats must contain png, pdf, or svg")
    groups = load_groups(sources, columns=columns, x_column=x_column, delimiter=delimiter, max_steps=max_steps)
    selected_columns = _selected_columns([run for runs in groups.values() for run in runs], columns)
    stems = [re.sub(r"[^\w.-]+", "_", column) for column in selected_columns]
    if len(set(stems)) != len(stems):
        raise ValueError("Metric names produce conflicting output filenames")
    output_paths: list[Path] = []
    for column, stem in zip(selected_columns, stems, strict=True):
        for individual_runs in (False, True):
            mode = "individual_runs" if individual_runs else "grouped"
            figure = plot_groups(
                groups,
                columns=(column,),
                individual_runs=individual_runs,
                smooth=smooth,
                title=title,
                colors=colors,
                linestyles=linestyles,
                ylabel_overrides=ylabel_overrides,
                reference_values=reference_values,
                font_size=font_size,
            )
            try:
                for fmt in formats:
                    output_paths.append(_save_figure(figure, Path(output_dir) / f"{stem}_{mode}.{fmt}", dpi=dpi))
            finally:
                plt.close(figure)
    return ExperimentPlotResult(groups=groups, output_paths=output_paths)
