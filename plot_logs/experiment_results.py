from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from plot_logs.plot_logs import is_supported_log_path, open_log_text, parse_scalar, parse_x_value


DEFAULT_X_COLUMN = "timesteps"
EP_REW_EMA_COLUMN = "ep_rew_ema"
EP_SUCCESS_RATE_EMA_COLUMN = "ep_success_rate_ema"
DEFAULT_DPI = 300
DEFAULT_RUN_LENGTH_LIMIT = 100_000_000
GROUP_PALETTE: tuple[str, ...] = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
)
INDIVIDUAL_RUN_LINE_WIDTH = 0.8
INDIVIDUAL_RUN_ALPHA = 0.9
SHORT_RUN_MARKER_SIZE = 24
SHORT_RUN_MARKER_EDGE_WIDTH = 0.6
GROUP_LINE_WIDTH = 1.0
THEORETICAL_MAXIMUM_LINE_WIDTH = 1.0
THEORETICAL_MAXIMUM_COLOR = "#444444"
THEORETICAL_MAXIMUM_LABEL = "Theoretical maximum"


@dataclass(slots=True)
class ExperimentRunLog:
    group_name: str
    run_name: str
    path: Path
    x_values: np.ndarray
    series: dict[str, np.ndarray]


@dataclass(slots=True)
class ExperimentGroup:
    name: str
    display_name: str
    runs: list[ExperimentRunLog]


@dataclass(slots=True)
class ExperimentPlotResult:
    groups: list[ExperimentGroup]
    output_paths: list[Path]


@dataclass(frozen=True, slots=True)
class ExperimentPlotSelection:
    name: str
    group_names: tuple[str, ...]
    title_suffix: str | None = None
    required_group_names: tuple[str, ...] = ()
    output_subdir: str | None = None


@dataclass(frozen=True, slots=True)
class MetricPlotSpec:
    column: str
    output_stem: str
    title: str
    ylabel: str


RETURN_EMA_PLOT = MetricPlotSpec(
    column=EP_REW_EMA_COLUMN,
    output_stem="ep_rew_ema",
    title="Episode Reward EMA",
    ylabel=EP_REW_EMA_COLUMN,
)
SUCCESS_RATE_EMA_PLOT = MetricPlotSpec(
    column=EP_SUCCESS_RATE_EMA_COLUMN,
    output_stem=EP_SUCCESS_RATE_EMA_COLUMN,
    title="Episode Success Rate EMA",
    ylabel="Success rate (%)",
)
PLOT_SPECS: tuple[MetricPlotSpec, ...] = (
    RETURN_EMA_PLOT,
    SUCCESS_RATE_EMA_PLOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot grouped and individual experiment return EMA curves from an experiment run folder. "
            "The expected layout is EXPERIMENT_RUN_DIR/GROUP/RUN/log.csv[.gz|.bz2|.xz|.zip]."
        )
    )
    parser.add_argument(
        "experiment_run_dir",
        type=Path,
        help="Path to the run folder containing one subdirectory per group.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated PNG files.",
    )
    parser.add_argument(
        "--group-order",
        nargs="*",
        default=None,
        help="Optional group label order. Groups not listed here are appended alphabetically.",
    )
    parser.add_argument(
        "--x-column",
        default=DEFAULT_X_COLUMN,
        help=f"X-axis column. Defaults to {DEFAULT_X_COLUMN!r}.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"PNG DPI. Defaults to {DEFAULT_DPI}.",
    )
    parser.add_argument(
        "--theoretical-maximum",
        type=float,
        default=None,
        help="Optional horizontal theoretical maximum line to draw on generated plots.",
    )
    parser.add_argument(
        "--run-length-limit",
        type=int,
        default=DEFAULT_RUN_LENGTH_LIMIT,
        help=f"Expected individual run length. Defaults to {DEFAULT_RUN_LENGTH_LIMIT}.",
    )
    return parser.parse_args()


def find_log_file(run_dir: Path) -> Path | None:
    preferred_names = (
        "log.csv",
        "log.csv.gz",
        "log.csv.bz2",
        "log.csv.xz",
        "log.csv.zip",
        "log.zip",
    )
    for name in preferred_names:
        candidate = run_dir / name
        if is_supported_log_path(candidate):
            return candidate

    candidates = sorted(
        path
        for path in run_dir.iterdir()
        if path.is_file() and path.stem.startswith("log") and is_supported_log_path(path)
    )
    return candidates[0] if candidates else None


def iter_group_log_files(group_dir: Path) -> Iterable[tuple[str, Path]]:
    direct_log = find_log_file(group_dir)
    if direct_log is not None:
        yield group_dir.name, direct_log

    for run_dir in sorted((path for path in group_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
        log_path = find_log_file(run_dir)
        if log_path is not None:
            yield run_dir.name, log_path


def iter_source_log_files(source_path: Path) -> Iterable[tuple[str, Path]]:
    if source_path.is_file():
        if not is_supported_log_path(source_path):
            raise ValueError(f"Unsupported log path: {source_path}")
        yield source_path.parent.name, source_path
        return

    if not source_path.is_dir():
        raise FileNotFoundError(source_path)

    yield from iter_group_log_files(source_path)


def resolve_source_path(source_path: Path, *, experiment_run_dir: Path) -> Path:
    expanded_path = source_path.expanduser()
    if expanded_path.is_absolute():
        return expanded_path.resolve()
    return (experiment_run_dir / expanded_path).resolve()


def ordered_group_names(discovered_names: Iterable[str], group_order: Sequence[str] | None) -> list[str]:
    discovered = set(discovered_names)
    ordered: list[str] = []
    if group_order is not None:
        for name in group_order:
            if name in discovered and name not in ordered:
                ordered.append(name)
    ordered.extend(name for name in sorted(discovered) if name not in ordered)
    return ordered


def load_experiment_run_log(
    *,
    path: Path,
    group_name: str,
    run_name: str,
    x_column: str,
    return_columns: Sequence[str],
    optional_return_columns: Sequence[str] = (),
) -> ExperimentRunLog:
    x_values: list[float] = []
    series: dict[str, list[float]] = {column: [] for column in return_columns}
    with open_log_text(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")

        missing_return_columns = [column for column in (x_column, *return_columns) if column not in reader.fieldnames]
        if missing_return_columns:
            missing = ", ".join(missing_return_columns)
            raise ValueError(f"Missing required columns in {path}: {missing}")
        present_optional_return_columns = [column for column in optional_return_columns if column in reader.fieldnames]
        missing_optional_return_columns = [column for column in optional_return_columns if column not in reader.fieldnames]
        for column in present_optional_return_columns:
            series[column] = []

        for row_index, row in enumerate(reader, start=2):
            x_value, _x_is_datetime = parse_x_value(row.get(x_column), x_column, path, row_index)
            if math.isnan(x_value):
                continue
            x_values.append(x_value)
            for column in series:
                series[column].append(parse_scalar(row.get(column), column, path, row_index))

    if missing_optional_return_columns:
        for column in missing_optional_return_columns:
            series[column] = [float("nan")] * len(x_values)

    if not x_values:
        raise ValueError(f"{path} contains no usable rows")

    arrays = {column: np.asarray(values, dtype=float) for column, values in series.items()}
    return ExperimentRunLog(
        group_name=group_name,
        run_name=run_name,
        path=path,
        x_values=np.asarray(x_values, dtype=float),
        series=arrays,
    )


def load_experiment_groups(
    experiment_run_dir: Path,
    *,
    group_order: Sequence[str] | None = None,
    x_column: str = DEFAULT_X_COLUMN,
    display_name_overrides: Mapping[str, str] | None = None,
    extra_group_sources: Mapping[str, Sequence[Path]] | None = None,
) -> list[ExperimentGroup]:
    experiment_run_dir = experiment_run_dir.expanduser().resolve()
    if not experiment_run_dir.is_dir():
        raise NotADirectoryError(experiment_run_dir)

    grouped_logs: dict[str, list[tuple[str, Path]]] = {}
    for group_dir in sorted((path for path in experiment_run_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
        run_logs = list(iter_group_log_files(group_dir))
        if run_logs:
            grouped_logs[group_dir.name] = run_logs

    if extra_group_sources is not None:
        for group_name, source_paths in extra_group_sources.items():
            group_logs = grouped_logs.setdefault(group_name, [])
            initial_group_log_count = len(group_logs)
            seen_log_paths = {path.resolve() for _run_name, path in group_logs}
            for source_path in source_paths:
                resolved_source_path = resolve_source_path(source_path, experiment_run_dir=experiment_run_dir)
                if not resolved_source_path.exists():
                    continue
                for run_name, log_path in iter_source_log_files(resolved_source_path):
                    resolved_log_path = log_path.resolve()
                    if resolved_log_path in seen_log_paths:
                        continue
                    group_logs.append((run_name, log_path))
                    seen_log_paths.add(resolved_log_path)
            if len(group_logs) == initial_group_log_count == 0:
                grouped_logs.pop(group_name, None)
                continue
            group_logs.sort(key=lambda item: (item[0], item[1].as_posix()))

    if not grouped_logs:
        raise ValueError(f"No run logs found under {experiment_run_dir}")

    # if display_name_overrides is not None:
    #     unknown_names = sorted(name for name in display_name_overrides if name not in grouped_logs)
    #     if unknown_names:
    #         unknown_names_display = ", ".join(unknown_names)
    #         raise ValueError(f"display_name_overrides contains unknown group names: {unknown_names_display}")

    return_columns = (EP_REW_EMA_COLUMN,)
    optional_return_columns = (EP_SUCCESS_RATE_EMA_COLUMN,)
    groups: list[ExperimentGroup] = []
    for group_name in ordered_group_names(grouped_logs, group_order):
        runs = [
            load_experiment_run_log(
                path=path,
                group_name=group_name,
                run_name=run_name,
                x_column=x_column,
                return_columns=return_columns,
                optional_return_columns=optional_return_columns,
            )
            for run_name, path in grouped_logs[group_name]
        ]
        groups.append(
            ExperimentGroup(
                name=group_name,
                display_name=display_name_overrides.get(group_name, group_name)
                if display_name_overrides is not None
                else group_name,
                runs=runs,
            )
        )
    return groups


def group_label(group: ExperimentGroup) -> str:
    return f"{group.display_name} (n={len(group.runs)})"


def group_colors(groups: Sequence[ExperimentGroup]) -> dict[str, tuple[float, float, float, float]]:
    fallback_cmap = plt.get_cmap("tab10")
    return {
        group.name: matplotlib.colors.to_rgba(
            GROUP_PALETTE[index] if index < len(GROUP_PALETTE) else fallback_cmap(index % fallback_cmap.N)
        )
        for index, group in enumerate(groups)
    }


def selected_groups_by_name(
    groups: Sequence[ExperimentGroup],
    group_names: Sequence[str],
) -> list[ExperimentGroup]:
    groups_by_name = {group.name: group for group in groups}
    return [groups_by_name[name] for name in group_names if name in groups_by_name]


def add_theoretical_maximum_line(axis: Axes, value: float | None) -> Line2D | None:
    if value is None:
        return None
    return axis.axhline(
        value,
        color=THEORETICAL_MAXIMUM_COLOR,
        linestyle="--",
        linewidth=THEORETICAL_MAXIMUM_LINE_WIDTH,
        label=THEORETICAL_MAXIMUM_LABEL,
    )


def add_group_legend(
    axis: Axes,
    groups: Sequence[ExperimentGroup],
    colors: dict[str, tuple[float, float, float, float]],
    *,
    theoretical_maximum_line: Line2D | None = None,
) -> None:
    handles = [
        Line2D([0], [0], color=colors[group.name], lw=GROUP_LINE_WIDTH, label=group_label(group))
        for group in groups
    ]
    if theoretical_maximum_line is not None:
        handles.append(theoretical_maximum_line)
    axis.legend(handles=handles, loc="best")


def save_figure(figure: Figure, output_path: Path, *, dpi: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def final_finite_run_point(run: ExperimentRunLog, column: str) -> tuple[float, float] | None:
    y_values = run.series[column]
    finite_mask = np.isfinite(run.x_values) & np.isfinite(y_values)
    finite_indices = np.flatnonzero(finite_mask)
    if finite_indices.size == 0:
        return None
    final_index = int(finite_indices[-1])
    return float(run.x_values[final_index]), float(y_values[final_index])


def metric_has_finite_values(groups: Sequence[ExperimentGroup], column: str) -> bool:
    return any(
        np.isfinite(run.series[column]).any()
        for group in groups
        for run in group.runs
    )


def plot_individual_metric(
    groups: Sequence[ExperimentGroup],
    output_dir: Path,
    *,
    x_column: str,
    dpi: int,
    theoretical_maximum: float | None = None,
    run_length_limit: int = DEFAULT_RUN_LENGTH_LIMIT,
    metric: MetricPlotSpec,
    output_name: str | None = None,
    title: str | None = None,
    colors: dict[str, tuple[float, float, float, float]] | None = None,
) -> Path:
    colors = group_colors(groups) if colors is None else colors
    figure, axis = plt.subplots(figsize=(16, 9))
    short_run_threshold = 0.99 * run_length_limit
    for group in groups:
        color = colors[group.name]
        for run in group.runs:
            axis.plot(
                run.x_values,
                run.series[metric.column],
                color=color,
                alpha=INDIVIDUAL_RUN_ALPHA,
                linewidth=INDIVIDUAL_RUN_LINE_WIDTH,
            )
            if run.x_values[-1] < short_run_threshold:
                final_point = final_finite_run_point(run, metric.column)
                if final_point is not None:
                    axis.scatter(
                        *final_point,
                        color=color,
                        edgecolors="black",
                        linewidths=SHORT_RUN_MARKER_EDGE_WIDTH,
                        s=SHORT_RUN_MARKER_SIZE,
                        zorder=3,
                    )

    axis.set_title(title or f"{metric.title} Per Run")
    axis.set_xlabel(x_column)
    axis.set_ylabel(metric.ylabel)
    axis.grid(alpha=0.25)
    theoretical_maximum_line = add_theoretical_maximum_line(axis, theoretical_maximum)
    add_group_legend(axis, groups, colors, theoretical_maximum_line=theoretical_maximum_line)
    figure.tight_layout()
    resolved_output_name = output_name or f"{metric.output_stem}_individual_runs.png"
    return save_figure(figure, output_dir / resolved_output_name, dpi=dpi)


def finite_interp(x_source: np.ndarray, y_source: np.ndarray, x_target: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(x_source) & np.isfinite(y_source)
    if finite_mask.sum() < 2:
        return np.full_like(x_target, np.nan, dtype=float)
    x_finite = x_source[finite_mask]
    y_finite = y_source[finite_mask]
    order = np.argsort(x_finite)
    x_sorted = x_finite[order]
    y_sorted = y_finite[order]
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    unique_y = y_sorted[unique_indices]
    result = np.full_like(x_target, np.nan, dtype=float)
    in_range = (x_target >= unique_x[0]) & (x_target <= unique_x[-1])
    result[in_range] = np.interp(x_target[in_range], unique_x, unique_y)
    return result


def nan_mean(values: np.ndarray, axis: int) -> np.ndarray:
    finite_mask = np.isfinite(values)
    counts = finite_mask.sum(axis=axis)
    totals = np.where(finite_mask, values, 0.0).sum(axis=axis)
    return np.divide(totals, counts, out=np.full_like(totals, np.nan, dtype=float), where=counts > 0)


def nan_std(values: np.ndarray, mean_values: np.ndarray, axis: int) -> np.ndarray:
    finite_mask = np.isfinite(values)
    expanded_mean = np.expand_dims(mean_values, axis=axis)
    counts = finite_mask.sum(axis=axis)
    squared_deltas = np.where(finite_mask, np.square(values - expanded_mean), 0.0).sum(axis=axis)
    variance = np.divide(
        squared_deltas,
        counts,
        out=np.full_like(squared_deltas, np.nan, dtype=float),
        where=counts > 0,
    )
    return np.sqrt(variance)


def group_x_values(runs: Sequence[ExperimentRunLog]) -> np.ndarray:
    values = sorted({
        float(value)
        for run in runs
        for value in run.x_values
        if np.isfinite(value)
    })
    return np.asarray(values, dtype=float)


def group_metric_mean_and_std(
    runs: Sequence[ExperimentRunLog],
    *,
    column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = group_x_values(runs)
    if x_values.size == 0:
        return x_values, x_values, x_values

    run_emas = np.vstack([
        finite_interp(run.x_values, run.series[column], x_values)
        for run in runs
    ])
    mean_values = nan_mean(run_emas, axis=0)
    return x_values, mean_values, nan_std(run_emas, mean_values, axis=0)


def plot_group_metric(
    groups: Sequence[ExperimentGroup],
    output_dir: Path,
    *,
    x_column: str,
    dpi: int,
    theoretical_maximum: float | None = None,
    metric: MetricPlotSpec,
    output_name: str | None = None,
    title: str | None = None,
    colors: dict[str, tuple[float, float, float, float]] | None = None,
) -> Path:
    colors = group_colors(groups) if colors is None else colors
    figure, axis = plt.subplots(figsize=(16, 9))
    for group in groups:
        x_values, mean_values, std_values = group_metric_mean_and_std(group.runs, column=metric.column)
        if x_values.size == 0:
            continue
        color = colors[group.name]
        axis.plot(x_values, mean_values, color=color, linewidth=GROUP_LINE_WIDTH, label=group_label(group))
        if len(group.runs) > 1:
            axis.fill_between(
                x_values,
                mean_values - std_values,
                mean_values + std_values,
                color=color,
                alpha=0.12,
                linewidth=0,
            )

    axis.set_title(title or f"{metric.title} By Group")
    axis.set_xlabel(x_column)
    axis.set_ylabel(metric.ylabel)
    axis.grid(alpha=0.25)
    add_theoretical_maximum_line(axis, theoretical_maximum)
    axis.legend(loc="best")
    figure.tight_layout()
    resolved_output_name = output_name or f"{metric.output_stem}_grouped.png"
    return save_figure(figure, output_dir / resolved_output_name, dpi=dpi)


def plot_experiment_selection(
    *,
    selection: ExperimentPlotSelection,
    groups: Sequence[ExperimentGroup],
    output_dir: Path,
    x_column: str,
    dpi: int,
    theoretical_maximum: float | None,
    run_length_limit: int,
    colors: dict[str, tuple[float, float, float, float]],
) -> list[Path]:
    selected_groups = selected_groups_by_name(groups, selection.group_names)
    selected_group_names = {group.name for group in selected_groups}
    if any(group_name not in selected_group_names for group_name in selection.required_group_names):
        return []
    if len(selected_groups) < 2:
        return []

    selection_output_dir = output_dir
    if selection.output_subdir is not None:
        selection_output_dir /= selection.output_subdir

    title_suffix = f" - {selection.title_suffix or selection.name.replace('_', ' ').title()}"
    output_paths: list[Path] = []
    for metric in PLOT_SPECS:
        if not metric_has_finite_values(selected_groups, metric.column):
            continue
        output_stem = f"{metric.output_stem}_{selection.name}"
        output_paths.extend([
            plot_individual_metric(
                selected_groups,
                selection_output_dir,
                x_column=x_column,
                dpi=dpi,
                theoretical_maximum=theoretical_maximum,
                run_length_limit=run_length_limit,
                metric=metric,
                output_name=f"{output_stem}_individual_runs.png",
                title=f"{metric.title} Per Run{title_suffix}",
                colors=colors,
            ),
            plot_group_metric(
                selected_groups,
                selection_output_dir,
                x_column=x_column,
                dpi=dpi,
                theoretical_maximum=theoretical_maximum,
                metric=metric,
                output_name=f"{output_stem}_grouped.png",
                title=f"{metric.title} By Group{title_suffix}",
                colors=colors,
            ),
        ])
    return output_paths


def plot_experiment_results(
    experiment_run_dir: Path,
    output_dir: Path,
    *,
    group_order: Sequence[str] | None = None,
    x_column: str = DEFAULT_X_COLUMN,
    dpi: int = DEFAULT_DPI,
    theoretical_maximum: float | None = None,
    run_length_limit: int = DEFAULT_RUN_LENGTH_LIMIT,
    display_name_overrides: Mapping[str, str] | None = None,
    extra_group_sources: Mapping[str, Sequence[Path]] | None = None,
    extra_plot_selections: Sequence[ExperimentPlotSelection] | None = None,
) -> ExperimentPlotResult:
    groups = load_experiment_groups(
        experiment_run_dir,
        group_order=group_order,
        x_column=x_column,
        display_name_overrides=display_name_overrides,
        extra_group_sources=extra_group_sources,
    )
    output_dir = output_dir.expanduser().resolve()
    colors = group_colors(groups)
    output_paths: list[Path] = []
    for metric in PLOT_SPECS:
        if not metric_has_finite_values(groups, metric.column):
            continue
        output_paths.extend([
            plot_individual_metric(
                groups,
                output_dir,
                x_column=x_column,
                dpi=dpi,
                theoretical_maximum=theoretical_maximum,
                run_length_limit=run_length_limit,
                metric=metric,
                colors=colors,
            ),
            plot_group_metric(
                groups,
                output_dir,
                x_column=x_column,
                dpi=dpi,
                theoretical_maximum=theoretical_maximum,
                metric=metric,
                colors=colors,
            ),
        ])
    if extra_plot_selections is not None:
        for selection in extra_plot_selections:
            output_paths.extend(
                plot_experiment_selection(
                    selection=selection,
                    groups=groups,
                    output_dir=output_dir,
                    x_column=x_column,
                    dpi=dpi,
                    theoretical_maximum=theoretical_maximum,
                    run_length_limit=run_length_limit,
                    colors=colors,
                )
            )
    return ExperimentPlotResult(groups=groups, output_paths=output_paths)


def main() -> int:
    args = parse_args()
    result = plot_experiment_results(
        args.experiment_run_dir,
        args.output_dir,
        group_order=args.group_order,
        x_column=args.x_column,
        dpi=args.dpi,
        theoretical_maximum=args.theoretical_maximum,
        run_length_limit=args.run_length_limit,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
