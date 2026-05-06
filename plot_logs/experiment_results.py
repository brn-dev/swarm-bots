from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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
EP_REW_MEAN_COLUMN = "ep_rew__mean"
EP_REW_STD_COLUMN = "ep_rew__std"
DEFAULT_DPI = 300
DEFAULT_FINAL_LOSS_COLUMNS: tuple[str, ...] = (
    "act_loss__mean",
    "val_loss__mean",
    "val_loss_scaled__mean",
    "world_model_loss_scaled__mean",
    "wm_loss__mean",
    "wm_loss_scaled__mean",
    "scalar_loss_scaled__mean",
    "angle_loss_scaled__mean",
    "rot6d_loss_scaled__mean",
    "binary_loss_scaled__mean",
)
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
GROUP_LINE_WIDTH = 1.0


@dataclass(slots=True)
class ExperimentRunLog:
    group_name: str
    run_name: str
    path: Path
    x_values: np.ndarray
    series: dict[str, np.ndarray]
    final_values: dict[str, float]


@dataclass(slots=True)
class ExperimentGroup:
    name: str
    runs: list[ExperimentRunLog]


@dataclass(slots=True)
class ExperimentPlotResult:
    groups: list[ExperimentGroup]
    output_paths: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot grouped experiment return curves and final losses from an experiment run folder. "
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
    final_loss_columns: Sequence[str],
) -> ExperimentRunLog:
    x_values: list[float] = []
    series: dict[str, list[float]] = {column: [] for column in (*return_columns, *final_loss_columns)}
    with open_log_text(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")

        missing_return_columns = [column for column in (x_column, *return_columns) if column not in reader.fieldnames]
        if missing_return_columns:
            missing = ", ".join(missing_return_columns)
            raise ValueError(f"Missing required columns in {path}: {missing}")

        available_loss_columns = [column for column in final_loss_columns if column in reader.fieldnames]
        series = {column: [] for column in (*return_columns, *available_loss_columns)}

        for row_index, row in enumerate(reader, start=2):
            x_value, _x_is_datetime = parse_x_value(row.get(x_column), x_column, path, row_index)
            if math.isnan(x_value):
                continue
            x_values.append(x_value)
            for column in series:
                series[column].append(parse_scalar(row.get(column), column, path, row_index))

    if not x_values:
        raise ValueError(f"{path} contains no usable rows")

    arrays = {column: np.asarray(values, dtype=float) for column, values in series.items()}
    final_values = {
        column: float(finite_values[-1])
        for column, values in arrays.items()
        if column in final_loss_columns
        for finite_values in [values[np.isfinite(values)]]
        if finite_values.size > 0
    }
    return ExperimentRunLog(
        group_name=group_name,
        run_name=run_name,
        path=path,
        x_values=np.asarray(x_values, dtype=float),
        series=arrays,
        final_values=final_values,
    )


def load_experiment_groups(
    experiment_run_dir: Path,
    *,
    group_order: Sequence[str] | None = None,
    x_column: str = DEFAULT_X_COLUMN,
    final_loss_columns: Sequence[str] = DEFAULT_FINAL_LOSS_COLUMNS,
) -> list[ExperimentGroup]:
    experiment_run_dir = experiment_run_dir.expanduser().resolve()
    if not experiment_run_dir.is_dir():
        raise NotADirectoryError(experiment_run_dir)

    grouped_logs: dict[str, list[tuple[str, Path]]] = {}
    for group_dir in sorted((path for path in experiment_run_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
        run_logs = list(iter_group_log_files(group_dir))
        if run_logs:
            grouped_logs[group_dir.name] = run_logs

    if not grouped_logs:
        raise ValueError(f"No run logs found under {experiment_run_dir}")

    return_columns = (EP_REW_EMA_COLUMN, EP_REW_MEAN_COLUMN, EP_REW_STD_COLUMN)
    groups: list[ExperimentGroup] = []
    for group_name in ordered_group_names(grouped_logs, group_order):
        runs = [
            load_experiment_run_log(
                path=path,
                group_name=group_name,
                run_name=run_name,
                x_column=x_column,
                return_columns=return_columns,
                final_loss_columns=final_loss_columns,
            )
            for run_name, path in grouped_logs[group_name]
        ]
        groups.append(ExperimentGroup(name=group_name, runs=runs))
    return groups


def group_label(group: ExperimentGroup) -> str:
    return f"{group.name} (n={len(group.runs)})"


def group_colors(groups: Sequence[ExperimentGroup]) -> dict[str, tuple[float, float, float, float]]:
    fallback_cmap = plt.get_cmap("tab10")
    return {
        group.name: matplotlib.colors.to_rgba(
            GROUP_PALETTE[index] if index < len(GROUP_PALETTE) else fallback_cmap(index % fallback_cmap.N)
        )
        for index, group in enumerate(groups)
    }


def add_group_legend(axis: Axes, groups: Sequence[ExperimentGroup], colors: dict[str, tuple[float, float, float, float]]) -> None:
    handles = [
        Line2D([0], [0], color=colors[group.name], lw=GROUP_LINE_WIDTH, label=group_label(group))
        for group in groups
    ]
    axis.legend(handles=handles, loc="best")


def save_figure(figure: Figure, output_path: Path, *, dpi: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_individual_ep_rew_ema(
    groups: Sequence[ExperimentGroup],
    output_dir: Path,
    *,
    x_column: str,
    dpi: int,
) -> Path:
    colors = group_colors(groups)
    figure, axis = plt.subplots(figsize=(16, 9))
    for group in groups:
        color = colors[group.name]
        for run in group.runs:
            axis.plot(
                run.x_values,
                run.series[EP_REW_EMA_COLUMN],
                color=color,
                alpha=INDIVIDUAL_RUN_ALPHA,
                linewidth=INDIVIDUAL_RUN_LINE_WIDTH,
            )

    axis.set_title("Episode Reward EMA Per Run")
    axis.set_xlabel(x_column)
    axis.set_ylabel(EP_REW_EMA_COLUMN)
    axis.grid(alpha=0.25)
    add_group_legend(axis, groups, colors)
    figure.tight_layout()
    return save_figure(figure, output_dir / "ep_rew_ema_individual_runs.png", dpi=dpi)


def plot_individual_ep_rew_mean_std(
    groups: Sequence[ExperimentGroup],
    output_dir: Path,
    *,
    x_column: str,
    dpi: int,
) -> Path:
    colors = group_colors(groups)
    figure, axis = plt.subplots(figsize=(16, 9))
    for group in groups:
        color = colors[group.name]
        for run in group.runs:
            mean_values = run.series[EP_REW_MEAN_COLUMN]
            std_values = run.series[EP_REW_STD_COLUMN]
            axis.plot(
                run.x_values,
                mean_values,
                color=color,
                alpha=INDIVIDUAL_RUN_ALPHA,
                linewidth=INDIVIDUAL_RUN_LINE_WIDTH,
            )
            axis.fill_between(
                run.x_values,
                mean_values - std_values,
                mean_values + std_values,
                color=color,
                alpha=0.045,
                linewidth=0,
            )

    axis.set_title("Episode Reward Mean +/- Std Per Run")
    axis.set_xlabel(x_column)
    axis.set_ylabel(EP_REW_MEAN_COLUMN)
    axis.grid(alpha=0.25)
    add_group_legend(axis, groups, colors)
    figure.tight_layout()
    return save_figure(figure, output_dir / "ep_rew_mean_std_individual_runs.png", dpi=dpi)


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


def group_mean_and_std(runs: Sequence[ExperimentRunLog]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = group_x_values(runs)
    if x_values.size == 0:
        return x_values, x_values, x_values

    run_means = np.vstack([
        finite_interp(run.x_values, run.series[EP_REW_MEAN_COLUMN], x_values)
        for run in runs
    ])
    run_stds = np.vstack([
        finite_interp(run.x_values, run.series[EP_REW_STD_COLUMN], x_values)
        for run in runs
    ])
    group_mean = nan_mean(run_means, axis=0)
    pooled_second_moment = nan_mean(np.square(run_stds) + np.square(run_means), axis=0)
    group_std = np.sqrt(np.maximum(pooled_second_moment - np.square(group_mean), 0.0))
    return x_values, group_mean, group_std


def group_ema_mean_and_std(runs: Sequence[ExperimentRunLog]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = group_x_values(runs)
    if x_values.size == 0:
        return x_values, x_values, x_values

    run_emas = np.vstack([
        finite_interp(run.x_values, run.series[EP_REW_EMA_COLUMN], x_values)
        for run in runs
    ])
    mean_values = nan_mean(run_emas, axis=0)
    return x_values, mean_values, nan_std(run_emas, mean_values, axis=0)


def plot_group_ep_rew_ema(
    groups: Sequence[ExperimentGroup],
    output_dir: Path,
    *,
    x_column: str,
    dpi: int,
) -> Path:
    colors = group_colors(groups)
    figure, axis = plt.subplots(figsize=(16, 9))
    for group in groups:
        x_values, mean_values, std_values = group_ema_mean_and_std(group.runs)
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

    axis.set_title("Episode Reward EMA By Group")
    axis.set_xlabel(x_column)
    axis.set_ylabel(EP_REW_EMA_COLUMN)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    return save_figure(figure, output_dir / "ep_rew_ema_grouped.png", dpi=dpi)


def plot_group_ep_rew_mean_std(
    groups: Sequence[ExperimentGroup],
    output_dir: Path,
    *,
    x_column: str,
    dpi: int,
) -> Path:
    colors = group_colors(groups)
    figure, axis = plt.subplots(figsize=(16, 9))
    for group in groups:
        x_values, mean_values, std_values = group_mean_and_std(group.runs)
        if x_values.size == 0:
            continue
        color = colors[group.name]
        axis.plot(x_values, mean_values, color=color, linewidth=GROUP_LINE_WIDTH, label=group_label(group))
        axis.fill_between(
            x_values,
            mean_values - std_values,
            mean_values + std_values,
            color=color,
            alpha=0.12,
            linewidth=0,
        )

    axis.set_title("Episode Reward Mean +/- Std By Group")
    axis.set_xlabel(x_column)
    axis.set_ylabel(EP_REW_MEAN_COLUMN)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    return save_figure(figure, output_dir / "ep_rew_mean_std_grouped.png", dpi=dpi)


def available_final_loss_columns(groups: Sequence[ExperimentGroup]) -> list[str]:
    available = {
        column
        for group in groups
        for run in group.runs
        for column in run.final_values
    }
    return [column for column in DEFAULT_FINAL_LOSS_COLUMNS if column in available]


def plot_final_losses(
    groups: Sequence[ExperimentGroup],
    output_dir: Path,
    *,
    dpi: int,
) -> Path | None:
    columns = available_final_loss_columns(groups)
    if not columns:
        return None

    ncols = 2 if len(columns) > 1 else 1
    nrows = math.ceil(len(columns) / ncols)
    figure, axes_array = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 4.5 * nrows), squeeze=False)
    axes = list(axes_array.ravel())
    colors = group_colors(groups)
    x_positions = np.arange(len(groups), dtype=float)
    x_labels = [group_label(group) for group in groups]

    for axis, column in zip(axes, columns, strict=False):
        for group_index, group in enumerate(groups):
            values = np.asarray(
                [run.final_values[column] for run in group.runs if column in run.final_values],
                dtype=float,
            )
            if values.size == 0:
                continue
            offsets = np.linspace(-0.16, 0.16, values.size) if values.size > 1 else np.asarray([0.0])
            axis.scatter(
                np.full(values.shape, x_positions[group_index]) + offsets,
                values,
                color=colors[group.name],
                alpha=0.75,
                s=35,
            )
            axis.scatter(
                [x_positions[group_index]],
                [float(np.nanmean(values))],
                color=colors[group.name],
                edgecolor="black",
                linewidth=0.8,
                marker="D",
                s=55,
                zorder=3,
            )
        axis.set_title(column)
        axis.set_xticks(x_positions, x_labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)

    for axis in axes[len(columns):]:
        axis.set_visible(False)

    figure.suptitle("Final Loss Values By Group")
    figure.tight_layout()
    return save_figure(figure, output_dir / "final_losses_by_group.png", dpi=dpi)


def plot_experiment_results(
    experiment_run_dir: Path,
    output_dir: Path,
    *,
    group_order: Sequence[str] | None = None,
    x_column: str = DEFAULT_X_COLUMN,
    dpi: int = DEFAULT_DPI,
) -> ExperimentPlotResult:
    groups = load_experiment_groups(
        experiment_run_dir,
        group_order=group_order,
        x_column=x_column,
    )
    output_dir = output_dir.expanduser().resolve()
    output_paths = [
        plot_individual_ep_rew_ema(groups, output_dir, x_column=x_column, dpi=dpi),
        plot_individual_ep_rew_mean_std(groups, output_dir, x_column=x_column, dpi=dpi),
        plot_group_ep_rew_ema(groups, output_dir, x_column=x_column, dpi=dpi),
        plot_group_ep_rew_mean_std(groups, output_dir, x_column=x_column, dpi=dpi),
    ]
    final_losses_path = plot_final_losses(groups, output_dir, dpi=dpi)
    if final_losses_path is not None:
        output_paths.append(final_losses_path)
    return ExperimentPlotResult(groups=groups, output_paths=output_paths)


def main() -> int:
    args = parse_args()
    result = plot_experiment_results(
        args.experiment_run_dir,
        args.output_dir,
        group_order=args.group_order,
        x_column=args.x_column,
        dpi=args.dpi,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
