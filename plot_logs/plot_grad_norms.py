from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.dates as mdates

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import plot_logs


@dataclass(slots=True)
class GradNormLog:
    path: Path
    x_column: str
    x_values: list[float]
    x_is_datetime: bool
    y_values: dict[str, list[float]]


DEFAULT_X_CANDIDATES: tuple[str, ...] = (
    "total_updates",
    "updates",
    "n_total_timesteps",
    "timesteps",
    "timestamp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot grad norm timeseries from one or more semicolon-delimited log files. "
            "Each input file gets its own figure."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more csv or compressed log files.",
    )
    parser.add_argument(
        "--stat",
        choices=("mean", "max", "min", "std"),
        default="mean",
        help="Summary statistic suffix to match (column suffix '__<stat>').",
    )
    parser.add_argument(
        "--group",
        choices=("auto", "none"),
        default="auto",
        help=(
            "Series grouping mode for subplots. "
            "'auto' separates 'grad_norm_wm_*' from other grad norm series."
        ),
    )
    parser.add_argument(
        "--x",
        dest="x_column",
        default=None,
        help="Optional X column. If omitted, a default candidate is auto-detected.",
    )
    parser.add_argument(
        "--pattern",
        default=r"^grad_norm(?:_.+)?__{stat}$",
        help=(
            "Regex used to select grad norm columns. "
            "Use '{stat}' as placeholder for --stat."
        ),
    )
    parser.add_argument(
        "--delimiter",
        default=";",
        help="CSV delimiter.",
    )
    parser.add_argument(
        "--title-prefix",
        default=None,
        help="Optional prefix for each figure title.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If provided, save one png per input file into this directory.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive plot windows.",
    )
    return parser.parse_args()


def parse_timestamp(value: str) -> datetime | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        if cleaned.endswith("Z"):
            try:
                return datetime.fromisoformat(f"{cleaned[:-1]}+00:00")
            except ValueError:
                return None
    return None


def parse_x_value(value: str | None, path: Path, row_index: int, column: str) -> tuple[float, bool]:
    if value is None or value == "":
        return float("nan"), False
    try:
        return float(value), False
    except ValueError:
        timestamp = parse_timestamp(value)
        if timestamp is None:
            raise ValueError(
                f"Non-scalar x value in {path} row {row_index} column {column}: {value!r}"
            )
        return mdates.date2num(timestamp), True


def parse_float(value: str | None, path: Path, row_index: int, column: str) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Non-scalar y value in {path} row {row_index} column {column}: {value!r}"
        ) from exc


def resolve_x_column(fieldnames: Iterable[str], requested: str | None) -> str:
    names = list(fieldnames)
    if requested is not None:
        if requested not in names:
            raise ValueError(f"Requested x column {requested!r} not found in csv header")
        return requested
    for candidate in DEFAULT_X_CANDIDATES:
        if candidate in names:
            return candidate
    raise ValueError(
        "Could not auto-detect x column. Use --x explicitly. "
        f"Tried: {', '.join(DEFAULT_X_CANDIDATES)}"
    )


def resolve_grad_columns(fieldnames: Iterable[str], pattern: re.Pattern[str]) -> list[str]:
    columns = [name for name in fieldnames if pattern.match(name)]
    if not columns:
        raise ValueError(f"No grad norm columns matched regex: {pattern.pattern!r}")
    return sorted(columns)


def load_log(
    path: Path,
    stat: str,
    x_column: str | None,
    column_pattern: str,
    delimiter: str,
) -> GradNormLog:
    regex = re.compile(column_pattern.format(stat=re.escape(stat)))
    with plot_logs.open_log_text(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")

        resolved_x_column = resolve_x_column(reader.fieldnames, x_column)
        grad_columns = resolve_grad_columns(reader.fieldnames, regex)

        x_values: list[float] = []
        x_is_datetime: bool | None = None
        y_values: dict[str, list[float]] = {column: [] for column in grad_columns}

        for row_index, row in enumerate(reader, start=2):
            x_value, is_datetime = parse_x_value(
                row.get(resolved_x_column),
                path=path,
                row_index=row_index,
                column=resolved_x_column,
            )
            if math.isnan(x_value):
                continue
            if x_is_datetime is None:
                x_is_datetime = is_datetime
            elif x_is_datetime != is_datetime:
                raise ValueError(
                    f"Mixed numeric and timestamp x values in {path} column {resolved_x_column}"
                )
            x_values.append(x_value)
            for column in grad_columns:
                y_values[column].append(
                    parse_float(row.get(column), path=path, row_index=row_index, column=column)
                )

    return GradNormLog(
        path=path,
        x_column=resolved_x_column,
        x_values=x_values,
        x_is_datetime=bool(x_is_datetime),
        y_values=y_values,
    )


def plot_log(log: GradNormLog, stat: str, title_prefix: str | None) -> plt.Figure:
    return plot_log_grouped(log=log, stat=stat, title_prefix=title_prefix, group_mode="none")


def build_groups(columns: list[str], stat: str, group_mode: str) -> dict[str, list[str]]:
    if group_mode == "none":
        return {"all": columns}
    if group_mode == "auto":
        wm = [
            column
            for column in columns
            if column.removesuffix(f"__{stat}").startswith("grad_norm_wm_")
        ]
        core = [column for column in columns if column not in wm]
        groups: dict[str, list[str]] = {}
        if core:
            groups["core"] = core
        if wm:
            groups["wm"] = wm
        return groups
    raise ValueError(f"Unsupported group mode: {group_mode!r}")


def plot_log_grouped(
    log: GradNormLog,
    stat: str,
    title_prefix: str | None,
    group_mode: str,
) -> plt.Figure:
    columns = sorted(log.y_values.keys())
    groups = build_groups(columns=columns, stat=stat, group_mode=group_mode)
    group_names = list(groups.keys())

    figure, axes = plt.subplots(
        nrows=len(group_names),
        ncols=1,
        sharex=True,
        figsize=(12, max(4, 3.5 * len(group_names))),
    )
    if len(group_names) == 1:
        axes = [axes]

    for axis, group_name in zip(axes, group_names, strict=True):
        for column in groups[group_name]:
            values = log.y_values[column]
            valid_x_values = [x for x, y in zip(log.x_values, values, strict=True) if math.isfinite(y)]
            valid_y_values = [y for y in values if math.isfinite(y)]
            if not valid_x_values:
                continue
            label = column.removesuffix(f"__{stat}")
            axis.plot(valid_x_values, valid_y_values, label=label, linewidth=1.5)
        axis.set_ylabel(f"{group_name} ({stat})")
        axis.grid(alpha=0.3)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="upper right", fontsize=8, ncol=2)

    if log.x_is_datetime:
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        for axis in axes:
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(formatter)

    axes[-1].set_xlabel(log.x_column)
    title = log.path.name if title_prefix is None else f"{title_prefix} | {log.path.name}"
    axes[0].set_title(title)
    figure.tight_layout()
    return figure


def build_output_path(output_dir: Path, source: Path, stat: str) -> Path:
    filename = f"{source.stem}_grad_norms_{stat}.png"
    return output_dir / filename


def main() -> int:
    args = parse_args()
    paths = [path.expanduser().resolve() for path in args.paths]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir: Path | None
    if args.output_dir is None:
        output_dir = None
    else:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    figures: list[plt.Figure] = []
    for path in paths:
        log = load_log(
            path=path,
            stat=args.stat,
            x_column=args.x_column,
            column_pattern=args.pattern,
            delimiter=args.delimiter,
        )
        figure = plot_log_grouped(
            log=log,
            stat=args.stat,
            title_prefix=args.title_prefix,
            group_mode=args.group,
        )
        figures.append(figure)
        if output_dir is not None:
            figure.savefig(build_output_path(output_dir, path, args.stat), dpi=150)

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
