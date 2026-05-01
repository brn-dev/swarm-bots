from __future__ import annotations

import argparse
import csv
import gzip
import io
import bz2
import lzma
import math
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TextIO
from contextlib import contextmanager
from zipfile import BadZipFile, ZipFile, ZipInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

"""
        WARNING: 95+% vibe coded
"""

@dataclass(slots=True)
class LogSeries:
    label: str
    x_values: list[float]
    x_is_datetime: bool
    y_values: dict[str, list[float]]
    y_std_values: dict[str, list[float]]
    y_skew_values: dict[str, list[float]]
    y_min_values: dict[str, list[float]]
    y_max_values: dict[str, list[float]]


SUPPORTED_LOG_SUFFIXES: tuple[str, ...] = (
    ".csv",
    ".zip",
    ".gz",
    ".bz2",
    ".xz",
)


def is_supported_log_path(path: Path) -> bool:
    if not path.is_file():
        return False
    return path.suffix.lower() in SUPPORTED_LOG_SUFFIXES


def resolve_zip_csv_member(path: Path, archive: ZipFile) -> ZipInfo:
    csv_members = [
        info
        for info in archive.infolist()
        if not info.is_dir() and Path(info.filename).suffix.lower() == ".csv"
    ]
    if not csv_members:
        raise ValueError(f"{path} does not contain a CSV file.")
    if len(csv_members) == 1:
        return csv_members[0]
    preferred_members = [
        info for info in csv_members if Path(info.filename).name.casefold() == "log.csv"
    ]
    if len(preferred_members) == 1:
        return preferred_members[0]
    member_names = ", ".join(info.filename for info in csv_members[:3])
    if len(csv_members) > 3:
        member_names = f"{member_names}, ..."
    raise ValueError(
        f"{path} contains multiple CSV files ({member_names}). Keep one CSV in the archive "
        f"or name the desired file log.csv."
    )


@contextmanager
def open_log_text(path: Path, newline: str | None = None) -> Iterator[TextIO]:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        try:
            with ZipFile(path) as archive:
                member = resolve_zip_csv_member(path, archive)
                with archive.open(member, mode="r") as raw_handle:
                    with io.TextIOWrapper(raw_handle, newline=newline) as text_handle:
                        yield text_handle
            return
        except BadZipFile as exc:
            raise ValueError(f"{path} is not a valid zip archive.") from exc
    if suffix == ".gz":
        with gzip.open(path, mode="rt", newline=newline) as handle:
            yield handle
        return
    if suffix == ".bz2":
        with bz2.open(path, mode="rt", newline=newline) as handle:
            yield handle
        return
    if suffix == ".xz":
        with lzma.open(path, mode="rt", newline=newline) as handle:
            yield handle
        return
    with path.open(mode="r", newline=newline) as handle:
        yield handle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot scalar columns from semicolon-delimited training logs.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more log csv files.",
    )
    parser.add_argument(
        "--x",
        required=True,
        dest="x_column",
        help="Column name for the X axis.",
    )
    parser.add_argument(
        "--y",
        required=True,
        nargs="+",
        dest="y_columns",
        help="One or more column names for the Y axis.",
    )
    parser.add_argument(
        "--std",
        nargs="+",
        dest="std_columns",
        default=None,
        help="Optional std column per Y column; use '_' when no std is available.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels matching the number of paths.",
    )
    parser.add_argument(
        "--delimiter",
        default=";",
        help="CSV delimiter used in the log files.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path to save the figure.",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=None,
        help="Relative vertical space per subplot; must match number of y columns.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window.",
    )
    return parser.parse_args()


def labels_unique(labels: Sequence[str]) -> bool:
    return len(set(labels)) == len(labels)


def label_from_parents(path: Path, depth: int) -> str:
    parents = path.parent.parts
    if depth <= 0:
        return path.stem
    if depth > len(parents):
        return str(path)
    return "/".join(parents[-depth:])


def default_labels(paths: Sequence[Path]) -> list[str]:
    for depth in range(0, 5):
        candidate_labels = [label_from_parents(path, depth) for path in paths]
        if labels_unique(candidate_labels):
            return candidate_labels
    return [str(path) for path in paths]


def build_labels(paths: Sequence[Path], labels: Sequence[str] | None) -> list[str]:
    if labels is None:
        return default_labels(paths)
    if len(labels) != len(paths):
        raise ValueError("labels count must match number of paths")
    return list(labels)


def parse_scalar(value: str | None, column: str, path: Path, row_index: int) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError as exc:
        message = (
            f"Non-scalar value in {path} row {row_index} column {column}: {value!r}"
        )
        raise ValueError(message) from exc


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


def parse_x_value(
    value: str | None,
    column: str,
    path: Path,
    row_index: int,
) -> tuple[float, bool]:
    if value is None or value == "":
        return float("nan"), False
    try:
        return float(value), False
    except ValueError:
        timestamp = parse_timestamp(value)
        if timestamp is None:
            message = (
                f"Non-scalar value in {path} row {row_index} column {column}: {value!r}"
            )
            raise ValueError(message)
        return mdates.date2num(timestamp), True


def validate_columns(
    available_columns: Iterable[str],
    required_columns: Iterable[str],
    path: Path,
) -> None:
    available_set = set(available_columns)
    missing_columns = [name for name in required_columns if name not in available_set]
    if missing_columns:
        missing_list = ", ".join(missing_columns)
        raise ValueError(f"Missing columns in {path}: {missing_list}")


def build_std_mapping(
    y_columns: Sequence[str],
    std_columns: Sequence[str] | None,
) -> dict[str, str | None]:
    if std_columns is None:
        return {column: None for column in y_columns}
    if len(std_columns) != len(y_columns):
        raise ValueError("std column count must match number of y columns")
    mapping: dict[str, str | None] = {}
    for y_column, std_column in zip(y_columns, std_columns, strict=True):
        if std_column in {"_", "none", "None"}:
            mapping[y_column] = None
        else:
            mapping[y_column] = std_column
    return mapping


def validate_ratios(
    ratios: Sequence[float] | None,
    y_columns: Sequence[str],
) -> list[float] | None:
    if ratios is None:
        return None
    if len(ratios) != len(y_columns):
        raise ValueError("ratios count must match number of y columns")
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("ratios must be positive")
    return list(ratios)


def load_log(
    path: Path,
    label: str,
    x_column: str,
    y_columns: Sequence[str],
    std_mapping: dict[str, str | None],
    delimiter: str,
    min_mapping: dict[str, str | None] | None = None,
    max_mapping: dict[str, str | None] | None = None,
    skew_mapping: dict[str, str | None] | None = None,
    from_index: int = 0,
    to_index: int | None = None,
) -> LogSeries:
    if from_index < 0:
        raise ValueError("From must be >= 0.")
    if to_index is not None and to_index < from_index:
        raise ValueError("To must be >= From.")
    x_values: list[float] = []
    x_is_datetime: bool | None = None
    y_values: dict[str, list[float]] = {column: [] for column in y_columns}
    y_std_values: dict[str, list[float]] = {
        column: [] for column in y_columns if std_mapping[column] is not None
    }
    resolved_skew_mapping = {column: None for column in y_columns}
    if skew_mapping:
        for column in y_columns:
            resolved_skew_mapping[column] = skew_mapping.get(column)
    y_skew_values: dict[str, list[float]] = {
        column: [] for column in y_columns if resolved_skew_mapping[column] is not None
    }
    resolved_min_mapping = {column: None for column in y_columns}
    if min_mapping:
        for column in y_columns:
            resolved_min_mapping[column] = min_mapping.get(column)
    resolved_max_mapping = {column: None for column in y_columns}
    if max_mapping:
        for column in y_columns:
            resolved_max_mapping[column] = max_mapping.get(column)
    y_min_values: dict[str, list[float]] = {
        column: [] for column in y_columns if resolved_min_mapping[column] is not None
    }
    y_max_values: dict[str, list[float]] = {
        column: [] for column in y_columns if resolved_max_mapping[column] is not None
    }
    valid_row_index = 0
    with open_log_text(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        required_columns = [x_column, *y_columns]
        for std_column in std_mapping.values():
            if std_column is not None:
                required_columns.append(std_column)
        for skew_column in resolved_skew_mapping.values():
            if skew_column is not None:
                required_columns.append(skew_column)
        for min_column in resolved_min_mapping.values():
            if min_column is not None:
                required_columns.append(min_column)
        for max_column in resolved_max_mapping.values():
            if max_column is not None:
                required_columns.append(max_column)
        validate_columns(reader.fieldnames, required_columns, path)
        for row_index, row in enumerate(reader, start=2):
            x_value, is_datetime = parse_x_value(row.get(x_column), x_column, path, row_index)
            if math.isnan(x_value):
                continue
            if x_is_datetime is None:
                x_is_datetime = is_datetime
            elif x_is_datetime != is_datetime:
                raise ValueError(
                    f"Mixed numeric and timestamp values in {path} column {x_column}."
                )
            if valid_row_index < from_index:
                valid_row_index += 1
                continue
            if to_index is not None and valid_row_index > to_index:
                continue
            valid_row_index += 1
            x_values.append(x_value)
            for column in y_columns:
                y_values[column].append(
                    parse_scalar(row.get(column), column, path, row_index)
                )
                std_column = std_mapping[column]
                if std_column is not None:
                    y_std_values[column].append(
                        parse_scalar(row.get(std_column), std_column, path, row_index)
                    )
                skew_column = resolved_skew_mapping[column]
                if skew_column is not None:
                    y_skew_values[column].append(
                        parse_scalar(row.get(skew_column), skew_column, path, row_index)
                    )
                min_column = resolved_min_mapping[column]
                if min_column is not None:
                    y_min_values[column].append(
                        parse_scalar(row.get(min_column), min_column, path, row_index)
                    )
                max_column = resolved_max_mapping[column]
                if max_column is not None:
                    y_max_values[column].append(
                        parse_scalar(row.get(max_column), max_column, path, row_index)
                    )
    return LogSeries(
        label=label,
        x_values=x_values,
        x_is_datetime=bool(x_is_datetime),
        y_values=y_values,
        y_std_values=y_std_values,
        y_skew_values=y_skew_values,
        y_min_values=y_min_values,
        y_max_values=y_max_values,
    )


def plot_logs(
    logs: Sequence[LogSeries],
    x_column: str,
    y_columns: Sequence[str],
    std_mapping: dict[str, str | None],
    ratios: Sequence[float] | None,
    title: str | None,
) -> plt.Figure:
    row_count = len(y_columns)
    height_ratios = list(ratios) if ratios is not None else None
    figure_height = 3 * (sum(height_ratios) if height_ratios is not None else row_count)
    figure, axes = plt.subplots(
        nrows=row_count,
        ncols=1,
        sharex=True,
        figsize=(10, max(3, figure_height)),
        gridspec_kw={"height_ratios": height_ratios} if height_ratios else None,
    )
    if row_count == 1:
        axes = [axes]
    x_is_datetime_flags = {log.x_is_datetime for log in logs}
    if len(x_is_datetime_flags) > 1:
        raise ValueError("Mixed numeric and timestamp values in X column.")
    x_is_datetime = next(iter(x_is_datetime_flags), False)
    for axis, column in zip(axes, y_columns, strict=True):
        for log in logs:
            axis.plot(log.x_values, log.y_values[column], label=log.label)
            if std_mapping[column] is not None:
                std_values = log.y_std_values.get(column)
                if std_values:
                    upper = [value + std for value, std in zip(log.y_values[column], std_values, strict=True)]
                    lower = [value - std for value, std in zip(log.y_values[column], std_values, strict=True)]
                    axis.fill_between(log.x_values, lower, upper, alpha=0.2)
        axis.set_ylabel(column)
        axis.grid(alpha=0.3)
    if x_is_datetime:
        for axis in axes:
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(formatter)
    axes[-1].set_xlabel(x_column)
    if title:
        figure.suptitle(title)
    if len(logs) > 1:
        axes[0].legend()
    figure.tight_layout()
    return figure


def main() -> int:
    args = parse_args()
    paths = [path.expanduser().resolve() for path in args.paths]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    labels = build_labels(paths, args.labels)
    std_mapping = build_std_mapping(args.y_columns, args.std_columns)
    ratios = validate_ratios(args.ratios, args.y_columns)
    logs = [
        load_log(
            path,
            label,
            args.x_column,
            args.y_columns,
            std_mapping,
            args.delimiter,
        )
        for path, label in zip(paths, labels, strict=True)
    ]
    figure = plot_logs(
        logs,
        args.x_column,
        args.y_columns,
        std_mapping,
        ratios,
        args.title,
    )
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        figure.savefig(output_path, dpi=150)
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
