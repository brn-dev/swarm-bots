"""Read scalar metrics from the CSV files written by SwarmBots learners."""

from __future__ import annotations

import bz2
import csv
import gzip
import io
import lzma
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO
from zipfile import ZipFile

import numpy as np

LogPath = str | Path
LogSources = LogPath | Sequence[LogPath]
GroupSources = LogPath | Mapping[str, LogSources]
DEFAULT_COLUMNS = ("ep_rew_ema", "ep_success_rate_ema")
LOG_NAMES = ("log.csv", "log.csv.gz", "log.csv.bz2", "log.csv.xz", "log.csv.zip", "log.zip")
SUPPORTED_SUFFIXES = {".csv", ".gz", ".bz2", ".xz", ".zip"}


@dataclass(slots=True)
class RunLog:
    path: Path
    label: str
    x_column: str
    x_values: np.ndarray
    series: dict[str, np.ndarray]
    x_is_datetime: bool = False


@dataclass(slots=True)
class GroupStatistics:
    x_values: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    counts: np.ndarray


def find_log_file(source: LogPath) -> Path:
    """Resolve a CSV/archive or a run directory, preferring its uncompressed log."""
    path = Path(source).expanduser()
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported log file: {path}")
        return path
    if not path.exists():
        raise FileNotFoundError(path)
    for name in LOG_NAMES:
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No training log found in {path}; expected log.csv or a compressed variant")


@contextmanager
def open_log_text(path: Path) -> Iterator[TextIO]:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with ZipFile(path) as archive:
            members = [
                member
                for member in archive.infolist()
                if not member.is_dir() and Path(member.filename).suffix.lower() == ".csv"
            ]
            preferred = [member for member in members if Path(member.filename).name.lower() == "log.csv"]
            if len(members) == 1:
                member = members[0]
            elif len(preferred) == 1:
                member = preferred[0]
            else:
                raise ValueError(f"{path} must contain one CSV, or exactly one CSV named log.csv")
            with archive.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as handle:
                yield handle
        return
    opener = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}.get(suffix, open)
    with opener(path, mode="rt", encoding="utf-8-sig", newline="") as handle:
        yield handle


def log_columns(source: LogPath, *, delimiter: str = ";") -> tuple[str, ...]:
    """List columns without loading the entire log."""
    path = find_log_file(source)
    with open_log_text(path) as handle:
        columns = csv.DictReader(handle, delimiter=delimiter).fieldnames
    if not columns:
        raise ValueError(f"{path} has no header row")
    return tuple(columns)


def _parse_scalar(value: str | None, *, path: Path, column: str, row: int) -> float:
    if value is None or not value.strip():
        return float("nan")
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"Non-scalar value in {path}, row {row}, column {column!r}: {value!r}") from error


def _parse_x(value: str, *, path: Path, column: str, row: int) -> tuple[float, bool]:
    try:
        return float(value), False
    except ValueError:
        from matplotlib.dates import date2num

        try:
            return float(date2num(datetime.fromisoformat(value.strip()))), True
        except ValueError as error:
            raise ValueError(f"Invalid X value in {path}, row {row}, column {column!r}: {value!r}") from error


def load_log(
    source: LogPath,
    *,
    columns: Sequence[str] | None = None,
    x_column: str = "timesteps",
    label: str | None = None,
    delimiter: str = ";",
    max_steps: float | None = None,
) -> RunLog:
    """Load selected scalar columns; defaults to available return/success EMAs.

    Blank cells are NaN. Rows with missing/nonfinite X values are skipped.
    Explicitly requested columns must exist. ``max_steps`` filters numeric X
    values before interpolation or plotting; it is unsupported for timestamps.
    """
    path = find_log_file(source)
    x_values: list[float] = []
    x_is_datetime: bool | None = None
    with open_log_text(path) as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        selected = (
            tuple(columns)
            if columns is not None
            else tuple(column for column in DEFAULT_COLUMNS if column in reader.fieldnames)
        )
        if not selected:
            raise ValueError(f"No metrics selected in {path}; pass columns= or inspect log_columns()")
        missing = set((x_column, *selected)) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
        series: dict[str, list[float]] = {column: [] for column in selected}
        for row_index, row in enumerate(reader, start=2):
            raw_x = row.get(x_column)
            if raw_x is None or not raw_x.strip():
                continue
            x_value, is_datetime = _parse_x(raw_x, path=path, column=x_column, row=row_index)
            if not np.isfinite(x_value):
                continue
            if x_is_datetime is not None and x_is_datetime != is_datetime:
                raise ValueError(f"Mixed numeric and timestamp values in {path}, column {x_column!r}")
            x_is_datetime = is_datetime
            if max_steps is not None:
                if is_datetime:
                    raise ValueError("max_steps requires a numeric X column")
                if x_value > max_steps:
                    continue
            x_values.append(x_value)
            for column, values in series.items():
                values.append(_parse_scalar(row.get(column), path=path, column=column, row=row_index))
    if not x_values:
        raise ValueError(f"{path} contains no usable rows after filtering")
    return RunLog(
        path=path,
        label=label if label is not None else path.parent.name,
        x_column=x_column,
        x_values=np.asarray(x_values),
        series={column: np.asarray(values) for column, values in series.items()},
        x_is_datetime=bool(x_is_datetime),
    )


def _source_paths(sources: LogSources) -> list[Path]:
    if isinstance(sources, (str, Path)):
        sources = [sources]
    paths: list[Path] = []
    for source in sources:
        path = Path(source).expanduser()
        try:
            paths.append(find_log_file(path))
        except FileNotFoundError:
            if not path.is_dir():
                raise
            for child in sorted(path.iterdir()):
                if child.is_dir():
                    try:
                        paths.append(find_log_file(child))
                    except FileNotFoundError:
                        continue
    unique_paths = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique_paths:
        raise ValueError(f"No run logs found in {sources}")
    return unique_paths


def load_groups(
    sources: GroupSources,
    *,
    columns: Sequence[str] | None = None,
    x_column: str = "timesteps",
    delimiter: str = ";",
    max_steps: float | None = None,
) -> dict[str, list[RunLog]]:
    """Load named groups, or discover ROOT/GROUP/log.csv and ROOT/GROUP/RUN/log.csv.

    A mapping preserves insertion order and accepts files, run directories, or
    directories containing runs. Discovery sorts groups/runs alphabetically.
    """
    if isinstance(sources, (str, Path)):
        root = Path(sources).expanduser()
        if not root.is_dir():
            raise NotADirectoryError(root)
        discovered: dict[str, list[Path]] = {}
        for child in sorted(root.iterdir()):
            if child.is_dir():
                try:
                    discovered[child.name] = _source_paths(child)
                except ValueError:
                    continue
        sources = discovered
    if not sources:
        raise ValueError("No experiment groups found")
    return {
        name: [
            load_log(path, columns=columns, x_column=x_column, delimiter=delimiter, max_steps=max_steps)
            for path in _source_paths(group_sources)
        ]
        for name, group_sources in sources.items()
    }


def smooth_values(values: np.ndarray, alpha: float | None) -> np.ndarray:
    """Apply an EMA to finite values, retaining missing cells as gaps."""
    if alpha is None:
        return values
    if not 0 < alpha <= 1:
        raise ValueError("smooth must be in (0, 1]")
    result = np.full_like(values, np.nan, dtype=float)
    previous: float | None = None
    for index, value in enumerate(values):
        if np.isfinite(value):
            previous = float(value) if previous is None else alpha * value + (1 - alpha) * previous
            result[index] = previous
    return result


def group_statistics(runs: Sequence[RunLog], column: str, *, smooth: float | None = None) -> GroupStatistics:
    """Interpolate runs on their combined X grid, with no endpoint extrapolation.

    Missing metric samples are ignored during interpolation. Duplicate X values
    use the last finite value, allowing resumed logs. Standard deviation is the
    population standard deviation of contributing runs at each X value.
    """
    if not runs:
        raise ValueError("At least one run is required")
    if len({run.x_is_datetime for run in runs}) > 1:
        raise ValueError("Cannot combine numeric and timestamp X values")
    x_values = np.unique(np.concatenate([run.x_values[np.isfinite(run.x_values)] for run in runs]))
    interpolated = np.full((len(runs), len(x_values)), np.nan)
    for row, run in enumerate(runs):
        values = smooth_values(run.series.get(column, np.full_like(run.x_values, np.nan)), smooth)
        finite = np.isfinite(run.x_values) & np.isfinite(values)
        order = np.argsort(run.x_values[finite], kind="stable")
        run_x, run_y = run.x_values[finite][order], values[finite][order]
        if run_x.size == 0:
            continue
        # Stable sorting plus reversed uniqueness keeps the latest resumed sample.
        _, reverse_indices = np.unique(run_x[::-1], return_index=True)
        indices = np.sort(len(run_x) - 1 - reverse_indices)
        run_x, run_y = run_x[indices], run_y[indices]
        in_range = (x_values >= run_x[0]) & (x_values <= run_x[-1])
        interpolated[row, in_range] = np.interp(x_values[in_range], run_x, run_y)
    finite = np.isfinite(interpolated)
    counts = finite.sum(axis=0)
    totals = np.where(finite, interpolated, 0).sum(axis=0)
    mean = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)
    squared_deltas = np.where(finite, (interpolated - mean) ** 2, 0).sum(axis=0)
    variance = np.divide(squared_deltas, counts, out=np.full_like(totals, np.nan), where=counts > 0)
    return GroupStatistics(x_values=x_values, mean=mean, std=np.sqrt(variance), counts=counts)
