from __future__ import annotations

import bz2
import gzip
import io
import lzma
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal, TextIO, TypeAlias
from zipfile import BadZipFile, ZipFile, ZipInfo

import pandas as pd


PathLike: TypeAlias = str | Path
PathInput: TypeAlias = PathLike | Sequence[PathLike]
RowSelection: TypeAlias = slice | int | Sequence[int] | range | None
AggregationName: TypeAlias = Literal["mean", "max", "min", "sum", "median", "std", "var"]

SUPPORTED_LOG_SUFFIXES: tuple[str, ...] = (
    ".csv",
    ".zip",
    ".gz",
    ".bz2",
    ".xz",
)
DEFAULT_DELIMITER = ";"
SOURCE_PATH_COLUMN = "source_path"
SOURCE_NAME_COLUMN = "source_name"


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


def _normalize_path(path: PathLike) -> Path:
    resolved_path = Path(path).expanduser().resolve()
    if resolved_path.exists() and not resolved_path.is_file():
        raise IsADirectoryError(resolved_path)
    if not resolved_path.exists():
        raise FileNotFoundError(resolved_path)
    if not is_supported_log_path(resolved_path):
        raise ValueError(f"Unsupported log path: {resolved_path}")
    return resolved_path


def _normalize_paths(path_or_paths: PathInput) -> list[Path]:
    if isinstance(path_or_paths, (str, Path)):
        return [_normalize_path(path_or_paths)]
    paths = [_normalize_path(path) for path in path_or_paths]
    if not paths:
        raise ValueError("At least one log path is required.")
    return paths


def _normalize_columns(columns: Sequence[str] | None) -> list[str] | None:
    if columns is None:
        return None
    normalized_columns = list(columns)
    if not normalized_columns:
        raise ValueError("columns must not be empty")
    return normalized_columns


def _is_simple_positive_slice(rows: RowSelection) -> bool:
    return (
        isinstance(rows, slice)
        and (rows.step is None or rows.step == 1)
        and (rows.start is None or rows.start >= 0)
        and (rows.stop is None or rows.stop >= 0)
    )


def _slice_start(rows: slice) -> int:
    return 0 if rows.start is None else rows.start


def _slice_stop(rows: slice) -> int | None:
    return rows.stop


def _append_source_columns(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, SOURCE_NAME_COLUMN, path.name)
    result.insert(0, SOURCE_PATH_COLUMN, str(path))
    return result


def _read_log_frame(
    path: Path,
    *,
    columns: Sequence[str] | None,
    rows: RowSelection,
    delimiter: str,
) -> pd.DataFrame:
    selected_columns = _normalize_columns(columns)
    if _is_simple_positive_slice(rows):
        row_slice = rows
        start = _slice_start(row_slice)
        stop = _slice_stop(row_slice)
        skiprows = range(1, 1 + start) if start > 0 else None
        nrows = None if stop is None else max(0, stop - start)
        with open_log_text(path, newline="") as handle:
            frame = pd.read_csv(
                handle,
                sep=delimiter,
                usecols=selected_columns,
                skiprows=skiprows,
                nrows=nrows,
            )
        if selected_columns is not None:
            frame = frame.loc[:, selected_columns]
        if start > 0 and not frame.empty:
            frame.index = pd.RangeIndex(start=start, stop=start + len(frame))
        return frame

    with open_log_text(path, newline="") as handle:
        frame = pd.read_csv(handle, sep=delimiter, usecols=selected_columns)
    if selected_columns is not None:
        frame = frame.loc[:, selected_columns]
    return select_log_frame(frame, rows=rows)


def _read_log_columns(path: Path, *, delimiter: str) -> list[str]:
    with open_log_text(path, newline="") as handle:
        frame = pd.read_csv(handle, sep=delimiter, nrows=0)
    return list(frame.columns)


def select_log_frame(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    rows: RowSelection = None,
) -> pd.DataFrame:
    selected_columns = _normalize_columns(columns)
    result = frame
    if selected_columns is not None:
        missing_columns = [column for column in selected_columns if column not in result.columns]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Missing columns in frame: {missing}")
        result = result.loc[:, selected_columns]
    if rows is None:
        return result
    if isinstance(rows, int):
        return result.iloc[[rows]].copy()
    if isinstance(rows, range):
        return result.iloc[list(rows)].copy()
    if isinstance(rows, slice):
        return result.iloc[rows].copy()
    return result.iloc[list(rows)].copy()


def fetch_log_frames(
    path_or_paths: PathInput,
    *,
    columns: Sequence[str] | None = None,
    rows: RowSelection = None,
    delimiter: str = DEFAULT_DELIMITER,
    include_source_column: bool | None = None,
) -> list[pd.DataFrame]:
    paths = _normalize_paths(path_or_paths)
    add_source_columns = len(paths) > 1 if include_source_column is None else include_source_column
    frames = [
        _fetch_single_log_frame(
            path,
            columns=columns,
            rows=rows,
            delimiter=delimiter,
            add_source_columns=add_source_columns,
        )
        for path in paths
    ]
    return frames


def list_log_columns(
    path_or_paths: PathInput,
    *,
    delimiter: str = DEFAULT_DELIMITER,
    per_source: bool = False,
) -> list[str] | dict[Path, list[str]]:
    paths = _normalize_paths(path_or_paths)
    columns_by_path = {path: _read_log_columns(path, delimiter=delimiter) for path in paths}
    if per_source:
        return columns_by_path

    seen_columns: set[str] = set()
    ordered_columns: list[str] = []
    for path in paths:
        for column in columns_by_path[path]:
            if column in seen_columns:
                continue
            seen_columns.add(column)
            ordered_columns.append(column)
    return ordered_columns


def _fetch_single_log_frame(
    path: Path,
    *,
    columns: Sequence[str] | None,
    rows: RowSelection,
    delimiter: str,
    add_source_columns: bool,
) -> pd.DataFrame:
    frame = _read_log_frame(
        path,
        columns=columns,
        rows=rows,
        delimiter=delimiter,
    )
    if add_source_columns:
        frame = _append_source_columns(frame, path)
    return frame


def fetch_log_frame(
    path_or_paths: PathInput,
    *,
    columns: Sequence[str] | None = None,
    rows: RowSelection = None,
    delimiter: str = DEFAULT_DELIMITER,
    include_source_column: bool | None = None,
) -> pd.DataFrame:
    frames = fetch_log_frames(
        path_or_paths,
        columns=columns,
        rows=rows,
        delimiter=delimiter,
        include_source_column=include_source_column,
    )
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def fetch_log_columns(
    path_or_paths: PathInput,
    columns: Sequence[str],
    *,
    delimiter: str = DEFAULT_DELIMITER,
    rows: RowSelection = None,
    include_source_column: bool | None = None,
) -> pd.DataFrame:
    return fetch_log_frame(
        path_or_paths,
        columns=columns,
        rows=rows,
        delimiter=delimiter,
        include_source_column=include_source_column,
    )


def fetch_log_rows(
    path_or_paths: PathInput,
    rows: RowSelection,
    *,
    columns: Sequence[str] | None = None,
    delimiter: str = DEFAULT_DELIMITER,
    include_source_column: bool | None = None,
) -> pd.DataFrame:
    return fetch_log_frame(
        path_or_paths,
        columns=columns,
        rows=rows,
        delimiter=delimiter,
        include_source_column=include_source_column,
    )


def fetch_log_head(
    path_or_paths: PathInput,
    count: int = 100,
    *,
    columns: Sequence[str] | None = None,
    delimiter: str = DEFAULT_DELIMITER,
    include_source_column: bool | None = None,
) -> pd.DataFrame:
    if count < 0:
        raise ValueError("count must be >= 0")
    return fetch_log_frame(
        path_or_paths,
        columns=columns,
        rows=slice(0, count),
        delimiter=delimiter,
        include_source_column=include_source_column,
    )


def fetch_log_tail(
    path_or_paths: PathInput,
    count: int = 100,
    *,
    columns: Sequence[str] | None = None,
    delimiter: str = DEFAULT_DELIMITER,
    include_source_column: bool | None = None,
) -> pd.DataFrame:
    if count < 0:
        raise ValueError("count must be >= 0")
    if count == 0:
        return fetch_log_frame(
            path_or_paths,
            columns=columns,
            rows=slice(0, 0),
            delimiter=delimiter,
            include_source_column=include_source_column,
        )
    return fetch_log_frame(
        path_or_paths,
        columns=columns,
        rows=slice(-count, None),
        delimiter=delimiter,
        include_source_column=include_source_column,
    )


def fetch_log_range(
    path_or_paths: PathInput,
    start: int,
    stop: int | None,
    *,
    columns: Sequence[str] | None = None,
    delimiter: str = DEFAULT_DELIMITER,
    include_source_column: bool | None = None,
) -> pd.DataFrame:
    return fetch_log_frame(
        path_or_paths,
        columns=columns,
        rows=slice(start, stop),
        delimiter=delimiter,
        include_source_column=include_source_column,
    )


def _selected_numeric_columns(
    frame: pd.DataFrame,
    columns: Sequence[str] | None,
) -> list[str]:
    if columns is None:
        selected_columns = list(frame.select_dtypes(include="number").columns)
    else:
        selected_columns = list(columns)
        missing_columns = [column for column in selected_columns if column not in frame.columns]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Missing columns in frame: {missing}")

    numeric_columns = [
        column for column in selected_columns if pd.api.types.is_numeric_dtype(frame[column])
    ]
    non_numeric_columns = [column for column in selected_columns if column not in numeric_columns]
    if non_numeric_columns:
        non_numeric = ", ".join(non_numeric_columns)
        raise ValueError(f"Aggregation only supports numeric columns: {non_numeric}")
    if not numeric_columns:
        raise ValueError("No numeric columns available for aggregation")
    return numeric_columns


def _aggregate_series(series: pd.Series, operation: AggregationName) -> float:
    if series.empty:
        return float("nan")
    aggregated_value = getattr(series, operation)()
    if pd.isna(aggregated_value):
        return float("nan")
    return float(aggregated_value)


def _summarize_frame(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None,
    operations: Sequence[AggregationName],
) -> dict[str, float | int]:
    numeric_columns = _selected_numeric_columns(frame, columns)
    summary: dict[str, float | int] = {
        "row_count": int(len(frame)),
    }
    for column in numeric_columns:
        series = frame[column]
        for operation in operations:
            summary[f"{column}__{operation}"] = _aggregate_series(series, operation)
    return summary


def fetch_log_aggregate(
    path_or_paths: PathInput,
    *,
    columns: Sequence[str] | None = None,
    rows: RowSelection = None,
    operations: Sequence[AggregationName] = ("mean", "max"),
    delimiter: str = DEFAULT_DELIMITER,
    per_source: bool | None = None,
    include_source_column: bool | None = None,
) -> pd.DataFrame:
    if not operations:
        raise ValueError("operations must not be empty")
    invalid_operations = [operation for operation in operations if operation not in {"mean", "max", "min", "sum", "median", "std", "var"}]
    if invalid_operations:
        invalid = ", ".join(invalid_operations)
        raise ValueError(f"Unsupported aggregation operations: {invalid}")

    paths = _normalize_paths(path_or_paths)
    resolved_per_source = len(paths) > 1 if per_source is None else per_source

    if resolved_per_source:
        rows_data: list[dict[str, float | int | str]] = []
        for path in paths:
            frame = _read_log_frame(path, columns=columns, rows=rows, delimiter=delimiter)
            row = _summarize_frame(frame, columns=columns, operations=operations)
            row[SOURCE_PATH_COLUMN] = str(path)
            row[SOURCE_NAME_COLUMN] = path.name
            row["source_count"] = 1
            rows_data.append(row)
        return pd.DataFrame(rows_data)

    frame = fetch_log_frame(
        paths,
        columns=columns,
        rows=rows,
        delimiter=delimiter,
        include_source_column=include_source_column,
    )
    summary = _summarize_frame(frame, columns=columns, operations=operations)
    summary["source_count"] = len(paths)
    if include_source_column:
        if len(paths) == 1:
            summary[SOURCE_PATH_COLUMN] = str(paths[0])
            summary[SOURCE_NAME_COLUMN] = paths[0].name
        else:
            summary[SOURCE_PATH_COLUMN] = "<combined>"
            summary[SOURCE_NAME_COLUMN] = "<combined>"
    return pd.DataFrame([summary])


def fetch_log_mean(
    path_or_paths: PathInput,
    *,
    columns: Sequence[str] | None = None,
    rows: RowSelection = None,
    delimiter: str = DEFAULT_DELIMITER,
    per_source: bool | None = None,
    include_source_column: bool | None = None,
) -> pd.DataFrame:
    return fetch_log_aggregate(
        path_or_paths,
        columns=columns,
        rows=rows,
        operations=("mean",),
        delimiter=delimiter,
        per_source=per_source,
        include_source_column=include_source_column,
    )


def fetch_log_max(
    path_or_paths: PathInput,
    *,
    columns: Sequence[str] | None = None,
    rows: RowSelection = None,
    delimiter: str = DEFAULT_DELIMITER,
    per_source: bool | None = None,
    include_source_column: bool | None = None,
) -> pd.DataFrame:
    return fetch_log_aggregate(
        path_or_paths,
        columns=columns,
        rows=rows,
        operations=("max",),
        delimiter=delimiter,
        per_source=per_source,
        include_source_column=include_source_column,
    )


def fetch_log_min(
    path_or_paths: PathInput,
    *,
    columns: Sequence[str] | None = None,
    rows: RowSelection = None,
    delimiter: str = DEFAULT_DELIMITER,
    per_source: bool | None = None,
    include_source_column: bool | None = None,
) -> pd.DataFrame:
    return fetch_log_aggregate(
        path_or_paths,
        columns=columns,
        rows=rows,
        operations=("min",),
        delimiter=delimiter,
        per_source=per_source,
        include_source_column=include_source_column,
    )


def fetch_log_sum(
    path_or_paths: PathInput,
    *,
    columns: Sequence[str] | None = None,
    rows: RowSelection = None,
    delimiter: str = DEFAULT_DELIMITER,
    per_source: bool | None = None,
    include_source_column: bool | None = None,
) -> pd.DataFrame:
    return fetch_log_aggregate(
        path_or_paths,
        columns=columns,
        rows=rows,
        operations=("sum",),
        delimiter=delimiter,
        per_source=per_source,
        include_source_column=include_source_column,
    )
