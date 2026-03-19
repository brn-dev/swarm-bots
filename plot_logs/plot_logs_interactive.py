from __future__ import annotations

import ast
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import plot_logs

STATE_PATH = Path(__file__).resolve().with_name(".plot_logs_recent_config.json")
PRESETS_PATH = Path(__file__).resolve().with_name(".plot_logs_presets.json")
PATH_ORDER_MODES = (
    "Added",
    "Created (oldest first)",
    "Created (newest first)",
)
DEFAULT_PATH_ORDER_MODE = PATH_ORDER_MODES[2]

"""
        WARNING: 95+% vibe coded
"""

@dataclass(slots=True)
class PlotRow:
    index_var: tk.StringVar
    index_combo: ttk.Combobox
    y_var: tk.StringVar
    y_combo: ttk.Combobox
    height_var: tk.StringVar
    height_entry: ttk.Entry
    ema_var: tk.StringVar
    ema_container: ttk.Frame
    ema_entry: ttk.Entry
    ema_only_var: tk.BooleanVar
    ema_only_check: ttk.Checkbutton
    std_container: ttk.Frame
    std_var: tk.BooleanVar
    std_check: ttk.Checkbutton
    std_ema_var: tk.BooleanVar
    std_ema_check: ttk.Checkbutton
    skew_container: ttk.Frame
    skew_var: tk.BooleanVar
    skew_check: ttk.Checkbutton
    skew_ema_var: tk.BooleanVar
    skew_ema_check: ttk.Checkbutton
    min_container: ttk.Frame
    min_var: tk.BooleanVar
    min_check: ttk.Checkbutton
    min_ema_var: tk.BooleanVar
    min_ema_check: ttk.Checkbutton
    max_container: ttk.Frame
    max_var: tk.BooleanVar
    max_check: ttk.Checkbutton
    max_ema_var: tk.BooleanVar
    max_ema_check: ttk.Checkbutton
    remove_button: ttk.Button


class AsEma:
    __slots__ = ()

    def __repr__(self) -> str:
        return "AS_EMA"


AS_EMA: AsEma = AsEma()


@dataclass(frozen=True, slots=True)
class PresetEntry:
    y_column: str
    height: float
    ema: float | None = None
    ema_only: bool = False
    nbins: int | None = None
    pooling: int | None = None
    std: bool | AsEma = False
    skew: bool | AsEma = False
    min: bool | AsEma = False
    max: bool | AsEma = False


@dataclass(frozen=True, slots=True)
class PlotPreset:
    name: str
    entries: tuple[PresetEntry, ...]
    x_column: str | None = None


@dataclass(slots=True)
class HistogramSeries:
    label: str
    x_values: list[float]
    x_is_datetime: bool
    bin_edges: list[float] | None
    values: list[list[float]]
    edges_status: str


def parse_preset_summary_option(
    raw_value: object,
    preset_name: str,
    y_column: str,
    field_name: str,
) -> bool | AsEma:
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value == "AS_EMA":
        return AS_EMA
    raise ValueError(
        f"Preset {preset_name!r} entry {y_column!r} field {field_name!r} must be "
        f"true, false, or 'AS_EMA'."
    )


def as_int_like(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def is_histogram_preset_column(y_column: str) -> bool:
    return y_column.endswith("__histogram_freqs") or y_column.endswith("__histogram_edges")


def parse_preset_ema(
    raw_value: object,
    preset_name: str,
    y_column: str,
) -> float | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        raise ValueError(
            f"Preset {preset_name!r} entry {y_column!r} field 'ema' must be a number between 0 and 1."
        )
    if isinstance(raw_value, (int, float)):
        alpha = float(raw_value)
        if 0.0 < alpha < 1.0:
            return alpha
        raise ValueError(
            f"Preset {preset_name!r} entry {y_column!r} field 'ema' must be between 0 and 1."
        )
    raise ValueError(
        f"Preset {preset_name!r} entry {y_column!r} field 'ema' must be a number between 0 and 1."
    )


def parse_preset_positive_int(
    raw_value: object,
    preset_name: str,
    y_column: str,
) -> int | None:
    if raw_value is None:
        return None
    parsed = as_int_like(raw_value)
    if parsed is None or parsed <= 0:
        raise ValueError(
            f"Preset {preset_name!r} entry {y_column!r} must use positive integers for histogram parameters."
        )
    return parsed


def parse_preset_ema_fields(
    raw_entry: dict[str, object],
    preset_name: str,
    y_column: str,
) -> tuple[float | None, int | None, int | None]:
    if "ema_binspool" in raw_entry:
        raise ValueError(
            f"Preset {preset_name!r} entry {y_column!r} uses deprecated field 'ema_binspool'. Use 'ema' and/or 'nbins'/'pooling'."
        )
    if "bins_pool" in raw_entry:
        raise ValueError(
            f"Preset {preset_name!r} entry {y_column!r} uses deprecated field 'bins_pool'. Use 'nbins' and 'pooling'."
        )
    has_ema = "ema" in raw_entry
    has_nbins = "nbins" in raw_entry
    has_pooling = "pooling" in raw_entry
    is_histogram = is_histogram_preset_column(y_column)
    if is_histogram and has_ema:
        raise ValueError(
            f"Preset {preset_name!r} entry {y_column!r}: 'ema' is not allowed for histogram columns."
        )
    if not is_histogram and (has_nbins or has_pooling):
        raise ValueError(
            f"Preset {preset_name!r} entry {y_column!r}: 'nbins' and 'pooling' are only allowed for histogram columns."
        )
    if is_histogram:
        nbins = parse_preset_positive_int(raw_entry.get("nbins"), preset_name, y_column)
        pooling = parse_preset_positive_int(raw_entry.get("pooling"), preset_name, y_column)
        if nbins is None and pooling is not None:
            raise ValueError(
                f"Preset {preset_name!r} entry {y_column!r}: 'pooling' requires 'nbins'."
            )
        if nbins is not None and pooling is None:
            pooling = 1
        return None, nbins, pooling
    return parse_preset_ema(raw_entry.get("ema"), preset_name, y_column), None, None


def parse_preset_entry(raw_entry: object, preset_name: str, entry_index: int) -> PresetEntry:
    if not isinstance(raw_entry, dict):
        raise ValueError(
            f"Preset {preset_name!r} entry {entry_index} must be an object."
        )
    y_column = raw_entry.get("y_column")
    if not isinstance(y_column, str) or not y_column:
        raise ValueError(
            f"Preset {preset_name!r} entry {entry_index} has invalid 'y_column'."
        )
    height = raw_entry.get("height")
    if isinstance(height, bool) or not isinstance(height, (int, float)):
        raise ValueError(
            f"Preset {preset_name!r} entry {y_column!r} has invalid 'height'."
        )
    raw_ema_only = raw_entry.get("ema_only", False)
    if not isinstance(raw_ema_only, bool):
        raise ValueError(
            f"Preset {preset_name!r} entry {y_column!r} field 'ema_only' must be true or false."
        )
    ema, nbins, pooling = parse_preset_ema_fields(raw_entry, preset_name, y_column)
    return PresetEntry(
        y_column=y_column,
        height=float(height),
        ema=ema,
        ema_only=raw_ema_only,
        nbins=nbins,
        pooling=pooling,
        std=parse_preset_summary_option(
            raw_entry.get("std", False),
            preset_name=preset_name,
            y_column=y_column,
            field_name="std",
        ),
        skew=parse_preset_summary_option(
            raw_entry.get("skew", False),
            preset_name=preset_name,
            y_column=y_column,
            field_name="skew",
        ),
        min=parse_preset_summary_option(
            raw_entry.get("min", False),
            preset_name=preset_name,
            y_column=y_column,
            field_name="min",
        ),
        max=parse_preset_summary_option(
            raw_entry.get("max", False),
            preset_name=preset_name,
            y_column=y_column,
            field_name="max",
        ),
    )


def parse_plot_preset(raw_preset: object, preset_index: int) -> PlotPreset:
    if not isinstance(raw_preset, dict):
        raise ValueError(f"Preset index {preset_index} must be an object.")
    name = raw_preset.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Preset index {preset_index} has invalid 'name'.")
    x_column = raw_preset.get("x_column")
    if x_column is not None and not isinstance(x_column, str):
        raise ValueError(f"Preset {name!r} has invalid 'x_column'.")
    raw_entries = raw_preset.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(f"Preset {name!r} must contain a non-empty 'entries' list.")
    entries = tuple(
        parse_preset_entry(raw_entry, preset_name=name, entry_index=entry_index)
        for entry_index, raw_entry in enumerate(raw_entries, start=1)
    )
    return PlotPreset(name=name, entries=entries, x_column=x_column)


def load_plot_presets(path: Path) -> tuple[PlotPreset, ...]:
    if not path.exists():
        return ()
    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    raw_presets = raw_payload.get("presets") if isinstance(raw_payload, dict) else raw_payload
    if not isinstance(raw_presets, list):
        raise ValueError("Preset file must contain a list or {'presets': [...]} format.")
    return tuple(
        parse_plot_preset(raw_preset, preset_index=preset_index)
        for preset_index, raw_preset in enumerate(raw_presets, start=1)
    )


def load_plot_presets_safe(path: Path) -> tuple[PlotPreset, ...]:
    try:
        return load_plot_presets(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Failed to load plot presets from {path}: {exc}", file=sys.stderr)
        return ()


PLOT_PRESETS: tuple[PlotPreset, ...] = load_plot_presets_safe(PRESETS_PATH)


def read_columns(path: Path, delimiter: str) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} has no header row") from exc
    columns = [name.strip() for name in header if name.strip()]
    if not columns:
        raise ValueError(f"{path} has an empty header row")
    return columns


def resolve_common_columns(paths: Sequence[Path], delimiter: str) -> tuple[list[str], bool]:
    columns_by_path = [read_columns(path, delimiter) for path in paths]
    base_columns = columns_by_path[0]
    base_set = set(base_columns)
    mismatch = any(set(columns) != base_set for columns in columns_by_path[1:])
    intersection = set(base_columns)
    for columns in columns_by_path[1:]:
        intersection &= set(columns)
    if not intersection:
        return [], mismatch
    ordered = [column for column in base_columns if column in intersection]
    return ordered, mismatch


def parse_histogram_list(
    raw: str | None,
    column: str,
    path: Path,
    row_index: int,
) -> list[float]:
    if raw is None or raw.strip() == "":
        raise ValueError(
            f"Missing histogram data in {path} row {row_index} column {column}."
        )
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Invalid histogram data in {path} row {row_index} column {column}: {raw!r}"
        ) from exc
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(
            f"Histogram data in {path} row {row_index} column {column} is not a list."
        )
    values: list[float] = []
    for value in parsed:
        try:
            values.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Histogram data in {path} row {row_index} column {column} is not numeric."
            ) from exc
    if not values:
        raise ValueError(
            f"Histogram data in {path} row {row_index} column {column} is empty."
        )
    return values


def is_missing_histogram_cell(raw: str | None) -> bool:
    if raw is None:
        return True
    stripped = raw.strip()
    if stripped == "":
        return True
    return stripped.lower() in {"none", "null", "nan"}


def build_edges_from_centers(centers: Sequence[float]) -> list[float]:
    if not centers:
        raise ValueError("Cannot build edges from empty centers.")
    if len(centers) == 1:
        center = centers[0]
        return [center - 0.5, center + 0.5]
    edges = [centers[0] - (centers[1] - centers[0]) / 2]
    for index in range(len(centers) - 1):
        left = centers[index]
        right = centers[index + 1]
        edges.append((left + right) / 2)
    edges.append(centers[-1] + (centers[-1] - centers[-2]) / 2)
    return edges


def exponential_moving_average(values: Sequence[float], alpha: float) -> list[float]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("EMA alpha must be between 0 and 1.")
    pre_exponential_samples = max(1, int(round(1.0 / alpha)))
    result: list[float] = []
    ema_value = math.nan
    pre_exponential_count = 0
    pre_exponential_sum = 0.0
    for value in values:
        if math.isnan(value):
            result.append(math.nan)
            continue
        if pre_exponential_count < pre_exponential_samples:
            pre_exponential_count += 1
            pre_exponential_sum += value
            ema_value = pre_exponential_sum / pre_exponential_count
            result.append(ema_value)
            continue
        ema_value = alpha * value + (1.0 - alpha) * ema_value
        result.append(ema_value)
    return result


def histogram_edges_match(reference: Sequence[float], candidate: Sequence[float]) -> bool:
    if len(reference) != len(candidate):
        return False
    return all(
        math.isclose(left, right, rel_tol=1e-3, abs_tol=1e-4)
        for left, right in zip(reference, candidate, strict=True)
    )


def validate_histogram_edges(edges: Sequence[float]) -> None:
    if len(edges) < 2:
        raise ValueError("Histogram edges must contain at least two values.")
    for left, right in zip(edges, edges[1:]):
        if not right > left:
            raise ValueError("Histogram edges must be strictly increasing.")


def ensure_strict_histogram_edges(edges: Sequence[float]) -> list[float]:
    if len(edges) < 2:
        raise ValueError("Histogram edges must contain at least two values.")
    for left, right in zip(edges, edges[1:]):
        if right < left:
            raise ValueError("Histogram edges must be non-decreasing.")
    repaired = [float(edges[0])]
    for edge in edges[1:]:
        value = float(edge)
        if value <= repaired[-1]:
            value = math.nextafter(repaired[-1], math.inf)
        repaired.append(value)
    return repaired


def histogram_value_range(series_list: Sequence[HistogramSeries]) -> tuple[float, float]:
    min_value = math.inf
    max_value = -math.inf
    for series in series_list:
        for row in series.values:
            for value in row:
                if math.isnan(value):
                    continue
                min_value = min(min_value, value)
                max_value = max(max_value, value)
    if min_value == math.inf or max_value == -math.inf:
        return 0.0, 1.0
    if min_value == max_value:
        return min_value - 1.0, max_value + 1.0
    return min_value, max_value


def log_histogram_values(values: Sequence[Sequence[float]]) -> list[list[float]]:
    return [
        [
            math.log(value + 0.005) if not math.isnan(value) else math.nan
            for value in row
        ]
        for row in values
    ]


def resolve_histogram_edges(series: HistogramSeries) -> list[float]:
    bin_count = len(series.values[0])
    if series.bin_edges is None:
        return [float(index) for index in range(bin_count + 1)]
    return normalize_histogram_edges(series.bin_edges, bin_count)


def transpose_histogram(values: Sequence[Sequence[float]]) -> list[list[float]]:
    if not values:
        return []
    bin_count = len(values[0])
    return [
        [values[time_index][bin_index] for time_index in range(len(values))]
        for bin_index in range(bin_count)
    ]


def normalize_histogram_edges(edges: Sequence[float], bin_count: int) -> list[float]:
    if len(edges) == bin_count + 1:
        normalized = list(edges)
    elif len(edges) == bin_count:
        normalized = build_edges_from_centers(edges)
    else:
        raise ValueError(
            f"Histogram edges length {len(edges)} does not match bins {bin_count}."
        )
    return ensure_strict_histogram_edges(normalized)


def linear_edges_from_range(min_edge: float, max_edge: float, bin_count: int) -> list[float]:
    if bin_count <= 0:
        raise ValueError("bin_count must be positive.")
    if math.isclose(min_edge, max_edge, rel_tol=0.0, abs_tol=0.0):
        span = 1.0
        min_edge -= span / 2
        max_edge += span / 2
    step = (max_edge - min_edge) / bin_count
    return [min_edge + step * index for index in range(bin_count + 1)]


def rebin_histogram_row(
    source_values: Sequence[float],
    source_edges: Sequence[float],
    target_edges: Sequence[float],
) -> list[float]:
    source_bin_count = len(source_values)
    if len(source_edges) != source_bin_count + 1:
        raise ValueError("Source histogram edges do not match source bin count.")
    target_bin_count = len(target_edges) - 1
    if target_bin_count <= 0:
        raise ValueError("Target histogram edges must define at least one bin.")
    validate_histogram_edges(source_edges)
    validate_histogram_edges(target_edges)
    rebinned = [0.0] * target_bin_count
    source_index = 0
    target_index = 0
    while source_index < source_bin_count and target_index < target_bin_count:
        source_left = source_edges[source_index]
        source_right = source_edges[source_index + 1]
        target_left = target_edges[target_index]
        target_right = target_edges[target_index + 1]
        overlap_left = max(source_left, target_left)
        overlap_right = min(source_right, target_right)
        if overlap_right > overlap_left:
            source_width = source_right - source_left
            rebinned[target_index] += (
                source_values[source_index] * (overlap_right - overlap_left) / source_width
            )
        if source_right <= target_right:
            source_index += 1
        if target_right <= source_right:
            target_index += 1
    return rebinned


def pool_histogram_along_x(
    x_values: Sequence[float],
    values: Sequence[Sequence[float]],
    pooling: int,
) -> tuple[list[float], list[list[float]]]:
    if pooling <= 1 or not values:
        return list(x_values), [list(row) for row in values]
    full_window_count = len(values) // pooling
    if full_window_count == 0:
        return [], []
    bin_count = len(values[0])
    pooled_x_values: list[float] = []
    pooled_values: list[list[float]] = []
    for start in range(0, full_window_count * pooling, pooling):
        x_chunk = x_values[start : start + pooling]
        value_chunk = values[start : start + pooling]
        pooled_x_values.append(sum(x_chunk) / len(x_chunk))
        pooled_values.append(
            [
                sum(row[bin_index] for row in value_chunk) / len(value_chunk)
                for bin_index in range(bin_count)
            ]
        )
    return pooled_x_values, pooled_values


def histogram_status_message(histogram_data: dict[str, list[HistogramSeries]]) -> str | None:
    drifting: list[str] = []
    for column, series_list in histogram_data.items():
        if any(series.edges_status.startswith("drifting") for series in series_list):
            drifting.append(column)
    if not drifting:
        return None
    joined = ", ".join(drifting)
    return f"Histogram edges drift for {joined}; values were rebinned onto a shared edge grid."


def resolve_x_datetime_flag(
    logs: Sequence[plot_logs.LogSeries],
    histogram_data: dict[str, list[HistogramSeries]],
    x_column: str,
) -> bool:
    flags: set[bool] = set()
    flags.update(log.x_is_datetime for log in logs)
    for series_list in histogram_data.values():
        flags.update(series.x_is_datetime for series in series_list)
    if len(flags) > 1:
        raise ValueError(f"Mixed numeric and timestamp values in X column {x_column}.")
    return next(iter(flags), False)


def load_histogram_series(
    path: Path,
    label: str,
    x_column: str,
    freqs_column: str,
    edges_column: str | None,
    delimiter: str,
    target_bin_count: int | None = None,
    x_pooling: int = 1,
    from_index: int = 0,
    to_index: int | None = None,
) -> HistogramSeries:
    if from_index < 0:
        raise ValueError("From must be >= 0.")
    if to_index is not None and to_index < from_index:
        raise ValueError("To must be >= From.")
    x_values: list[float] = []
    x_is_datetime: bool | None = None
    values: list[list[float]] = []
    bin_edges: list[float] | None = None
    edges_status = "missing" if edges_column is None else "edges"
    source_rows: list[tuple[list[float] | None, list[float] | None]] = []
    max_source_bin_count = 0
    global_min_edge: float | None = None
    global_max_edge: float | None = None
    reference_edges: list[float] | None = None
    saw_drifting_edges = False
    saw_missing_row_edges = False
    valid_row_index = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        required_columns = [x_column, freqs_column]
        if edges_column is not None:
            required_columns.append(edges_column)
        plot_logs.validate_columns(reader.fieldnames, required_columns, path)
        for row_index, row in enumerate(reader, start=2):
            x_value, is_datetime = plot_logs.parse_x_value(
                row.get(x_column),
                x_column,
                path,
                row_index,
            )
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
            raw_freqs = row.get(freqs_column)
            if is_missing_histogram_cell(raw_freqs):
                x_values.append(x_value)
                source_rows.append((None, None))
                continue
            freqs = parse_histogram_list(raw_freqs, freqs_column, path, row_index)
            source_bin_count = len(freqs)
            max_source_bin_count = max(max_source_bin_count, source_bin_count)
            source_edges: list[float] | None = None
            if edges_column is not None:
                raw_edges = row.get(edges_column)
                if is_missing_histogram_cell(raw_edges):
                    source_edges = None
                    saw_missing_row_edges = True
                else:
                    edges = parse_histogram_list(raw_edges, edges_column, path, row_index)
                    normalized = normalize_histogram_edges(edges, source_bin_count)
                    validate_histogram_edges(normalized)
                    source_edges = normalized
                    if reference_edges is None:
                        reference_edges = normalized
                    elif not histogram_edges_match(reference_edges, normalized):
                        saw_drifting_edges = True
            x_values.append(x_value)
            if source_edges is None and edges_column is None:
                source_edges = [float(index) for index in range(source_bin_count + 1)]
            if source_edges is not None:
                global_min_edge = (
                    source_edges[0]
                    if global_min_edge is None
                    else min(global_min_edge, source_edges[0])
                )
                global_max_edge = (
                    source_edges[-1]
                    if global_max_edge is None
                    else max(global_max_edge, source_edges[-1])
                )
            source_rows.append((freqs, source_edges))
    if not x_values:
        raise ValueError(f"No histogram data in {path} for {freqs_column}.")
    if max_source_bin_count <= 0:
        raise ValueError(f"No histogram data in {path} for {freqs_column}.")
    resolved_target_bin_count = (
        target_bin_count if target_bin_count is not None else max_source_bin_count
    )
    if global_min_edge is None or global_max_edge is None:
        global_min_edge = 0.0
        global_max_edge = float(max_source_bin_count)
    target_edges = linear_edges_from_range(
        global_min_edge,
        global_max_edge,
        resolved_target_bin_count,
    )
    values = []
    for freqs, source_edges in source_rows:
        if freqs is None or source_edges is None:
            values.append([0.0] * resolved_target_bin_count)
            continue
        values.append(rebin_histogram_row(freqs, source_edges, target_edges))
    bin_edges = target_edges
    if edges_column is not None:
        if saw_drifting_edges and saw_missing_row_edges:
            edges_status = "drifting-missing"
        elif saw_drifting_edges:
            edges_status = "drifting-range"
        elif saw_missing_row_edges:
            edges_status = "missing"
        else:
            edges_status = "edges"
    if x_pooling <= 0:
        raise ValueError("Histogram pooling must be >= 1.")
    if x_pooling > 1:
        x_values, values = pool_histogram_along_x(x_values, values, x_pooling)
        if not x_values:
            raise ValueError(
                f"Histogram pooling ({x_pooling}) is larger than available timesteps "
                f"in {path} for {freqs_column}."
            )
    return HistogramSeries(
        label=label,
        x_values=x_values,
        x_is_datetime=bool(x_is_datetime),
        bin_edges=bin_edges,
        values=values,
        edges_status=edges_status,
    )


class PlotLogsInteractiveApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Plot Logs Interactive")
        self.root.minsize(900, 600)
        self.root.columnconfigure(0, minsize=500)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.paths: list[Path] = []
        self.path_groups: dict[Path, str] = {}
        self.available_columns: list[str] = []
        self.plot_rows: list[PlotRow] = []
        self.typeahead_buffers: dict[tk.Widget, str] = {}
        self.typeahead_after_ids: dict[tk.Widget, str] = {}
        self.combo_values_getters: dict[ttk.Combobox, Callable[[], Sequence[str]]] = {}
        self.listbox_owner: dict[str, ttk.Combobox] = {}
        self.histogram_bin_count_cache: dict[
            tuple[tuple[str, ...], str, str],
            int | None,
        ] = {}
        self.previous_y_by_row: dict[int, str] = {}
        self.controls_canvas: tk.Canvas | None = None
        self.controls_canvas_window: int | None = None
        self.figure: plt.Figure | None = None
        self.canvas: FigureCanvasTkAgg | None = None
        self.toolbar: NavigationToolbar2Tk | None = None
        self.updating_row_index_widgets = False
        self.active_plot_tab_index = 0
        self.plot_tab_payloads: list[dict[str, object] | None] = [None]
        self.plot_tabs_frame: ttk.Frame | None = None
        self.plot_tab_grid_column_count = 0
        self.plot_tab_buttons: list[ttk.Button] = []
        self.plot_add_tab_button: ttk.Button | None = None

        self.delimiter_var = tk.StringVar(value=";")
        self.auto_refresh_interval_var = tk.StringVar(value="0")
        self.title_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Select CSV files to begin.")
        self.global_ema_var = tk.StringVar()
        self.global_ema_only_var = tk.BooleanVar(value=False)
        self.group_var = tk.StringVar()
        self.from_var = tk.StringVar(value="0")
        self.to_var = tk.StringVar(value="")
        self.line_alpha_var = tk.DoubleVar(value=1.0)
        self.line_width_var = tk.DoubleVar(value=0.75)
        self.dark_mode_var = tk.BooleanVar(value=True)
        self.file_opacity_var = tk.StringVar(value="1.0")
        self.file_opacity_by_path: dict[Path, float] = {}
        self.file_color_var = tk.StringVar(value="")
        self.file_color_by_path: dict[Path, str] = {}
        self.path_enabled: dict[Path, bool] = {}
        self.path_order_var = tk.StringVar(value=DEFAULT_PATH_ORDER_MODE)
        self.path_added_order: dict[Path, int] = {}
        self.path_added_order_counter = 0
        self.csv_source_folder: Path | None = None
        self.csv_source_folder_var = tk.StringVar(value="Source folder: (none)")
        self.reload_folder_button: ttk.Button | None = None
        self.light_figure_palette = {
            "figure_face": matplotlib.rcParams["figure.facecolor"],
            "axes_face": matplotlib.rcParams["axes.facecolor"],
            "text": matplotlib.rcParams["text.color"],
            "spine": matplotlib.rcParams["axes.edgecolor"],
            "grid": matplotlib.rcParams["grid.color"],
            "legend_face": matplotlib.rcParams["legend.facecolor"],
        }
        self.auto_refresh_after_id: str | None = None
        self.global_ema_entry: ttk.Entry | None = None
        self.global_ema_container: ttk.Frame | None = None
        self.global_ema_only_check: ttk.Checkbutton | None = None

        self._build_layout()
        self.auto_refresh_interval_var.trace_add("write", self.on_auto_refresh_interval_change)
        self.apply_theme(self.dark_mode_var.get())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind_all("<KeyPress>", self.on_global_keypress, add=True)
        self.root.bind_all("<space>", self.on_spacebar, add=True)
        self.add_plot_row()
        self.capture_current_plot_tab_payload()

    def _build_layout(self) -> None:
        controls_container = ttk.Frame(self.root)
        controls_container.grid(row=0, column=0, sticky="nsew")
        controls_container.rowconfigure(0, weight=1)
        controls_container.columnconfigure(0, weight=1)

        self.controls_canvas = tk.Canvas(controls_container, highlightthickness=0, borderwidth=0)
        self.controls_canvas.grid(row=0, column=0, sticky="nsew")
        controls_scrollbar = ttk.Scrollbar(
            controls_container,
            orient="vertical",
            command=self.controls_canvas.yview,
        )
        controls_scrollbar.grid(row=0, column=1, sticky="ns")
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

        controls_frame = ttk.Frame(self.controls_canvas, padding=8)
        self.controls_canvas_window = self.controls_canvas.create_window(
            (0, 0),
            window=controls_frame,
            anchor="nw",
        )
        controls_frame.columnconfigure(0, weight=1)
        controls_frame.bind("<Configure>", self.on_controls_frame_configure)
        self.controls_canvas.bind("<Configure>", self.on_controls_canvas_configure)
        self.controls_canvas.bind("<Enter>", self.on_controls_canvas_enter)
        self.controls_canvas.bind("<Leave>", self.on_controls_canvas_leave)

        plot_frame = ttk.Frame(self.root, padding=8)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.rowconfigure(1, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        settings_frame = ttk.LabelFrame(controls_frame, text="Settings", padding=(4, 4))
        settings_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        settings_frame.columnconfigure(1, weight=1)

        ttk.Label(settings_frame, text="Delimiter").grid(row=0, column=0, sticky="w")
        delimiter_entry = ttk.Entry(settings_frame, textvariable=self.delimiter_var, width=4)
        delimiter_entry.grid(row=0, column=1, sticky="w")

        ttk.Label(settings_frame, text="Title").grid(row=1, column=0, sticky="w", pady=(6, 0))
        title_entry = ttk.Entry(settings_frame, textvariable=self.title_var)
        title_entry.grid(row=1, column=1, sticky="ew", pady=(6, 0))

        ttk.Label(settings_frame, text="Line Opacity").grid(row=2, column=0, sticky="w", pady=(6, 0))
        opacity_scale = ttk.Scale(
            settings_frame,
            from_=0.0,
            to=1.0,
            variable=self.line_alpha_var,
            command=self.on_opacity_change,
        )
        opacity_scale.grid(row=2, column=1, sticky="ew", pady=(6, 0))

        ttk.Label(settings_frame, text="Line Width").grid(row=3, column=0, sticky="w", pady=(6, 0))
        width_scale = ttk.Scale(
            settings_frame,
            from_=0.1,
            to=1.0,
            variable=self.line_width_var,
            command=self.on_line_width_change,
        )
        width_scale.grid(row=3, column=1, sticky="ew", pady=(6, 0))

        ttk.Label(settings_frame, text="Dark Mode").grid(row=4, column=0, sticky="w", pady=(6, 0))
        dark_mode_check = ttk.Checkbutton(
            settings_frame,
            text="Dark Mode",
            variable=self.dark_mode_var,
            command=self.on_theme_toggle,
        )
        dark_mode_check.grid(row=4, column=1, sticky="w", pady=(6, 0))

        settings_buttons = ttk.Frame(settings_frame)
        settings_buttons.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        settings_buttons.columnconfigure(0, weight=1, uniform="settings_buttons")
        settings_buttons.columnconfigure(1, weight=1, uniform="settings_buttons")
        settings_buttons.columnconfigure(2, weight=1, uniform="settings_buttons")
        save_config_button = ttk.Button(
            settings_buttons,
            text="Save Config",
            command=self.save_config_to_file,
        )
        save_config_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        load_config_button = ttk.Button(
            settings_buttons,
            text="Load Config",
            command=self.load_config_from_file,
        )
        load_config_button.grid(row=0, column=1, sticky="ew", padx=(0, 4))
        load_recent_button = ttk.Button(
            settings_buttons,
            text="Load Recent",
            command=self.load_saved_paths,
        )
        load_recent_button.grid(row=0, column=2, sticky="ew")

        files_frame = ttk.LabelFrame(controls_frame, text="CSV Files", padding=(4, 4))
        files_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        files_frame.columnconfigure(0, weight=1)

        source_folder_label = ttk.Label(
            files_frame,
            textvariable=self.csv_source_folder_var,
            wraplength=240,
            justify="left",
        )
        source_folder_label.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        files_order_frame = ttk.Frame(files_frame)
        files_order_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        files_order_frame.columnconfigure(1, weight=1)
        ttk.Label(files_order_frame, text="Order").grid(row=0, column=0, sticky="w")
        files_order_combo = ttk.Combobox(
            files_order_frame,
            state="readonly",
            values=PATH_ORDER_MODES,
            textvariable=self.path_order_var,
        )
        files_order_combo.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        files_order_combo.bind("<<ComboboxSelected>>", self.on_path_order_change)

        self.files_listbox = tk.Listbox(files_frame, height=6, selectmode="extended")
        self.files_listbox.grid(row=2, column=0, sticky="ew")
        self.files_listbox.bind("<<ListboxSelect>>", self.on_file_selection)
        files_scrollbar = ttk.Scrollbar(files_frame, orient="vertical", command=self.files_listbox.yview)
        files_scrollbar.grid(row=2, column=1, sticky="ns")
        self.files_listbox.configure(yscrollcommand=files_scrollbar.set)

        files_buttons = ttk.Frame(files_frame)
        files_buttons.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        files_buttons.columnconfigure(0, weight=1, uniform="files_buttons")
        files_buttons.columnconfigure(1, weight=1, uniform="files_buttons")
        add_button = ttk.Button(files_buttons, text="Add CSV Files", command=self.add_files)
        add_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        remove_button = ttk.Button(files_buttons, text="Remove Selected", command=self.remove_selected_files)
        remove_button.grid(row=0, column=1, sticky="ew")
        move_up_button = ttk.Button(files_buttons, text="Move Up", command=lambda: self.move_selected_files(-1))
        move_up_button.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(4, 0))
        move_down_button = ttk.Button(files_buttons, text="Move Down", command=lambda: self.move_selected_files(1))
        move_down_button.grid(row=1, column=1, sticky="ew", pady=(4, 0))
        enable_button = ttk.Button(
            files_buttons,
            text="Enable Selected",
            command=lambda: self.set_enabled_for_selection(True),
        )
        enable_button.grid(row=2, column=0, sticky="ew", padx=(0, 4), pady=(4, 0))
        disable_button = ttk.Button(
            files_buttons,
            text="Disable Selected",
            command=lambda: self.set_enabled_for_selection(False),
        )
        disable_button.grid(row=2, column=1, sticky="ew", pady=(4, 0))
        self.reload_folder_button = ttk.Button(
            files_buttons,
            text="Reload Folder CSVs",
            command=self.reload_folder_csv_files,
            state="disabled",
        )
        self.reload_folder_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        plot_grad_norms_button = ttk.Button(
            files_buttons,
            text="Plot Grad Norms",
            command=self.plot_grad_norms,
        )
        plot_grad_norms_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        meta_frame = ttk.Frame(files_frame)
        meta_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        meta_frame.columnconfigure(1, weight=1)

        ttk.Label(meta_frame, text="Group").grid(row=0, column=0, sticky="w")
        group_entry = ttk.Entry(meta_frame, textvariable=self.group_var)
        group_entry.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        set_group_button = ttk.Button(meta_frame, text="Set", width=5, command=self.set_group_for_selection)
        set_group_button.grid(row=0, column=2, sticky="w")
        clear_group_button = ttk.Button(meta_frame, text="Clear", width=5, command=self.clear_group_for_selection)
        clear_group_button.grid(row=0, column=3, sticky="w", padx=(6, 0))

        ttk.Label(meta_frame, text="Opacity").grid(row=1, column=0, sticky="w", pady=(6, 0))
        opacity_entry = ttk.Entry(meta_frame, textvariable=self.file_opacity_var)
        opacity_entry.grid(row=1, column=1, sticky="ew", padx=(6, 6), pady=(6, 0))
        set_opacity_button = ttk.Button(
            meta_frame,
            text="Set",
            width=5,
            command=self.set_opacity_for_selection,
        )
        set_opacity_button.grid(row=1, column=2, sticky="w", pady=(6, 0))
        reset_opacity_button = ttk.Button(
            meta_frame,
            text="Reset",
            width=5,
            command=lambda: self.set_opacity_for_selection(reset=True),
        )
        reset_opacity_button.grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(6, 0))

        ttk.Label(meta_frame, text="Color").grid(row=2, column=0, sticky="w", pady=(6, 0))
        color_entry = ttk.Entry(meta_frame, textvariable=self.file_color_var)
        color_entry.grid(row=2, column=1, sticky="ew", padx=(6, 6), pady=(6, 0))
        pick_color_button = ttk.Button(
            meta_frame,
            text="Pick",
            width=5,
            command=self.pick_color_for_selection,
        )
        pick_color_button.grid(row=2, column=2, sticky="w", pady=(6, 0))
        clear_color_button = ttk.Button(
            meta_frame,
            text="Clear",
            width=5,
            command=lambda: self.set_color_for_selection(reset=True),
        )
        clear_color_button.grid(row=2, column=3, sticky="w", padx=(6, 0), pady=(6, 0))

        plots_frame = ttk.LabelFrame(controls_frame, text="Plots", padding=(4, 4))
        plots_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        plots_frame.columnconfigure(0, weight=1)

        self.plot_tabs_frame = ttk.Frame(plots_frame)
        self.plot_tabs_frame.grid(row=0, column=0, sticky="ew")
        self.rebuild_plot_tab_controls()

        x_column_frame = ttk.Frame(plots_frame)
        x_column_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        x_column_frame.columnconfigure(1, weight=1)
        ttk.Label(x_column_frame, text="X Column").grid(row=0, column=0, sticky="w")
        self.x_combo = ttk.Combobox(x_column_frame, state="disabled")
        self.x_combo.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.bind_typeahead(self.x_combo, lambda: self.available_columns)
        range_frame = ttk.Frame(x_column_frame)
        range_frame.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Label(x_column_frame, text="From").grid(row=1, column=0, sticky="w", pady=(6, 0))
        from_entry = ttk.Entry(range_frame, textvariable=self.from_var, width=8)
        from_entry.grid(row=0, column=0, sticky="w")
        ttk.Label(range_frame, text="To").grid(row=0, column=1, sticky="w", padx=(8, 0))
        to_entry = ttk.Entry(range_frame, textvariable=self.to_var, width=8)
        to_entry.grid(row=0, column=2, sticky="w", padx=(6, 0))

        self.plots_container = ttk.Frame(plots_frame)
        self.plots_container.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.plots_container.columnconfigure(1, weight=1, minsize=100)
        self.plots_container.columnconfigure(2, weight=0)
        self.plots_container.columnconfigure(3, weight=0)
        self.plots_container.columnconfigure(4, weight=0, minsize=20)
        self.plots_container.columnconfigure(5, weight=0, minsize=20)
        self.plots_container.columnconfigure(6, weight=0, minsize=20)
        self.plots_container.columnconfigure(7, weight=0)

        ttk.Label(self.plots_container, text="#").grid(row=0, column=0, sticky="w")
        ttk.Label(self.plots_container, text="Y Column").grid(row=0, column=1, sticky="w", padx=(10, 4))
        ttk.Label(self.plots_container, text="Height").grid(row=0, column=2, sticky="w", padx=(6, 4))
        ttk.Label(
            self.plots_container,
            text="EMA (only)\n/\nnbins;pool\n(log)",
            justify="center",
            anchor="center",
        ).grid(
            row=0,
            column=3,
            sticky="n",
            padx=(2, 4),
        )
        ttk.Label(self.plots_container, text="STD\n(ema)").grid(row=0, column=4, sticky="w", padx=(1, 1))
        ttk.Label(self.plots_container, text="Min\n(ema)").grid(row=0, column=5, sticky="w", padx=(1, 1))
        ttk.Label(self.plots_container, text="Max\n(ema)").grid(row=0, column=6, sticky="w", padx=(1, 1))
        ttk.Label(self.plots_container, text="").grid(row=0, column=7, sticky="w", padx=(6, 4))

        plots_buttons = ttk.Frame(plots_frame)
        plots_buttons.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        plots_buttons.columnconfigure(0, weight=1, uniform="plots_buttons")
        plots_buttons.columnconfigure(1, weight=1, uniform="plots_buttons")
        plots_buttons.columnconfigure(2, weight=1, uniform="plots_buttons")
        add_plot_button = ttk.Button(plots_buttons, text="Add Plot", command=self.add_plot_row)
        add_plot_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        clear_plot_button = ttk.Button(plots_buttons, text="Clear Plots", command=self.clear_plot_rows)
        clear_plot_button.grid(row=0, column=1, sticky="ew", padx=(0, 4))
        refresh_button = ttk.Button(plots_buttons, text="Refresh Columns", command=self.refresh_columns)
        refresh_button.grid(row=0, column=2, sticky="ew")
        self.toggle_std_button = ttk.Button(
            self.plots_container,
            text="↑",
            command=self.toggle_all_std,
            width=2,
        )
        self.toggle_min_button = ttk.Button(
            self.plots_container,
            text="↑",
            command=self.toggle_all_min,
            width=2,
        )
        self.toggle_max_button = ttk.Button(
            self.plots_container,
            text="↑",
            command=self.toggle_all_max,
            width=2,
        )
        self.global_ema_container = ttk.Frame(self.plots_container)
        self.global_ema_entry = ttk.Entry(
            self.global_ema_container,
            textvariable=self.global_ema_var,
            width=5,
        )
        self.global_ema_only_check = ttk.Checkbutton(
            self.global_ema_container,
            text="",
            variable=self.global_ema_only_var,
            padding=0,
            command=self.on_global_ema_only_toggle,
        )
        self.global_ema_entry.grid(row=0, column=0, sticky="w")
        self.global_ema_only_check.grid(row=0, column=1, padx=(3, 0))
        self.global_ema_entry.bind("<Return>", self.on_global_ema_submit)
        self.global_ema_entry.bind("<FocusOut>", self.on_global_ema_focus_out)

        if PLOT_PRESETS:
            presets_label = ttk.Label(plots_frame, text="Presets")
            presets_label.grid(row=4, column=0, sticky="w", pady=(8, 0))
            presets_frame = ttk.Frame(plots_frame)
            presets_frame.grid(row=5, column=0, sticky="ew", pady=(4, 0))
            preset_columns = 3
            for column in range(preset_columns):
                presets_frame.columnconfigure(column, weight=1)
            for index, preset in enumerate(PLOT_PRESETS):
                button = ttk.Button(
                    presets_frame,
                    text=preset.name,
                    command=lambda target=preset: self.apply_preset(target),
                )
                button.grid(
                    row=index // preset_columns,
                    column=index % preset_columns,
                    sticky="ew",
                    padx=2,
                    pady=2,
                )

        action_frame = ttk.Frame(controls_frame)
        action_frame.grid(row=3, column=0, sticky="ew")
        action_frame.columnconfigure(0, weight=1)
        plot_button = ttk.Button(action_frame, text="Plot", command=self.plot, style="Accent.TButton")
        plot_button.grid(row=0, column=0, sticky="ew")
        ttk.Label(action_frame, text="Auto refresh (s)").grid(row=0, column=1, sticky="w", padx=(8, 4))
        auto_refresh_entry = ttk.Entry(
            action_frame,
            textvariable=self.auto_refresh_interval_var,
            width=6,
        )
        auto_refresh_entry.grid(row=0, column=2, sticky="w")

        status_label = ttk.Label(controls_frame, textvariable=self.status_var, wraplength=240)
        status_label.grid(row=4, column=0, sticky="w", pady=(6, 0))

        self.toolbar_frame = ttk.Frame(plot_frame)
        self.toolbar_frame.grid(row=0, column=0, sticky="ew")
        self.canvas_frame = ttk.Frame(plot_frame)
        self.canvas_frame.grid(row=1, column=0, sticky="nsew")

    def on_controls_frame_configure(self, _event: tk.Event) -> None:
        if not self.controls_canvas:
            return
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

    def on_controls_canvas_configure(self, event: tk.Event) -> None:
        if not self.controls_canvas or self.controls_canvas_window is None:
            return
        self.controls_canvas.itemconfigure(self.controls_canvas_window, width=event.width)

    def on_controls_canvas_enter(self, _event: tk.Event) -> None:
        self.root.bind_all("<MouseWheel>", self.on_controls_mousewheel, add=True)

    def on_controls_canvas_leave(self, _event: tk.Event) -> None:
        self.root.unbind_all("<MouseWheel>")

    def on_controls_mousewheel(self, event: tk.Event) -> None:
        if not self.controls_canvas or event.delta == 0:
            return
        if isinstance(event.widget, tk.Listbox):
            return
        self.controls_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def add_paths(self, found_paths: Sequence[Path], source_label: str) -> None:
        if not found_paths:
            self.set_status(f"No CSV files found in the selected {source_label}.")
            return
        added_paths: list[Path] = []
        for path in sorted(found_paths):
            if path not in self.paths:
                self.paths.append(path)
                self.path_added_order_counter += 1
                self.path_added_order[path] = self.path_added_order_counter
                added_paths.append(path)
            self.file_opacity_by_path.setdefault(path, 1.0)
            self.file_color_by_path.setdefault(path, "")
            self.path_enabled.setdefault(path, True)
        if len(added_paths) > 5:
            for path in added_paths:
                self.path_enabled[path] = False
        self.refresh_file_list()
        self.refresh_columns(preserve_state=True)
        if not added_paths:
            self.set_status(f"All CSV files in the selected {source_label} are already added.")
        elif len(added_paths) > 5:
            self.set_status(f"Added {len(added_paths)} CSV files (disabled by default).")
        else:
            self.set_status(f"Added {len(added_paths)} CSV files.")

    def list_csv_files_in_folder(self, root_path: Path) -> list[Path]:
        return [
            path.resolve()
            for path in root_path.rglob("*.csv")
            if path.is_file()
        ]

    def set_csv_source_folder(self, folder: Path | None) -> None:
        self.csv_source_folder = folder
        if folder is None:
            self.csv_source_folder_var.set("Source folder: (none)")
        else:
            try:
                folder_display = str(folder.relative_to(REPO_ROOT))
            except ValueError:
                folder_display = str(folder)
            self.csv_source_folder_var.set(f"Source folder: {folder_display}")
        if self.reload_folder_button is not None:
            state = "normal" if folder is not None else "disabled"
            self.reload_folder_button.configure(state=state)

    def rebuild_added_order_from_paths(self) -> None:
        for index, path in enumerate(self.paths, start=1):
            self.path_added_order[path] = index
        self.path_added_order_counter = len(self.paths)

    def path_created_timestamp(self, path: Path) -> float | None:
        try:
            return path.stat().st_ctime
        except OSError:
            return None

    def sort_paths_by_selected_order(self) -> None:
        order_mode = self.path_order_var.get()
        if order_mode == PATH_ORDER_MODES[0]:
            self.paths.sort(key=lambda path: self.path_added_order.get(path, 0))
            return
        if order_mode == PATH_ORDER_MODES[1]:
            def oldest_sort_key(path: Path) -> tuple[bool, float, int]:
                created_at = self.path_created_timestamp(path)
                return (
                    created_at is None,
                    created_at if created_at is not None else 0.0,
                    self.path_added_order.get(path, 0),
                )
            self.paths.sort(
                key=oldest_sort_key
            )
            return
        if order_mode == PATH_ORDER_MODES[2]:
            def newest_sort_key(path: Path) -> tuple[bool, float, int]:
                created_at = self.path_created_timestamp(path)
                return (
                    created_at is None,
                    -(created_at if created_at is not None else 0.0),
                    self.path_added_order.get(path, 0),
                )
            self.paths.sort(
                key=newest_sort_key
            )
            return
        self.path_order_var.set(PATH_ORDER_MODES[0])
        self.paths.sort(key=lambda path: self.path_added_order.get(path, 0))

    def on_path_order_change(self, _event: tk.Event | None = None) -> None:
        self.refresh_file_list()

    def add_files(self) -> None:
        choice = messagebox.askyesnocancel(
            "Add CSV Files",
            "Choose source:\nYes = pick CSV files\nNo = pick a folder",
            parent=self.root,
        )
        if choice is None:
            return
        if choice:
            selected_paths = filedialog.askopenfilenames(
                title="Select CSV files",
                initialdir=str(REPO_ROOT),
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if not selected_paths:
                return
            found_paths = [
                Path(path).resolve()
                for path in selected_paths
                if Path(path).is_file() and Path(path).suffix.lower() == ".csv"
            ]
            self.add_paths(found_paths, "files")
            return
        directory = filedialog.askdirectory(
            title="Select folder containing CSV files",
            initialdir=str(REPO_ROOT),
        )
        if not directory:
            return
        root_path = Path(directory).resolve()
        found_paths = self.list_csv_files_in_folder(root_path)
        self.set_csv_source_folder(root_path)
        self.add_paths(found_paths, "folder")

    def reload_folder_csv_files(self) -> None:
        if self.csv_source_folder is None:
            self.set_status("No source folder selected yet.")
            return
        if not self.csv_source_folder.is_dir():
            self.show_error(f"Source folder not found: {self.csv_source_folder}")
            self.set_csv_source_folder(None)
            return
        found_paths = self.list_csv_files_in_folder(self.csv_source_folder)
        before_count = len(set(self.paths))
        self.add_paths(found_paths, "folder")
        added_count = len(set(self.paths)) - before_count
        if added_count == 0 and found_paths:
            self.set_status("No new CSV files found in the saved source folder.")
        elif found_paths:
            self.set_status(f"Reloaded folder and added {added_count} new CSV files.")

    def remove_selected_files(self) -> None:
        selected_indices = list(self.files_listbox.curselection())
        if not selected_indices:
            return
        for index in sorted(selected_indices, reverse=True):
            self.path_added_order.pop(self.paths[index], None)
            self.path_groups.pop(self.paths[index], None)
            self.file_opacity_by_path.pop(self.paths[index], None)
            self.file_color_by_path.pop(self.paths[index], None)
            self.path_enabled.pop(self.paths[index], None)
            del self.paths[index]
        if self.path_order_var.get() == PATH_ORDER_MODES[0]:
            self.rebuild_added_order_from_paths()
        self.refresh_file_list()
        self.refresh_columns(preserve_state=True)

    def refresh_file_list(self) -> None:
        self.sort_paths_by_selected_order()
        self.files_listbox.delete(0, tk.END)
        for path in self.paths:
            self.files_listbox.insert(tk.END, self.display_path(path))
        self.on_file_selection(None)

    def display_path(self, path: Path) -> str:
        group = self.path_groups.get(path)
        opacity = self.opacity_for_path(path)
        color = self.file_color_by_path.get(path, "")
        enabled = self.path_enabled.get(path, True)
        try:
            display = str(path.relative_to(REPO_ROOT))
        except ValueError:
            display = str(path)
        parts: list[str] = []
        if not enabled:
            parts.append("[disabled]")
        if group:
            parts.append(f"[{group}]")
        if not math.isclose(opacity, 1.0):
            parts.append(f"[α={self.format_opacity(opacity)}]")
        if color:
            parts.append(f"[c={color}]")
        if parts:
            return f"{' '.join(parts)} {display}"
        return display

    def format_opacity(self, value: float) -> str:
        text = f"{value:.2f}"
        return text.rstrip("0").rstrip(".")

    def opacity_for_path(self, path: Path) -> float:
        value = self.file_opacity_by_path.get(path, 1.0)
        return max(0.0, min(1.0, value))

    def on_file_selection(self, _event: tk.Event | None) -> None:
        selected_indices = list(self.files_listbox.curselection())
        if not selected_indices:
            self.group_var.set("")
            self.file_opacity_var.set("")
            self.file_color_var.set("")
            return
        groups = {self.path_groups.get(self.paths[index], "") for index in selected_indices}
        if len(groups) == 1:
            self.group_var.set(groups.pop())
        else:
            self.group_var.set("")
        opacities = {
            self.opacity_for_path(self.paths[index]) for index in selected_indices
        }
        if len(opacities) == 1:
            self.file_opacity_var.set(self.format_opacity(opacities.pop()))
        else:
            self.file_opacity_var.set("")
        colors = {
            self.file_color_by_path.get(self.paths[index], "") for index in selected_indices
        }
        colors.discard("")
        if len(colors) == 1:
            self.file_color_var.set(colors.pop())
        else:
            self.file_color_var.set("")

    def move_selected_files(self, direction: int) -> None:
        if direction not in {-1, 1}:
            return
        selected_indices = list(self.files_listbox.curselection())
        if not selected_indices:
            return
        selected_set = set(selected_indices)
        if direction < 0:
            for index in range(1, len(self.paths)):
                if index in selected_set and (index - 1) not in selected_set:
                    self.paths[index - 1], self.paths[index] = (
                        self.paths[index],
                        self.paths[index - 1],
                    )
                    selected_set.remove(index)
                    selected_set.add(index - 1)
        else:
            for index in range(len(self.paths) - 2, -1, -1):
                if index in selected_set and (index + 1) not in selected_set:
                    self.paths[index + 1], self.paths[index] = (
                        self.paths[index],
                        self.paths[index + 1],
                    )
                    selected_set.remove(index)
                    selected_set.add(index + 1)
        if self.path_order_var.get() == PATH_ORDER_MODES[0]:
            self.rebuild_added_order_from_paths()
        self.refresh_file_list()
        for index in sorted(selected_set):
            self.files_listbox.selection_set(index)

    def set_group_for_selection(self) -> None:
        selected_indices = list(self.files_listbox.curselection())
        if not selected_indices:
            return
        group = self.group_var.get().strip()
        if not group:
            self.show_error("Group name cannot be empty.")
            return
        for index in selected_indices:
            self.path_groups[self.paths[index]] = group
        self.refresh_file_list()
        for index in selected_indices:
            self.files_listbox.selection_set(index)

    def clear_group_for_selection(self) -> None:
        selected_indices = list(self.files_listbox.curselection())
        if not selected_indices:
            return
        for index in selected_indices:
            self.path_groups.pop(self.paths[index], None)
        self.group_var.set("")
        self.refresh_file_list()
        for index in selected_indices:
            self.files_listbox.selection_set(index)

    def set_opacity_for_selection(self, reset: bool = False) -> None:
        selected_indices = list(self.files_listbox.curselection())
        if not selected_indices:
            self.show_error("Select at least one file to set opacity.")
            return
        if reset:
            opacity = 1.0
        else:
            raw_value = self.file_opacity_var.get().strip()
            if not raw_value:
                self.show_error("Opacity cannot be empty.")
                return
            try:
                opacity = float(raw_value)
            except ValueError:
                self.show_error("Opacity must be a number between 0 and 1.")
                return
            if not (0.0 <= opacity <= 1.0):
                self.show_error("Opacity must be between 0 and 1.")
                return
        for index in selected_indices:
            self.file_opacity_by_path[self.paths[index]] = opacity
        self.refresh_file_list()
        for index in selected_indices:
            self.files_listbox.selection_set(index)

    def set_color_for_selection(self, reset: bool = False, color: str | None = None) -> None:
        selected_indices = list(self.files_listbox.curselection())
        if not selected_indices:
            self.show_error("Select at least one file to set color.")
            return
        if reset:
            color_value = ""
        else:
            raw_value = color or self.file_color_var.get().strip()
            if not raw_value:
                self.show_error("Color cannot be empty.")
                return
            color_value = raw_value.lower()
            if not self.is_valid_color(color_value):
                self.show_error("Color must be a hex value like #RRGGBB.")
                return
        for index in selected_indices:
            if color_value:
                self.file_color_by_path[self.paths[index]] = color_value
            else:
                self.file_color_by_path.pop(self.paths[index], None)
        self.refresh_file_list()
        for index in selected_indices:
            self.files_listbox.selection_set(index)

    def pick_color_for_selection(self) -> None:
        initial = self.file_color_var.get().strip()
        if not self.is_valid_color(initial):
            initial = None
        _rgb, hex_value = colorchooser.askcolor(color=initial, title="Pick line color")
        if not hex_value:
            return
        self.file_color_var.set(hex_value)
        self.set_color_for_selection(color=hex_value)

    def is_valid_color(self, value: str | None) -> bool:
        if not value:
            return False
        if not value.startswith("#") or len(value) != 7:
            return False
        return all(char in "0123456789abcdefABCDEF" for char in value[1:])

    def enabled_paths(self) -> list[Path]:
        return [path for path in self.paths if self.path_enabled.get(path, True)]

    def plot_grad_norms(self) -> None:
        enabled_paths = self.enabled_paths()
        if not self.paths:
            self.show_error("No CSV files selected.")
            return
        if not enabled_paths:
            self.show_error("Enable at least one CSV file to plot grad norms.")
            return

        script_path = Path(__file__).resolve().with_name("plot_grad_norms.py")
        if not script_path.exists():
            self.show_error(f"Grad norm plotting script not found: {script_path}")
            return

        delimiter = self.delimiter_var.get()
        if len(delimiter) != 1:
            self.show_error("Delimiter must be a single character.")
            return

        command = [
            sys.executable,
            str(script_path),
            "--delimiter",
            delimiter,
            *[str(path) for path in enabled_paths],
        ]

        try:
            process_env = os.environ.copy()
            process_env["MPLBACKEND"] = "TkAgg"
            subprocess.Popen(command, cwd=str(REPO_ROOT), env=process_env)
        except OSError as exc:
            self.show_error(f"Failed to launch grad norm plotter: {exc}")
            return
        self.set_status("Opened grad norm plotter for enabled CSV files.")

    def set_enabled_for_selection(self, enabled: bool) -> None:
        selected_indices = list(self.files_listbox.curselection())
        if not selected_indices:
            return
        for index in selected_indices:
            self.path_enabled[self.paths[index]] = enabled
        self.refresh_file_list()
        for index in selected_indices:
            self.files_listbox.selection_set(index)
        self.refresh_columns(preserve_state=True)

    def build_paths_payload(self) -> list[dict[str, str | float | bool]]:
        payload: list[dict[str, str | float | bool]] = []
        for path in self.paths:
            try:
                stored_path = str(path.relative_to(REPO_ROOT))
            except ValueError:
                stored_path = str(path)
            payload.append(
                {
                    "path": stored_path,
                    "group": self.path_groups.get(path, ""),
                    "opacity": self.opacity_for_path(path),
                    "color": self.file_color_by_path.get(path, ""),
                    "enabled": self.path_enabled.get(path, True),
                }
            )
        return payload

    def build_plot_payload(self) -> dict[str, object]:
        rows: list[dict[str, object]] = []
        for row in self.plot_rows:
            y_value = row.y_combo.get().strip()
            if not y_value:
                continue
            entry: dict[str, object] = {
                "y": y_value,
                "height": row.height_var.get().strip(),
                "ema": row.ema_var.get().strip(),
                "ema_only": row.ema_only_var.get(),
                "std": row.std_var.get(),
                "std_ema": row.std_ema_var.get(),
                "min": row.min_var.get(),
                "min_ema": row.min_ema_var.get(),
                "max": row.max_var.get(),
                "max_ema": row.max_ema_var.get(),
            }
            rows.append(entry)
        return {
            "x": self.x_combo.get().strip(),
            "from": self.from_var.get().strip(),
            "to": self.to_var.get().strip(),
            "rows": rows,
        }

    def capture_current_plot_tab_payload(self) -> None:
        if not self.plot_tab_payloads:
            self.plot_tab_payloads = [None]
            self.active_plot_tab_index = 0
        self.plot_tab_payloads[self.active_plot_tab_index] = self.build_plot_payload()
        self.refresh_plot_tab_buttons()

    def build_plot_tabs_payload(self) -> dict[str, object]:
        self.capture_current_plot_tab_payload()
        return {
            "active": self.active_plot_tab_index + 1,
            "tabs": list(self.plot_tab_payloads),
        }

    def build_config_payload(self) -> dict[str, object]:
        source_folder: str | None = None
        if self.csv_source_folder is not None:
            try:
                source_folder = str(self.csv_source_folder.relative_to(REPO_ROOT))
            except ValueError:
                source_folder = str(self.csv_source_folder)
        return {
            "paths": self.build_paths_payload(),
            "plots": self.build_plot_payload(),
            "plot_tabs": self.build_plot_tabs_payload(),
            "auto_refresh_interval": self.auto_refresh_interval_var.get().strip(),
            "csv_source_folder": source_folder,
            "path_order": self.path_order_var.get(),
        }

    def apply_auto_refresh_interval_payload(self, payload: dict[str, object]) -> None:
        interval_value = payload.get("auto_refresh_interval")
        if isinstance(interval_value, (int, float)):
            self.auto_refresh_interval_var.set(str(interval_value))
        elif isinstance(interval_value, str):
            self.auto_refresh_interval_var.set(interval_value.strip() or "0")

    def apply_path_order_payload(self, payload: dict[str, object]) -> None:
        order_value = payload.get("path_order")
        if isinstance(order_value, str) and order_value in PATH_ORDER_MODES:
            self.path_order_var.set(order_value)
        else:
            self.path_order_var.set(DEFAULT_PATH_ORDER_MODE)

    def apply_csv_source_folder_payload(self, payload: dict[str, object]) -> str | None:
        raw_source_folder = payload.get("csv_source_folder")
        if not isinstance(raw_source_folder, str) or not raw_source_folder.strip():
            self.set_csv_source_folder(None)
            return None
        folder_candidate = Path(raw_source_folder)
        if not folder_candidate.is_absolute():
            folder_candidate = (REPO_ROOT / folder_candidate).resolve()
        if not folder_candidate.is_dir():
            self.set_csv_source_folder(None)
            return raw_source_folder
        self.set_csv_source_folder(folder_candidate)
        return None

    def apply_paths_payload(self, payload: object) -> tuple[int, int]:
        if not isinstance(payload, list):
            raise ValueError("Saved paths file is invalid.")
        loaded_paths: list[Path] = []
        loaded_groups: dict[Path, str] = {}
        loaded_opacities: dict[Path, float] = {}
        loaded_colors: dict[Path, str] = {}
        loaded_enabled: dict[Path, bool] = {}
        missing_paths: list[str] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            raw_path = entry.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                continue
            group = entry.get("group")
            group_value = group if isinstance(group, str) else ""
            opacity_value = entry.get("opacity")
            opacity = 1.0
            if isinstance(opacity_value, (int, float)):
                opacity = float(opacity_value)
            elif isinstance(opacity_value, str):
                try:
                    opacity = float(opacity_value)
                except ValueError:
                    opacity = 1.0
            color_value = entry.get("color")
            color = color_value if isinstance(color_value, str) else ""
            enabled_value = entry.get("enabled")
            enabled = enabled_value if isinstance(enabled_value, bool) else True
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = (REPO_ROOT / candidate).resolve()
            if candidate.exists():
                if candidate not in loaded_paths:
                    loaded_paths.append(candidate)
                if group_value:
                    loaded_groups[candidate] = group_value
                if 0.0 <= opacity <= 1.0:
                    loaded_opacities[candidate] = opacity
                if self.is_valid_color(color):
                    loaded_colors[candidate] = color.lower()
                loaded_enabled[candidate] = enabled
            else:
                missing_paths.append(raw_path)
        self.paths = loaded_paths
        self.path_groups = loaded_groups
        self.file_opacity_by_path = loaded_opacities
        self.file_color_by_path = loaded_colors
        self.path_enabled = loaded_enabled
        self.path_added_order = {path: index for index, path in enumerate(self.paths, start=1)}
        self.path_added_order_counter = len(self.paths)
        self.refresh_file_list()
        self.refresh_columns()
        return len(loaded_paths), len(missing_paths)

    def apply_plot_payload(self, payload: object) -> tuple[int, list[str], str | None]:
        if not isinstance(payload, dict):
            raise ValueError("Saved plot configuration is invalid.")
        self.from_var.set("0")
        self.to_var.set("")
        rows_payload = payload.get("rows")
        if rows_payload is None:
            return 0, [], None
        if not isinstance(rows_payload, list):
            raise ValueError("Saved plot configuration is invalid.")
        missing_columns: list[str] = []
        missing_x: str | None = None
        if rows_payload:
            self.clear_plot_rows()
            for _ in range(len(rows_payload) - 1):
                self.add_plot_row()
            for row, row_data in zip(self.plot_rows, rows_payload, strict=True):
                if not isinstance(row_data, dict):
                    continue
                y_value = row_data.get("y")
                y_valid = False
                if isinstance(y_value, str) and y_value:
                    if self.available_columns and y_value in self.available_columns:
                        row.y_combo.set(y_value)
                        y_valid = True
                    else:
                        missing_columns.append(y_value)
                height_value = row_data.get("height")
                if isinstance(height_value, (int, float)):
                    row.height_var.set(self.format_ratio(float(height_value)))
                elif isinstance(height_value, str):
                    row.height_var.set(height_value)
                ema_value = row_data.get("ema")
                if isinstance(ema_value, (int, float)):
                    row.ema_var.set(str(ema_value))
                elif isinstance(ema_value, str):
                    row.ema_var.set(ema_value)
                ema_only_value = row_data.get("ema_only")
                if isinstance(ema_only_value, bool):
                    row.ema_only_var.set(ema_only_value)
                if y_valid:
                    desired_std = row_data.get("std") if isinstance(row_data.get("std"), bool) else None
                    desired_std_ema = (
                        row_data.get("std_ema") if isinstance(row_data.get("std_ema"), bool) else None
                    )
                    desired_min = row_data.get("min") if isinstance(row_data.get("min"), bool) else None
                    desired_min_ema = (
                        row_data.get("min_ema") if isinstance(row_data.get("min_ema"), bool) else None
                    )
                    desired_max = row_data.get("max") if isinstance(row_data.get("max"), bool) else None
                    desired_max_ema = (
                        row_data.get("max_ema") if isinstance(row_data.get("max_ema"), bool) else None
                    )
                    self.update_summary_checkboxes(
                        row,
                        desired_std=desired_std,
                        desired_std_ema=desired_std_ema,
                        desired_min=desired_min,
                        desired_min_ema=desired_min_ema,
                        desired_max=desired_max,
                        desired_max_ema=desired_max_ema,
                        preserve_existing=True,
                    )
                    self.update_row_ema_field(row, preserve_existing=True)
        else:
            self.clear_plot_rows()
        x_value = payload.get("x")
        if isinstance(x_value, str) and x_value:
            if self.available_columns and x_value in self.available_columns:
                self.x_combo.set(x_value)
            else:
                missing_x = x_value
        from_value = payload.get("from")
        to_value = payload.get("to")
        if from_value is None:
            from_value = payload.get("skip")
        if isinstance(from_value, (int, float)):
            self.from_var.set(str(int(from_value)))
        elif isinstance(from_value, str):
            self.from_var.set(from_value.strip() or "0")
        if isinstance(to_value, (int, float)):
            self.to_var.set(str(int(to_value)))
        elif isinstance(to_value, str):
            self.to_var.set(to_value.strip())
        return len(rows_payload), missing_columns, missing_x

    def load_plot_tab_into_ui(self, tab_index: int) -> tuple[list[str], str | None]:
        if tab_index < 0 or tab_index >= len(self.plot_tab_payloads):
            raise ValueError("Invalid tab index.")
        payload = self.plot_tab_payloads[tab_index]
        if payload is None:
            self.clear_plot_rows()
            self.x_combo.set("")
            self.from_var.set("0")
            self.to_var.set("")
            return [], None
        _row_count, missing_columns, missing_x = self.apply_plot_payload(payload)
        return missing_columns, missing_x

    def rebuild_plot_tab_controls(self) -> None:
        if self.plot_tabs_frame is None:
            return
        for child in self.plot_tabs_frame.winfo_children():
            child.destroy()
        for column in range(self.plot_tab_grid_column_count):
            self.plot_tabs_frame.columnconfigure(column, weight=0, uniform="")
        self.plot_tab_buttons = []
        tab_count = len(self.plot_tab_payloads)
        for tab_index in range(tab_count):
            self.plot_tabs_frame.columnconfigure(tab_index, weight=1, uniform="plot_tabs")
            tab_button = ttk.Button(
                self.plot_tabs_frame,
                text=f"T{tab_index + 1}",
                style="PlotTab.TButton",
            )
            tab_button.grid(
                row=0,
                column=tab_index,
                sticky="ew",
                padx=(0 if tab_index == 0 else 2, 0),
            )
            tab_button.bind(
                "<ButtonRelease-1>",
                lambda event, target=tab_index: self.on_plot_tab_button_click(event, target),
                add=True,
            )
            tab_button.bind(
                "<ButtonRelease-2>",
                lambda event, target=tab_index: self.on_plot_tab_middle_click(event, target),
                add=True,
            )
            self.plot_tab_buttons.append(tab_button)
        plus_column = tab_count
        self.plot_tabs_frame.columnconfigure(plus_column, weight=0, uniform="")
        self.plot_add_tab_button = ttk.Button(
            self.plot_tabs_frame,
            text="+",
            width=3,
            command=self.add_plot_tab,
            style="PlotTabAdd.TButton",
        )
        self.plot_add_tab_button.grid(row=0, column=plus_column, sticky="e", padx=(4, 0))
        self.plot_tab_grid_column_count = plus_column + 1

    def refresh_plot_tab_buttons(self) -> None:
        tab_count = len(self.plot_tab_payloads)
        for tab_index, button in enumerate(self.plot_tab_buttons):
            payload = self.plot_tab_payloads[tab_index]
            row_count = 0
            if isinstance(payload, dict):
                rows_payload = payload.get("rows")
                if isinstance(rows_payload, list):
                    row_count = len(rows_payload)
            suffix = f" ({row_count})" if row_count else ""
            style_name = (
                "PlotTabActive.TButton"
                if tab_index == self.active_plot_tab_index
                else "PlotTab.TButton"
            )
            button.configure(text=f"T{tab_index + 1}{suffix}", style=style_name)

    def on_plot_tab_button_click(self, _event: tk.Event, tab_index: int) -> str:
        self.on_plot_tab_selected(tab_index)
        return "break"

    def on_plot_tab_middle_click(self, _event: tk.Event, tab_index: int) -> str:
        self.close_plot_tab(tab_index)
        return "break"

    def add_plot_tab(self) -> None:
        self.capture_current_plot_tab_payload()
        self.plot_tab_payloads.append(None)
        self.active_plot_tab_index = len(self.plot_tab_payloads) - 1
        self.rebuild_plot_tab_controls()
        self.load_plot_tab_into_ui(self.active_plot_tab_index)
        if "timesteps" in self.available_columns:
            self.x_combo.set("timesteps")
        elif "iteration" in self.available_columns:
            self.x_combo.set("iteration")
        elif self.available_columns:
            self.x_combo.set(self.available_columns[0])
        else:
            self.x_combo.set("")
        self.refresh_plot_tab_buttons()
        self.set_status(f"Created T{self.active_plot_tab_index + 1}.")

    def close_plot_tab(self, tab_index: int) -> None:
        tab_count = len(self.plot_tab_payloads)
        if tab_index < 0 or tab_index >= tab_count:
            return
        if tab_count <= 1:
            self.set_status("Cannot close the last tab.")
            return
        self.capture_current_plot_tab_payload()
        self.plot_tab_payloads.pop(tab_index)
        active_tab_changed = False
        if tab_index < self.active_plot_tab_index:
            self.active_plot_tab_index -= 1
        elif tab_index == self.active_plot_tab_index:
            active_tab_changed = True
            if self.active_plot_tab_index >= len(self.plot_tab_payloads):
                self.active_plot_tab_index = len(self.plot_tab_payloads) - 1
        self.rebuild_plot_tab_controls()
        missing_columns: list[str] = []
        missing_x: str | None = None
        if active_tab_changed:
            try:
                missing_columns, missing_x = self.load_plot_tab_into_ui(self.active_plot_tab_index)
            except ValueError as exc:
                self.show_error(str(exc))
                return
        self.refresh_plot_tab_buttons()
        status_parts = [f"Closed T{tab_index + 1}."]
        if active_tab_changed:
            status_parts.append(f"Now on T{self.active_plot_tab_index + 1}.")
        if missing_columns:
            unique_missing = ", ".join(sorted(set(missing_columns)))
            status_parts.append(f"Missing plot columns: {unique_missing}.")
        if missing_x:
            status_parts.append(f"Missing X column: {missing_x}.")
        self.set_status(" ".join(status_parts))

    def on_plot_tab_selected(self, tab_index: int) -> None:
        if tab_index < 0 or tab_index >= len(self.plot_tab_payloads):
            return
        if tab_index == self.active_plot_tab_index:
            return
        previous_index = self.active_plot_tab_index
        self.capture_current_plot_tab_payload()
        self.active_plot_tab_index = tab_index
        try:
            missing_columns, missing_x = self.load_plot_tab_into_ui(tab_index)
        except ValueError as exc:
            self.active_plot_tab_index = previous_index
            self.refresh_plot_tab_buttons()
            self.show_error(str(exc))
            return
        status_parts = [f"Switched to T{tab_index + 1}."]
        if missing_columns:
            unique_missing = ", ".join(sorted(set(missing_columns)))
            status_parts.append(f"Missing plot columns: {unique_missing}.")
        if missing_x:
            status_parts.append(f"Missing X column: {missing_x}.")
        self.refresh_plot_tab_buttons()
        self.set_status(" ".join(status_parts))

    def apply_plot_tabs_payload(self, payload: object) -> tuple[list[str], str | None]:
        if not isinstance(payload, dict):
            raise ValueError("Saved plot tab configuration is invalid.")
        raw_tabs = payload.get("tabs")
        if not isinstance(raw_tabs, list):
            raise ValueError("Saved plot tab configuration is invalid.")
        tab_payloads: list[dict[str, object] | None] = []
        for tab_payload in raw_tabs:
            if not isinstance(tab_payload, dict):
                tab_payloads.append(None)
                continue
            rows_payload = tab_payload.get("rows")
            if not isinstance(rows_payload, list):
                tab_payloads.append(None)
                continue
            x_value = tab_payload.get("x")
            from_value = tab_payload.get("from")
            to_value = tab_payload.get("to")
            if from_value is None:
                from_value = tab_payload.get("skip")
            tab_payloads.append({
                "x": x_value if isinstance(x_value, str) else "",
                "from": (
                    str(int(from_value))
                    if isinstance(from_value, (int, float))
                    else (from_value if isinstance(from_value, str) else "0")
                ),
                "to": (
                    str(int(to_value))
                    if isinstance(to_value, (int, float))
                    else (to_value if isinstance(to_value, str) else "")
                ),
                "rows": rows_payload,
            })
        if not tab_payloads:
            tab_payloads = [None]
        active_index = 0
        active_value = as_int_like(payload.get("active"))
        if active_value is not None:
            candidate_index = active_value - 1
            if 0 <= candidate_index < len(tab_payloads):
                active_index = candidate_index
        self.plot_tab_payloads = tab_payloads
        self.active_plot_tab_index = active_index
        self.rebuild_plot_tab_controls()
        missing_columns, missing_x = self.load_plot_tab_into_ui(active_index)
        self.refresh_plot_tab_buttons()
        return missing_columns, missing_x

    def apply_plots_payload_from_config(self, payload: dict[str, object]) -> tuple[list[str], str | None]:
        tab_payload = payload.get("plot_tabs")
        if tab_payload is not None:
            return self.apply_plot_tabs_payload(tab_payload)
        plots_payload = payload.get("plots")
        if plots_payload is None:
            return [], None
        _plot_count, missing_plot_columns, missing_x = self.apply_plot_payload(plots_payload)
        self.plot_tab_payloads = [None]
        self.active_plot_tab_index = 0
        self.rebuild_plot_tab_controls()
        self.capture_current_plot_tab_payload()
        return missing_plot_columns, missing_x

    def loaded_paths_message(self, loaded_count: int, missing_count: int, label: str) -> str:
        label_capitalized = label[:1].upper() + label[1:] if label else label
        if loaded_count == 0:
            if missing_count:
                return f"{label_capitalized} missing on disk."
            return f"No {label} available."
        if missing_count:
            return f"Loaded {loaded_count} {label}. Missing {missing_count}."
        return f"Loaded {loaded_count} {label}."

    def set_loaded_paths_status(self, loaded_count: int, missing_count: int, label: str) -> None:
        self.set_status(self.loaded_paths_message(loaded_count, missing_count, label))

    def save_selected_paths(self) -> None:
        payload = self.build_config_payload()
        try:
            STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            self.show_error(f"Failed to save selected paths: {exc}")

    def save_config_to_file(self) -> None:
        path_str = filedialog.asksaveasfilename(
            title="Save CSV Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(REPO_ROOT),
        )
        if not path_str:
            return
        payload = self.build_config_payload()
        path = Path(path_str)
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            self.show_error(f"Failed to save CSV configuration: {exc}")
            return
        self.set_status(f"Saved CSV configuration to {path.name}.")

    def load_saved_paths(self) -> None:
        if not STATE_PATH.exists():
            self.set_status("No saved paths found.")
            return
        try:
            payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.show_error(f"Failed to load saved paths: {exc}")
            return
        if isinstance(payload, list):
            self.set_csv_source_folder(None)
            self.path_order_var.set(DEFAULT_PATH_ORDER_MODE)
            try:
                loaded_count, missing_count = self.apply_paths_payload(payload)
            except ValueError as exc:
                self.show_error(str(exc))
                return
            self.set_loaded_paths_status(loaded_count, missing_count, "saved paths")
            return
        if not isinstance(payload, dict):
            self.show_error("Saved configuration is invalid.")
            return
        paths_payload = payload.get("paths")
        try:
            loaded_count, missing_count = self.apply_paths_payload(paths_payload)
        except ValueError as exc:
            self.show_error(str(exc))
            return
        missing_plot_columns: list[str] = []
        missing_x: str | None = None
        try:
            missing_plot_columns, missing_x = self.apply_plots_payload_from_config(payload)
        except ValueError as exc:
            self.show_error(str(exc))
            return
        self.apply_auto_refresh_interval_payload(payload)
        missing_source_folder = self.apply_csv_source_folder_payload(payload)
        self.apply_path_order_payload(payload)
        self.refresh_file_list()
        status_parts = [
            self.loaded_paths_message(loaded_count, missing_count, "saved paths")
        ]
        if missing_plot_columns:
            unique_missing = ", ".join(sorted(set(missing_plot_columns)))
            status_parts.append(f"Missing plot columns: {unique_missing}.")
        if missing_x:
            status_parts.append(f"Missing X column: {missing_x}.")
        if missing_source_folder:
            status_parts.append(f"Missing source folder: {missing_source_folder}.")
        self.set_status(" ".join(status_parts))

    def load_config_from_file(self) -> None:
        path_str = filedialog.askopenfilename(
            title="Load CSV Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(REPO_ROOT),
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.show_error(f"Failed to load CSV configuration: {exc}")
            return
        if isinstance(payload, list):
            self.set_csv_source_folder(None)
            self.path_order_var.set(DEFAULT_PATH_ORDER_MODE)
            try:
                loaded_count, missing_count = self.apply_paths_payload(payload)
            except ValueError as exc:
                self.show_error(str(exc))
                return
            self.set_loaded_paths_status(loaded_count, missing_count, f"paths from {path.name}")
            return
        if not isinstance(payload, dict):
            self.show_error("Saved configuration is invalid.")
            return
        paths_payload = payload.get("paths")
        try:
            loaded_count, missing_count = self.apply_paths_payload(paths_payload)
        except ValueError as exc:
            self.show_error(str(exc))
            return
        missing_plot_columns: list[str] = []
        missing_x: str | None = None
        try:
            missing_plot_columns, missing_x = self.apply_plots_payload_from_config(payload)
        except ValueError as exc:
            self.show_error(str(exc))
            return
        self.apply_auto_refresh_interval_payload(payload)
        missing_source_folder = self.apply_csv_source_folder_payload(payload)
        self.apply_path_order_payload(payload)
        self.refresh_file_list()
        status_parts = [
            self.loaded_paths_message(loaded_count, missing_count, f"paths from {path.name}")
        ]
        if missing_plot_columns:
            unique_missing = ", ".join(sorted(set(missing_plot_columns)))
            status_parts.append(f"Missing plot columns: {unique_missing}.")
        if missing_x:
            status_parts.append(f"Missing X column: {missing_x}.")
        if missing_source_folder:
            status_parts.append(f"Missing source folder: {missing_source_folder}.")
        self.set_status(" ".join(status_parts))

    def refresh_columns(self, preserve_state: bool = False) -> None:
        enabled_paths = self.enabled_paths()
        if not self.paths:
            self.available_columns = []
            self.update_column_options(preserve_state=preserve_state)
            self.set_status("Select CSV files to load columns.")
            return
        if not enabled_paths:
            self.available_columns = []
            self.update_column_options(preserve_state=preserve_state)
            self.set_status("Enable at least one CSV file to load columns.")
            return
        delimiter = self.delimiter_var.get()
        if len(delimiter) != 1:
            self.show_error("Delimiter must be a single character.")
            return
        try:
            columns, mismatch = resolve_common_columns(enabled_paths, delimiter)
        except ValueError as exc:
            self.show_error(str(exc))
            return
        self.histogram_bin_count_cache.clear()
        if not columns:
            self.available_columns = []
            self.update_column_options(preserve_state=preserve_state)
            self.show_error("No common columns across selected files.")
            return
        self.available_columns = columns
        self.update_column_options(preserve_state=preserve_state)
        if mismatch:
            self.set_status(
                f"Column mismatch across files. Using {len(columns)} shared columns."
            )
        else:
            self.set_status(f"Loaded {len(columns)} columns.")

    def update_column_options(self, preserve_state: bool = False) -> None:
        columns = self.available_columns
        if columns:
            self.x_combo.configure(values=columns, state="readonly")
            if self.x_combo.get() not in columns:
                if "timesteps" in columns:
                    self.x_combo.set("timesteps")
                else:
                    self.x_combo.set(columns[0])
        else:
            self.x_combo.configure(values=[], state="disabled")
            self.x_combo.set("")

        for index, row in enumerate(self.plot_rows):
            self.update_row_options(
                row,
                columns,
                prefer_default=index == 0,
                preserve_state=preserve_state,
            )

    def update_row_options(
        self,
        row: PlotRow,
        columns: Sequence[str],
        prefer_default: bool,
        preserve_state: bool = False,
    ) -> None:
        if columns:
            row.y_combo.configure(values=columns, state="readonly")
            current_value = row.y_combo.get()
            selection_preserved = current_value in columns
            if not selection_preserved:
                if prefer_default and "ep_rew_ema" in columns:
                    row.y_combo.set("ep_rew_ema")
                else:
                    row.y_combo.set(columns[0])
            self.on_row_column_selected(
                row,
                preserve_existing=preserve_state and selection_preserved,
            )
        else:
            row.y_combo.configure(values=[], state="disabled")
            if not preserve_state:
                row.y_combo.set("")
                row.std_var.set(False)
                row.std_ema_var.set(False)
                row.skew_var.set(False)
                row.skew_ema_var.set(False)
                row.min_var.set(False)
                row.min_ema_var.set(False)
                row.max_var.set(False)
                row.max_ema_var.set(False)
            row.std_check.state(["disabled"])
            row.std_ema_check.state(["disabled"])
            row.skew_check.state(["disabled"])
            row.skew_ema_check.state(["disabled"])
            row.min_check.state(["disabled"])
            row.min_ema_check.state(["disabled"])
            row.max_check.state(["disabled"])
            row.max_ema_check.state(["disabled"])

    def add_plot_row(self) -> None:
        row_index = len(self.plot_rows) + 1
        index_var = tk.StringVar(value=str(row_index))
        index_combo = ttk.Combobox(
            self.plots_container,
            textvariable=index_var,
            state="readonly",
            width=2,
            values=[str(index) for index in range(1, len(self.plot_rows) + 2)],
        )
        self.disable_combobox_scroll_input(index_combo)
        index_combo.grid(row=row_index, column=0, sticky="w", pady=2)

        y_var = tk.StringVar()
        y_combo = ttk.Combobox(self.plots_container, textvariable=y_var, state="disabled", width=28)
        y_combo.grid(row=row_index, column=1, sticky="ew", padx=(10, 4), pady=2)
        self.bind_typeahead(y_combo, lambda: self.available_columns)

        height_var = tk.StringVar(value="1")
        height_entry = ttk.Entry(self.plots_container, textvariable=height_var, width=5)
        height_entry.grid(row=row_index, column=2, sticky="w", padx=(6, 4), pady=2)

        ema_var = tk.StringVar()
        ema_container = ttk.Frame(self.plots_container)
        ema_container.grid(row=row_index, column=3, sticky="w", padx=(2, 4), pady=2)
        ema_entry = ttk.Entry(ema_container, textvariable=ema_var, width=5)
        ema_entry.grid(row=0, column=0, sticky="w")

        ema_only_var = tk.BooleanVar(value=False)
        ema_only_check = ttk.Checkbutton(ema_container, text="", variable=ema_only_var, padding=0)
        ema_only_check.grid(row=0, column=1, padx=(3, 0))

        std_container = ttk.Frame(self.plots_container)
        std_container.grid(row=row_index, column=4, padx=(1, 1), pady=2, sticky="w")
        std_var = tk.BooleanVar(value=False)
        std_check = ttk.Checkbutton(std_container, text="", variable=std_var, padding=0)
        std_check.grid(row=0, column=0)
        std_ema_var = tk.BooleanVar(value=False)
        std_ema_check = ttk.Checkbutton(
            std_container,
            text="",
            variable=std_ema_var,
            padding=0,
            style="SmallEma.TCheckbutton",
        )
        std_ema_check.grid(row=0, column=1)

        skew_container = ttk.Frame(self.plots_container)
        skew_var = tk.BooleanVar(value=False)
        skew_check = ttk.Checkbutton(skew_container, text="", variable=skew_var, padding=0)
        skew_check.grid(row=0, column=0)
        skew_ema_var = tk.BooleanVar(value=False)
        skew_ema_check = ttk.Checkbutton(
            skew_container,
            text="",
            variable=skew_ema_var,
            padding=0,
            style="SmallEma.TCheckbutton",
        )
        skew_ema_check.grid(row=0, column=1)

        min_container = ttk.Frame(self.plots_container)
        min_container.grid(row=row_index, column=5, padx=(1, 1), pady=2, sticky="w")
        min_var = tk.BooleanVar(value=False)
        min_check = ttk.Checkbutton(min_container, text="", variable=min_var, padding=0)
        min_check.grid(row=0, column=0)
        min_ema_var = tk.BooleanVar(value=False)
        min_ema_check = ttk.Checkbutton(
            min_container,
            text="",
            variable=min_ema_var,
            padding=0,
            style="SmallEma.TCheckbutton",
        )
        min_ema_check.grid(row=0, column=1)

        max_container = ttk.Frame(self.plots_container)
        max_container.grid(row=row_index, column=6, padx=(1, 1), pady=2, sticky="w")
        max_var = tk.BooleanVar(value=False)
        max_check = ttk.Checkbutton(max_container, text="", variable=max_var, padding=0)
        max_check.grid(row=0, column=0)
        max_ema_var = tk.BooleanVar(value=False)
        max_ema_check = ttk.Checkbutton(
            max_container,
            text="",
            variable=max_ema_var,
            padding=0,
            style="SmallEma.TCheckbutton",
        )
        max_ema_check.grid(row=0, column=1)

        remove_button = ttk.Button(self.plots_container, text="🗑️", width=2)
        remove_button.grid(row=row_index, column=7, sticky="e", pady=2)

        row = PlotRow(
            index_var=index_var,
            index_combo=index_combo,
            y_var=y_var,
            y_combo=y_combo,
            height_var=height_var,
            height_entry=height_entry,
            ema_var=ema_var,
            ema_container=ema_container,
            ema_entry=ema_entry,
            ema_only_var=ema_only_var,
            ema_only_check=ema_only_check,
            std_container=std_container,
            std_var=std_var,
            std_check=std_check,
            std_ema_var=std_ema_var,
            std_ema_check=std_ema_check,
            skew_container=skew_container,
            skew_var=skew_var,
            skew_check=skew_check,
            skew_ema_var=skew_ema_var,
            skew_ema_check=skew_ema_check,
            min_container=min_container,
            min_var=min_var,
            min_check=min_check,
            min_ema_var=min_ema_var,
            min_ema_check=min_ema_check,
            max_container=max_container,
            max_var=max_var,
            max_check=max_check,
            max_ema_var=max_ema_var,
            max_ema_check=max_ema_check,
            remove_button=remove_button,
        )
        index_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event, target=row: self.on_row_index_selected(target),
        )
        remove_button.configure(command=lambda target=row: self.remove_plot_row(target))
        y_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event, target=row: self.on_row_column_selected(target),
        )
        self.plot_rows.append(row)
        self.update_row_options(row, self.available_columns, prefer_default=len(self.plot_rows) == 1)
        self.refresh_row_labels()

    def on_row_column_selected(self, row: PlotRow, preserve_existing: bool = False) -> None:
        previous_y = self.previous_y_by_row.get(id(row), "")
        current_y = row.y_combo.get()
        transitioned_from_histogram = (
            bool(previous_y)
            and self.is_histogram_column(previous_y)
            and not self.is_histogram_column(current_y)
        )
        self.update_summary_checkboxes(row, preserve_existing=preserve_existing)
        if transitioned_from_histogram:
            row.ema_var.set("")
        if not transitioned_from_histogram:
            self.update_row_ema_field(row, preserve_existing=preserve_existing)
        self.previous_y_by_row[id(row)] = current_y

    def parse_histogram_input(
        self,
        raw_value: str,
        row_index: int,
    ) -> tuple[int, int] | None:
        if not raw_value:
            return None
        parts = [part.strip() for part in raw_value.split(";")]
        if len(parts) == 1:
            parts.append("1")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            self.show_error(
                f"Histogram input must be 'nbins;pooling' with positive integers (row {row_index})."
            )
            return None
        bin_count = self.parse_positive_int(parts[0], row_index)
        pooling = self.parse_positive_int(parts[1], row_index)
        if bin_count is None or pooling is None:
            return None
        return bin_count, pooling

    def parse_histogram_input_silent(self, raw_value: str) -> tuple[int, int] | None:
        if not raw_value:
            return None
        parts = [part.strip() for part in raw_value.split(";")]
        if len(parts) == 1:
            parts.append("1")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None
        bin_count = self.parse_positive_int_silent(parts[0])
        pooling = self.parse_positive_int_silent(parts[1])
        if bin_count is None or pooling is None:
            return None
        return bin_count, pooling

    def parse_positive_int(self, raw_value: str, row_index: int) -> int | None:
        try:
            numeric_value = float(raw_value)
        except ValueError:
            self.show_error(
                f"Histogram input must be 'nbins;pooling' with positive integers (row {row_index})."
            )
            return None
        if numeric_value <= 0.0 or not numeric_value.is_integer():
            self.show_error(
                f"Histogram input must be 'nbins;pooling' with positive integers (row {row_index})."
            )
            return None
        return int(numeric_value)

    def parse_positive_int_silent(self, raw_value: str) -> int | None:
        try:
            numeric_value = float(raw_value)
        except ValueError:
            return None
        if numeric_value <= 0.0 or not numeric_value.is_integer():
            return None
        return int(numeric_value)

    def as_positive_int(self, value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, float) and value > 0.0 and value.is_integer():
            return int(value)
        return None

    def as_ema_alpha(self, value: object) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if 0.0 < numeric_value < 1.0:
            return numeric_value
        return None

    def update_row_ema_field(self, row: PlotRow, preserve_existing: bool = False) -> None:
        y_value = row.y_combo.get()
        if not y_value:
            return
        raw_value = row.ema_var.get().strip()
        if self.is_histogram_column(y_value):
            if raw_value:
                parsed = self.parse_histogram_input_silent(raw_value)
                if parsed is not None:
                    bin_count, pooling = parsed
                    row.ema_var.set(f"{bin_count};{pooling}")
                    return
                if preserve_existing:
                    return
                row.ema_var.set("")
            elif preserve_existing:
                return
            bin_count = self.infer_histogram_bin_count(y_value)
            if bin_count is not None:
                row.ema_var.set(f"{bin_count};1")
            return
        if preserve_existing or not raw_value:
            return
        try:
            alpha = float(raw_value)
        except ValueError:
            row.ema_var.set("")
            return
        if not 0.0 < alpha < 1.0:
            row.ema_var.set("")

    def infer_histogram_bin_count(self, y_value: str) -> int | None:
        resolved = self.resolve_histogram_columns(y_value)
        if resolved is None:
            return None
        freqs_column, _edges_column = resolved
        enabled_paths = self.enabled_paths()
        if not enabled_paths:
            return None
        delimiter = self.delimiter_var.get()
        if len(delimiter) != 1:
            return None
        cache_key = (
            tuple(str(path) for path in enabled_paths),
            delimiter,
            freqs_column,
        )
        cached = self.histogram_bin_count_cache.get(cache_key)
        if cache_key in self.histogram_bin_count_cache:
            return cached
        inferred: int | None = None
        for path in enabled_paths:
            try:
                with path.open(newline="") as handle:
                    reader = csv.DictReader(handle, delimiter=delimiter)
                    if reader.fieldnames is None or freqs_column not in reader.fieldnames:
                        continue
                    for row_index, csv_row in enumerate(reader, start=2):
                        raw_freqs = csv_row.get(freqs_column)
                        if is_missing_histogram_cell(raw_freqs):
                            continue
                        try:
                            inferred = len(
                                parse_histogram_list(
                                    raw_freqs,
                                    freqs_column,
                                    path,
                                    row_index,
                                )
                            )
                        except ValueError:
                            continue
                        break
            except OSError:
                continue
            if inferred is not None:
                break
        self.histogram_bin_count_cache[cache_key] = inferred
        return inferred

    def update_std_checkbox(
        self,
        row: PlotRow,
        desired: bool | None = None,
        desired_ema: bool | None = None,
    ) -> None:
        y_value = row.y_combo.get()
        candidate = self.std_column_for(y_value)
        if candidate is None:
            row.std_var.set(False)
            row.std_ema_var.set(False)
            row.std_check.state(["disabled"])
            row.std_ema_check.state(["disabled"])
        else:
            row.std_check.state(["!disabled"])
            row.std_ema_check.state(["!disabled"])
            if desired is None:
                row.std_var.set(True)
            else:
                row.std_var.set(desired)
            if desired_ema is None:
                row.std_ema_var.set(False)
            else:
                row.std_ema_var.set(desired_ema)

    def update_skew_checkbox(
        self,
        row: PlotRow,
        desired: bool | None = None,
        desired_ema: bool | None = None,
    ) -> None:
        y_value = row.y_combo.get()
        candidate = self.skew_column_for(y_value)
        if candidate is None:
            row.skew_var.set(False)
            row.skew_ema_var.set(False)
            row.skew_check.state(["disabled"])
            row.skew_ema_check.state(["disabled"])
        else:
            row.skew_check.state(["!disabled"])
            row.skew_ema_check.state(["!disabled"])
            if desired is None:
                row.skew_var.set(False)
            else:
                row.skew_var.set(desired)
            if desired_ema is None:
                row.skew_ema_var.set(False)
            else:
                row.skew_ema_var.set(desired_ema)

    def update_minmax_checkboxes(
        self,
        row: PlotRow,
        desired_min: bool | None = None,
        desired_max: bool | None = None,
    ) -> None:
        self.update_min_checkbox(row, desired=desired_min)
        self.update_max_checkbox(row, desired=desired_max)

    def update_summary_checkboxes(
        self,
        row: PlotRow,
        desired_std: bool | None = None,
        desired_std_ema: bool | None = None,
        desired_min: bool | None = None,
        desired_min_ema: bool | None = None,
        desired_max: bool | None = None,
        desired_max_ema: bool | None = None,
        preserve_existing: bool = False,
    ) -> None:
        if preserve_existing:
            if desired_std is None:
                desired_std = row.std_var.get()
            if desired_std_ema is None:
                desired_std_ema = row.std_ema_var.get()
            if desired_min is None:
                desired_min = row.min_var.get()
            if desired_min_ema is None:
                desired_min_ema = row.min_ema_var.get()
            if desired_max is None:
                desired_max = row.max_var.get()
            if desired_max_ema is None:
                desired_max_ema = row.max_ema_var.get()
        self.update_std_checkbox(row, desired=desired_std, desired_ema=desired_std_ema)
        self.update_min_checkbox(row, desired=desired_min, desired_ema=desired_min_ema)
        self.update_max_checkbox(row, desired=desired_max, desired_ema=desired_max_ema)

    def std_column_for(self, y_value: str) -> str | None:
        if not y_value or not y_value.endswith("__mean"):
            return None
        candidate = f"{y_value.removesuffix('__mean')}__std"
        if candidate in self.available_columns:
            return candidate
        return None

    def skew_column_for(self, y_value: str) -> str | None:
        if not y_value or not y_value.endswith("__mean"):
            return None
        candidate = f"{y_value.removesuffix('__mean')}__skew"
        if candidate in self.available_columns:
            return candidate
        return None

    def min_column_for(self, y_value: str) -> str | None:
        if not y_value or not y_value.endswith("__mean"):
            return None
        candidate = f"{y_value.removesuffix('__mean')}__min"
        if candidate in self.available_columns:
            return candidate
        return None

    def max_column_for(self, y_value: str) -> str | None:
        if not y_value or not y_value.endswith("__mean"):
            return None
        candidate = f"{y_value.removesuffix('__mean')}__max"
        if candidate in self.available_columns:
            return candidate
        return None

    def update_min_checkbox(
        self,
        row: PlotRow,
        desired: bool | None = None,
        desired_ema: bool | None = None,
    ) -> None:
        y_value = row.y_combo.get()
        min_candidate = self.min_column_for(y_value)
        if min_candidate is None:
            row.min_var.set(False)
            row.min_ema_var.set(False)
            row.min_check.state(["disabled"])
            row.min_ema_check.state(["disabled"])
        else:
            row.min_check.state(["!disabled"])
            row.min_ema_check.state(["!disabled"])
            if desired is None:
                row.min_var.set(False)
            else:
                row.min_var.set(desired)
            if desired_ema is None:
                row.min_ema_var.set(False)
            else:
                row.min_ema_var.set(desired_ema)

    def update_max_checkbox(
        self,
        row: PlotRow,
        desired: bool | None = None,
        desired_ema: bool | None = None,
    ) -> None:
        y_value = row.y_combo.get()
        max_candidate = self.max_column_for(y_value)
        if max_candidate is None:
            row.max_var.set(False)
            row.max_ema_var.set(False)
            row.max_check.state(["disabled"])
            row.max_ema_check.state(["disabled"])
        else:
            row.max_check.state(["!disabled"])
            row.max_ema_check.state(["!disabled"])
            if desired is None:
                row.max_var.set(False)
            else:
                row.max_var.set(desired)
            if desired_ema is None:
                row.max_ema_var.set(False)
            else:
                row.max_ema_var.set(desired_ema)

    def is_histogram_column(self, column: str) -> bool:
        return column.endswith("__histogram_freqs") or column.endswith("__histogram_edges")

    def resolve_histogram_columns(self, column: str) -> tuple[str, str | None] | None:
        if column.endswith("__histogram_freqs"):
            freqs_column = column
            edges_column = f"{column.removesuffix('__histogram_freqs')}__histogram_edges"
            if edges_column not in self.available_columns:
                edges_column = None
            return freqs_column, edges_column
        if column.endswith("__histogram_edges"):
            edges_column = column
            freqs_column = f"{column.removesuffix('__histogram_edges')}__histogram_freqs"
            if freqs_column not in self.available_columns:
                return None
            return freqs_column, edges_column
        return None

    def bind_typeahead(
        self,
        combo: ttk.Combobox,
        values_getter: Callable[[], Sequence[str]],
    ) -> None:
        self.combo_values_getters[combo] = values_getter
        self.register_listbox(combo)
        self.disable_combobox_scroll_input(combo)
        combo.bind(
            "<KeyPress>",
            lambda event, target=combo, getter=values_getter: self.on_typeahead(event, target, getter),
            add=True,
        )

    def disable_combobox_scroll_input(self, combo: ttk.Combobox) -> None:
        combo.bind("<MouseWheel>", self.on_combobox_mousewheel)
        combo.bind("<Button-4>", self.on_combobox_mousewheel)
        combo.bind("<Button-5>", self.on_combobox_mousewheel)

    def on_combobox_mousewheel(self, _event: tk.Event) -> str:
        return "break"

    def on_typeahead(
        self,
        event: tk.Event,
        combo: ttk.Combobox,
        values_getter: Callable[[], Sequence[str]],
    ) -> None:
        if event.keysym in {"Up", "Down", "Left", "Right", "Home", "End", "Prior", "Next", "Tab", "Escape", "Return"}:
            return
        buffer = self.typeahead_buffers.get(combo, "")
        if event.keysym == "BackSpace":
            buffer = buffer[:-1]
        else:
            char = event.char
            if not char and len(event.keysym) == 1:
                char = event.keysym
            if char and char.isprintable():
                buffer += char
            else:
                return
        self.typeahead_buffers[combo] = buffer
        after_id = self.typeahead_after_ids.get(combo)
        if after_id:
            self.root.after_cancel(after_id)
        self.typeahead_after_ids[combo] = self.root.after(
            800, lambda: self.clear_typeahead(combo)
        )
        if not buffer:
            return
        values = list(values_getter())
        if not values:
            return
        needle = buffer.lower()
        for index, value in enumerate(values):
            if value.lower().startswith(needle):
                combo.current(index)
                combo.event_generate("<<ComboboxSelected>>")
                return

    def clear_typeahead(self, combo: ttk.Combobox) -> None:
        self.typeahead_buffers.pop(combo, None)
        self.typeahead_after_ids.pop(combo, None)

    def register_listbox(self, combo: ttk.Combobox) -> None:
        try:
            popdown = combo.tk.call("ttk::combobox::PopdownWindow", combo)
        except tk.TclError:
            return
        listbox_path = f"{popdown}.f.l"
        self.listbox_owner[listbox_path] = combo

    def on_global_keypress(self, event: tk.Event) -> str | None:
        focus_name = str(self.root.tk.call("focus"))
        if not focus_name:
            return None
        try:
            focus_class = str(self.root.tk.call("winfo", "class", focus_name))
        except tk.TclError:
            return None
        if focus_class != "Listbox":
            return None
        combo = self.listbox_owner.get(focus_name)
        if combo is None:
            return None
        values_getter = self.combo_values_getters.get(combo)
        if values_getter is None:
            return None
        self.on_typeahead(event, combo, values_getter)
        try:
            listbox_widget = self.root.nametowidget(focus_name)
        except KeyError:
            return "break"
        self.sync_listbox_selection(listbox_widget, combo)
        return "break"

    def on_spacebar(self, event: tk.Event) -> str | None:
        focus_widget = event.widget
        focus_class = focus_widget.winfo_class()
        if focus_class in {"Entry", "TEntry", "Text"}:
            return None
        self.plot()
        return "break"

    def sync_listbox_selection(self, listbox: tk.Listbox, combo: ttk.Combobox) -> None:
        current_index = combo.current()
        if current_index < 0:
            return
        listbox.selection_clear(0, tk.END)
        listbox.selection_set(current_index)
        listbox.activate(current_index)
        listbox.see(current_index)

    def remove_plot_row(self, row: PlotRow) -> None:
        for widget in (
            row.index_combo,
            row.y_combo,
            row.height_entry,
            row.ema_container,
            row.std_container,
            row.skew_container,
            row.min_container,
            row.max_container,
            row.remove_button,
        ):
            widget.destroy()
        if row in self.plot_rows:
            self.plot_rows.remove(row)
        self.previous_y_by_row.pop(id(row), None)
        self.refresh_row_labels()

    def clear_plot_rows(self) -> None:
        for row in self.plot_rows:
            for widget in (
                row.index_combo,
                row.y_combo,
                row.height_entry,
                row.ema_container,
                row.std_container,
                row.skew_container,
                row.min_container,
                row.max_container,
                row.remove_button,
            ):
                widget.destroy()
        self.plot_rows.clear()
        self.previous_y_by_row.clear()
        self.add_plot_row()

    def apply_preset(self, preset: PlotPreset) -> None:
        if not self.available_columns:
            self.show_error("Load columns before applying presets.")
            return
        missing_columns = [
            entry.y_column
            for entry in preset.entries
            if entry.y_column not in self.available_columns
        ]
        if missing_columns:
            self.show_error(
                f"Preset missing columns: {', '.join(missing_columns)}"
            )
            return
        if preset.x_column:
            if preset.x_column not in self.available_columns:
                self.show_error(f"Preset X column missing: {preset.x_column}")
                return
            self.x_combo.set(preset.x_column)
        self.clear_plot_rows()
        for _ in range(len(preset.entries) - 1):
            self.add_plot_row()
        missing_std: list[str] = []
        missing_min: list[str] = []
        missing_max: list[str] = []
        invalid_histogram_bins: list[str] = []
        invalid_scalar_ema: list[str] = []
        invalid_summary_ema: list[str] = []
        invalid_ema_only: list[str] = []
        for row, entry in zip(self.plot_rows, preset.entries, strict=True):
            row.y_combo.set(entry.y_column)
            row.height_var.set(self.format_ratio(entry.height))
            has_row_ema = False
            is_histogram = self.is_histogram_column(entry.y_column)
            if is_histogram:
                if entry.nbins is None:
                    row.ema_var.set("")
                else:
                    bins = self.as_positive_int(entry.nbins)
                    pooling_value = entry.pooling if entry.pooling is not None else 1
                    pooling = self.as_positive_int(pooling_value)
                    if bins is None or pooling is None:
                        row.ema_var.set("")
                        invalid_histogram_bins.append(entry.y_column)
                    else:
                        row.ema_var.set(f"{bins};{pooling}")
            else:
                if entry.ema is None:
                    row.ema_var.set("")
                else:
                    alpha = self.as_ema_alpha(entry.ema)
                    if alpha is None:
                        row.ema_var.set("")
                        invalid_scalar_ema.append(entry.y_column)
                    else:
                        row.ema_var.set(str(alpha))
                        has_row_ema = True
            row.ema_only_var.set(entry.ema_only if is_histogram else entry.ema_only and has_row_ema)
            if entry.ema_only and not has_row_ema and not is_histogram:
                invalid_ema_only.append(entry.y_column)
            std_enabled, std_use_ema = self.resolve_preset_summary_option(entry.std)
            min_enabled, min_use_ema = self.resolve_preset_summary_option(entry.min)
            max_enabled, max_use_ema = self.resolve_preset_summary_option(entry.max)
            if std_enabled and self.std_column_for(entry.y_column) is None:
                missing_std.append(entry.y_column)
            if min_enabled and self.min_column_for(entry.y_column) is None:
                missing_min.append(entry.y_column)
            if max_enabled and self.max_column_for(entry.y_column) is None:
                missing_max.append(entry.y_column)
            if std_use_ema and not has_row_ema:
                std_use_ema = False
                invalid_summary_ema.append(f"{entry.y_column}: std")
            if min_use_ema and not has_row_ema:
                min_use_ema = False
                invalid_summary_ema.append(f"{entry.y_column}: min")
            if max_use_ema and not has_row_ema:
                max_use_ema = False
                invalid_summary_ema.append(f"{entry.y_column}: max")
            self.update_summary_checkboxes(
                row,
                desired_std=std_enabled,
                desired_std_ema=std_use_ema,
                desired_min=min_enabled,
                desired_min_ema=min_use_ema,
                desired_max=max_enabled,
                desired_max_ema=max_use_ema,
            )
            self.update_row_ema_field(row, preserve_existing=True)
        status_parts = [f"Preset {preset.name!r} loaded."]
        if missing_std:
            status_parts.append(
                f"Missing std columns for: {', '.join(sorted(set(missing_std)))}."
            )
        if missing_min:
            status_parts.append(
                f"Missing min columns for: {', '.join(sorted(set(missing_min)))}."
            )
        if missing_max:
            status_parts.append(
                f"Missing max columns for: {', '.join(sorted(set(missing_max)))}."
            )
        if invalid_histogram_bins:
            status_parts.append(
                "Histogram input must be 'nbins;pooling' with positive integers; ignored for: "
                f"{', '.join(sorted(set(invalid_histogram_bins)))}."
            )
        if invalid_scalar_ema:
            status_parts.append(
                "Scalar EMA must be between 0 and 1; ignored for: "
                f"{', '.join(sorted(set(invalid_scalar_ema)))}."
            )
        if invalid_ema_only:
            status_parts.append(
                "EMA-only requires a row EMA alpha; ignored for: "
                f"{', '.join(sorted(set(invalid_ema_only)))}."
            )
        if invalid_summary_ema:
            status_parts.append(
                "Summary EMA toggles require a row EMA alpha; ignored for: "
                f"{', '.join(sorted(set(invalid_summary_ema)))}."
            )
        self.set_status(" ".join(status_parts))

    def resolve_preset_summary_option(self, value: bool | AsEma) -> tuple[bool, bool]:
        if value is AS_EMA:
            return True, True
        if value:
            return True, False
        return False, False

    def format_ratio(self, ratio: float) -> str:
        if ratio.is_integer():
            return str(int(ratio))
        return str(ratio)

    def on_row_index_selected(self, row: PlotRow) -> None:
        if self.updating_row_index_widgets:
            return
        if row not in self.plot_rows:
            return
        try:
            target_index = int(row.index_var.get()) - 1
        except ValueError:
            self.refresh_row_labels()
            return
        current_index = self.plot_rows.index(row)
        clamped_target = max(0, min(len(self.plot_rows) - 1, target_index))
        if clamped_target == current_index:
            self.refresh_row_labels()
            return
        moved_row = self.plot_rows.pop(current_index)
        self.plot_rows.insert(clamped_target, moved_row)
        self.refresh_row_labels()

    def refresh_row_labels(self) -> None:
        row_positions = [str(index) for index in range(1, len(self.plot_rows) + 1)]
        self.updating_row_index_widgets = True
        try:
            for index, row in enumerate(self.plot_rows, start=1):
                row.index_combo.configure(values=row_positions)
                row.index_var.set(str(index))
                row.index_combo.grid_configure(row=index, column=0)
                row.y_combo.grid_configure(row=index, column=1)
                row.height_entry.grid_configure(row=index, column=2)
                row.ema_container.grid_configure(row=index, column=3)
                row.std_container.grid_configure(row=index, column=4)
                row.min_container.grid_configure(row=index, column=5)
                row.max_container.grid_configure(row=index, column=6)
                row.remove_button.grid_configure(row=index, column=7)
        finally:
            self.updating_row_index_widgets = False
        footer_row = len(self.plot_rows) + 1
        if self.global_ema_container is not None:
            self.global_ema_container.grid(
                row=footer_row,
                column=3,
                sticky="w",
                padx=(2, 4),
                pady=(4, 0),
            )
        self.toggle_std_button.grid(row=footer_row, column=4, pady=(4, 0))
        self.toggle_min_button.grid(row=footer_row, column=5, pady=(4, 0))
        self.toggle_max_button.grid(row=footer_row, column=6, pady=(4, 0))

    def on_global_ema_submit(self, _event: tk.Event) -> str:
        self.apply_global_ema_to_rows(show_error=True)
        return "break"

    def on_global_ema_focus_out(self, _event: tk.Event) -> None:
        self.apply_global_ema_to_rows(show_error=False)

    def on_global_ema_only_toggle(self) -> None:
        self.apply_global_ema_only_to_rows()

    def apply_global_ema_to_rows(self, show_error: bool) -> None:
        raw_alpha = self.global_ema_var.get().strip()
        if raw_alpha == "":
            cleared_count = 0
            skipped_histogram_count = 0
            for row in self.plot_rows:
                y_value = row.y_combo.get()
                if y_value and self.is_histogram_column(y_value):
                    skipped_histogram_count += 1
                    continue
                row.ema_var.set("")
                cleared_count += 1
            if skipped_histogram_count:
                self.set_status(
                    "Cleared EMA alpha for "
                    f"{cleared_count} rows (kept {skipped_histogram_count} histogram nbins;pooling)."
                )
            else:
                self.set_status(f"Cleared EMA alpha for {cleared_count} rows.")
            return
        try:
            alpha = float(raw_alpha)
        except ValueError:
            if show_error:
                self.show_error("Global EMA alpha must be a number between 0 and 1.")
            return
        if not 0.0 < alpha < 1.0:
            if show_error:
                self.show_error("Global EMA alpha must be between 0 and 1.")
            return
        applied_count = 0
        skipped_count = 0
        for row in self.plot_rows:
            y_value = row.y_combo.get()
            if y_value and self.is_histogram_column(y_value):
                skipped_count += 1
                continue
            row.ema_var.set(raw_alpha)
            applied_count += 1
        if skipped_count:
            self.set_status(
                f"Applied EMA alpha to {applied_count} rows (skipped {skipped_count} histogram rows)."
            )
        else:
            self.set_status(f"Applied EMA alpha to {applied_count} rows.")

    def apply_global_ema_only_to_rows(self) -> None:
        enabled = self.global_ema_only_var.get()
        for row in self.plot_rows:
            row.ema_only_var.set(enabled)
        if enabled:
            self.set_status("Enabled EMA (only) for all rows.")
        else:
            self.set_status("Disabled EMA (only) for all rows.")

    def toggle_all_std(self) -> None:
        self.cycle_summary_column(
            available=self.std_column_for,
            is_enabled=lambda row: row.std_var.get(),
            is_ema_enabled=lambda row: row.std_ema_var.get(),
            apply=lambda row, enabled, ema: self.update_std_checkbox(
                row,
                desired=enabled,
                desired_ema=ema,
            ),
        )

    def toggle_all_min(self) -> None:
        self.cycle_summary_column(
            available=self.min_column_for,
            is_enabled=lambda row: row.min_var.get(),
            is_ema_enabled=lambda row: row.min_ema_var.get(),
            apply=lambda row, enabled, ema: self.update_min_checkbox(
                row,
                desired=enabled,
                desired_ema=ema,
            ),
        )

    def toggle_all_max(self) -> None:
        self.cycle_summary_column(
            available=self.max_column_for,
            is_enabled=lambda row: row.max_var.get(),
            is_ema_enabled=lambda row: row.max_ema_var.get(),
            apply=lambda row, enabled, ema: self.update_max_checkbox(
                row,
                desired=enabled,
                desired_ema=ema,
            ),
        )

    def cycle_summary_column(
        self,
        available: Callable[[str], str | None],
        is_enabled: Callable[[PlotRow], bool],
        is_ema_enabled: Callable[[PlotRow], bool],
        apply: Callable[[PlotRow, bool, bool], None],
    ) -> None:
        rows = [
            row
            for row in self.plot_rows
            if available(row.y_combo.get()) is not None
        ]
        if not rows:
            return

        all_off = all(not is_enabled(row) for row in rows)
        all_on_no_ema = all(is_enabled(row) and not is_ema_enabled(row) for row in rows)
        all_on_ema = all(is_enabled(row) and is_ema_enabled(row) for row in rows)

        if all_off:
            target_enabled = True
            target_ema = False
        elif all_on_no_ema:
            target_enabled = True
            target_ema = True
        elif all_on_ema:
            target_enabled = False
            target_ema = False
        else:
            target_enabled = False
            target_ema = False

        for row in rows:
            apply(row, target_enabled, target_ema)

    def set_all_min(self, enabled: bool) -> None:
        for row in self.plot_rows:
            self.update_min_checkbox(row, desired=enabled, desired_ema=False)

    def set_all_max(self, enabled: bool) -> None:
        for row in self.plot_rows:
            self.update_max_checkbox(row, desired=enabled, desired_ema=False)

    def set_all_std(self, enabled: bool) -> None:
        for row in self.plot_rows:
            self.update_std_checkbox(row, desired=enabled, desired_ema=False)

    def group_keys_for_labels(self, paths: Sequence[Path], labels: Sequence[str]) -> list[str]:
        return [
            self.path_groups.get(path, label)
            for path, label in zip(paths, labels, strict=True)
        ]

    def group_color_map(self, group_keys: Sequence[str]) -> dict[str, str]:
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not colors:
            colors = list(matplotlib.colors.TABLEAU_COLORS.values())
        if not colors:
            return {}
        color_map: dict[str, str] = {}
        color_index = 0
        for key in group_keys:
            if key in color_map:
                continue
            color_map[key] = colors[color_index % len(colors)]
            color_index += 1
        return color_map

    def plot(self) -> None:
        auto_refresh_interval = self.parse_auto_refresh_interval(show_error=True)
        if auto_refresh_interval is None:
            return
        enabled_paths = self.enabled_paths()
        if not self.paths:
            self.show_error("No CSV files selected.")
            return
        if not enabled_paths:
            self.show_error("Enable at least one CSV file to plot.")
            return
        delimiter = self.delimiter_var.get()
        if len(delimiter) != 1:
            self.show_error("Delimiter must be a single character.")
            return
        x_column = self.x_combo.get()
        if not x_column:
            self.show_error("Pick an X column.")
            return
        row_range = self.parse_row_range(show_error=True)
        if row_range is None:
            return
        from_index, to_index = row_range
        if not self.validate_from_index(enabled_paths, delimiter, from_index):
            return
        missing_rows = [index for index, row in enumerate(self.plot_rows, start=1) if not row.y_combo.get()]
        if missing_rows:
            self.show_error("Every plot row needs a Y column selected.")
            return
        y_columns = [row.y_combo.get() for row in self.plot_rows]
        if not y_columns:
            self.show_error("Add at least one plot row.")
            return
        if len(set(y_columns)) != len(y_columns):
            self.show_error("Y columns must be unique.")
            return
        ratios: list[float] = []
        for index, row in enumerate(self.plot_rows, start=1):
            raw_ratio = row.height_var.get().strip()
            if not raw_ratio:
                ratio = 1.0
            else:
                try:
                    ratio = float(raw_ratio)
                except ValueError:
                    self.show_error(f"Height ratio must be a number (row {index}).")
                    return
            ratios.append(ratio)
        try:
            ratios = plot_logs.validate_ratios(ratios, y_columns) or []
        except ValueError as exc:
            self.show_error(str(exc))
            return

        ema_mapping: dict[str, float] = {}
        histogram_bin_counts: dict[str, int] = {}
        histogram_pooling: dict[str, int] = {}
        ema_only_columns: set[str] = set()
        for index, row in enumerate(self.plot_rows, start=1):
            y_value = row.y_combo.get()
            raw_ema = row.ema_var.get().strip()
            if self.is_histogram_column(y_value):
                if row.ema_only_var.get():
                    ema_only_columns.add(y_value)
                parsed_histogram = self.parse_histogram_input(raw_ema, index)
                if parsed_histogram is None:
                    if raw_ema:
                        return
                    continue
                bin_count, pooling = parsed_histogram
                histogram_bin_counts[y_value] = bin_count
                histogram_pooling[y_value] = pooling
                continue
            if not raw_ema:
                continue
            try:
                alpha = float(raw_ema)
            except ValueError:
                self.show_error(f"EMA alpha must be a number between 0 and 1 (row {index}).")
                return
            if not 0.0 < alpha < 1.0:
                self.show_error(f"EMA alpha must be between 0 and 1 (row {index}).")
                return
            ema_mapping[y_value] = alpha
            if row.ema_only_var.get():
                ema_only_columns.add(y_value)
        histogram_columns = [column for column in y_columns if self.is_histogram_column(column)]
        histogram_specs: dict[str, tuple[str, str | None, int | None, int]] = {}
        for column in histogram_columns:
            resolved = self.resolve_histogram_columns(column)
            if resolved is None:
                base = column.removesuffix("__histogram_edges")
                self.show_error(
                    f"Histogram column {column} needs {base}__histogram_freqs."
                )
                return
            histogram_specs[column] = (
                resolved[0],
                resolved[1],
                histogram_bin_counts.get(column),
                histogram_pooling.get(column, 1),
            )

        scalar_columns = [column for column in y_columns if column not in histogram_specs]
        scalar_column_set = set(scalar_columns)
        std_mapping: dict[str, str | None] = {column: None for column in scalar_columns}
        min_mapping: dict[str, str | None] = {column: None for column in scalar_columns}
        max_mapping: dict[str, str | None] = {column: None for column in scalar_columns}
        std_ema_columns: set[str] = set()
        min_ema_columns: set[str] = set()
        max_ema_columns: set[str] = set()
        for index, row in enumerate(self.plot_rows, start=1):
            y_value = row.y_combo.get()
            if not y_value or y_value not in scalar_column_set:
                continue
            if row.std_var.get():
                std_column = self.std_column_for(y_value)
                if std_column is None:
                    self.show_error(f"No std column found for {y_value}.")
                    return
                std_mapping[y_value] = std_column
                if row.std_ema_var.get():
                    if y_value not in ema_mapping:
                        self.show_error(f"STD EMA requires an EMA alpha in row {index}.")
                        return
                    std_ema_columns.add(y_value)
            if row.min_var.get():
                min_column = self.min_column_for(y_value)
                if min_column is None:
                    self.show_error(f"No min column found for {y_value}.")
                    return
                min_mapping[y_value] = min_column
                if row.min_ema_var.get():
                    if y_value not in ema_mapping:
                        self.show_error(f"Min EMA requires an EMA alpha in row {index}.")
                        return
                    min_ema_columns.add(y_value)
            if row.max_var.get():
                max_column = self.max_column_for(y_value)
                if max_column is None:
                    self.show_error(f"No max column found for {y_value}.")
                    return
                max_mapping[y_value] = max_column
                if row.max_ema_var.get():
                    if y_value not in ema_mapping:
                        self.show_error(f"Max EMA requires an EMA alpha in row {index}.")
                        return
                    max_ema_columns.add(y_value)

        labels = plot_logs.build_labels(enabled_paths, None)
        file_opacities = [self.opacity_for_path(path) for path in enabled_paths]
        file_colors = [self.file_color_by_path.get(path, "") for path in enabled_paths]
        group_keys: list[str] | None = None
        logs: list[plot_logs.LogSeries] = []
        if scalar_columns:
            try:
                logs = [
                    plot_logs.load_log(
                        path,
                        label,
                        x_column,
                        scalar_columns,
                        std_mapping,
                        delimiter,
                        min_mapping=min_mapping,
                        max_mapping=max_mapping,
                        from_index=from_index,
                        to_index=to_index,
                    )
                    for path, label in zip(enabled_paths, labels, strict=True)
                ]
                group_keys = self.group_keys_for_labels(enabled_paths, labels)
            except (ValueError, FileNotFoundError) as exc:
                self.show_error(str(exc))
                return

        histogram_data: dict[str, list[HistogramSeries]] = {}
        for column, (freqs_column, edges_column, target_bin_count, x_pooling) in histogram_specs.items():
            try:
                histogram_data[column] = [
                    load_histogram_series(
                        path,
                        label,
                        x_column,
                        freqs_column,
                        edges_column,
                        delimiter,
                        target_bin_count=target_bin_count,
                        x_pooling=x_pooling,
                        from_index=from_index,
                        to_index=to_index,
                    )
                    for path, label in zip(enabled_paths, labels, strict=True)
                ]
            except (ValueError, FileNotFoundError) as exc:
                self.show_error(str(exc))
                return

        title = self.title_var.get().strip() or None
        try:
            figure = self.build_figure(
                logs=logs,
                histogram_data=histogram_data,
                x_column=x_column,
                y_columns=y_columns,
                std_mapping=std_mapping,
                std_ema_columns=std_ema_columns or None,
                min_mapping=min_mapping,
                min_ema_columns=min_ema_columns or None,
                max_mapping=max_mapping,
                max_ema_columns=max_ema_columns or None,
                ema_mapping=ema_mapping or None,
                ema_only_columns=ema_only_columns or None,
                ratios=ratios or None,
                title=title,
                group_keys=group_keys,
                file_opacities=file_opacities,
                file_colors=file_colors,
            )
        except ValueError as exc:
            self.show_error(str(exc))
            return
        self.apply_line_opacity(figure, self.line_alpha_var.get())
        self.apply_line_width(figure, self.line_width_var.get())
        self.render_figure(figure)
        self.save_selected_paths()
        status_message = histogram_status_message(histogram_data)
        if status_message:
            self.set_status(f"Plot updated. {status_message}")
        else:
            self.set_status("Plot updated.")
        self.schedule_auto_refresh(auto_refresh_interval)

    def on_auto_refresh_interval_change(self, *_args: object) -> None:
        self.schedule_auto_refresh()

    def parse_auto_refresh_interval(self, show_error: bool) -> float | None:
        raw_interval = self.auto_refresh_interval_var.get().strip()
        if not raw_interval:
            return 0.0
        try:
            interval = float(raw_interval)
        except ValueError:
            if show_error:
                self.show_error("Auto refresh interval must be a number >= 0.")
            return None
        if interval < 0:
            if show_error:
                self.show_error("Auto refresh interval must be >= 0.")
            return None
        return interval

    def parse_row_range(self, show_error: bool) -> tuple[int, int | None] | None:
        raw_from = self.from_var.get().strip()
        raw_to = self.to_var.get().strip()
        if not raw_from:
            from_index = 0
        else:
            try:
                from_index = int(raw_from)
            except ValueError:
                if show_error:
                    self.show_error("From must be an integer >= 0.")
                return None
        if from_index < 0:
            if show_error:
                self.show_error("From must be an integer >= 0.")
            return None
        to_index: int | None
        if not raw_to:
            to_index = None
        else:
            try:
                to_index = int(raw_to)
            except ValueError:
                if show_error:
                    self.show_error("To must be an integer >= From.")
                return None
            if to_index < from_index:
                if show_error:
                    self.show_error("To must be an integer >= From.")
                return None
        return from_index, to_index

    def csv_data_row_count(self, path: Path, delimiter: str) -> int:
        with path.open(newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            try:
                next(reader)
            except StopIteration:
                return 0
            return sum(1 for row in reader if any(cell.strip() for cell in row))

    def validate_from_index(self, enabled_paths: Sequence[Path], delimiter: str, from_index: int) -> bool:
        min_entry_count: int | None = None
        min_entry_path: Path | None = None
        for path in enabled_paths:
            try:
                entry_count = self.csv_data_row_count(path, delimiter)
            except (OSError, csv.Error) as exc:
                self.show_error(f"Failed to read {path}: {exc}")
                return False
            if min_entry_count is None or entry_count < min_entry_count:
                min_entry_count = entry_count
                min_entry_path = path
        if min_entry_count is None:
            return True
        if from_index >= min_entry_count:
            self.show_error(
                f"From index {from_index} is out of range. "
                f"{min_entry_path.name if min_entry_path is not None else 'The smallest file'} "
                f"has {min_entry_count} entries (valid start: 0 to {max(0, min_entry_count - 1)})."
            )
            return False
        return True

    def schedule_auto_refresh(self, interval_seconds: float | None = None) -> None:
        self.cancel_auto_refresh()
        if self.figure is None:
            return
        resolved_interval = (
            self.parse_auto_refresh_interval(show_error=False)
            if interval_seconds is None
            else interval_seconds
        )
        if resolved_interval is None or resolved_interval <= 0:
            return
        delay_ms = max(1, int(resolved_interval * 1000))
        self.auto_refresh_after_id = self.root.after(delay_ms, self.on_auto_refresh_timer)

    def cancel_auto_refresh(self) -> None:
        if self.auto_refresh_after_id is None:
            return
        self.root.after_cancel(self.auto_refresh_after_id)
        self.auto_refresh_after_id = None

    def on_auto_refresh_timer(self) -> None:
        self.auto_refresh_after_id = None
        interval = self.parse_auto_refresh_interval(show_error=False)
        if interval is None or interval <= 0:
            return
        self.plot()

    def build_figure(
        self,
        logs: Sequence[plot_logs.LogSeries],
        histogram_data: dict[str, list[HistogramSeries]],
        x_column: str,
        y_columns: Sequence[str],
        std_mapping: dict[str, str | None],
        std_ema_columns: set[str] | None,
        min_mapping: dict[str, str | None],
        min_ema_columns: set[str] | None,
        max_mapping: dict[str, str | None],
        max_ema_columns: set[str] | None,
        ema_mapping: dict[str, float] | None,
        ema_only_columns: set[str] | None,
        ratios: Sequence[float] | None,
        title: str | None,
        group_keys: Sequence[str] | None,
        file_opacities: Sequence[float] | None,
        file_colors: Sequence[str] | None,
    ) -> plt.Figure:
        base_ratios = list(ratios) if ratios is not None else [1.0] * len(y_columns)
        axis_specs: list[tuple[str, HistogramSeries | None, float]] = []
        height_ratios: list[float] = []
        for column, ratio in zip(y_columns, base_ratios, strict=True):
            if column in histogram_data:
                for series in histogram_data[column]:
                    axis_specs.append((column, series, ratio))
                    height_ratios.append(ratio)
            else:
                axis_specs.append((column, None, ratio))
                height_ratios.append(ratio)

        row_count = len(axis_specs)
        figure_height = 3 * (sum(height_ratios) if height_ratios else 1)
        figure, axes = plt.subplots(
            nrows=row_count,
            ncols=1,
            sharex=True,
            figsize=(10, max(3, figure_height)),
            gridspec_kw={"height_ratios": height_ratios} if height_ratios else None,
        )
        if row_count == 1:
            axes = [axes]
        if group_keys is None:
            group_keys = [log.label for log in logs]
        if file_opacities is None or len(file_opacities) != len(logs):
            file_opacities = [1.0] * len(logs)
        if file_colors is None or len(file_colors) != len(logs):
            file_colors = [""] * len(logs)
        group_color_map = self.group_color_map(group_keys) if logs else {}
        ema_only_columns = ema_only_columns or set()
        std_ema_columns = std_ema_columns or set()
        min_ema_columns = min_ema_columns or set()
        max_ema_columns = max_ema_columns or set()
        hist_ranges = {
            column: histogram_value_range(
                series_list
                if column not in ema_only_columns
                else [
                    HistogramSeries(
                        label=series.label,
                        x_values=series.x_values,
                        x_is_datetime=series.x_is_datetime,
                        bin_edges=series.bin_edges,
                        values=log_histogram_values(series.values),
                        edges_status=series.edges_status,
                    )
                    for series in series_list
                ]
            )
            for column, series_list in histogram_data.items()
        }
        hist_axes: dict[str, list[plt.Axes]] = {}
        hist_meshes: dict[str, matplotlib.collections.QuadMesh] = {}
        first_scalar_axis: plt.Axes | None = None
        for axis, (column, series, _ratio) in zip(axes, axis_specs, strict=True):
            if series is None:
                ema_alpha = ema_mapping.get(column) if ema_mapping else None
                only_ema = ema_alpha is not None and column in ema_only_columns
                force_summary_ema = only_ema and ema_alpha is not None
                for log, group_key, base_alpha, file_color in zip(
                    logs,
                    group_keys,
                    file_opacities,
                    file_colors,
                    strict=True,
                ):
                    color = file_color or group_color_map.get(group_key)
                    raw_values = log.y_values[column]
                    ema_values = (
                        exponential_moving_average(raw_values, ema_alpha)
                        if ema_alpha is not None
                        else None
                    )
                    line_color = color
                    if not only_ema:
                        line = axis.plot(
                            log.x_values,
                            raw_values,
                            label=log.label,
                            color=color,
                        )[0]
                        line._plot_alpha_base = base_alpha
                        line_color = line.get_color()
                    if ema_values is not None:
                        ema_line = axis.plot(
                            log.x_values,
                            ema_values,
                            label=log.label if only_ema else "_ema",
                            color=line_color,
                            linestyle="--",
                        )[0]
                        ema_line._plot_alpha_base = base_alpha
                    std_column = std_mapping.get(column)
                    if std_column is not None:
                        std_values = log.y_std_values.get(column)
                        if std_values:
                            center_values = raw_values
                            should_ema_std = (
                                ema_alpha is not None
                                and (column in std_ema_columns or force_summary_ema)
                            )
                            if should_ema_std:
                                std_values = exponential_moving_average(std_values, ema_alpha)
                                if ema_values is not None:
                                    center_values = ema_values
                            upper = [
                                center + std
                                for center, std in zip(
                                    center_values,
                                    std_values,
                                    strict=True,
                                )
                            ]
                            lower = [
                                center - std
                                for center, std in zip(
                                    center_values,
                                    std_values,
                                    strict=True,
                                )
                            ]
                            axis.fill_between(
                                log.x_values,
                                lower,
                                upper,
                                alpha=0.2 * base_alpha,
                                color=line_color,
                            )
                    min_column = min_mapping.get(column)
                    if min_column is not None:
                        min_values = log.y_min_values.get(column)
                        if min_values:
                            should_ema_min = (
                                ema_alpha is not None
                                and (column in min_ema_columns or force_summary_ema)
                            )
                            if should_ema_min:
                                min_values = exponential_moving_average(min_values, ema_alpha)
                            min_line = axis.plot(
                                log.x_values,
                                min_values,
                                color=line_color,
                                linestyle="--",
                                label="_min",
                            )[0]
                            min_line._plot_alpha_base = base_alpha
                    max_column = max_mapping.get(column)
                    if max_column is not None:
                        max_values = log.y_max_values.get(column)
                        if max_values:
                            should_ema_max = (
                                ema_alpha is not None
                                and (column in max_ema_columns or force_summary_ema)
                            )
                            if should_ema_max:
                                max_values = exponential_moving_average(max_values, ema_alpha)
                            max_line = axis.plot(
                                log.x_values,
                                max_values,
                                color=line_color,
                                linestyle=":",
                                label="_max",
                            )[0]
                            max_line._plot_alpha_base = base_alpha
                axis.set_ylabel(column)
                axis.grid(alpha=0.3)
                if first_scalar_axis is None:
                    first_scalar_axis = axis
                continue

            x_edges = build_edges_from_centers(series.x_values)
            y_edges = resolve_histogram_edges(series)
            row_values = (
                log_histogram_values(series.values)
                if column in ema_only_columns
                else series.values
            )
            values = transpose_histogram(row_values)
            vmin, vmax = hist_ranges[column]
            mesh = axis.pcolormesh(
                x_edges,
                y_edges,
                values,
                shading="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            axis.set_ylabel(f"{column} (log)" if column in ema_only_columns else column)
            if len(histogram_data[column]) > 1:
                axis.set_title(series.label)
            hist_axes.setdefault(column, []).append(axis)
            hist_meshes[column] = mesh

        if resolve_x_datetime_flag(logs, histogram_data, x_column):
            for axis in axes:
                locator = mdates.AutoDateLocator()
                formatter = mdates.ConciseDateFormatter(locator)
                axis.xaxis.set_major_locator(locator)
                axis.xaxis.set_major_formatter(formatter)

        axes[-1].set_xlabel(x_column)
        if title:
            figure.suptitle(title)
        if first_scalar_axis is not None and len(logs) > 1:
            first_scalar_axis.legend()
        for column, axes_for_column in hist_axes.items():
            mesh = hist_meshes[column]
            colorbar = figure.colorbar(mesh, ax=axes_for_column, pad=0.01)
            colorbar.set_label("log(freq)" if column in ema_only_columns else "freq")
        figure.tight_layout()
        return figure

    def render_figure(self, figure: plt.Figure) -> None:
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.toolbar:
            self.toolbar.destroy()
            self.toolbar = None
        if self.figure:
            plt.close(self.figure)
        self.figure = figure
        self.canvas = FigureCanvasTkAgg(figure, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.apply_toolbar_theme()
        self.apply_figure_theme(self.figure)

    def on_opacity_change(self, _value: str) -> None:
        if self.figure is None:
            return
        self.apply_line_opacity(self.figure, self.line_alpha_var.get())
        if self.canvas:
            self.canvas.draw_idle()

    def on_line_width_change(self, _value: str) -> None:
        if self.figure is None:
            return
        self.apply_line_width(self.figure, self.line_width_var.get())
        if self.canvas:
            self.canvas.draw_idle()

    def apply_line_opacity(self, figure: plt.Figure, alpha: float) -> None:
        clamped = max(0.0, min(1.0, alpha))
        for axis in figure.get_axes():
            for line in axis.get_lines():
                base_alpha = getattr(line, "_plot_alpha_base", 1.0)
                line.set_alpha(clamped * base_alpha)

    def apply_line_width(self, figure: plt.Figure, width: float) -> None:
        clamped = max(0.1, width)
        for axis in figure.get_axes():
            for line in axis.get_lines():
                line.set_linewidth(clamped)

    def on_theme_toggle(self) -> None:
        self.apply_theme(self.dark_mode_var.get())
        if self.figure and self.canvas:
            self.apply_figure_theme(self.figure)
            self.canvas.draw_idle()

    def apply_theme(self, dark_mode: bool) -> None:
        palette = self.tk_palette(dark_mode)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=palette["panel_bg"])
        style.configure("TLabel", background=palette["panel_bg"], foreground=palette["fg"])
        style.configure(
            "TLabelframe",
            background=palette["panel_bg"],
            foreground=palette["fg"],
            bordercolor=palette["border"],
        )
        style.configure(
            "TLabelframe.Label",
            background=palette["panel_bg"],
            foreground=palette["fg"],
        )
        style.configure(
            "TButton",
            background=palette["panel_bg"],
            foreground=palette["fg"],
        )
        style.map(
            "TButton",
            background=[("active", palette["accent"]), ("disabled", palette["panel_bg"])],
            foreground=[("active", palette["accent_fg"]), ("disabled", palette["disabled_fg"])],
        )
        style.configure(
            "Accent.TButton",
            background=palette["accent"],
            foreground=palette["accent_fg"],
            padding=(14, 8),
            font=("TkDefaultFont", 11, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", palette["accent"]), ("disabled", palette["panel_bg"])],
            foreground=[("active", palette["accent_fg"]), ("disabled", palette["disabled_fg"])],
        )
        style.configure(
            "PlotTab.TButton",
            background=palette["panel_bg"],
            foreground=palette["fg"],
            padding=(8, 5),
        )
        style.map(
            "PlotTab.TButton",
            background=[("active", palette["accent"]), ("disabled", palette["panel_bg"])],
            foreground=[("active", palette["accent_fg"]), ("disabled", palette["disabled_fg"])],
        )
        style.configure(
            "PlotTabActive.TButton",
            background=palette["accent"],
            foreground=palette["accent_fg"],
            padding=(8, 5),
            font=("TkDefaultFont", 9, "bold"),
        )
        style.map(
            "PlotTabActive.TButton",
            background=[("active", palette["accent"]), ("disabled", palette["panel_bg"])],
            foreground=[("active", palette["accent_fg"]), ("disabled", palette["disabled_fg"])],
        )
        style.configure(
            "PlotTabClose.TButton",
            background=palette["panel_bg"],
            foreground=palette["fg"],
            padding=(4, 5),
        )
        style.map(
            "PlotTabClose.TButton",
            background=[("active", palette["accent"]), ("disabled", palette["panel_bg"])],
            foreground=[("active", palette["accent_fg"]), ("disabled", palette["disabled_fg"])],
        )
        style.configure(
            "PlotTabCloseActive.TButton",
            background=palette["accent"],
            foreground=palette["accent_fg"],
            padding=(4, 5),
            font=("TkDefaultFont", 9, "bold"),
        )
        style.map(
            "PlotTabCloseActive.TButton",
            background=[("active", palette["accent"]), ("disabled", palette["panel_bg"])],
            foreground=[("active", palette["accent_fg"]), ("disabled", palette["disabled_fg"])],
        )
        style.configure(
            "PlotTabAdd.TButton",
            background=palette["panel_bg"],
            foreground=palette["fg"],
            padding=(8, 5),
        )
        style.map(
            "PlotTabAdd.TButton",
            background=[("active", palette["accent"]), ("disabled", palette["panel_bg"])],
            foreground=[("active", palette["accent_fg"]), ("disabled", palette["disabled_fg"])],
        )
        style.configure(
            "TEntry",
            fieldbackground=palette["entry_bg"],
            foreground=palette["fg"],
            background=palette["panel_bg"],
        )
        style.configure(
            "TCombobox",
            fieldbackground=palette["entry_bg"],
            foreground=palette["fg"],
            background=palette["panel_bg"],
            arrowcolor=palette["fg"],
        )
        style.map(
            "TCombobox",
            fieldbackground=[
                ("readonly", palette["entry_bg"]),
                ("disabled", palette["panel_bg"]),
            ],
            foreground=[("readonly", palette["fg"]), ("disabled", palette["disabled_fg"])],
            background=[("readonly", palette["panel_bg"]), ("disabled", palette["panel_bg"])],
        )
        style.configure(
            "TCheckbutton",
            background=palette["panel_bg"],
            foreground=palette["fg"],
        )
        style.configure(
            "SmallEma.TCheckbutton",
            background=palette["panel_bg"],
            foreground=palette["fg"],
            padding=0,
        )
        try:
            style.configure("SmallEma.TCheckbutton", indicatorsize=9)
        except tk.TclError:
            pass
        style.configure("TScale", background=palette["panel_bg"])
        style.configure("TScrollbar", background=palette["panel_bg"])

        self.root.configure(background=palette["root_bg"])
        if self.controls_canvas:
            self.controls_canvas.configure(
                background=palette["panel_bg"],
                highlightbackground=palette["panel_bg"],
            )
        self.files_listbox.configure(
            background=palette["list_bg"],
            foreground=palette["fg"],
            selectbackground=palette["select_bg"],
            selectforeground=palette["select_fg"],
            highlightbackground=palette["border"],
            highlightcolor=palette["border"],
        )
        self.root.option_add("*TCombobox*Listbox.background", palette["list_bg"])
        self.root.option_add("*TCombobox*Listbox.foreground", palette["fg"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", palette["select_bg"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", palette["select_fg"])
        self.refresh_plot_tab_buttons()
        self.apply_toolbar_theme()

    def apply_toolbar_theme(self) -> None:
        if not self.toolbar:
            return
        palette = self.tk_palette(self.dark_mode_var.get())
        try:
            self.toolbar.configure(background=palette["panel_bg"])
        except tk.TclError:
            pass
        for widget in self.toolbar.winfo_children():
            options = {
                "background": palette["panel_bg"],
                "foreground": palette["fg"],
                "highlightbackground": palette["panel_bg"],
            }
            widget_keys = widget.keys()
            if "activebackground" in widget_keys:
                options["activebackground"] = palette["accent"]
            if "activeforeground" in widget_keys:
                options["activeforeground"] = palette["accent_fg"]
            try:
                widget.configure(**options)
            except tk.TclError:
                continue

    def apply_figure_theme(self, figure: plt.Figure) -> None:
        dark_mode = self.dark_mode_var.get()
        palette = self.figure_palette(dark_mode)
        figure.set_facecolor(palette["figure_face"])
        if figure._suptitle is not None:
            figure._suptitle.set_color(palette["text"])
        for axis in figure.get_axes():
            axis.set_facecolor(palette["axes_face"])
            axis.tick_params(colors=palette["text"])
            axis.xaxis.label.set_color(palette["text"])
            axis.yaxis.label.set_color(palette["text"])
            axis.title.set_color(palette["text"])
            for spine in axis.spines.values():
                spine.set_color(palette["spine"])
            for gridline in axis.get_xgridlines() + axis.get_ygridlines():
                gridline.set_color(palette["grid"])
            legend = axis.get_legend()
            if legend:
                frame = legend.get_frame()
                frame.set_facecolor(palette["legend_face"])
                frame.set_edgecolor(palette["spine"])
                for text in legend.get_texts():
                    text.set_color(palette["text"])

    def tk_palette(self, dark_mode: bool) -> dict[str, str]:
        if dark_mode:
            return {
                "root_bg": "#1e1e1e",
                "panel_bg": "#252526",
                "fg": "#dcdcdc",
                "disabled_fg": "#6f6f6f",
                "entry_bg": "#2d2d2d",
                "list_bg": "#2a2a2a",
                "select_bg": "#3a8dff",
                "select_fg": "#ffffff",
                "border": "#3c3c3c",
                "accent": "#3a8dff",
                "accent_fg": "#ffffff",
            }
        return {
            "root_bg": "#f0f0f0",
            "panel_bg": "#f6f6f6",
            "fg": "#1f1f1f",
            "disabled_fg": "#8a8a8a",
            "entry_bg": "#ffffff",
            "list_bg": "#ffffff",
            "select_bg": "#4a76d1",
            "select_fg": "#ffffff",
            "border": "#c4c4c4",
            "accent": "#4a76d1",
            "accent_fg": "#ffffff",
        }

    def figure_palette(self, dark_mode: bool) -> dict[str, str]:
        if dark_mode:
            return {
                "figure_face": "#1b1b1b",
                "axes_face": "#222222",
                "text": "#e6e6e6",
                "spine": "#5a5a5a",
                "grid": "#3a3a3a",
                "legend_face": "#2d2d2d",
            }
        legend_face = self.light_figure_palette["legend_face"]
        if legend_face == "inherit":
            legend_face = self.light_figure_palette["axes_face"]
        return {
            "figure_face": self.light_figure_palette["figure_face"],
            "axes_face": self.light_figure_palette["axes_face"],
            "text": self.light_figure_palette["text"],
            "spine": self.light_figure_palette["spine"],
            "grid": self.light_figure_palette["grid"],
            "legend_face": legend_face,
        }

    def show_error(self, message: str) -> None:
        messagebox.showerror("Plot Logs Interactive", message)
        self.set_status(message)

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def on_close(self) -> None:
        for after_id in self.typeahead_after_ids.values():
            self.root.after_cancel(after_id)
        self.cancel_auto_refresh()
        if self.figure:
            plt.close(self.figure)
            self.figure = None
        self.root.quit()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    PlotLogsInteractiveApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
