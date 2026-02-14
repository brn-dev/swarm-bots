from __future__ import annotations

import ast
import csv
import json
import math
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

STATE_PATH = Path(__file__).resolve().with_name(".plot_logs_config.json")


@dataclass(slots=True)
class PlotRow:
    index_label: ttk.Label
    y_var: tk.StringVar
    y_combo: ttk.Combobox
    height_var: tk.StringVar
    height_entry: ttk.Entry
    ema_var: tk.StringVar
    ema_entry: ttk.Entry
    std_var: tk.BooleanVar
    std_check: ttk.Checkbutton
    skew_var: tk.BooleanVar
    skew_check: ttk.Checkbutton
    min_var: tk.BooleanVar
    min_check: ttk.Checkbutton
    max_var: tk.BooleanVar
    max_check: ttk.Checkbutton
    remove_button: ttk.Button


@dataclass(frozen=True, slots=True)
class PresetEntry:
    y_column: str
    height: float
    ema: float | None = None
    std: bool = False
    skew: bool = False
    min: bool = False
    max: bool = False


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


PLOT_PRESETS: tuple[PlotPreset, ...] = (
    PlotPreset(
        name="Episode Stats",
        entries=(
            PresetEntry("ep_rew_ema", 3.0, ema=0.02),
            PresetEntry("ep_rew__mean", 1.0, std=True, ema=0.02),
            PresetEntry("ep_len__mean", 1.0, std=True, ema=0.02),
        ),
    ),
    PlotPreset(
        name="Actions",
        entries=(
            PresetEntry("ep_rew_ema", 2.0, ema=0.02),
            PresetEntry("act0__histogram_freqs", 3.0),
            PresetEntry("act1__histogram_freqs", 1.0),
            PresetEntry("act0__mean", 1.0, std=True),
            PresetEntry("std0__mean", 1.0, std=True),
            PresetEntry("act1__mean", 1.0, std=True),
        ),
    ),
    PlotPreset(
        name="PPO Metrics",
        entries=(
            PresetEntry("ep_rew_ema", 2.0, ema=0.02),
            PresetEntry("approx_kl__mean", 1.0, std=True),
            PresetEntry("clip_frac__mean", 1.0, std=True),
            PresetEntry("ratio__std", 1.0),
            PresetEntry("expl_var", 1.0),
            PresetEntry("learning_rate", 1.0),
        ),
    ),
    PlotPreset(
        name="PPO Losses",
        entries=(
            PresetEntry("ep_rew_ema", 2.0, ema=0.02),
            PresetEntry("act_loss__mean", 1.0, std=True),
            PresetEntry("val_loss__mean", 1.0, std=True),
            PresetEntry("ent_loss__mean", 1.0, std=True),
            PresetEntry("grad_norm__mean", 1.0, std=True),
            PresetEntry("grad_clip_frac", 1.0),
            PresetEntry("learning_rate", 1.0),
        ),
    ),
    PlotPreset(
        name="PPO Full",
        entries=(
            PresetEntry("ep_rew_ema", 2.0, ema=0.02),
            PresetEntry("approx_kl__mean", 1.0, std=True),
            PresetEntry("clip_frac__mean", 1.0, std=True),
            PresetEntry("ratio__std", 1.0),
            PresetEntry("expl_var", 1.0),
            PresetEntry("val_loss__mean", 1.0, std=True),
            PresetEntry("act_loss__mean", 1.0, std=True),
            PresetEntry("ent_loss__mean", 1.0, std=True),
            PresetEntry("grad_norm__mean", 1.0, std=True),
            PresetEntry("grad_clip_frac", 1.0),
            PresetEntry("learning_rate", 1.0),
        ),
    ),
    PlotPreset(
        name="NOP Losses",
        entries=(
            PresetEntry("ep_rew_ema", 2.0, ema=0.02),
            PresetEntry("scalar_loss__mean", 1.0, std=True),
            PresetEntry("angle_loss__mean", 1.0, std=True),
            PresetEntry("rot6d_loss__mean", 1.0, std=True),
            PresetEntry("binary_loss__mean", 1.0, std=True),
        ),
    ),
    PlotPreset(
        name="Performance",
        entries=(
            PresetEntry("fps", 1.0, ema=0.02),
            PresetEntry("updates", 1.0),
            PresetEntry("policy_forward_time__mean", 1.0, std=True),
            PresetEntry("env_step_time__mean", 1.0, std=True),
            PresetEntry("rollout_time", 1.0),
            PresetEntry("sampling_time__mean", 1.0, std=True),
            PresetEntry("update_time__mean", 1.0, std=True),
            PresetEntry("train_time", 1.0),
        ),
    ),
)


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
    result: list[float] = []
    prev = math.nan
    for value in values:
        if math.isnan(value):
            result.append(math.nan)
            continue
        if math.isnan(prev):
            prev = value
        else:
            prev = alpha * value + (1.0 - alpha) * prev
        result.append(prev)
    return result


def histogram_edges_match(reference: Sequence[float], candidate: Sequence[float]) -> bool:
    if len(reference) != len(candidate):
        return False
    return all(
        math.isclose(left, right, rel_tol=1e-3, abs_tol=1e-4)
        for left, right in zip(reference, candidate, strict=True)
    )


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
        return list(edges)
    if len(edges) == bin_count:
        return build_edges_from_centers(edges)
    raise ValueError(
        f"Histogram edges length {len(edges)} does not match bins {bin_count}."
    )


def linear_edges_from_range(min_edge: float, max_edge: float, bin_count: int) -> list[float]:
    if bin_count <= 0:
        raise ValueError("bin_count must be positive.")
    if math.isclose(min_edge, max_edge, rel_tol=0.0, abs_tol=0.0):
        span = 1.0
        min_edge -= span / 2
        max_edge += span / 2
    step = (max_edge - min_edge) / bin_count
    return [min_edge + step * index for index in range(bin_count + 1)]


def histogram_status_message(histogram_data: dict[str, list[HistogramSeries]]) -> str | None:
    drifting: list[str] = []
    for column, series_list in histogram_data.items():
        if any(series.edges_status == "drifting" for series in series_list):
            drifting.append(column)
    if not drifting:
        return None
    joined = ", ".join(drifting)
    return f"Histogram edges drift for {joined}; using range-based edges."


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
) -> HistogramSeries:
    x_values: list[float] = []
    x_is_datetime: bool | None = None
    values: list[list[float]] = []
    bin_edges: list[float] | None = None
    edges_status = "missing" if edges_column is None else "edges"
    expected_bins: int | None = None
    min_edge: float | None = None
    max_edge: float | None = None
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
            freqs = parse_histogram_list(row.get(freqs_column), freqs_column, path, row_index)
            if expected_bins is None:
                expected_bins = len(freqs)
            elif len(freqs) != expected_bins:
                raise ValueError(
                    f"Histogram bins changed in {path} column {freqs_column} at row {row_index}."
                )
            if edges_column is not None:
                edges = parse_histogram_list(row.get(edges_column), edges_column, path, row_index)
                normalized = normalize_histogram_edges(edges, expected_bins)
                if bin_edges is None:
                    bin_edges = normalized
                elif not histogram_edges_match(bin_edges, normalized):
                    edges_status = "drifting"
                min_edge = normalized[0] if min_edge is None else min(min_edge, normalized[0])
                max_edge = normalized[-1] if max_edge is None else max(max_edge, normalized[-1])
            x_values.append(x_value)
            values.append(freqs)
    if not x_values:
        raise ValueError(f"No histogram data in {path} for {freqs_column}.")
    if edges_column is not None and edges_status == "drifting":
        if expected_bins is None or min_edge is None or max_edge is None:
            raise ValueError(f"Histogram edges missing in {path} for {freqs_column}.")
        bin_edges = linear_edges_from_range(min_edge, max_edge, expected_bins)
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
        self.controls_canvas: tk.Canvas | None = None
        self.controls_canvas_window: int | None = None
        self.figure: plt.Figure | None = None
        self.canvas: FigureCanvasTkAgg | None = None
        self.toolbar: NavigationToolbar2Tk | None = None

        self.delimiter_var = tk.StringVar(value=";")
        self.title_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Select CSV files to begin.")
        self.group_var = tk.StringVar()
        self.line_alpha_var = tk.DoubleVar(value=1.0)
        self.line_width_var = tk.DoubleVar(value=0.75)
        self.dark_mode_var = tk.BooleanVar(value=True)
        self.file_opacity_var = tk.StringVar(value="1.0")
        self.file_opacity_by_path: dict[Path, float] = {}
        self.file_color_var = tk.StringVar(value="")
        self.file_color_by_path: dict[Path, str] = {}
        self.path_enabled: dict[Path, bool] = {}
        self.light_figure_palette = {
            "figure_face": matplotlib.rcParams["figure.facecolor"],
            "axes_face": matplotlib.rcParams["axes.facecolor"],
            "text": matplotlib.rcParams["text.color"],
            "spine": matplotlib.rcParams["axes.edgecolor"],
            "grid": matplotlib.rcParams["grid.color"],
            "legend_face": matplotlib.rcParams["legend.facecolor"],
        }

        self._build_layout()
        self.apply_theme(self.dark_mode_var.get())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind_all("<KeyPress>", self.on_global_keypress, add=True)
        self.root.bind_all("<space>", self.on_spacebar, add=True)
        self.add_plot_row()

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

        self.files_listbox = tk.Listbox(files_frame, height=6, selectmode="extended")
        self.files_listbox.grid(row=0, column=0, sticky="ew")
        self.files_listbox.bind("<<ListboxSelect>>", self.on_file_selection)
        files_scrollbar = ttk.Scrollbar(files_frame, orient="vertical", command=self.files_listbox.yview)
        files_scrollbar.grid(row=0, column=1, sticky="ns")
        self.files_listbox.configure(yscrollcommand=files_scrollbar.set)

        files_buttons = ttk.Frame(files_frame)
        files_buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
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

        meta_frame = ttk.Frame(files_frame)
        meta_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
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

        x_column_frame = ttk.Frame(plots_frame)
        x_column_frame.grid(row=0, column=0, sticky="ew")
        x_column_frame.columnconfigure(1, weight=1)
        ttk.Label(x_column_frame, text="X Column").grid(row=0, column=0, sticky="w")
        self.x_combo = ttk.Combobox(x_column_frame, state="disabled")
        self.x_combo.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.bind_typeahead(self.x_combo, lambda: self.available_columns)

        self.plots_container = ttk.Frame(plots_frame)
        self.plots_container.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.plots_container.columnconfigure(1, weight=1, minsize=100)
        self.plots_container.columnconfigure(2, weight=0)
        self.plots_container.columnconfigure(3, weight=0)
        self.plots_container.columnconfigure(4, weight=0, minsize=20)
        self.plots_container.columnconfigure(5, weight=0, minsize=20)
        self.plots_container.columnconfigure(6, weight=0, minsize=20)
        self.plots_container.columnconfigure(7, weight=0, minsize=20)
        self.plots_container.columnconfigure(8, weight=0)

        ttk.Label(self.plots_container, text="#").grid(row=0, column=0, sticky="w")
        ttk.Label(self.plots_container, text="Y Column").grid(row=0, column=1, sticky="w", padx=(10, 4))
        ttk.Label(self.plots_container, text="Height").grid(row=0, column=2, sticky="w", padx=(6, 4))
        ttk.Label(self.plots_container, text="EMA").grid(row=0, column=3, sticky="w", padx=(2, 4))
        ttk.Label(self.plots_container, text="STD").grid(row=0, column=4, sticky="w", padx=(1, 1))
        ttk.Label(self.plots_container, text="Skew").grid(row=0, column=5, sticky="w", padx=(1, 1))
        ttk.Label(self.plots_container, text="Min").grid(row=0, column=6, sticky="w", padx=(1, 1))
        ttk.Label(self.plots_container, text="Max").grid(row=0, column=7, sticky="w", padx=(1, 1))
        ttk.Label(self.plots_container, text="").grid(row=0, column=8, sticky="w", padx=(6, 4))

        plots_buttons = ttk.Frame(plots_frame)
        plots_buttons.grid(row=2, column=0, sticky="ew", pady=(6, 0))
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
            text="A",
            command=self.toggle_all_std,
            width=2,
        )
        self.toggle_skew_button = ttk.Button(
            self.plots_container,
            text="A",
            command=self.toggle_all_skew,
            width=2,
        )
        self.toggle_min_button = ttk.Button(
            self.plots_container,
            text="A",
            command=self.toggle_all_min,
            width=2,
        )
        self.toggle_max_button = ttk.Button(
            self.plots_container,
            text="A",
            command=self.toggle_all_max,
            width=2,
        )

        if PLOT_PRESETS:
            presets_label = ttk.Label(plots_frame, text="Presets")
            presets_label.grid(row=3, column=0, sticky="w", pady=(8, 0))
            presets_frame = ttk.Frame(plots_frame)
            presets_frame.grid(row=4, column=0, sticky="ew", pady=(4, 0))
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
        plot_button = ttk.Button(action_frame, text="Plot", command=self.plot, style="Accent.TButton")
        plot_button.grid(row=0, column=0, sticky="ew")

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

    def add_files(self) -> None:
        directory = filedialog.askdirectory(
            title="Select folder containing log.csv files",
            initialdir=str(REPO_ROOT),
        )
        if not directory:
            return
        root_path = Path(directory).resolve()
        found_paths = [
            path.resolve()
            for path in root_path.rglob("log.csv")
            if path.is_file()
        ]
        if not found_paths:
            self.set_status("No log.csv files found in the selected folder.")
            return
        added_paths: list[Path] = []
        for path in sorted(found_paths):
            if path not in self.paths:
                self.paths.append(path)
                added_paths.append(path)
            self.file_opacity_by_path.setdefault(path, 1.0)
            self.file_color_by_path.setdefault(path, "")
            self.path_enabled.setdefault(path, True)
        if len(added_paths) > 5:
            for path in added_paths:
                self.path_enabled[path] = False
        self.refresh_file_list()
        self.refresh_columns()
        if not added_paths:
            self.set_status("All log.csv files in the selected folder are already added.")
        elif len(added_paths) > 5:
            self.set_status(
                f"Added {len(added_paths)} log.csv files (disabled by default)."
            )
        else:
            self.set_status(f"Added {len(added_paths)} log.csv files.")

    def remove_selected_files(self) -> None:
        selected_indices = list(self.files_listbox.curselection())
        if not selected_indices:
            return
        for index in sorted(selected_indices, reverse=True):
            self.path_groups.pop(self.paths[index], None)
            self.file_opacity_by_path.pop(self.paths[index], None)
            self.file_color_by_path.pop(self.paths[index], None)
            self.path_enabled.pop(self.paths[index], None)
            del self.paths[index]
        self.refresh_file_list()
        self.refresh_columns()

    def refresh_file_list(self) -> None:
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
                "std": row.std_var.get(),
                "skew": row.skew_var.get(),
                "min": row.min_var.get(),
                "max": row.max_var.get(),
            }
            rows.append(entry)
        return {
            "x": self.x_combo.get().strip(),
            "rows": rows,
        }

    def build_config_payload(self) -> dict[str, object]:
        return {
            "paths": self.build_paths_payload(),
            "plots": self.build_plot_payload(),
        }

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
        self.refresh_file_list()
        self.refresh_columns()
        return len(loaded_paths), len(missing_paths)

    def apply_plot_payload(self, payload: object) -> tuple[int, list[str], str | None]:
        if not isinstance(payload, dict):
            raise ValueError("Saved plot configuration is invalid.")
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
                if y_valid:
                    desired_std = row_data.get("std") if isinstance(row_data.get("std"), bool) else None
                    desired_skew = row_data.get("skew") if isinstance(row_data.get("skew"), bool) else None
                    desired_min = row_data.get("min") if isinstance(row_data.get("min"), bool) else None
                    desired_max = row_data.get("max") if isinstance(row_data.get("max"), bool) else None
                    self.update_summary_checkboxes(
                        row,
                        desired_std=desired_std,
                        desired_skew=desired_skew,
                        desired_min=desired_min,
                        desired_max=desired_max,
                        preserve_existing=True,
                    )
        else:
            self.clear_plot_rows()
        x_value = payload.get("x")
        if isinstance(x_value, str) and x_value:
            if self.available_columns and x_value in self.available_columns:
                self.x_combo.set(x_value)
            else:
                missing_x = x_value
        return len(rows_payload), missing_columns, missing_x

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
        plots_payload = payload.get("plots")
        if plots_payload is not None:
            try:
                _plot_count, missing_plot_columns, missing_x = self.apply_plot_payload(
                    plots_payload
                )
            except ValueError as exc:
                self.show_error(str(exc))
                return
        status_parts = [
            self.loaded_paths_message(loaded_count, missing_count, "saved paths")
        ]
        if missing_plot_columns:
            unique_missing = ", ".join(sorted(set(missing_plot_columns)))
            status_parts.append(f"Missing plot columns: {unique_missing}.")
        if missing_x:
            status_parts.append(f"Missing X column: {missing_x}.")
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
        plots_payload = payload.get("plots")
        if plots_payload is not None:
            try:
                _plot_count, missing_plot_columns, missing_x = self.apply_plot_payload(
                    plots_payload
                )
            except ValueError as exc:
                self.show_error(str(exc))
                return
        status_parts = [
            self.loaded_paths_message(loaded_count, missing_count, f"paths from {path.name}")
        ]
        if missing_plot_columns:
            unique_missing = ", ".join(sorted(set(missing_plot_columns)))
            status_parts.append(f"Missing plot columns: {unique_missing}.")
        if missing_x:
            status_parts.append(f"Missing X column: {missing_x}.")
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
            self.update_summary_checkboxes(
                row,
                preserve_existing=preserve_state and selection_preserved,
            )
        else:
            row.y_combo.configure(values=[], state="disabled")
            if not preserve_state:
                row.y_combo.set("")
                row.std_var.set(False)
                row.skew_var.set(False)
                row.min_var.set(False)
                row.max_var.set(False)
            row.std_check.state(["disabled"])
            row.skew_check.state(["disabled"])
            row.min_check.state(["disabled"])
            row.max_check.state(["disabled"])

    def add_plot_row(self) -> None:
        row_index = len(self.plot_rows) + 1
        index_label = ttk.Label(self.plots_container, text=str(row_index))
        index_label.grid(row=row_index, column=0, sticky="w", pady=2)

        y_var = tk.StringVar()
        y_combo = ttk.Combobox(self.plots_container, textvariable=y_var, state="disabled", width=28)
        y_combo.grid(row=row_index, column=1, sticky="ew", padx=(10, 4), pady=2)
        self.bind_typeahead(y_combo, lambda: self.available_columns)

        height_var = tk.StringVar(value="1")
        height_entry = ttk.Entry(self.plots_container, textvariable=height_var, width=5)
        height_entry.grid(row=row_index, column=2, sticky="w", padx=(6, 4), pady=2)

        ema_var = tk.StringVar()
        ema_entry = ttk.Entry(self.plots_container, textvariable=ema_var, width=5)
        ema_entry.grid(row=row_index, column=3, sticky="w", padx=(2, 4), pady=2)

        std_var = tk.BooleanVar(value=False)
        std_check = ttk.Checkbutton(self.plots_container, text="", variable=std_var, padding=0)
        std_check.grid(row=row_index, column=4, padx=(1, 1), pady=2)

        skew_var = tk.BooleanVar(value=False)
        skew_check = ttk.Checkbutton(self.plots_container, text="", variable=skew_var, padding=0)
        skew_check.grid(row=row_index, column=5, padx=(1, 1), pady=2)

        min_var = tk.BooleanVar(value=False)
        min_check = ttk.Checkbutton(self.plots_container, text="", variable=min_var, padding=0)
        min_check.grid(row=row_index, column=6, padx=(1, 1), pady=2)

        max_var = tk.BooleanVar(value=False)
        max_check = ttk.Checkbutton(self.plots_container, text="", variable=max_var, padding=0)
        max_check.grid(row=row_index, column=7, padx=(1, 1), pady=2)

        remove_button = ttk.Button(self.plots_container, text="🗑️", width=2)
        remove_button.grid(row=row_index, column=8, sticky="e", pady=2)

        row = PlotRow(
            index_label=index_label,
            y_var=y_var,
            y_combo=y_combo,
            height_var=height_var,
            height_entry=height_entry,
            ema_var=ema_var,
            ema_entry=ema_entry,
            std_var=std_var,
            std_check=std_check,
            skew_var=skew_var,
            skew_check=skew_check,
            min_var=min_var,
            min_check=min_check,
            max_var=max_var,
            max_check=max_check,
            remove_button=remove_button,
        )
        remove_button.configure(command=lambda target=row: self.remove_plot_row(target))
        y_combo.bind("<<ComboboxSelected>>", lambda _event, target=row: self.update_summary_checkboxes(target))
        self.plot_rows.append(row)
        self.update_row_options(row, self.available_columns, prefer_default=len(self.plot_rows) == 1)
        self.refresh_row_labels()

    def update_std_checkbox(self, row: PlotRow, desired: bool | None = None) -> None:
        y_value = row.y_combo.get()
        candidate = self.std_column_for(y_value)
        if candidate is None:
            row.std_var.set(False)
            row.std_check.state(["disabled"])
        else:
            row.std_check.state(["!disabled"])
            if desired is None:
                row.std_var.set(True)
            else:
                row.std_var.set(desired)

    def update_skew_checkbox(self, row: PlotRow, desired: bool | None = None) -> None:
        y_value = row.y_combo.get()
        candidate = self.skew_column_for(y_value)
        if candidate is None:
            row.skew_var.set(False)
            row.skew_check.state(["disabled"])
        else:
            row.skew_check.state(["!disabled"])
            if desired is None:
                row.skew_var.set(False)
            else:
                row.skew_var.set(desired)

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
        desired_skew: bool | None = None,
        desired_min: bool | None = None,
        desired_max: bool | None = None,
        preserve_existing: bool = False,
    ) -> None:
        if preserve_existing:
            if desired_std is None:
                desired_std = row.std_var.get()
            if desired_skew is None:
                desired_skew = row.skew_var.get()
            if desired_min is None:
                desired_min = row.min_var.get()
            if desired_max is None:
                desired_max = row.max_var.get()
        self.update_std_checkbox(row, desired=desired_std)
        self.update_skew_checkbox(row, desired=desired_skew)
        self.update_min_checkbox(row, desired=desired_min)
        self.update_max_checkbox(row, desired=desired_max)

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

    def update_min_checkbox(self, row: PlotRow, desired: bool | None = None) -> None:
        y_value = row.y_combo.get()
        min_candidate = self.min_column_for(y_value)
        if min_candidate is None:
            row.min_var.set(False)
            row.min_check.state(["disabled"])
        else:
            row.min_check.state(["!disabled"])
            if desired is None:
                row.min_var.set(False)
            else:
                row.min_var.set(desired)

    def update_max_checkbox(self, row: PlotRow, desired: bool | None = None) -> None:
        y_value = row.y_combo.get()
        max_candidate = self.max_column_for(y_value)
        if max_candidate is None:
            row.max_var.set(False)
            row.max_check.state(["disabled"])
        else:
            row.max_check.state(["!disabled"])
            if desired is None:
                row.max_var.set(False)
            else:
                row.max_var.set(desired)

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
        combo.bind(
            "<KeyPress>",
            lambda event, target=combo, getter=values_getter: self.on_typeahead(event, target, getter),
            add=True,
        )

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
            row.index_label,
            row.y_combo,
            row.height_entry,
            row.ema_entry,
            row.std_check,
            row.skew_check,
            row.min_check,
            row.max_check,
            row.remove_button,
        ):
            widget.destroy()
        if row in self.plot_rows:
            self.plot_rows.remove(row)
        self.refresh_row_labels()

    def clear_plot_rows(self) -> None:
        for row in self.plot_rows:
            for widget in (
                row.index_label,
                row.y_combo,
                row.height_entry,
                row.ema_entry,
                row.std_check,
                row.skew_check,
                row.min_check,
                row.max_check,
                row.remove_button,
            ):
                widget.destroy()
        self.plot_rows.clear()
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
        missing_skew: list[str] = []
        missing_min: list[str] = []
        missing_max: list[str] = []
        invalid_ema: list[str] = []
        for row, entry in zip(self.plot_rows, preset.entries, strict=True):
            row.y_combo.set(entry.y_column)
            row.height_var.set(self.format_ratio(entry.height))
            if entry.ema is None:
                row.ema_var.set("")
            else:
                row.ema_var.set(str(entry.ema))
                if self.is_histogram_column(entry.y_column):
                    invalid_ema.append(entry.y_column)
            if entry.std and self.std_column_for(entry.y_column) is None:
                missing_std.append(entry.y_column)
            if entry.skew and self.skew_column_for(entry.y_column) is None:
                missing_skew.append(entry.y_column)
            if entry.min and self.min_column_for(entry.y_column) is None:
                missing_min.append(entry.y_column)
            if entry.max and self.max_column_for(entry.y_column) is None:
                missing_max.append(entry.y_column)
            self.update_summary_checkboxes(
                row,
                desired_std=entry.std,
                desired_skew=entry.skew,
                desired_min=entry.min,
                desired_max=entry.max,
            )
        status_parts = [f"Preset {preset.name!r} loaded."]
        if missing_std:
            status_parts.append(
                f"Missing std columns for: {', '.join(sorted(set(missing_std)))}."
            )
        if missing_skew:
            status_parts.append(
                f"Missing skew columns for: {', '.join(sorted(set(missing_skew)))}."
            )
        if missing_min:
            status_parts.append(
                f"Missing min columns for: {', '.join(sorted(set(missing_min)))}."
            )
        if missing_max:
            status_parts.append(
                f"Missing max columns for: {', '.join(sorted(set(missing_max)))}."
            )
        if invalid_ema:
            status_parts.append(
                f"EMA ignored for histogram columns: {', '.join(sorted(set(invalid_ema)))}."
            )
        self.set_status(" ".join(status_parts))

    def format_ratio(self, ratio: float) -> str:
        if ratio.is_integer():
            return str(int(ratio))
        return str(ratio)

    def refresh_row_labels(self) -> None:
        for index, row in enumerate(self.plot_rows, start=1):
            row.index_label.configure(text=str(index))
            row.index_label.grid_configure(row=index, column=0)
            row.y_combo.grid_configure(row=index, column=1)
            row.height_entry.grid_configure(row=index, column=2)
            row.ema_entry.grid_configure(row=index, column=3)
            row.std_check.grid_configure(row=index, column=4)
            row.skew_check.grid_configure(row=index, column=5)
            row.min_check.grid_configure(row=index, column=6)
            row.max_check.grid_configure(row=index, column=7)
            row.remove_button.grid_configure(row=index, column=8)
        footer_row = len(self.plot_rows) + 1
        self.toggle_std_button.grid(row=footer_row, column=4, pady=(4, 0))
        self.toggle_skew_button.grid(row=footer_row, column=5, pady=(4, 0))
        self.toggle_min_button.grid(row=footer_row, column=6, pady=(4, 0))
        self.toggle_max_button.grid(row=footer_row, column=7, pady=(4, 0))

    def toggle_all_std(self) -> None:
        desired = not any(
            row.std_var.get()
            for row in self.plot_rows
            if self.std_column_for(row.y_combo.get()) is not None
        )
        self.set_all_std(desired)

    def toggle_all_skew(self) -> None:
        desired = not any(
            row.skew_var.get()
            for row in self.plot_rows
            if self.skew_column_for(row.y_combo.get()) is not None
        )
        self.set_all_skew(desired)

    def toggle_all_min(self) -> None:
        desired = not any(
            row.min_var.get()
            for row in self.plot_rows
            if self.min_column_for(row.y_combo.get()) is not None
        )
        self.set_all_min(desired)

    def toggle_all_max(self) -> None:
        desired = not any(
            row.max_var.get()
            for row in self.plot_rows
            if self.max_column_for(row.y_combo.get()) is not None
        )
        self.set_all_max(desired)

    def set_all_min(self, enabled: bool) -> None:
        for row in self.plot_rows:
            self.update_min_checkbox(row, desired=enabled)

    def set_all_max(self, enabled: bool) -> None:
        for row in self.plot_rows:
            self.update_max_checkbox(row, desired=enabled)

    def set_all_std(self, enabled: bool) -> None:
        for row in self.plot_rows:
            self.update_std_checkbox(row, desired=enabled)

    def set_all_skew(self, enabled: bool) -> None:
        for row in self.plot_rows:
            self.update_skew_checkbox(row, desired=enabled)

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
        for index, row in enumerate(self.plot_rows, start=1):
            raw_ema = row.ema_var.get().strip()
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
            y_value = row.y_combo.get()
            if self.is_histogram_column(y_value):
                self.show_error(f"EMA is only supported for scalar plots (row {index}).")
                return
            ema_mapping[y_value] = alpha
        histogram_columns = [column for column in y_columns if self.is_histogram_column(column)]
        histogram_specs: dict[str, tuple[str, str | None]] = {}
        for column in histogram_columns:
            resolved = self.resolve_histogram_columns(column)
            if resolved is None:
                base = column.removesuffix("__histogram_edges")
                self.show_error(
                    f"Histogram column {column} needs {base}__histogram_freqs."
                )
                return
            histogram_specs[column] = resolved

        scalar_columns = [column for column in y_columns if column not in histogram_specs]
        scalar_column_set = set(scalar_columns)
        std_mapping: dict[str, str | None] = {column: None for column in scalar_columns}
        skew_mapping: dict[str, str | None] = {column: None for column in scalar_columns}
        min_mapping: dict[str, str | None] = {column: None for column in scalar_columns}
        max_mapping: dict[str, str | None] = {column: None for column in scalar_columns}
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
            if row.skew_var.get():
                skew_column = self.skew_column_for(y_value)
                if skew_column is None:
                    self.show_error(f"No skew column found for {y_value}.")
                    return
                skew_mapping[y_value] = skew_column
            if row.min_var.get():
                min_column = self.min_column_for(y_value)
                if min_column is None:
                    self.show_error(f"No min column found for {y_value}.")
                    return
                min_mapping[y_value] = min_column
            if row.max_var.get():
                max_column = self.max_column_for(y_value)
                if max_column is None:
                    self.show_error(f"No max column found for {y_value}.")
                    return
                max_mapping[y_value] = max_column

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
                        skew_mapping=skew_mapping,
                        min_mapping=min_mapping,
                        max_mapping=max_mapping,
                    )
                    for path, label in zip(enabled_paths, labels, strict=True)
                ]
                group_keys = self.group_keys_for_labels(enabled_paths, labels)
            except (ValueError, FileNotFoundError) as exc:
                self.show_error(str(exc))
                return

        histogram_data: dict[str, list[HistogramSeries]] = {}
        for column, (freqs_column, edges_column) in histogram_specs.items():
            try:
                histogram_data[column] = [
                    load_histogram_series(
                        path,
                        label,
                        x_column,
                        freqs_column,
                        edges_column,
                        delimiter,
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
                skew_mapping=skew_mapping,
                min_mapping=min_mapping,
                max_mapping=max_mapping,
                ema_mapping=ema_mapping or None,
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

    def build_figure(
        self,
        logs: Sequence[plot_logs.LogSeries],
        histogram_data: dict[str, list[HistogramSeries]],
        x_column: str,
        y_columns: Sequence[str],
        std_mapping: dict[str, str | None],
        skew_mapping: dict[str, str | None],
        min_mapping: dict[str, str | None],
        max_mapping: dict[str, str | None],
        ema_mapping: dict[str, float] | None,
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
        hist_ranges = {
            column: histogram_value_range(series_list)
            for column, series_list in histogram_data.items()
        }
        hist_axes: dict[str, list[plt.Axes]] = {}
        hist_meshes: dict[str, matplotlib.collections.QuadMesh] = {}
        first_scalar_axis: plt.Axes | None = None
        for axis, (column, series, _ratio) in zip(axes, axis_specs, strict=True):
            if series is None:
                ema_alpha = ema_mapping.get(column) if ema_mapping else None
                skew_axis: plt.Axes | None = None
                if skew_mapping.get(column) is not None:
                    skew_axis = axis.twinx()
                    skew_axis.set_ylabel("skew")
                    skew_axis.grid(False)
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
                    line = axis.plot(
                        log.x_values,
                        raw_values,
                        label=log.label,
                        color=color,
                    )[0]
                    line._plot_alpha_base = base_alpha
                    if ema_values is not None:
                        ema_line = axis.plot(
                            log.x_values,
                            ema_values,
                            label="_ema",
                            color=line.get_color(),
                            linestyle="--",
                        )[0]
                        ema_line._plot_alpha_base = base_alpha
                    std_column = std_mapping.get(column)
                    if std_column is not None:
                        std_values = log.y_std_values.get(column)
                        if std_values:
                            upper = [
                                value + std
                                for value, std in zip(
                                    raw_values,
                                    std_values,
                                    strict=True,
                                )
                            ]
                            lower = [
                                value - std
                                for value, std in zip(
                                    raw_values,
                                    std_values,
                                    strict=True,
                                )
                            ]
                            axis.fill_between(
                                log.x_values,
                                lower,
                                upper,
                                alpha=0.2 * base_alpha,
                                color=line.get_color(),
                            )
                    if skew_axis is not None:
                        skew_values = log.y_skew_values.get(column)
                        if skew_values:
                            if ema_alpha is not None:
                                skew_values = exponential_moving_average(skew_values, ema_alpha)
                            skew_line = skew_axis.plot(
                                log.x_values,
                                skew_values,
                                color=line.get_color(),
                                linestyle="-.",
                                label="_skew",
                            )[0]
                            skew_line._plot_alpha_base = base_alpha
                    min_column = min_mapping.get(column)
                    if min_column is not None:
                        min_values = log.y_min_values.get(column)
                        if min_values:
                            if ema_alpha is not None:
                                min_values = exponential_moving_average(min_values, ema_alpha)
                            min_line = axis.plot(
                                log.x_values,
                                min_values,
                                color=line.get_color(),
                                linestyle="--",
                                label="_min",
                            )[0]
                            min_line._plot_alpha_base = base_alpha
                    max_column = max_mapping.get(column)
                    if max_column is not None:
                        max_values = log.y_max_values.get(column)
                        if max_values:
                            if ema_alpha is not None:
                                max_values = exponential_moving_average(max_values, ema_alpha)
                            max_line = axis.plot(
                                log.x_values,
                                max_values,
                                color=line.get_color(),
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
            values = transpose_histogram(series.values)
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
            axis.set_ylabel(column)
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
            figure.colorbar(mesh, ax=axes_for_column, pad=0.01)
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
