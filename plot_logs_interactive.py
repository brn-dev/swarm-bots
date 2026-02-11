from __future__ import annotations

import ast
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import plot_logs


@dataclass(slots=True)
class PlotRow:
    index_label: ttk.Label
    y_var: tk.StringVar
    y_combo: ttk.Combobox
    height_var: tk.StringVar
    height_entry: ttk.Entry
    std_var: tk.BooleanVar
    std_check: ttk.Checkbutton
    remove_button: ttk.Button


@dataclass(frozen=True, slots=True)
class PresetEntry:
    y_column: str
    height: float
    std: bool


@dataclass(frozen=True, slots=True)
class PlotPreset:
    name: str
    entries: tuple[PresetEntry, ...]
    x_column: str | None = None


@dataclass(slots=True)
class HistogramSeries:
    label: str
    x_values: list[float]
    bin_edges: list[float] | None
    values: list[list[float]]
    edges_status: str


PLOT_PRESETS: tuple[PlotPreset, ...] = (
    PlotPreset(
        name="Episode Stats",
        entries=(
            PresetEntry("ep_rew_ema", 3.0, False),
            PresetEntry("ep_rew__mean", 1.0, True),
            PresetEntry("ep_len__mean", 1.0, True),
        ),
    ),
    PlotPreset(
        name="Actions",
        entries=(
            PresetEntry("ep_rew_ema", 5.0, False),
            PresetEntry("act0__histogram_freqs", 3.0, False),
            PresetEntry("act1__histogram_freqs", 1.0, False),
            PresetEntry("act0__mean", 1.0, True),
            PresetEntry("std0__mean", 1.0, True),
            PresetEntry("act1__mean", 1.0, True),
        ),
    ),
    PlotPreset(
        name="PPO Metrics",
        entries=(
            PresetEntry("ep_rew_ema", 3.0, False),
            PresetEntry("approx_kl__mean", 1.0, True),
            PresetEntry("clip_frac__mean", 1.0, True),
            PresetEntry("ratio__std", 1.0, False),
            PresetEntry("expl_var", 1.0, False),
            PresetEntry("grad_norm__mean", 1.0, True),
            PresetEntry("grad_clip_frac", 1.0, False),
            PresetEntry("learning_rate", 1.0, False),
        ),
    ),
    PlotPreset(
        name="PPO Losses",
        entries=(
            PresetEntry("ep_rew_ema", 3.0, False),
            PresetEntry("act_loss__mean", 1.0, True),
            PresetEntry("val_loss__mean", 1.0, True),
            PresetEntry("ent_loss__mean", 1.0, True),
            PresetEntry("grad_norm__mean", 1.0, True),
            PresetEntry("grad_clip_frac", 1.0, False),
        ),
    ),
    PlotPreset(
        name="PPO Full",
        entries=(
            PresetEntry("ep_rew_ema", 2.0, False),
            PresetEntry("approx_kl__mean", 1.0, True),
            PresetEntry("clip_frac__mean", 1.0, True),
            PresetEntry("ratio__std", 1.0, False),
            PresetEntry("expl_var", 1.0, False),
            PresetEntry("val_loss__mean", 1.0, True),
            PresetEntry("act_loss__mean", 1.0, True),
            PresetEntry("ent_loss__mean", 1.0, True),
            PresetEntry("grad_norm__mean", 1.0, True),
            PresetEntry("grad_clip_frac", 1.0, False),
            PresetEntry("learning_rate", 1.0, False),
        ),
    ),
    PlotPreset(
        name="NOP Losses",
        entries=(
            PresetEntry("ep_rew_ema", 5.0, False),
            PresetEntry("scalar_loss__mean", 1.0, True),
            PresetEntry("angle_loss__mean", 1.0, True),
            PresetEntry("rot6d_loss__mean", 1.0, True),
            PresetEntry("binary_loss__mean", 1.0, True),
        ),
    ),
    PlotPreset(
        name="Performance",
        entries=(
            PresetEntry("fps", 1.0, False),
            PresetEntry("updates", 1.0, False),
            PresetEntry("policy_forward_time__mean", 1.0, True),
            PresetEntry("env_step_time__mean", 1.0, True),
            PresetEntry("rollout_time", 1.0, False),
            PresetEntry("sampling_time__mean", 1.0, True),
            PresetEntry("update_time__mean", 1.0, True),
            PresetEntry("train_time", 1.0, False),
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


def load_histogram_series(
    path: Path,
    label: str,
    x_column: str,
    freqs_column: str,
    edges_column: str | None,
    delimiter: str,
) -> HistogramSeries:
    x_values: list[float] = []
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
            x_value = plot_logs.parse_scalar(row.get(x_column), x_column, path, row_index)
            if math.isnan(x_value):
                continue
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
        self.available_columns: list[str] = []
        self.plot_rows: list[PlotRow] = []
        self.typeahead_buffers: dict[tk.Widget, str] = {}
        self.typeahead_after_ids: dict[tk.Widget, str] = {}
        self.combo_values_getters: dict[ttk.Combobox, Callable[[], Sequence[str]]] = {}
        self.listbox_owner: dict[str, ttk.Combobox] = {}
        self.figure: plt.Figure | None = None
        self.canvas: FigureCanvasTkAgg | None = None
        self.toolbar: NavigationToolbar2Tk | None = None

        self.delimiter_var = tk.StringVar(value=";")
        self.title_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Select CSV files to begin.")
        self.line_alpha_var = tk.DoubleVar(value=1.0)
        self.line_width_var = tk.DoubleVar(value=0.75)
        self.dark_mode_var = tk.BooleanVar(value=True)
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
        controls_frame = ttk.Frame(self.root, padding=8)
        controls_frame.grid(row=0, column=0, sticky="ns")
        controls_frame.columnconfigure(0, weight=1)

        plot_frame = ttk.Frame(self.root, padding=8)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.rowconfigure(1, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        files_frame = ttk.LabelFrame(controls_frame, text="CSV Files")
        files_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        files_frame.columnconfigure(0, weight=1)

        self.files_listbox = tk.Listbox(files_frame, height=6, selectmode="extended")
        self.files_listbox.grid(row=0, column=0, sticky="ew")
        files_scrollbar = ttk.Scrollbar(files_frame, orient="vertical", command=self.files_listbox.yview)
        files_scrollbar.grid(row=0, column=1, sticky="ns")
        self.files_listbox.configure(yscrollcommand=files_scrollbar.set)

        files_buttons = ttk.Frame(files_frame)
        files_buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        add_button = ttk.Button(files_buttons, text="Add CSV Files", command=self.add_files)
        add_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        remove_button = ttk.Button(files_buttons, text="Remove Selected", command=self.remove_selected_files)
        remove_button.grid(row=0, column=1, sticky="ew")
        refresh_button = ttk.Button(files_buttons, text="Refresh Columns", command=self.refresh_columns)
        refresh_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        settings_frame = ttk.LabelFrame(controls_frame, text="Settings")
        settings_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        settings_frame.columnconfigure(1, weight=1)

        ttk.Label(settings_frame, text="Delimiter").grid(row=0, column=0, sticky="w")
        delimiter_entry = ttk.Entry(settings_frame, textvariable=self.delimiter_var, width=4)
        delimiter_entry.grid(row=0, column=1, sticky="w")

        ttk.Label(settings_frame, text="X Column").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.x_combo = ttk.Combobox(settings_frame, state="disabled")
        self.x_combo.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        self.bind_typeahead(self.x_combo, lambda: self.available_columns)

        ttk.Label(settings_frame, text="Title").grid(row=2, column=0, sticky="w", pady=(6, 0))
        title_entry = ttk.Entry(settings_frame, textvariable=self.title_var)
        title_entry.grid(row=2, column=1, sticky="ew", pady=(6, 0))

        ttk.Label(settings_frame, text="Line Opacity").grid(row=3, column=0, sticky="w", pady=(6, 0))
        opacity_scale = ttk.Scale(
            settings_frame,
            from_=0.0,
            to=1.0,
            variable=self.line_alpha_var,
            command=self.on_opacity_change,
        )
        opacity_scale.grid(row=3, column=1, sticky="ew", pady=(6, 0))

        ttk.Label(settings_frame, text="Line Width").grid(row=4, column=0, sticky="w", pady=(6, 0))
        width_scale = ttk.Scale(
            settings_frame,
            from_=0.1,
            to=1.0,
            variable=self.line_width_var,
            command=self.on_line_width_change,
        )
        width_scale.grid(row=4, column=1, sticky="ew", pady=(6, 0))

        dark_mode_check = ttk.Checkbutton(
            settings_frame,
            text="Dark Mode",
            variable=self.dark_mode_var,
            command=self.on_theme_toggle,
        )
        dark_mode_check.grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))

        plots_frame = ttk.LabelFrame(controls_frame, text="Plots")
        plots_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        plots_frame.columnconfigure(0, weight=1)

        self.plots_container = ttk.Frame(plots_frame)
        self.plots_container.grid(row=0, column=0, sticky="ew", pady=(4, 0))
        self.plots_container.columnconfigure(1, weight=1)
        self.plots_container.columnconfigure(2, weight=0)
        self.plots_container.columnconfigure(3, weight=0)
        self.plots_container.columnconfigure(4, weight=0)

        ttk.Label(self.plots_container, text="#").grid(row=0, column=0, sticky="w")
        ttk.Label(self.plots_container, text="Y Column").grid(row=0, column=1, sticky="w", padx=(10, 4))
        ttk.Label(self.plots_container, text="Height").grid(row=0, column=2, sticky="w", padx=(6, 4))
        ttk.Label(self.plots_container, text="STD").grid(row=0, column=3, sticky="w", padx=(6, 4))
        ttk.Label(self.plots_container, text="").grid(row=0, column=4, sticky="w", padx=(6, 4))

        plots_buttons = ttk.Frame(plots_frame)
        plots_buttons.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        add_plot_button = ttk.Button(plots_buttons, text="Add Plot", command=self.add_plot_row)
        add_plot_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        clear_plot_button = ttk.Button(plots_buttons, text="Clear Plots", command=self.clear_plot_rows)
        clear_plot_button.grid(row=0, column=1, sticky="ew")

        if PLOT_PRESETS:
            presets_label = ttk.Label(plots_frame, text="Presets")
            presets_label.grid(row=2, column=0, sticky="w", pady=(8, 0))
            presets_frame = ttk.Frame(plots_frame)
            presets_frame.grid(row=3, column=0, sticky="ew", pady=(4, 0))
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

    def add_files(self) -> None:
        filenames = filedialog.askopenfilenames(
            title="Select log CSV files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not filenames:
            return
        for filename in filenames:
            path = Path(filename).resolve()
            if path not in self.paths:
                self.paths.append(path)
        self.refresh_file_list()
        self.refresh_columns()

    def remove_selected_files(self) -> None:
        selected_indices = list(self.files_listbox.curselection())
        if not selected_indices:
            return
        for index in sorted(selected_indices, reverse=True):
            del self.paths[index]
        self.refresh_file_list()
        self.refresh_columns()

    def refresh_file_list(self) -> None:
        self.files_listbox.delete(0, tk.END)
        for path in self.paths:
            self.files_listbox.insert(tk.END, self.display_path(path))

    def display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    def refresh_columns(self) -> None:
        if not self.paths:
            self.available_columns = []
            self.update_column_options()
            self.set_status("Select CSV files to load columns.")
            return
        delimiter = self.delimiter_var.get()
        if len(delimiter) != 1:
            self.show_error("Delimiter must be a single character.")
            return
        try:
            columns, mismatch = resolve_common_columns(self.paths, delimiter)
        except ValueError as exc:
            self.show_error(str(exc))
            return
        if not columns:
            self.available_columns = []
            self.update_column_options()
            self.show_error("No common columns across selected files.")
            return
        self.available_columns = columns
        self.update_column_options()
        if mismatch:
            self.set_status(
                f"Column mismatch across files. Using {len(columns)} shared columns."
            )
        else:
            self.set_status(f"Loaded {len(columns)} columns.")

    def update_column_options(self) -> None:
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
            self.update_row_options(row, columns, prefer_default=index == 0)

    def update_row_options(self, row: PlotRow, columns: Sequence[str], prefer_default: bool) -> None:
        if columns:
            row.y_combo.configure(values=columns, state="readonly")
            if row.y_combo.get() not in columns:
                if prefer_default and "ep_rew_ema" in columns:
                    row.y_combo.set("ep_rew_ema")
                else:
                    row.y_combo.set(columns[0])
            self.update_std_checkbox(row)
        else:
            row.y_combo.configure(values=[], state="disabled")
            row.y_combo.set("")
            row.std_var.set(False)
            row.std_check.state(["disabled"])

    def add_plot_row(self) -> None:
        row_index = len(self.plot_rows) + 1
        index_label = ttk.Label(self.plots_container, text=str(row_index))
        index_label.grid(row=row_index, column=0, sticky="w", pady=2)

        y_var = tk.StringVar()
        y_combo = ttk.Combobox(self.plots_container, textvariable=y_var, state="disabled")
        y_combo.grid(row=row_index, column=1, sticky="ew", padx=(10, 4), pady=2)
        self.bind_typeahead(y_combo, lambda: self.available_columns)

        height_var = tk.StringVar(value="1")
        height_entry = ttk.Entry(self.plots_container, textvariable=height_var, width=5)
        height_entry.grid(row=row_index, column=2, sticky="w", padx=(6, 4), pady=2)

        std_var = tk.BooleanVar(value=False)
        std_check = ttk.Checkbutton(self.plots_container, text="Use", variable=std_var)
        std_check.grid(row=row_index, column=3, sticky="w", padx=(6, 4), pady=2)

        remove_button = ttk.Button(self.plots_container, text="Remove")
        remove_button.grid(row=row_index, column=4, sticky="e", pady=2)

        row = PlotRow(
            index_label=index_label,
            y_var=y_var,
            y_combo=y_combo,
            height_var=height_var,
            height_entry=height_entry,
            std_var=std_var,
            std_check=std_check,
            remove_button=remove_button,
        )
        remove_button.configure(command=lambda target=row: self.remove_plot_row(target))
        y_combo.bind("<<ComboboxSelected>>", lambda _event, target=row: self.update_std_checkbox(target))
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

    def std_column_for(self, y_value: str) -> str | None:
        if not y_value or not y_value.endswith("__mean"):
            return None
        candidate = f"{y_value.removesuffix('__mean')}__std"
        if candidate in self.available_columns:
            return candidate
        return None

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
        for widget in (row.index_label, row.y_combo, row.height_entry, row.std_check, row.remove_button):
            widget.destroy()
        if row in self.plot_rows:
            self.plot_rows.remove(row)
        self.refresh_row_labels()

    def clear_plot_rows(self) -> None:
        for row in self.plot_rows:
            for widget in (row.index_label, row.y_combo, row.height_entry, row.std_check, row.remove_button):
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
        for row, entry in zip(self.plot_rows, preset.entries, strict=True):
            row.y_combo.set(entry.y_column)
            row.height_var.set(self.format_ratio(entry.height))
            if entry.std and self.std_column_for(entry.y_column) is None:
                missing_std.append(entry.y_column)
            self.update_std_checkbox(row, desired=entry.std)
        if missing_std:
            self.set_status(
                "Preset "
                f"{preset.name!r} loaded. Missing std columns for: "
                f"{', '.join(missing_std)}."
            )
        else:
            self.set_status(f"Preset {preset.name!r} loaded.")

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
            row.std_check.grid_configure(row=index, column=3)
            row.remove_button.grid_configure(row=index, column=4)

    def plot(self) -> None:
        if not self.paths:
            self.show_error("No CSV files selected.")
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
        for row in self.plot_rows:
            y_value = row.y_combo.get()
            if not y_value or y_value not in scalar_column_set:
                continue
            if row.std_var.get():
                std_column = self.std_column_for(y_value)
                if std_column is None:
                    self.show_error(f"No std column found for {y_value}.")
                    return
                std_mapping[y_value] = std_column

        labels = plot_logs.build_labels(self.paths, None)
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
                    )
                    for path, label in zip(self.paths, labels, strict=True)
                ]
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
                    for path, label in zip(self.paths, labels, strict=True)
                ]
            except (ValueError, FileNotFoundError) as exc:
                self.show_error(str(exc))
                return

        title = self.title_var.get().strip() or None
        figure = self.build_figure(
            logs=logs,
            histogram_data=histogram_data,
            x_column=x_column,
            y_columns=y_columns,
            std_mapping=std_mapping,
            ratios=ratios or None,
            title=title,
        )
        self.apply_line_opacity(figure, self.line_alpha_var.get())
        self.apply_line_width(figure, self.line_width_var.get())
        self.render_figure(figure)
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
        ratios: Sequence[float] | None,
        title: str | None,
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
        hist_ranges = {
            column: histogram_value_range(series_list)
            for column, series_list in histogram_data.items()
        }
        hist_axes: dict[str, list[plt.Axes]] = {}
        hist_meshes: dict[str, matplotlib.collections.QuadMesh] = {}
        first_scalar_axis: plt.Axes | None = None
        for axis, (column, series, _ratio) in zip(axes, axis_specs, strict=True):
            if series is None:
                for log in logs:
                    axis.plot(log.x_values, log.y_values[column], label=log.label)
                    std_column = std_mapping.get(column)
                    if std_column is not None:
                        std_values = log.y_std_values.get(column)
                        if std_values:
                            upper = [
                                value + std
                                for value, std in zip(
                                    log.y_values[column],
                                    std_values,
                                    strict=True,
                                )
                            ]
                            lower = [
                                value - std
                                for value, std in zip(
                                    log.y_values[column],
                                    std_values,
                                    strict=True,
                                )
                            ]
                            axis.fill_between(log.x_values, lower, upper, alpha=0.2)
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
                line.set_alpha(clamped)

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
