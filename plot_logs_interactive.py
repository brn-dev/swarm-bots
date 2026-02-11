from __future__ import annotations

import csv
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

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind_all("<KeyPress>", self.on_global_keypress, add=True)
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

        action_frame = ttk.Frame(controls_frame)
        action_frame.grid(row=3, column=0, sticky="ew")
        plot_button = ttk.Button(action_frame, text="Plot", command=self.plot)
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

    def update_std_checkbox(self, row: PlotRow) -> None:
        y_value = row.y_combo.get()
        candidate = self.std_column_for(y_value)
        if candidate is None:
            row.std_var.set(False)
            row.std_check.state(["disabled"])
        else:
            row.std_check.state(["!disabled"])
            row.std_var.set(True)

    def std_column_for(self, y_value: str) -> str | None:
        if not y_value or not y_value.endswith("__mean"):
            return None
        candidate = f"{y_value.removesuffix('__mean')}__std"
        if candidate in self.available_columns:
            return candidate
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
        std_mapping: dict[str, str | None] = {}
        for row in self.plot_rows:
            y_value = row.y_combo.get()
            if not y_value:
                continue
            if row.std_var.get():
                std_column = self.std_column_for(y_value)
                if std_column is None:
                    self.show_error(f"No std column found for {y_value}.")
                    return
                std_mapping[y_value] = std_column
            else:
                std_mapping[y_value] = None

        labels = plot_logs.build_labels(self.paths, None)
        try:
            logs = [
                plot_logs.load_log(
                    path,
                    label,
                    x_column,
                    y_columns,
                    std_mapping,
                    delimiter,
                )
                for path, label in zip(self.paths, labels, strict=True)
            ]
        except (ValueError, FileNotFoundError) as exc:
            self.show_error(str(exc))
            return

        title = self.title_var.get().strip() or None
        figure = plot_logs.plot_logs(
            logs,
            x_column,
            y_columns,
            std_mapping,
            ratios=ratios or None,
            title=title,
        )
        self.apply_line_opacity(figure, self.line_alpha_var.get())
        self.apply_line_width(figure, self.line_width_var.get())
        self.render_figure(figure)
        self.set_status("Plot updated.")

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
