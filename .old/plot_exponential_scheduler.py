from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swarmbots.learn.scheduling.exponential_scheduler import ExponentialScheduler
from swarmbots.learn.scheduling.schedulers import ScheduleUnit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Tk plot for ExponentialScheduler.")
    parser.add_argument("--duration", type=int, default=100)
    parser.add_argument("--start-value", type=float, default=0.0)
    parser.add_argument("--final-value", type=float, default=1.0)
    parser.add_argument("--base", type=float, default=math.e)
    parser.add_argument("--min-base", type=float, default=0.05)
    parser.add_argument("--max-base", type=float, default=50.0)
    return parser.parse_args()


def build_curve(
        duration: int,
        start_value: float,
        final_value: float,
        base: float,
) -> tuple[np.ndarray, np.ndarray]:
    scheduler = ExponentialScheduler(
        unit=ScheduleUnit.ITERATIONS,
        duration=duration,
        start_value=start_value,
        final_value=final_value,
        base=base,
    )
    steps = np.arange(duration + 1, dtype=float)
    values = np.array(
        [
            scheduler.schedule(
                progress=step / duration if duration > 0 else 1.0,
                old_value=float("nan"),
                state={},
                metrics={},
            )["new_value"]
            for step in steps
        ],
        dtype=float,
    )
    return steps, values


class ExponentialSchedulerPlotApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.root.title("Exponential Scheduler Plot")
        self.root.geometry("1100x720")

        self.duration_var = tk.StringVar(value=str(args.duration))
        self.start_value_var = tk.StringVar(value=f"{args.start_value:g}")
        self.final_value_var = tk.StringVar(value=f"{args.final_value:g}")
        self.base_var = tk.DoubleVar(value=args.base)
        self.base_text_var = tk.StringVar(value=f"{args.base:.3f}")
        self.min_base = args.min_base
        self.max_base = args.max_base

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.canvas: FigureCanvasTkAgg | None = None
        self.toolbar: NavigationToolbar2Tk | None = None

        self._build_layout()
        self.redraw_plot()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(self.root, padding=16)
        controls.grid(row=0, column=0, sticky="ns")

        plot_frame = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        ttk.Label(controls, text="Duration").grid(row=0, column=0, sticky="w")
        duration_entry = ttk.Entry(controls, textvariable=self.duration_var, width=12)
        duration_entry.grid(row=1, column=0, sticky="ew", pady=(0, 12))

        ttk.Label(controls, text="Start value").grid(row=2, column=0, sticky="w")
        start_entry = ttk.Entry(controls, textvariable=self.start_value_var, width=12)
        start_entry.grid(row=3, column=0, sticky="ew", pady=(0, 12))

        ttk.Label(controls, text="Final value").grid(row=4, column=0, sticky="w")
        final_entry = ttk.Entry(controls, textvariable=self.final_value_var, width=12)
        final_entry.grid(row=5, column=0, sticky="ew", pady=(0, 16))

        ttk.Label(controls, text="Base").grid(row=6, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.base_text_var).grid(row=7, column=0, sticky="w")

        base_scale = ttk.Scale(
            controls,
            from_=self.min_base,
            to=self.max_base,
            variable=self.base_var,
            orient="horizontal",
            command=self._on_base_change,
        )
        base_scale.grid(row=8, column=0, sticky="ew", pady=(4, 8))

        ttk.Label(
            controls,
            text=f"Slider range: {self.min_base:g} to {self.max_base:g}",
        ).grid(row=9, column=0, sticky="w")

        apply_button = ttk.Button(controls, text="Apply values", command=self._on_apply)
        apply_button.grid(row=10, column=0, sticky="ew", pady=(16, 8))

        reset_button = ttk.Button(controls, text="Reset base", command=self._reset_base)
        reset_button.grid(row=11, column=0, sticky="ew")

        info_label = ttk.Label(
            controls,
            text="Drag the base slider.\nPress Enter in an input field to replot.",
            justify="left",
        )
        info_label.grid(row=12, column=0, sticky="w", pady=(20, 0))

        for entry in (duration_entry, start_entry, final_entry):
            entry.bind("<Return>", self._on_apply)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=1, column=0, sticky="ew")

    def _on_base_change(self, _value: str) -> None:
        self.base_text_var.set(f"{self.base_var.get():.3f}")
        self.redraw_plot()

    def _on_apply(self, _event: object | None = None) -> None:
        try:
            self._validated_inputs()
        except ValueError as exc:
            messagebox.showerror("Invalid scheduler values", str(exc))
            return
        self.redraw_plot()

    def _reset_base(self) -> None:
        self.base_var.set(math.e)
        self.base_text_var.set(f"{self.base_var.get():.3f}")
        self.redraw_plot()

    def _validated_inputs(self) -> tuple[int, float, float, float]:
        try:
            duration = int(self.duration_var.get())
        except ValueError as exc:
            raise ValueError(f"duration must be an integer, got {self.duration_var.get()!r}") from exc
        if duration < 0:
            raise ValueError(f"duration must be >= 0, got {duration}")

        try:
            start_value = float(self.start_value_var.get())
        except ValueError as exc:
            raise ValueError(
                f"start value must be a number, got {self.start_value_var.get()!r}"
            ) from exc

        try:
            final_value = float(self.final_value_var.get())
        except ValueError as exc:
            raise ValueError(
                f"final value must be a number, got {self.final_value_var.get()!r}"
            ) from exc

        base = float(self.base_var.get())
        if not math.isfinite(base) or base <= 0:
            raise ValueError(f"base must be finite and > 0, got {base}")
        return duration, start_value, final_value, base

    def redraw_plot(self) -> None:
        try:
            duration, start_value, final_value, base = self._validated_inputs()
        except ValueError:
            return

        steps, values = build_curve(
            duration=duration,
            start_value=start_value,
            final_value=final_value,
            base=base,
        )

        self.axis.clear()
        self.axis.plot(steps, values, color="#1f77b4", linewidth=2, label="exponential")
        self.axis.scatter(
            [0, max(duration, 0)],
            [start_value, final_value],
            color="#d62728",
            zorder=3,
            label="endpoints",
        )
        self.axis.set_title("ExponentialScheduler")
        self.axis.set_xlabel("iteration")
        self.axis.set_ylabel("value")
        self.axis.grid(alpha=0.3)
        self.axis.legend(loc="best")
        self.axis.text(
            0.02,
            0.98,
            f"base={base:.3f}\nduration={duration}",
            transform=self.axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

        x_max = max(duration, 1)
        self.axis.set_xlim(0, x_max)

        value_min = min(float(np.min(values)), start_value, final_value)
        value_max = max(float(np.max(values)), start_value, final_value)
        padding = max((value_max - value_min) * 0.08, 0.1)
        self.axis.set_ylim(value_min - padding, value_max + padding)

        if self.canvas is not None:
            self.canvas.draw_idle()


def main() -> int:
    args = parse_args()
    if args.min_base <= 0:
        raise ValueError(f"min-base must be > 0, got {args.min_base}")
    if args.max_base <= args.min_base:
        raise ValueError(
            f"max-base must be greater than min-base, got {args.max_base} <= {args.min_base}"
        )
    if not args.min_base <= args.base <= args.max_base:
        raise ValueError(
            f"base must be inside slider range [{args.min_base}, {args.max_base}], got {args.base}"
        )

    root = tk.Tk()
    ExponentialSchedulerPlotApp(root=root, args=args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
