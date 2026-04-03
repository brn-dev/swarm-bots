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

from swarmbots.learn.scheduling.cosine_scheduler import CosineScheduler
from swarmbots.learn.scheduling.schedulers import ScheduleUnit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Tk plot for CosineAnnealingScheduler.")
    parser.add_argument("--duration", type=int, default=100)
    parser.add_argument("--start-value", type=float, default=1.0)
    parser.add_argument("--final-value", type=float, default=0.0)
    parser.add_argument("--bias", type=float, default=1.0)
    parser.add_argument("--sharpness", type=float, default=1.0)
    parser.add_argument("--min-shape", type=float, default=0.1)
    parser.add_argument("--max-shape", type=float, default=5.0)
    return parser.parse_args()


def build_curve(
        duration: int,
        start_value: float,
        final_value: float,
        bias: float,
        sharpness: float,
) -> tuple[np.ndarray, np.ndarray]:
    scheduler = CosineScheduler(
        unit=ScheduleUnit.ITERATIONS,
        duration=duration,
        start_value=start_value,
        final_value=final_value,
        bias=bias,
        sharpness=sharpness,
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


class CosineAnnealingSchedulerPlotApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.root.title("Cosine Annealing Scheduler Plot")
        self.root.geometry("1100x720")

        self.duration_var = tk.StringVar(value=str(args.duration))
        self.start_value_var = tk.StringVar(value=f"{args.start_value:g}")
        self.final_value_var = tk.StringVar(value=f"{args.final_value:g}")
        self.bias_var = tk.DoubleVar(value=args.bias)
        self.bias_text_var = tk.StringVar(value=f"{args.bias:.3f}")
        self.sharpness_var = tk.DoubleVar(value=args.sharpness)
        self.sharpness_text_var = tk.StringVar(value=f"{args.sharpness:.3f}")
        self.min_shape = args.min_shape
        self.max_shape = args.max_shape

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

        ttk.Label(controls, text="Bias").grid(row=6, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.bias_text_var).grid(row=7, column=0, sticky="w")
        bias_scale = ttk.Scale(
            controls,
            from_=self.min_shape,
            to=self.max_shape,
            variable=self.bias_var,
            orient="horizontal",
            command=self._on_bias_change,
        )
        bias_scale.grid(row=8, column=0, sticky="ew", pady=(4, 8))

        ttk.Label(controls, text="Sharpness").grid(row=9, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.sharpness_text_var).grid(row=10, column=0, sticky="w")
        sharpness_scale = ttk.Scale(
            controls,
            from_=self.min_shape,
            to=self.max_shape,
            variable=self.sharpness_var,
            orient="horizontal",
            command=self._on_sharpness_change,
        )
        sharpness_scale.grid(row=11, column=0, sticky="ew", pady=(4, 8))

        ttk.Label(
            controls,
            text=f"Slider range: {self.min_shape:g} to {self.max_shape:g}",
        ).grid(row=12, column=0, sticky="w")

        apply_button = ttk.Button(controls, text="Apply values", command=self._on_apply)
        apply_button.grid(row=13, column=0, sticky="ew", pady=(16, 8))

        reset_button = ttk.Button(controls, text="Reset shapes", command=self._reset_shapes)
        reset_button.grid(row=14, column=0, sticky="ew")

        info_label = ttk.Label(
            controls,
            text="Bias warps time.\nSharpness changes curve intensity.",
            justify="left",
        )
        info_label.grid(row=15, column=0, sticky="w", pady=(20, 0))

        for entry in (duration_entry, start_entry, final_entry):
            entry.bind("<Return>", self._on_apply)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=1, column=0, sticky="ew")

    def _on_apply(self, _event: object | None = None) -> None:
        try:
            self._validated_inputs()
        except ValueError as exc:
            messagebox.showerror("Invalid scheduler values", str(exc))
            return
        self.redraw_plot()

    def _on_bias_change(self, _value: str) -> None:
        self.bias_text_var.set(f"{self.bias_var.get():.3f}")
        self.redraw_plot()

    def _on_sharpness_change(self, _value: str) -> None:
        self.sharpness_text_var.set(f"{self.sharpness_var.get():.3f}")
        self.redraw_plot()

    def _reset_shapes(self) -> None:
        self.bias_var.set(1.0)
        self.sharpness_var.set(1.0)
        self.bias_text_var.set(f"{self.bias_var.get():.3f}")
        self.sharpness_text_var.set(f"{self.sharpness_var.get():.3f}")
        self.redraw_plot()

    def _validated_inputs(self) -> tuple[int, float, float, float, float]:
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

        bias = float(self.bias_var.get())
        if not math.isfinite(bias) or bias <= 0:
            raise ValueError(f"bias must be finite and > 0, got {bias}")

        sharpness = float(self.sharpness_var.get())
        if not math.isfinite(sharpness) or sharpness <= 0:
            raise ValueError(f"sharpness must be finite and > 0, got {sharpness}")

        return duration, start_value, final_value, bias, sharpness

    def redraw_plot(self) -> None:
        try:
            duration, start_value, final_value, bias, sharpness = self._validated_inputs()
        except ValueError:
            return

        steps, values = build_curve(
            duration=duration,
            start_value=start_value,
            final_value=final_value,
            bias=bias,
            sharpness=sharpness,
        )

        self.axis.clear()
        self.axis.plot(steps, values, color="#1f77b4", linewidth=2, label="cosine annealing")
        self.axis.scatter(
            [0, max(duration, 0)],
            [start_value, final_value],
            color="#d62728",
            zorder=3,
            label="endpoints",
        )
        self.axis.set_title("CosineAnnealingScheduler")
        self.axis.set_xlabel("iteration")
        self.axis.set_ylabel("value")
        self.axis.grid(alpha=0.3)
        self.axis.legend(loc="best")
        self.axis.text(
            0.02,
            0.98,
            f"duration={duration}\nbias={bias:.3f}\nsharpness={sharpness:.3f}",
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
    if args.min_shape <= 0:
        raise ValueError(f"min-shape must be > 0, got {args.min_shape}")
    if args.max_shape <= args.min_shape:
        raise ValueError(
            f"max-shape must be greater than min-shape, got {args.max_shape} <= {args.min_shape}"
        )
    if not args.min_shape <= args.bias <= args.max_shape:
        raise ValueError(
            f"bias must be inside slider range [{args.min_shape}, {args.max_shape}], got {args.bias}"
        )
    if not args.min_shape <= args.sharpness <= args.max_shape:
        raise ValueError(
            f"sharpness must be inside slider range [{args.min_shape}, {args.max_shape}], got {args.sharpness}"
        )
    root = tk.Tk()
    CosineAnnealingSchedulerPlotApp(root=root, args=args)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
