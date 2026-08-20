from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from plot_logs.experiment_results import (
    DEFAULT_RUN_LENGTH_LIMIT,
    EP_REW_EMA_COLUMN,
    EP_SUCCESS_RATE_EMA_COLUMN,
    ExperimentGroup,
    ExperimentRunLog,
    run_metric_series,
)

DEFAULT_TAIL_POINTS = 10
DEFAULT_SUCCESS_RATE_THRESHOLDS = (25.0, 50.0, 75.0, 90.0, 95.0, 98.0)


def run_tail_mean(
    run: ExperimentRunLog,
    column: str,
    *,
    tail_points: int,
    run_length_limit: int = DEFAULT_RUN_LENGTH_LIMIT,
    cut_at_limit: bool = False,
) -> float | None:
    _, values = run_metric_series(
        run,
        column,
        run_length_limit=run_length_limit,
        cut_at_limit=cut_at_limit,
    )
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None
    return float(np.mean(finite_values[-tail_points:]))


def summarize_metric(
    runs: Sequence[ExperimentRunLog],
    column: str,
    *,
    tail_points: int,
    run_length_limit: int = DEFAULT_RUN_LENGTH_LIMIT,
    cut_at_limit: bool = False,
) -> dict[str, int | float | None]:
    run_values: list[float] = []
    for run in runs:
        value = run_tail_mean(
            run,
            column,
            tail_points=tail_points,
            run_length_limit=run_length_limit,
            cut_at_limit=cut_at_limit,
        )
        if value is not None:
            run_values.append(value)
    if not run_values:
        return {"number_of_runs": 0, "mean": None, "std": None}

    values = np.asarray(run_values, dtype=float)
    return {
        "number_of_runs": len(run_values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def summarize_threshold_timesteps(
    runs: Sequence[ExperimentRunLog],
    column: str,
    *,
    thresholds: Sequence[float],
    run_length_limit: int = DEFAULT_RUN_LENGTH_LIMIT,
    cut_at_limit: bool = False,
) -> dict[str, dict[str, int | float | None]]:
    timesteps_by_threshold = {threshold: [] for threshold in thresholds}
    for run in runs:
        x_values, values = run_metric_series(
            run,
            column,
            run_length_limit=run_length_limit,
            cut_at_limit=cut_at_limit,
        )
        finite_mask = np.isfinite(x_values) & np.isfinite(values)
        for threshold in thresholds:
            threshold_indices = np.flatnonzero(finite_mask & (values >= threshold))
            if threshold_indices.size > 0:
                timesteps_by_threshold[threshold].append(
                    float(x_values[int(threshold_indices[0])])
                )

    summaries: dict[str, dict[str, int | float | None]] = {}
    for threshold in thresholds:
        timesteps = timesteps_by_threshold[threshold]
        if not timesteps:
            summaries[f"{threshold:g}"] = {
                "number_of_runs": 0,
                "mean": None,
                "std": None,
            }
            continue

        timestep_values = np.asarray(timesteps, dtype=float)
        summaries[f"{threshold:g}"] = {
            "number_of_runs": len(timesteps),
            "mean": float(np.mean(timestep_values)),
            "std": float(np.std(timestep_values)),
        }
    return summaries


def summarize_experiment_groups(
    groups: Sequence[ExperimentGroup],
    *,
    tail_points: int = DEFAULT_TAIL_POINTS,
    success_rate_thresholds: Sequence[float] = DEFAULT_SUCCESS_RATE_THRESHOLDS,
    run_length_limit: int = DEFAULT_RUN_LENGTH_LIMIT,
    run_length_limit_overrides: Mapping[str, int] | None = None,
    cut_at_limit: bool = False,
) -> dict[str, Any]:
    if tail_points < 1:
        raise ValueError("tail_points must be at least 1")
    if not all(np.isfinite(threshold) for threshold in success_rate_thresholds):
        raise ValueError("success_rate_thresholds must be finite")

    variants: dict[str, Any] = {}
    for group in groups:
        group_run_length_limit = (
            run_length_limit
            if run_length_limit_overrides is None
            else run_length_limit_overrides.get(group.name, run_length_limit)
        )
        variants[group.name] = {
            "display_name": group.display_name,
            "number_of_runs": len(group.runs),
            EP_REW_EMA_COLUMN: summarize_metric(
                group.runs,
                EP_REW_EMA_COLUMN,
                tail_points=tail_points,
                run_length_limit=group_run_length_limit,
                cut_at_limit=cut_at_limit,
            ),
            EP_SUCCESS_RATE_EMA_COLUMN: summarize_metric(
                group.runs,
                EP_SUCCESS_RATE_EMA_COLUMN,
                tail_points=tail_points,
                run_length_limit=group_run_length_limit,
                cut_at_limit=cut_at_limit,
            ),
            "success_rate_threshold_timesteps": summarize_threshold_timesteps(
                group.runs,
                EP_SUCCESS_RATE_EMA_COLUMN,
                thresholds=success_rate_thresholds,
                run_length_limit=group_run_length_limit,
                cut_at_limit=cut_at_limit,
            ),
        }

    return {
        "tail_points_per_run": tail_points,
        "success_rate_thresholds": list(success_rate_thresholds),
        "run_length_limit": run_length_limit if cut_at_limit else None,
        "run_length_limit_overrides": (
            dict(run_length_limit_overrides or {}) if cut_at_limit else {}
        ),
        "standard_deviation": "population",
        "variants": variants,
    }


def write_experiment_summary(summary: dict[str, Any], output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path
