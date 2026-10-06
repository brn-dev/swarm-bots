"""Scriptable training-log plots. Install with ``pip install 'swarmbots[plot]'``."""

from swarmbots.plotting.logs import (
    GroupStatistics,
    RunLog,
    group_statistics,
    load_groups,
    load_log,
    log_columns,
)

from swarmbots.plotting.plot_logs import ExperimentPlotResult, plot_experiment_results, plot_groups, plot_logs

__all__ = [
    "ExperimentPlotResult",
    "GroupStatistics",
    "RunLog",
    "group_statistics",
    "load_groups",
    "load_log",
    "log_columns",
    "plot_experiment_results",
    "plot_groups",
    "plot_logs",
]
