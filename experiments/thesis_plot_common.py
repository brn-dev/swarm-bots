from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from plot_logs.experiment_results import ExperimentPlotResult, plot_experiment_results

THESIS_GROUP_ORDER = (
    "mappo",
    "mat_qcx",
    "mat_ind",
    "mat_orig",
    "tmasac_baseline",
    "slstm_two_small_actor_state_critic",
)
THESIS_DISPLAY_NAMES = {
    "mappo": "MAPPO",
    "mat_qcx": "MAT-QCX + NOP",
    "mat_ind": "MAT-Independent + NOP",
    "mat_orig": "MAT original actor decoder + NOP",
    "tmasac_baseline": "TMASAC",
    "slstm_two_small_actor_state_critic": "TMASAC + sLSTM",
}
THESIS_RUN_LENGTH = 100_000_000


def plot_thesis_experiment_results(
    *,
    experiment_run_dir: Path,
    output_dir: Path,
    extra_group_sources: Mapping[str, Sequence[Path]],
) -> ExperimentPlotResult:
    return plot_experiment_results(
        experiment_run_dir,
        output_dir,
        group_order=THESIS_GROUP_ORDER,
        display_name_overrides=THESIS_DISPLAY_NAMES,
        extra_group_sources=extra_group_sources,
        run_length_limit=THESIS_RUN_LENGTH,
        cut_at_limit=True,
    )
