from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from plot_logs.experiment_results import (
    ExperimentPlotResult,
    ExperimentPlotSelection,
    plot_experiment_results,
)

THESIS_MAIN_GROUP_ORDER = (
    "mappo",
    "mat_qcx",
    "mat_ind",
    "mat_orig",
    "tmasac_baseline",
    "slstm_two_small_actor_state_critic",
)
THESIS_GROUP_ORDER = (
    *THESIS_MAIN_GROUP_ORDER,
    "mat_qcx_gsde",
    "mat_qcx_no_nop",
    "tmasac_baseline_predicted_std",
    "tmasac_baseline_no_nop",
    "slstm_two_small_actor_state_critic_predicted_std",
    "slstm_two_small_actor_state_critic_no_nop",
    "lstm_two_small_actor_state_critic",
)
THESIS_ABLATION_PLOTS = (
    ExperimentPlotSelection(
        name="mat_qcx_no_nop",
        group_names=("mat_qcx", "mat_qcx_no_nop"),
        required_group_names=("mat_qcx_no_nop",),
        title_suffix="MAT-QCX NOP Ablation",
        output_subdir="no_nop/mat_qcx",
    ),
    ExperimentPlotSelection(
        name="tmasac_no_nop",
        group_names=("tmasac_baseline", "tmasac_baseline_no_nop"),
        required_group_names=("tmasac_baseline_no_nop",),
        title_suffix="TMASAC NOP Ablation",
        output_subdir="no_nop/tmasac",
    ),
    ExperimentPlotSelection(
        name="slstm_tmasac_no_nop",
        group_names=(
            "slstm_two_small_actor_state_critic",
            "slstm_two_small_actor_state_critic_no_nop",
        ),
        required_group_names=("slstm_two_small_actor_state_critic_no_nop",),
        title_suffix="sLSTM-TMASAC NOP Ablation",
        output_subdir="no_nop/slstm_tmasac",
    ),
    ExperimentPlotSelection(
        name="mat_qcx_gsde",
        group_names=("mat_qcx", "mat_qcx_gsde"),
        required_group_names=("mat_qcx_gsde",),
        title_suffix="MAT-QCX gSDE",
        output_subdir="gaussian_action_distributions/mat_qcx",
    ),
    ExperimentPlotSelection(
        name="tmasac_predicted_std",
        group_names=("tmasac_baseline", "tmasac_baseline_predicted_std"),
        required_group_names=("tmasac_baseline_predicted_std",),
        title_suffix="TMASAC Predicted Standard Deviation",
        output_subdir="gaussian_action_distributions/tmasac",
    ),
    ExperimentPlotSelection(
        name="slstm_tmasac_predicted_std",
        group_names=(
            "slstm_two_small_actor_state_critic",
            "slstm_two_small_actor_state_critic_predicted_std",
        ),
        required_group_names=(
            "slstm_two_small_actor_state_critic_predicted_std",
        ),
        title_suffix="sLSTM-TMASAC Predicted Standard Deviation",
        output_subdir="gaussian_action_distributions/slstm_tmasac",
    ),
    ExperimentPlotSelection(
        name="tmasac_temporal_model",
        group_names=(
            "slstm_two_small_actor_state_critic",
            "lstm_two_small_actor_state_critic",
        ),
        required_group_names=("lstm_two_small_actor_state_critic",),
        title_suffix="TMASAC Temporal Model Ablation",
        output_subdir="temporal_model/tmasac",
    ),
)
THESIS_DISPLAY_NAMES = {
    "mappo": "MAPPO",
    "mat_qcx": "MAT-QCX + NOP",
    "mat_ind": "MAT-Independent + NOP",
    "mat_orig": "MAT original actor decoder + NOP",
    "tmasac_baseline": "TMASAC",
    "slstm_two_small_actor_state_critic": "TMASAC + sLSTM",
    "mat_qcx_gsde": "MAT-QCX + NOP, gSDE",
    "mat_qcx_no_nop": "MAT-QCX, no NOP",
    "tmasac_baseline_predicted_std": "TMASAC + NOP, predicted std",
    "tmasac_baseline_no_nop": "TMASAC, no NOP",
    "slstm_two_small_actor_state_critic_predicted_std": (
        "TMASAC + sLSTM + NOP, predicted std"
    ),
    "slstm_two_small_actor_state_critic_no_nop": "TMASAC + sLSTM, no NOP",
    "lstm_two_small_actor_state_critic": "TMASAC + LSTM",
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
        main_group_names=THESIS_MAIN_GROUP_ORDER,
        extra_plot_selections=THESIS_ABLATION_PLOTS,
        run_length_limit=THESIS_RUN_LENGTH,
        cut_at_limit=True,
    )
