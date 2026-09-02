from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from experiments.post_thesis_action_dist_common import (
    ACTION_DIST_VARIANTS,
    BASELINE_GROUP,
    action_dist_group_name,
)
from experiments.thesis_plot_common import (
    THESIS_GROUP_COLOR_OVERRIDES,
    THESIS_GROUP_LINESTYLE_OVERRIDES,
    THESIS_GROUP_MARKER_OVERRIDES,
    THESIS_RUN_LENGTH,
)
from experiments.thesis_result_summary import summarize_thesis_groups
from plot_logs.experiment_results import (
    ExperimentPlotResult,
    load_experiment_groups,
    plot_experiment_results,
)

GROUP_ORDER = (
    BASELINE_GROUP,
    *(action_dist_group_name(variant) for variant in ACTION_DIST_VARIANTS),
)
DISPLAY_NAME_OVERRIDES = {
    BASELINE_GROUP: "TMASAC, Gumbel sign-magnitude Beta",
    action_dist_group_name("bernstein_6"): "TMASAC, Bernstein (degree 6)",
    action_dist_group_name("bernstein_8"): "TMASAC, Bernstein (degree 8)",
    action_dist_group_name("rqs_4"): "TMASAC, RQS (4 bins)",
    action_dist_group_name("rqs_6"): "TMASAC, RQS (6 bins)",
}
GROUP_COLOR_OVERRIDES = {
    BASELINE_GROUP: THESIS_GROUP_COLOR_OVERRIDES[BASELINE_GROUP],
    action_dist_group_name("bernstein_6"): "#E69F00",
    action_dist_group_name("bernstein_8"): "#D55E00",
    action_dist_group_name("rqs_4"): "#009E73",
    action_dist_group_name("rqs_6"): "#CC79A7",
}
GROUP_LINESTYLE_OVERRIDES = {
    BASELINE_GROUP: THESIS_GROUP_LINESTYLE_OVERRIDES[BASELINE_GROUP],
    **dict.fromkeys(GROUP_ORDER[1:], "-"),
}
GROUP_MARKER_OVERRIDES = {
    BASELINE_GROUP: THESIS_GROUP_MARKER_OVERRIDES[BASELINE_GROUP],
    action_dist_group_name("bernstein_6"): "o",
    action_dist_group_name("bernstein_8"): "s",
    action_dist_group_name("rqs_4"): "^",
    action_dist_group_name("rqs_6"): "D",
}


def make_thesis_tmasac_baseline_sources(
        *,
        thesis_experiment_run_dir: Path,
        thesis_extra_group_sources: Mapping[str, Sequence[Path]],
) -> tuple[Path, ...]:
    return (
        thesis_experiment_run_dir / BASELINE_GROUP,
        *thesis_extra_group_sources[BASELINE_GROUP],
    )


def plot_post_thesis_action_dist_results(
        *,
        experiment_run_dir: Path,
        output_dir: Path,
        baseline_sources: Sequence[Path],
        scenario_title: str,
) -> ExperimentPlotResult:
    return plot_experiment_results(
        experiment_run_dir,
        output_dir,
        group_order=GROUP_ORDER,
        group_filter=GROUP_ORDER,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        extra_group_sources={BASELINE_GROUP: baseline_sources},
        group_color_overrides=GROUP_COLOR_OVERRIDES,
        group_linestyle_overrides=GROUP_LINESTYLE_OVERRIDES,
        group_marker_overrides=GROUP_MARKER_OVERRIDES,
        run_length_limit=THESIS_RUN_LENGTH,
        cut_at_limit=True,
        title_suffix=scenario_title,
    )


def summarize_post_thesis_action_dist_results(
        *,
        experiment_run_dir: Path,
        baseline_sources: Sequence[Path],
        output_path: Path,
        tail_points: int,
) -> Path:
    groups = load_experiment_groups(
        experiment_run_dir,
        group_order=GROUP_ORDER,
        group_filter=GROUP_ORDER,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        extra_group_sources={BASELINE_GROUP: baseline_sources},
    )
    return summarize_thesis_groups(
        groups=groups,
        output_path=output_path,
        tail_points=tail_points,
    )
