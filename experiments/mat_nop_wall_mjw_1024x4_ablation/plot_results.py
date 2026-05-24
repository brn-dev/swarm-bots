from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import ExperimentPlotSelection, plot_experiment_results


EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / "mat_nop_swarm_bots_wall_mjw_1024x4_ablation"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
BASELINE_GROUP = "lr_beta_nop"
GROUP_ORDER = (
    BASELINE_GROUP,
    "gsde_nop",
    "beta_nop",
    "squashed_diag_gaussian_nop",
    "lr_beta_nop_wall_pass_skew_2",
    "lr_beta_nop_shuffle_agents",
    "lr_beta_nop_shuffle_agents_preserve_prefix",
    "lr_beta_no_nop",
    "sticky_lr_beta_nop_squared_relu",
    "lr_beta_nop_init_gain_0_01",
    "lr_beta_nop_init_gain_0_01_projections_1",
    "lr_beta_nop_init_gain_0_1",
    "lr_beta_nop_init_gain_0_1_projections_1",
    "lr_beta_nop_value_regressor_init_gain_0_1",
    "lr_beta_nop_context_tokens_only",
    "lr_beta_nop_no_agent_embeddings",
    "lr_beta_nop_context_tokens_only_no_agent_embeddings",
)
DISPLAY_NAME_OVERRIDES = {
    BASELINE_GROUP: "L/R Beta + NOP",
    "gsde_nop": "gSDE + NOP",
    "beta_nop": "Beta + NOP",
    "squashed_diag_gaussian_nop": "Squashed Diag Gaussian + NOP",
    "lr_beta_nop_wall_pass_skew_2": "L/R Beta + NOP, wall-pass skew 2",
    "lr_beta_nop_shuffle_agents": "L/R Beta + NOP, shuffled agents",
    "lr_beta_nop_shuffle_agents_preserve_prefix": (
        "L/R Beta + NOP, shuffled active-prefix agents"
    ),
    "lr_beta_no_nop": "L/R Beta, no NOP",
    "sticky_lr_beta_nop_squared_relu": "Sticky L/R Beta + NOP, Squared ReLU",
    "lr_beta_nop_init_gain_0_01": "L/R Beta + NOP, hidden init gain 0.01",
    "lr_beta_nop_init_gain_0_01_projections_1": (
        "L/R Beta + NOP, hidden init gain 0.01, projections 1.0"
    ),
    "lr_beta_nop_init_gain_0_1": "L/R Beta + NOP, hidden init gain 0.1",
    "lr_beta_nop_init_gain_0_1_projections_1": (
        "L/R Beta + NOP, hidden init gain 0.1, projections 1.0"
    ),
    "lr_beta_nop_value_regressor_init_gain_0_1": (
        "L/R Beta + NOP, value regressor init gain 0.1"
    ),
    "lr_beta_nop_context_tokens_only": "L/R Beta + NOP, context tokens only",
    "lr_beta_nop_no_agent_embeddings": "L/R Beta + NOP, no agent embeddings",
    "lr_beta_nop_context_tokens_only_no_agent_embeddings": (
        "L/R Beta + NOP, context tokens only, no agent embeddings"
    ),
}
EXTRA_GROUP_SOURCES = {
    BASELINE_GROUP: (
        REPO_ROOT / "runs" / "mat_nop_swarm_bots_wall_mjw_batch_env_sweep" / "1024x4",
    ),
}
PAIRWISE_VARIANT_GROUPS = tuple(group_name for group_name in GROUP_ORDER if group_name != BASELINE_GROUP)
VARIANT_FAMILY_SELECTIONS = (
    ExperimentPlotSelection(
        name="action_distribution_variants",
        group_names=(
            BASELINE_GROUP,
            "gsde_nop",
            "beta_nop",
            "squashed_diag_gaussian_nop",
        ),
        title_suffix="Action Distribution Variants",
        required_group_names=(BASELINE_GROUP,),
    ),
    ExperimentPlotSelection(
        name="lr_beta_ablation_variants",
        group_names=(
            BASELINE_GROUP,
            "lr_beta_nop_wall_pass_skew_2",
            "lr_beta_nop_shuffle_agents",
            "lr_beta_nop_shuffle_agents_preserve_prefix",
            "lr_beta_no_nop",
            "sticky_lr_beta_nop_squared_relu",
        ),
        title_suffix="L/R Beta Ablation Variants",
        required_group_names=(BASELINE_GROUP,),
    ),
    ExperimentPlotSelection(
        name="shuffle_variants",
        group_names=(
            BASELINE_GROUP,
            "lr_beta_nop_shuffle_agents",
            "lr_beta_nop_shuffle_agents_preserve_prefix",
        ),
        title_suffix="Shuffle Variants",
        required_group_names=(BASELINE_GROUP,),
    ),
    ExperimentPlotSelection(
        name="nop_ablation",
        group_names=(
            BASELINE_GROUP,
            "lr_beta_no_nop",
        ),
        title_suffix="NOP Ablation",
        required_group_names=(BASELINE_GROUP,),
    ),
    ExperimentPlotSelection(
        name="initialization_variants",
        group_names=(
            BASELINE_GROUP,
            "lr_beta_nop_init_gain_0_01",
            "lr_beta_nop_init_gain_0_01_projections_1",
            "lr_beta_nop_init_gain_0_1",
            "lr_beta_nop_init_gain_0_1_projections_1",
            "lr_beta_nop_value_regressor_init_gain_0_1",
        ),
        title_suffix="Initialization Variants",
        required_group_names=(BASELINE_GROUP,),
    ),
    ExperimentPlotSelection(
        name="context_agent_embedding_variants",
        group_names=(
            BASELINE_GROUP,
            "lr_beta_nop_context_tokens_only",
            "lr_beta_nop_no_agent_embeddings",
            "lr_beta_nop_context_tokens_only_no_agent_embeddings",
        ),
        title_suffix="Context-Only And Agent Embedding Variants",
        required_group_names=(BASELINE_GROUP,),
    ),
)
EXTRA_PLOT_SELECTIONS = tuple(
    ExperimentPlotSelection(
        name=f"pair_{BASELINE_GROUP}_vs_{variant_group}",
        group_names=(BASELINE_GROUP, variant_group),
        title_suffix=(
            f"{DISPLAY_NAME_OVERRIDES[BASELINE_GROUP]} vs "
            f"{DISPLAY_NAME_OVERRIDES.get(variant_group, variant_group)}"
        ),
        required_group_names=(BASELINE_GROUP,),
    )
    for variant_group in PAIRWISE_VARIANT_GROUPS
) + VARIANT_FAMILY_SELECTIONS
THEORETICAL_MAXIMUM = 10.5


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        theoretical_maximum=THEORETICAL_MAXIMUM,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        extra_group_sources=EXTRA_GROUP_SOURCES,
        extra_plot_selections=EXTRA_PLOT_SELECTIONS,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
