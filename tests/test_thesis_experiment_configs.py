from pathlib import Path
from unittest.mock import patch

import pytest

from experiments import thesis_experiment_common, thesis_plot_common
from experiments.thesis_mjw_find_opening import plot_results as find_opening_plot
from experiments.thesis_mjw_find_opening.scripts import common as find_opening_common
from experiments.thesis_mjw_po_wall_medium import plot_250m_results as po_wall_250m_plot
from experiments.thesis_mjw_po_wall_medium import plot_results as po_wall_plot
from experiments.thesis_mjw_po_wall_medium.scripts import common as po_wall_common
from experiments.thesis_mjw_po_wall_medium.scripts import common_250m as po_wall_250m_common
from experiments.thesis_parallel_env_ablation_po_wall_medium import (
    plot_results as parallel_env_plot,
)
from swarmbots.learn.algos.mat.mat_encoder import (
    MATEncoderConfig,
    resolve_transformer_ff_config,
)
from swarmbots.learn.nn_components.feed_forward import MLPConfig


@pytest.mark.parametrize(
    ("variant", "policy_variant", "continuous_action_dist", "use_nop"),
    (
        ("mappo", "mat_ind", "sign_magnitude_beta", True),
        ("mat_qcx", "mat_qcx", "sign_magnitude_beta", True),
        ("mat_ind", "mat_ind", "sign_magnitude_beta", True),
        ("r_mat_ind", "r_mat_ind", "sign_magnitude_beta", True),
        ("mat_orig", "mat_orig", "sign_magnitude_beta", True),
        ("mat_qcx_gsde", "mat_qcx", "gsde", True),
        ("mat_qcx_no_nop", "mat_qcx", "sign_magnitude_beta", False),
    ),
)
def test_ppo_thesis_variants_use_current_observation_and_action_defaults(
    variant: thesis_experiment_common.ThesisAlgorithmVariant,
    policy_variant: str,
    continuous_action_dist: str,
    use_nop: bool,
) -> None:
    entrypoint_path = Path(__file__)
    scenario_kwargs = {"continuous_connector_actions": True}

    with patch.object(thesis_experiment_common, "run_mjw_experiment") as run:
        thesis_experiment_common.run_thesis_experiment(
            experiment_run_name="test",
            scenario_name="find_opening",
            scenario_kwargs=scenario_kwargs,
            variant=variant,
            entrypoint_path=entrypoint_path,
        )

    run.assert_called_once_with(
        num_envs=1024,
        rollout_steps_per_env=4,
        variant_name=variant,
        entrypoint_path=entrypoint_path,
        continuous_action_dist=continuous_action_dist,
        policy_variant=policy_variant,
        mat_add_agent_embeddings=False,
        mat_use_agent_attention=variant != "mappo",
        nop_add_agent_embeddings_transition_model=False,
        use_nop=use_nop,
        use_transition_obs=False,
        experiment_run_name="test",
        scenario_name="find_opening",
        scenario_kwargs=scenario_kwargs,
        total_timesteps=100_000_000,
    )


@pytest.mark.parametrize(
    (
        "variant",
        "tmasac_variant",
        "continuous_action_dist",
        "use_nop",
        "include_slstm_memory_strength",
    ),
    (
        (
            "tmasac_baseline",
            "tmasac_baseline",
            "gumbel_softmax_sign_magnitude_beta",
            True,
            False,
        ),
        (
            "tmasac_shared_encoder",
            "tmasac_shared_encoder",
            "gumbel_softmax_sign_magnitude_beta",
            True,
            False,
        ),
        (
            "slstm_two_small_actor_state_critic",
            "slstm_two_small_actor_state_critic",
            "gumbel_softmax_sign_magnitude_beta",
            True,
            False,
        ),
        (
            "slstm_shared_encoder",
            "slstm_shared_encoder",
            "gumbel_softmax_sign_magnitude_beta",
            True,
            False,
        ),
        (
            "lstm_two_small_actor_state_critic",
            "lstm_two_small_actor_state_critic",
            "gumbel_softmax_sign_magnitude_beta",
            True,
            False,
        ),
        (
            "tmasac_baseline_predicted_std",
            "tmasac_baseline",
            "predicted_std",
            True,
            False,
        ),
        (
            "tmasac_baseline_no_nop",
            "tmasac_baseline",
            "gumbel_softmax_sign_magnitude_beta",
            False,
            False,
        ),
        (
            "slstm_two_small_actor_state_critic_predicted_std",
            "slstm_two_small_actor_state_critic",
            "predicted_std",
            True,
            False,
        ),
        (
            "slstm_two_small_actor_state_critic_no_nop",
            "slstm_two_small_actor_state_critic",
            "gumbel_softmax_sign_magnitude_beta",
            False,
            False,
        ),
        (
            "slstm_two_small_actor_state_critic_no_memory_strength",
            "slstm_two_small_actor_state_critic",
            "gumbel_softmax_sign_magnitude_beta",
            True,
            False,
        ),
    ),
)
def test_tmasac_thesis_variants_use_shared_current_architectures(
    variant: thesis_experiment_common.ThesisAlgorithmVariant,
    tmasac_variant: str,
    continuous_action_dist: str,
    use_nop: bool,
    include_slstm_memory_strength: bool,
) -> None:
    entrypoint_path = Path(__file__)
    scenario_kwargs = {"continuous_connector_actions": True}

    with patch.object(thesis_experiment_common, "run_tmasac_experiment") as run:
        thesis_experiment_common.run_thesis_experiment(
            experiment_run_name="test",
            scenario_name="wall",
            scenario_kwargs=scenario_kwargs,
            variant=variant,
            entrypoint_path=entrypoint_path,
        )

    run.assert_called_once_with(
        experiment_run_name="test",
        scenario_name="wall",
        scenario_kwargs=scenario_kwargs,
        variant=tmasac_variant,
        entrypoint_path=entrypoint_path,
        continuous_action_dist=continuous_action_dist,
        use_nop=use_nop,
        include_slstm_memory_strength=include_slstm_memory_strength,
        variant_name=variant,
    )


def test_thesis_suites_use_continuous_connector_scenario_configs() -> None:
    assert find_opening_common.SCENARIO_KWARGS == {"continuous_connector_actions": True}
    assert po_wall_common.SCENARIO_KWARGS["continuous_connector_actions"] is True


def test_thesis_suite_names_use_thesis_prefix() -> None:
    assert find_opening_common.EXPERIMENT_RUN_NAME == "thesis_mjw_find_opening"
    assert po_wall_common.EXPERIMENT_RUN_NAME == "thesis_mjw_po_wall_medium"
    assert find_opening_plot.EXPERIMENT_RUN_DIR.name == "thesis_mjw_find_opening"
    assert po_wall_plot.EXPERIMENT_RUN_DIR.name == "thesis_mjw_po_wall_medium"


def test_po_wall_original_suite_remains_at_100m_steps() -> None:
    with patch.object(thesis_experiment_common, "run_mjw_experiment") as run:
        po_wall_common.run_experiment(
            variant="mat_qcx",
            entrypoint_path=Path(__file__),
        )

    assert run.call_args.kwargs["total_timesteps"] == 100_000_000


@pytest.mark.parametrize("variant", ("mappo", "mat_qcx", "mat_ind", "mat_orig"))
def test_po_wall_250m_suite_has_a_long_horizon_run_for_each_ppo_variant(
    variant: po_wall_250m_common.LongHorizonAlgorithmVariant,
) -> None:
    with patch.object(thesis_experiment_common, "run_mjw_experiment") as run:
        po_wall_250m_common.run_experiment(
            variant=variant,
            entrypoint_path=Path(__file__),
        )

    assert run.call_args.kwargs["variant_name"] == variant
    assert run.call_args.kwargs["experiment_run_name"] == (
        "thesis_mjw_po_wall_medium_250m"
    )
    assert run.call_args.kwargs["total_timesteps"] == 250_000_000


def test_po_wall_250m_suite_has_separate_entrypoints_and_plot() -> None:
    scripts_dir = Path(po_wall_250m_common.__file__).parent
    assert {path.name for path in scripts_dir.glob("run_*_250m.py")} == {
        "run_mappo_250m.py",
        "run_mat_qcx_250m.py",
        "run_mat_ind_250m.py",
        "run_mat_orig_250m.py",
    }
    assert po_wall_250m_plot.EXPERIMENT_RUN_DIR.name == (
        po_wall_250m_common.EXPERIMENT_RUN_NAME
    )
    assert po_wall_250m_plot.OUTPUT_DIR.name == "250m"
    assert po_wall_250m_plot.GROUP_ORDER == ("mappo", "mat_ind", "mat_qcx", "mat_orig")


def test_thesis_plots_keep_ablations_out_of_main_plot_and_use_pair_comparisons() -> None:
    assert thesis_plot_common.THESIS_MAIN_GROUP_ORDER == (
        "mappo",
        "mat_ind",
        "mat_qcx",
        "mat_orig",
        "tmasac_baseline",
        "slstm_two_small_actor_state_critic",
    )
    assert {
        selection.name: selection.group_names
        for selection in thesis_plot_common.THESIS_ABLATION_PLOTS
    } == {
        "mat_ind_recurrence": ("mat_ind", "r_mat_ind"),
        "mat_qcx_no_nop": ("mat_qcx", "mat_qcx_no_nop"),
        "tmasac_no_nop": ("tmasac_baseline", "tmasac_baseline_no_nop"),
        "slstm_tmasac_no_nop": (
            "slstm_two_small_actor_state_critic",
            "slstm_two_small_actor_state_critic_no_nop",
        ),
        "tmasac_shared_encoder": (
            "tmasac_baseline",
            "tmasac_shared_encoder",
        ),
        "slstm_shared_encoder": (
            "slstm_two_small_actor_state_critic",
            "slstm_shared_encoder",
        ),
        "mat_qcx_gsde": ("mat_qcx", "mat_qcx_gsde"),
        "tmasac_predicted_std": (
            "tmasac_baseline",
            "tmasac_baseline_predicted_std",
        ),
        "slstm_tmasac_predicted_std": (
            "slstm_two_small_actor_state_critic",
            "slstm_two_small_actor_state_critic_predicted_std",
        ),
        "tmasac_temporal_model": (
            "slstm_two_small_actor_state_critic",
            "lstm_two_small_actor_state_critic",
        ),
        "slstm_tmasac_memory_strength": (
            "slstm_two_small_actor_state_critic",
            "slstm_two_small_actor_state_critic_no_memory_strength",
        ),
    }
    assert thesis_plot_common.THESIS_GROUP_ORDER[-1] == (
        "slstm_two_small_actor_state_critic_no_memory_strength"
    )

    enlarged_selection_names = {
        "mat_qcx_no_nop",
        "tmasac_no_nop",
        "slstm_tmasac_no_nop",
        "mat_qcx_gsde",
        "tmasac_predicted_std",
        "slstm_tmasac_predicted_std",
    }
    for selection in thesis_plot_common.THESIS_ABLATION_PLOTS:
        if selection.name in enlarged_selection_names:
            assert selection.font_size == 22
            assert selection.legend_font_size == 24
        else:
            assert selection.font_size == 16
            assert selection.legend_font_size == 18


def test_thesis_group_colors_match_requested_swaps() -> None:
    expected_main_colors = {
        "mappo": "#CC79A7",
        "mat_qcx": "#E69F00",
        "mat_ind": "#009E73",
        "mat_orig": "#D55E00",
        "tmasac_baseline": "#0072B2",
        "slstm_two_small_actor_state_critic": "#56B4E9",
    }
    assert {
        group_name: thesis_plot_common.THESIS_GROUP_COLOR_OVERRIDES[group_name]
        for group_name in thesis_plot_common.THESIS_MAIN_GROUP_ORDER
    } == expected_main_colors


def test_every_thesis_variant_has_a_scenario_independent_color() -> None:
    assert set(thesis_plot_common.THESIS_GROUP_COLOR_OVERRIDES) == set(
        thesis_plot_common.THESIS_GROUP_ORDER
    )


def test_thesis_ablation_pairs_use_distinct_colors() -> None:
    colors = thesis_plot_common.THESIS_GROUP_COLOR_OVERRIDES
    for selection in thesis_plot_common.THESIS_ABLATION_PLOTS:
        assert len({colors[group_name] for group_name in selection.group_names}) == len(
            selection.group_names
        )


def test_thesis_ablation_colors_do_not_reuse_main_colors() -> None:
    colors = thesis_plot_common.THESIS_GROUP_COLOR_OVERRIDES
    main_colors = {
        colors[group_name]
        for group_name in thesis_plot_common.THESIS_MAIN_GROUP_ORDER
    }
    ablation_colors = {
        colors[group_name]
        for group_name in thesis_plot_common.THESIS_GROUP_ORDER
        if group_name not in thesis_plot_common.THESIS_MAIN_GROUP_ORDER
    }
    assert main_colors.isdisjoint(ablation_colors)


def test_parallel_env_baseline_keeps_mat_qcx_color() -> None:
    assert set(parallel_env_plot.GROUP_COLOR_OVERRIDES) == set(
        parallel_env_plot.GROUP_ORDER
    )
    assert len(set(parallel_env_plot.GROUP_COLOR_OVERRIDES.values())) == len(
        parallel_env_plot.GROUP_ORDER
    )
    assert parallel_env_plot.GROUP_COLOR_OVERRIDES["mat_qcx_1024x4"] == (
        thesis_plot_common.THESIS_GROUP_COLOR_OVERRIDES["mat_qcx"]
    )


def test_thesis_main_labels_omit_nop_and_nop_ablations_label_both_sides() -> None:
    assert all(
        thesis_experiment_common.THESIS_VARIANT_CONFIGS[group_name].use_nop
        for group_name in thesis_plot_common.THESIS_MAIN_GROUP_ORDER
    )
    assert all(
        "NOP" not in thesis_plot_common.THESIS_DISPLAY_NAMES[group_name]
        for group_name in thesis_plot_common.THESIS_MAIN_GROUP_ORDER
    )
    nop_selections = thesis_plot_common.THESIS_ABLATION_PLOTS[:3]
    for selection in nop_selections:
        assert selection.display_name_overrides is not None
        assert set(selection.display_name_overrides) == set(selection.group_names)
        assert any(
            "+ NOP" in display_name
            for display_name in selection.display_name_overrides.values()
        )
        assert any(
            "no NOP" in display_name
            for display_name in selection.display_name_overrides.values()
        )


def test_thesis_squashed_gaussian_labels_use_reader_facing_name() -> None:
    assert thesis_plot_common.THESIS_DISPLAY_NAMES[
        "tmasac_baseline_predicted_std"
    ] == "TMASAC, squashed Gaussian"
    assert thesis_plot_common.THESIS_DISPLAY_NAMES[
        "slstm_two_small_actor_state_critic_predicted_std"
    ] == "TMASAC + sLSTM, squashed Gaussian"


def test_find_opening_plot_reuses_matching_tmasac_runs() -> None:
    assert find_opening_plot.EXTRA_GROUP_SOURCES == {
        "tmasac_baseline": (
            find_opening_plot.MATCHING_TMASAC_RUN_DIR / "tmasac_baseline",
        ),
        "slstm_two_small_actor_state_critic": (
            find_opening_plot.MATCHING_TMASAC_RUN_DIR
            / "slstm_two_small_actor_state_critic",
        ),
        "lstm_two_small_actor_state_critic": (
            find_opening_plot.MATCHING_TMASAC_RUN_DIR
            / "lstm_two_small_actor_state_critic",
        ),
    }


def test_implicit_recurrent_critic_feedforward_matches_current_small_mlp() -> None:
    old_implicit_config = MATEncoderConfig(
        d_model=256,
        dim_feedforward=512,
        transformer_ff_config=None,
    )

    assert resolve_transformer_ff_config(old_implicit_config) == MLPConfig(
        hidden_dims=[512]
    )


def test_po_wall_plot_reuses_matching_ppo_and_tmasac_runs() -> None:
    assert po_wall_plot.EXTRA_GROUP_SOURCES == {
        "mappo": (po_wall_plot.MATCHING_PPO_RUN_DIR / "mappo",),
        "mat_qcx": (po_wall_plot.MATCHING_PPO_RUN_DIR / "mat_qcx",),
        "mat_ind": (po_wall_plot.MATCHING_PPO_RUN_DIR / "mat_ind",),
        "mat_orig": (po_wall_plot.MATCHING_PPO_RUN_DIR / "mat_orig",),
        "tmasac_baseline": (
            po_wall_plot.MATCHING_TMASAC_RUN_DIR / "tmasac_lr=5e-5_bigger_mlps",
        ),
    }
