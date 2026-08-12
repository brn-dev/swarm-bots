from pathlib import Path
from unittest.mock import patch

import pytest

from experiments import thesis_experiment_common
from experiments.thesis_mjw_find_opening import plot_results as find_opening_plot
from experiments.thesis_mjw_find_opening.scripts import common as find_opening_common
from experiments.thesis_mjw_po_wall_medium import plot_results as po_wall_plot
from experiments.thesis_mjw_po_wall_medium.scripts import common as po_wall_common
from swarmbots.learn.algos.mat.mat_encoder import (
    MATEncoderConfig,
    resolve_transformer_ff_config,
)
from swarmbots.learn.nn_components.feed_forward import MLPConfig


@pytest.mark.parametrize(
    ("variant", "policy_variant"),
    (
        ("mappo", "mat_ind"),
        ("mat_qcx", "mat_qcx"),
        ("mat_ind", "mat_ind"),
        ("mat_orig", "mat_orig"),
    ),
)
def test_ppo_thesis_variants_use_current_observation_and_action_defaults(
    variant: thesis_experiment_common.ThesisAlgorithmVariant,
    policy_variant: str,
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
        policy_variant=policy_variant,
        mat_add_agent_embeddings=False,
        mat_use_agent_attention=variant != "mappo",
        nop_add_agent_embeddings_transition_model=False,
        use_nop=True,
        use_transition_obs=False,
        experiment_run_name="test",
        scenario_name="find_opening",
        scenario_kwargs=scenario_kwargs,
        total_timesteps=100_000_000,
    )


@pytest.mark.parametrize(
    "variant",
    ("tmasac_baseline", "slstm_two_small_actor_state_critic"),
)
def test_tmasac_thesis_variants_use_shared_current_architectures(
    variant: thesis_experiment_common.ThesisAlgorithmVariant,
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
        variant=variant,
        entrypoint_path=entrypoint_path,
    )


def test_thesis_suites_use_continuous_connector_scenario_configs() -> None:
    assert find_opening_common.SCENARIO_KWARGS == {"continuous_connector_actions": True}
    assert po_wall_common.SCENARIO_KWARGS["continuous_connector_actions"] is True


def test_thesis_suite_names_use_thesis_prefix() -> None:
    assert find_opening_common.EXPERIMENT_RUN_NAME == "thesis_mjw_find_opening"
    assert po_wall_common.EXPERIMENT_RUN_NAME == "thesis_mjw_po_wall_medium"
    assert find_opening_plot.EXPERIMENT_RUN_DIR.name == "thesis_mjw_find_opening"
    assert po_wall_plot.EXPERIMENT_RUN_DIR.name == "thesis_mjw_po_wall_medium"


def test_find_opening_plot_reuses_matching_tmasac_runs() -> None:
    assert find_opening_plot.EXTRA_GROUP_SOURCES == {
        "tmasac_baseline": (
            find_opening_plot.MATCHING_TMASAC_RUN_DIR / "tmasac_baseline",
        ),
        "slstm_two_small_actor_state_critic": (
            find_opening_plot.MATCHING_TMASAC_RUN_DIR
            / "slstm_two_small_actor_state_critic",
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
