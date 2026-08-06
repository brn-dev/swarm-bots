from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import experiments.mjw_bridge_po_tmasac.scripts.common as bridge_po_common
import experiments.mjw_bridge_static_tmasac.scripts.common as bridge_static_common
import experiments.mjw_multi_payload_goal_easy_tmasac.scripts.common as multi_payload_common
import experiments.mjw_vertical_reach_tmasac.scripts.common as vertical_reach_common
import experiments.tmasac_experiment_common as tmasac_common
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import (
    ActorStateCriticInputConfig,
)
from swarmbots.learn.algos.xlstm.slstm import (
    SLSTMTemporalSequenceModel,
    SLSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.nn_components.feed_forward import (
    GLUStackConfig,
    MLPConfig,
    SwiGLUConfig,
)


@pytest.mark.parametrize(
    ("variant", "expected_policy_variant"),
    [
        ("tmasac_baseline", "tmasac"),
        ("tmasac_swiglu", "tmasac"),
        ("slstm_two_small_actor_state_critic", "r_tmasac"),
        ("slstm_two_small_swiglu_actor_state_critic", "r_tmasac"),
    ],
)
def test_tmasac_experiment_variants_wire_expected_policy(
    monkeypatch: pytest.MonkeyPatch,
    variant: tmasac_common.TMASACExperimentVariant,
    expected_policy_variant: str,
) -> None:
    run_mjw_experiment = Mock()
    monkeypatch.setattr(tmasac_common, "run_mjw_experiment", run_mjw_experiment)

    tmasac_common.run_tmasac_experiment(
        experiment_run_name="test_suite",
        scenario_name="vertical_reach",
        scenario_kwargs={"continuous_connector_actions": True},
        variant=variant,
        entrypoint_path=Path(__file__),
    )

    kwargs = run_mjw_experiment.call_args.kwargs
    assert kwargs["policy_variant"] == expected_policy_variant
    assert kwargs["num_envs"] == 1024
    assert kwargs["rollout_steps_per_env"] == 1
    assert kwargs["sac_learning_rate"] == 5e-5
    assert kwargs["scenario_name"] == "vertical_reach"


def test_default_tmasac_uses_two_512_hidden_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_mjw_experiment = Mock()
    monkeypatch.setattr(tmasac_common, "run_mjw_experiment", run_mjw_experiment)

    _run_variant("tmasac_baseline")

    config = run_mjw_experiment.call_args.kwargs["mat_encoder_transformer_ff_config"]
    assert isinstance(config, MLPConfig)
    assert config.hidden_dims == [512, 512]


def test_feedforward_tmasac_swiglu_stacks_two_parameter_matched_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_mjw_experiment = Mock()
    monkeypatch.setattr(tmasac_common, "run_mjw_experiment", run_mjw_experiment)

    _run_variant("tmasac_swiglu")

    config = run_mjw_experiment.call_args.kwargs["mat_encoder_transformer_ff_config"]
    assert isinstance(config, SwiGLUConfig)
    assert config.hidden_dim == 344
    assert isinstance(config.stacked, GLUStackConfig)
    assert config.stacked.n_layers == 2
    assert config.stacked.pre_norm is nn.LayerNorm
    assert config.stacked.residual


@pytest.mark.parametrize(
    ("variant", "expected_config_type"),
    [
        ("slstm_two_small_actor_state_critic", MLPConfig),
        ("slstm_two_small_swiglu_actor_state_critic", SwiGLUConfig),
    ],
)
def test_slstm_variants_use_two_actor_feedforwards_and_actor_state_critic_input(
    monkeypatch: pytest.MonkeyPatch,
    variant: tmasac_common.TMASACExperimentVariant,
    expected_config_type: type[MLPConfig] | type[SwiGLUConfig],
) -> None:
    run_mjw_experiment = Mock()
    monkeypatch.setattr(tmasac_common, "run_mjw_experiment", run_mjw_experiment)

    _run_variant(variant)

    kwargs = run_mjw_experiment.call_args.kwargs
    assert isinstance(kwargs["mat_encoder_transformer_ff_config"], expected_config_type)
    assert isinstance(kwargs["rmat_actor_transformer_ff_config"], expected_config_type)
    assert kwargs["rmat_actor_inter_module_mlp"] is True
    assert kwargs["rmat_temporal_model_cls"] is SLSTMTemporalSequenceModel
    assert kwargs["rmat_temporal_model_config"] == SLSTMTemporalSequenceModelConfig(
        num_heads=4
    )
    assert kwargs["sac_batch_size"] == 16
    assert kwargs["sac_buffer_capacity_per_env"] == 1024
    assert kwargs["sac_temporal_state_storage_dtype"] is torch.float16
    assert kwargs[
        "r_tmasac_actor_state_critic_input_config"
    ] == ActorStateCriticInputConfig(
        projection_dim=256,
        projection_hidden_dims=(256,),
    )


def test_requested_scenario_suites_expose_expected_scenario_configs() -> None:
    assert bridge_static_common.SCENARIO_KWARGS == {
        "continuous_connector_actions": True
    }
    assert bridge_po_common.SCENARIO_KWARGS["bridge_x"].low == -2.0
    assert bridge_po_common.SCENARIO_KWARGS["bridge_x"].high == 2.0
    assert vertical_reach_common.SCENARIO_KWARGS == {
        "continuous_connector_actions": True
    }
    assert multi_payload_common.SCENARIO_KWARGS == {
        "max_payloads": 2,
        "active_payload_count_probs": {1: 0.5, 2: 0.5},
        "continuous_connector_actions": True,
    }


def test_shared_mjw_experiment_registry_constructs_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_scenario = object()
    default_bridge = Mock(return_value=expected_scenario)
    monkeypatch.setattr(tmasac_common, "run_mjw_experiment", Mock())

    import experiments.mjw_experiment_common as mjw_common

    monkeypatch.setattr(mjw_common, "default_bridge", default_bridge)
    scenario = mjw_common._make_scenario(
        scenario_name="bridge",
        scenario_kwargs={"continuous_connector_actions": True},
    )

    assert scenario is expected_scenario
    default_bridge.assert_called_once_with(continuous_connector_actions=True)


def _run_variant(variant: tmasac_common.TMASACExperimentVariant) -> None:
    tmasac_common.run_tmasac_experiment(
        experiment_run_name="test_suite",
        scenario_name="vertical_reach",
        scenario_kwargs={"continuous_connector_actions": True},
        variant=variant,
        entrypoint_path=Path(__file__),
    )
