from dataclasses import fields

import pytest
import torch
from torch import nn

from swarmbots.learn import list_model_scales
from swarmbots.learn.algos.mat import MATEncoderConfig
from swarmbots.learn.algos.r_mat import RMATEncoder
from swarmbots.learn.parameter_counts import count_policy_parameters
from test_model_scale import make_policy


def _parameter_shapes(module: nn.Module) -> dict[str, torch.Size]:
    return {name: parameter.shape for name, parameter in module.named_parameters()}


def _non_temporal_parameter_shapes(module: nn.Module) -> dict[str, torch.Size]:
    return {
        name.replace(".norm1.", ".attention_norm.").replace(".norm2.", ".feedforward_norm."): shape
        for name, shape in _parameter_shapes(module).items()
        if ".temporal_" not in name
    }


def _assert_additive_recurrent_encoder(baseline: nn.Module, recurrent: RMATEncoder) -> None:
    assert _non_temporal_parameter_shapes(recurrent) == _non_temporal_parameter_shapes(baseline)
    assert sum(parameter.numel() for parameter in recurrent.parameters()) > sum(
        parameter.numel() for parameter in baseline.parameters()
    )
    assert recurrent.config.temporal_residual
    assert recurrent.config.temporal_layer_norm
    assert recurrent.config.temporal_model_order == "inter_agent_attention_first"
    assert not recurrent.config.inter_module_mlp
    assert all(layer.inter_module_feedforward is None for layer in recurrent.layers)


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize("variant", ["tmatd3", "tmatd3_dec"])
def test_recurrent_td3_critics_share_mat_memory_allocation(variant: str, scale: str) -> None:
    with torch.device("meta"):
        td3 = make_policy(variant, scale, use_nop=False, td3_recurrent_actor=True, td3_recurrent_critic=True)
        mat = make_policy("mat_ind__lstm", scale, use_nop=False)
    assert td3.critic.encoder.config.temporal_hidden_dim == mat.critic_encoder.config.temporal_hidden_dim
    assert _parameter_shapes(td3.critic.encoder.layers) == _parameter_shapes(mat.critic_encoder.layers)


@pytest.mark.parametrize("scale", [None, *list_model_scales()])
@pytest.mark.parametrize(
    ("baseline_variant", "recurrent_variant"),
    [
        ("mat_ind", "mat_ind__lstm"),
        ("mat_qcx", "mat_qcx__lstm"),
    ],
)
def test_recurrent_presets_add_memory_without_resizing_the_base(
    scale: str | None, baseline_variant: str, recurrent_variant: str,
) -> None:
    with torch.device("meta"):
        baseline = make_policy(baseline_variant, scale, use_nop=False)
        recurrent = make_policy(recurrent_variant, scale, use_nop=False)

    for field in fields(MATEncoderConfig):
        assert getattr(recurrent.actor_encoder_config, field.name) == getattr(
            baseline.actor_encoder_config, field.name
        ), field.name
        assert getattr(recurrent.critic_encoder_config, field.name) == getattr(
            baseline.critic_encoder_config, field.name
        ), field.name
    _assert_additive_recurrent_encoder(baseline.actor_encoder, recurrent.actor_encoder)
    assert _parameter_shapes(recurrent.actor_head) == _parameter_shapes(baseline.actor_head)
    assert recurrent.config.critic_config == baseline.config.critic_config

    _assert_additive_recurrent_encoder(baseline.critic_encoder, recurrent.critic_encoder)
    assert _parameter_shapes(recurrent.critic) == _parameter_shapes(baseline.critic)
    assert recurrent.actor_encoder is not recurrent.critic_encoder
    if baseline_variant == "mat_qcx":
        assert recurrent.config.decoder_config == baseline.config.decoder_config
        assert _parameter_shapes(recurrent.decoder) == _parameter_shapes(baseline.decoder)


@pytest.mark.parametrize("scale", [None, *list_model_scales()])
@pytest.mark.parametrize("variant", ["tmasac__lstm", "tmasac__slstm"])
def test_recurrent_tmasac_preserves_feedforward_capacity_and_critic(
    scale: str | None, variant: str,
) -> None:
    with torch.device("meta"):
        baseline = make_policy("tmasac", scale, use_nop=False)
        recurrent = make_policy(variant, scale, use_nop=False)

    for field in fields(MATEncoderConfig):
        if scale is not None or field.name not in {"transformer_ff_config", "dim_feedforward"}:
            assert getattr(recurrent.actor_encoder_config, field.name) == getattr(baseline.actor_encoder_config, field.name)
        assert getattr(recurrent.critic_encoder_config, field.name) == getattr(baseline.critic_encoder_config, field.name)
    assert recurrent.actor_encoder.config.temporal_model_order == "inter_agent_attention_first"
    if scale is not None:
        _assert_additive_recurrent_encoder(baseline.actor_encoder, recurrent.actor_encoder)
    else:
        assert recurrent.actor_encoder.config.inter_module_mlp
        for reference_layer, layer in zip(baseline.actor_encoder.layers, recurrent.actor_encoder.layers, strict=True):
            assert _parameter_shapes(layer.self_attn) == _parameter_shapes(reference_layer.self_attn)
            reference_count = sum(parameter.numel() for parameter in reference_layer.feedforward.parameters())
            for feedforward in (layer.inter_module_feedforward, layer.feedforward):
                linears = [module for module in feedforward.modules() if isinstance(module, nn.Linear)]
                assert len(linears) == 2
                assert linears[0].in_features == linears[-1].out_features == layer.d_model
                assert sum(parameter.numel() for parameter in feedforward.parameters()) == pytest.approx(reference_count / 2, rel=0.01)
            assert not {id(parameter) for parameter in layer.inter_module_feedforward.parameters()} & {
                id(parameter) for parameter in layer.feedforward.parameters()
            }
            assert layer.temporal_residual and layer.temporal_layer_norm
    assert _parameter_shapes(recurrent.actor_head) == _parameter_shapes(baseline.actor_head)
    assert recurrent.config.critic_config == baseline.config.critic_config
    assert _parameter_shapes(recurrent.critic.q1) == _parameter_shapes(baseline.critic.q1)
    assert _parameter_shapes(recurrent.critic.q2) == _parameter_shapes(baseline.critic.q2)
    assert recurrent.config.actor_state_critic_input_config is not None
    assert _parameter_shapes(recurrent.critic.encoder.layers) == _parameter_shapes(baseline.critic.encoder.layers)


@pytest.mark.parametrize("scale", [None, *list_model_scales()])
def test_lstm_and_slstm_match_non_temporal_architecture_and_fixed_main_budget(scale: str | None) -> None:
    with torch.device("meta"):
        lstm = make_policy("tmasac__lstm", scale, use_nop=False)
        slstm = make_policy("tmasac__slstm", scale, use_nop=False)

    assert _non_temporal_parameter_shapes(lstm.actor_encoder) == _non_temporal_parameter_shapes(slstm.actor_encoder)
    assert _parameter_shapes(lstm.actor_head) == _parameter_shapes(slstm.actor_head)
    critic_shapes = [
        {name: shape for name, shape in _parameter_shapes(policy.critic).items()
         if not name.startswith("actor_state_encoder.")}
        for policy in (lstm, slstm)
    ]
    assert critic_shapes[0] == critic_shapes[1]
    if scale is not None:
        lstm_count = count_policy_parameters(lstm)["trainable_minus_nop"]
        slstm_count = count_policy_parameters(slstm)["trainable_minus_nop"]
        assert lstm_count == pytest.approx(slstm_count, rel=0.005)
    for policy in (lstm, slstm):
        state_width = policy.actor_encoder.layers[-1].temporal_model.hidden_dim
        first_linear = next(module for module in policy.critic.actor_state_encoder.modules() if isinstance(module, nn.Linear))
        assert first_linear.in_features == state_width * 2
