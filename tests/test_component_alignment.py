import pytest
import torch
from torch import nn

from swarmbots.learn import list_model_scales
from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig
from swarmbots.learn.nn_components.feed_forward import MLPConfig
from swarmbots.learn.nn_components.residual_mlp import ResidualMLP
from swarmbots.learn.presets.transformer import MATInitGains, MATNormalizationConfig
from test_model_scale import make_policy, wall_easy_actor_shape  # noqa: F401


def _parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.usefixtures("wall_easy_actor_shape")
def test_actor_encoders_match_processing_depth_width_and_allocation(scale: str) -> None:
    variants = (
        "mat_ind", "mat_dec", "mat_qcx", "mat_orig", "mappo__deepset_critic", "tmasac", "tmasac_dec",
        "tmatd3", "tmatd3_dec", "masac__deepset_critic", "matd3__deepset_critic", "maddpg__deepset_critic",
    )
    with torch.device("meta"):
        policies = [make_policy(variant, scale, n_agents=5, use_nop=False) for variant in variants]
    reference = policies[0].actor_encoder
    for policy in policies:
        encoder = policy.actor_encoder if hasattr(policy, "actor_encoder") else policy.actor.encoder
        assert len(encoder.layers) == len(reference.layers)
        assert _parameters(encoder) == pytest.approx(_parameters(reference), rel=0.001)
        for layer, reference_layer in zip(encoder.layers, reference.layers, strict=True):
            assert {
                name: parameter.shape for name, parameter in layer.feedforward.named_parameters()
            } == {name: parameter.shape for name, parameter in reference_layer.feedforward.named_parameters()}
            communication = layer.self_attn if layer.self_attn is not None else layer.attention_replacement
            assert _parameters(communication) == pytest.approx(_parameters(reference_layer.self_attn), rel=0.002)


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize("actor_head_gain", [1.0, 0.7])
def test_critic_communication_ablation_preserves_seeded_complete_actor(scale: str, actor_head_gain: float) -> None:
    gains = MATInitGains(actor_head=actor_head_gain)
    torch.manual_seed(123)
    attention = make_policy("mat_dec", scale, use_nop=False, mat_init_gains=gains)
    torch.manual_seed(123)
    deepset = make_policy("mappo__deepset_critic", scale, use_nop=False, mat_init_gains=gains)
    for first, second in (
        (attention.actor_encoder, deepset.actor_encoder),
        (attention.actor_head, deepset.actor.mlp),
        (attention.action_dist, deepset.action_dist),
    ):
        torch.testing.assert_close(first.state_dict(), second.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize(("first_variant", "second_variant", "actor_modules"), [
    ("tmasac_dec", "masac__deepset_critic", ("actor_encoder", "actor_head", "action_dist")),
    ("tmatd3_dec", "matd3__deepset_critic", ("actor", "actor_target")),
])
def test_off_policy_critic_communication_controls_preserve_seeded_actors(
    scale: str, first_variant: str, second_variant: str, actor_modules: tuple[str, ...],
) -> None:
    torch.manual_seed(123)
    first = make_policy(first_variant, scale, use_nop=False)
    torch.manual_seed(123)
    second = make_policy(second_variant, scale, use_nop=False)
    for name in actor_modules:
        torch.testing.assert_close(
            getattr(first, name).state_dict(), getattr(second, name).state_dict(), rtol=0, atol=0,
        )


@pytest.mark.parametrize("variant", ["mat_qcx", "mat_qcx__lstm", "tmasac"])
@pytest.mark.parametrize("normalize", [False, True])
def test_qcx_context_normalization_override_reaches_actual_layers(variant: str, normalize: bool) -> None:
    options = {"tmasac_actor_head_kind": "qcx"} if variant == "tmasac" else {}
    with torch.device("meta"):
        policy = make_policy(
            variant, use_nop=False, mat_normalization=MATNormalizationConfig(normalize_context_tokens=normalize),
            **options,
        )
    decoder = policy.actor_head.decoder if variant == "tmasac" else policy.decoder
    assert all(isinstance(layer.context_token_norm, nn.LayerNorm) is normalize for layer in decoder.layers)


def _local_head(policy: nn.Module) -> ResidualMLP:
    if hasattr(policy, "actor_head"):
        head = policy.actor_head
    elif hasattr(policy.actor, "mlp"):
        head = policy.actor.mlp
    else:
        head = policy.actor.head
    return head if isinstance(head, ResidualMLP) else head.head


@pytest.mark.parametrize("scale", list_model_scales())
def test_independent_actor_heads_share_a_local_residual_mlp(scale: str) -> None:
    variants = (
        "mat_ind", "mat_dec", "mappo__deepset_critic", "tmasac", "tmasac_dec",
        "tmatd3", "tmatd3_dec", "masac__deepset_critic", "matd3__deepset_critic", "maddpg__deepset_critic",
    )
    with torch.device("meta"):
        heads = [_local_head(make_policy(variant, scale, use_nop=False)) for variant in variants]
    reference = {name: parameter.shape for name, parameter in heads[0].named_parameters()}
    for head in heads:
        assert isinstance(head, ResidualMLP)
        assert not any(isinstance(module, nn.MultiheadAttention) for module in head.modules())
        assert isinstance(head.output_projection, nn.Identity)
        assert {name: parameter.shape for name, parameter in head.named_parameters()} == reference


def test_five_million_two_layer_heads_use_the_approved_dimensions() -> None:
    with torch.device("meta"):
        independent = make_policy("mat_ind", "5M-2L", use_nop=False)
        qcx = make_policy("mat_qcx", "5M-2L", use_nop=False)
    for policy in (independent, qcx):
        assert len(policy.actor_encoder.layers) == len(policy.critic_encoder.layers) == 2
        assert policy.actor_encoder.norm.normalized_shape == (256,)
    head = _local_head(independent)
    assert len(head.layers) == 1
    assert [(layer.in_features, layer.out_features) for layer in head.layers[0].feedforward.modules()
            if isinstance(layer, nn.Linear)] == [(256, 512), (512, 256)]
    assert len(qcx.decoder.layers) == 1
    decoder = qcx.decoder.layers[0]
    assert [(layer.in_features, layer.out_features) for layer in qcx.decoder.input_projection.modules()
            if isinstance(layer, nn.Linear)] == [(256, 160)]
    assert [(layer.in_features, layer.out_features) for layer in decoder.context_encoder.modules()
            if isinstance(layer, nn.Linear)] == [(320, 160)]
    assert decoder.query_context_attn.embed_dim == decoder.memory_attn.embed_dim == 160
    assert decoder.query_context_attn.num_heads == decoder.memory_attn.num_heads == 4
    assert decoder.memory_attn.kdim == decoder.memory_attn.vdim == 256
    assert (decoder.linear1.in_features, decoder.linear1.out_features, decoder.linear2.out_features) == (160, 392, 160)
    assert [(layer.in_features, layer.out_features) for layer in qcx.actor_head.modules()
            if isinstance(layer, nn.Linear)] == [(160, 256)]


@pytest.mark.parametrize("variant", ["mat_ind", "mappo__deepset_critic", "tmasac", "tmatd3"])
def test_local_actor_heads_keep_agents_independent_and_all_layers_trainable(variant: str) -> None:
    policy = make_policy(variant, "2.5M", use_nop=False)
    head = _local_head(policy)
    inputs = torch.randn(2, 3, 192, requires_grad=True)
    outputs = head(inputs)
    changed = inputs.detach().clone()
    changed[:, 1:] += 3
    torch.testing.assert_close(head(changed)[:, 0], outputs[:, 0], rtol=0, atol=0)
    (outputs * torch.randn_like(outputs)).sum().backward()
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in head.parameters())
    assert torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize("variant", ["mat_ind", "mappo__deepset_critic", "tmasac", "tmatd3"])
def test_local_actor_heads_preserve_the_residual_stream_when_updates_are_zero(variant: str) -> None:
    head = _local_head(make_policy(variant, "2.5M", use_nop=False))
    for layer in head.layers:
        for module in layer.feedforward.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
    inputs = torch.randn(2, 3, 192)
    torch.testing.assert_close(head(inputs), nn.functional.layer_norm(inputs, (192,)), rtol=0, atol=0)


def test_local_encoder_replacements_learn_without_mixing_agent_observations() -> None:
    encoder = MATEncoder(
        MATEncoderConfig(
            d_model=8, nhead=2, num_layers=2, use_agent_attention=False,
            transformer_ff_config=MLPConfig(hidden_dims=[16, 16]),
            attention_replacement_config=MLPConfig(hidden_dims=[16]),
        ),
        max_agents=3, local_obs_dim=5, global_obs_dim=0,
    )
    inputs = torch.randn(2, 3, 5)
    globals_ = torch.empty(2, 0)
    outputs = encoder(inputs, globals_)
    changed = inputs.clone()
    changed[:, 1:] += 3
    torch.testing.assert_close(encoder(changed, globals_)[:, 0], outputs[:, 0], rtol=0, atol=0)
    (outputs * torch.randn_like(outputs)).sum().backward()
    for layer in encoder.layers:
        assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in layer.attention_replacement.parameters())


@pytest.mark.parametrize("scale", list_model_scales())
def test_each_value_and_q_head_has_the_same_architecture_and_parameter_budget(scale: str) -> None:
    with torch.device("meta"):
        value = make_policy("mat_ind", scale, use_nop=False).critic
        critic = make_policy("tmasac", scale, use_nop=False).critic
    value_shapes = {name: parameter.shape for name, parameter in value.named_parameters()}
    for head in (critic.q1, critic.q2):
        assert _parameters(value) == _parameters(head)
        assert {name: parameter.shape for name, parameter in head.named_parameters()} == value_shapes
