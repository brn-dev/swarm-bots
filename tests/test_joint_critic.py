"""Behavioral contracts for sharing set encoders between clipped double-Q heads."""

from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.off_policy.joint_critic import JointCriticConfig, JointQNetwork, JointTwinCritic
from swarmbots.learn.algos.off_policy.transformer_critic import TransformerCriticConfig, TransformerTwinCritic
from swarmbots.learn.nn_components.deep_set import DeepSetInteractionConfig


@pytest.fixture(scope="module", autouse=True)
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_critic(independent_encoders: bool = False) -> JointTwinCritic:
    return JointTwinCritic(
        n_agents=3, local_obs_dim=2, global_obs_dim=1, hidden_local_vars_dim=1,
        hidden_global_vars_dim=1, action_dim=1,
        config=JointCriticConfig(
            kind="deepset", hidden_dims=(8,), element_hidden_dims=(8,),
            independent_encoders=independent_encoders,
            interaction_config=DeepSetInteractionConfig(num_layers=2),
        ),
    )


def observations() -> dict[str, torch.Tensor]:
    mask = torch.tensor([[True, True, False], [False, False, False]])
    result = {
        "local_obs": torch.randn(2, 3, 2), "global_obs": torch.randn(2, 1),
        "hidden_local_vars": torch.randn(2, 3, 1), "hidden_global_vars": torch.randn(2, 1),
        "actions": torch.randn(2, 3, 1), "agent_mask": mask,
    }
    for key in ("local_obs", "hidden_local_vars", "actions"):
        result[key][~mask] = float("nan")
    return result


@pytest.mark.parametrize("independent_encoders", [False, True])
def test_twin_critic_encodes_each_trunk_once_and_nop_uses_the_same_features(independent_encoders: bool) -> None:
    critic = make_critic(independent_encoders)
    inputs = observations()
    secondary_encoding = (
        patch.object(critic.encoder2, "encode", wraps=critic.encoder2.encode)
        if independent_encoders else nullcontext()
    )
    with (
        patch.object(critic.encoder, "encode", wraps=critic.encoder.encode) as encode,
        secondary_encoding as encode2,
    ):
        q1, q2, latents = critic(**inputs)
    assert encode.call_count == 1
    if independent_encoders:
        assert encode2.call_count == 1
    assert q1.shape == q2.shape == (2,)
    assert torch.isfinite(q1).all() and torch.isfinite(q2).all()
    assert latents.shape == (2, 3, 8 * (2 if independent_encoders else 1))
    assert critic.nop_source_latent_dim == latents.shape[-1]
    assert torch.isfinite(latents).all() and (latents[~inputs["agent_mask"]] == 0).all()
    encoded = critic.encode(
        local_inputs=inputs["local_obs"], global_inputs=inputs["global_obs"],
        **{key: value for key, value in inputs.items() if key not in {"local_obs", "global_obs"}},
    )
    torch.testing.assert_close(encoded, latents)
    permutation = torch.tensor([2, 0, 1])
    permuted = {
        key: value[:, permutation] if key in {"local_obs", "hidden_local_vars", "actions", "agent_mask"} else value
        for key, value in inputs.items()
    }
    permuted_q1, permuted_q2, permuted_latents = critic(**permuted)
    torch.testing.assert_close(permuted_q1, q1)
    torch.testing.assert_close(permuted_q2, q2)
    torch.testing.assert_close(permuted_latents, latents[:, permutation])


@pytest.mark.parametrize("independent_encoders", [False, True])
@pytest.mark.parametrize("head_name", ["q1", "q2"])
def test_each_q_loss_updates_its_head_and_the_expected_encoder(
    independent_encoders: bool, head_name: str,
) -> None:
    critic = make_critic(independent_encoders)
    inputs = observations()
    inputs["actions"].requires_grad_()
    q1, q2, _latents = critic(**inputs)
    (q1 if head_name == "q1" else q2).sum().backward()
    for name in ("q1", "q2"):
        gradients = [parameter.grad for parameter in getattr(critic, name).parameters()]
        if name == head_name:
            assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients)
        else:
            assert all(gradient is None for gradient in gradients)
    for name in ("encoder", "encoder2"):
        encoder = getattr(critic, name)
        if encoder is None:
            continue
        receives_gradient = not independent_encoders or name == ("encoder" if head_name == "q1" else "encoder2")
        gradients = [parameter.grad for parameter in encoder.parameters()]
        if receives_gradient:
            assert any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients)
        else:
            assert all(gradient is None for gradient in gradients)
    assert torch.isfinite(inputs["actions"].grad).all()
    assert (inputs["actions"].grad[~inputs["agent_mask"]] == 0).all()
    assert inputs["actions"].grad[inputs["agent_mask"]].abs().sum() > 0


def test_independent_encoder_ablation_adds_one_trunk_and_retains_independent_q_heads() -> None:
    shared = make_critic()
    independent = make_critic(True)
    def parameter_count(module: torch.nn.Module) -> int:
        return sum(parameter.numel() for parameter in module.parameters())

    assert parameter_count(independent) - parameter_count(shared) == parameter_count(shared.encoder)
    assert {id(parameter) for parameter in shared.q1.parameters()}.isdisjoint(
        id(parameter) for parameter in shared.q2.parameters()
    )


def aligned_config(*, gated: bool = False, joint_obs_embedding: bool = False) -> JointCriticConfig:
    encoder_config = MATEncoderConfig(
        d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
        normalize_tokens=True, joint_obs_embedding=joint_obs_embedding,
    )
    return JointCriticConfig(
        kind="deepset", encoder_config=encoder_config,
        interaction_config=DeepSetInteractionConfig(message_dim=40, gated=gated),
        head_config=TransformerCriticConfig(
            action_coembed_hidden_dims=[32], local_projection_hidden_dims=[16],
            value_regressor_hidden_dims=[16, 16],
            interaction_config=DeepSetInteractionConfig(message_dim=20, gated=gated),
        ),
    )


@pytest.mark.parametrize("gated", [False, True])
def test_encoder_communication_layers_have_independent_initial_values(gated: bool) -> None:
    config = aligned_config(gated=gated)
    config = replace(config, encoder_config=replace(config.encoder_config, transformer_ff_init_gain=1.0))
    critic = JointTwinCritic(config=config, **critic_dimensions())
    first, second = critic.encoder.network.layers
    first_parameters = dict(first.interaction.named_parameters())
    for name, parameter in second.interaction.named_parameters():
        assert parameter is not first_parameters[name]
        if parameter.ndim == 2:
            assert not torch.equal(parameter, first_parameters[name]), name


def critic_dimensions() -> dict[str, int]:
    return dict(
        n_agents=3, local_obs_dim=2, global_obs_dim=1, hidden_local_vars_dim=1,
        hidden_global_vars_dim=1, action_dim=1,
    )


@pytest.mark.parametrize("gated", [False, True])
def test_single_q_uses_the_same_trunk_and_private_readout_as_twin_q(gated: bool) -> None:
    config = aligned_config(gated=gated)
    single = JointQNetwork(config=config, **critic_dimensions())
    twin = JointTwinCritic(config=config, **critic_dimensions())
    single.encoder.load_state_dict(twin.encoder.state_dict())
    single.network.load_state_dict(twin.q1.state_dict())
    inputs = observations()
    single_q, single_latents = single.forward_with_latents(**inputs)
    twin_q, _q2, twin_latents = twin(**inputs)
    torch.testing.assert_close(single_q, twin_q)
    torch.testing.assert_close(single_latents, twin_latents)
    assert sum(parameter.numel() for parameter in twin.parameters()) == (
        sum(parameter.numel() for parameter in single.parameters())
        + sum(parameter.numel() for parameter in twin.q2.parameters())
    )
    assert len(single.network.deepset.interactions) == 2
    assert single.nop_source_latent_dim == config.encoder_config.d_model
    with torch.no_grad():
        for parameter in single.network.parameters():
            parameter.add_(0.3)
    _changed_q, unchanged_latents = single.forward_with_latents(**inputs)
    torch.testing.assert_close(unchanged_latents, single_latents)


@pytest.mark.parametrize("joint_obs_embedding", [False, True])
def test_transformer_and_deepset_share_every_component_except_the_communication_operator(
    joint_obs_embedding: bool,
) -> None:
    config = aligned_config(joint_obs_embedding=joint_obs_embedding)
    deepset = JointTwinCritic(config=config, **critic_dimensions())
    transformer = TransformerTwinCritic(
        n_agents=3, max_agents=3, local_input_dim=2, global_input_dim=1,
        hidden_local_vars_dim=1, hidden_global_vars_dim=1, action_dim=1,
        encoder_config=config.encoder_config, critic_config=config.head_config,
        act_fn_cls=torch.nn.GELU, dropout=0.0,
    )
    deepset_outer = {
        key: value for key, value in deepset.encoder.network.state_dict().items()
        if ".interaction." not in key
    }
    transformer_outer = {
        key: value for key, value in transformer.encoder.state_dict().items()
        if ".self_attn." not in key
    }
    assert {key: value.shape for key, value in deepset_outer.items()} == {
        key: value.shape for key, value in transformer_outer.items()
    }
    deepset.encoder.network.load_state_dict(transformer_outer, strict=False)
    deepset.q1.load_state_dict(transformer.q1.state_dict())
    deepset.q2.load_state_dict(transformer.q2.state_dict())
    with torch.no_grad():
        for layer in deepset.encoder.network.layers:
            layer.interaction.message_projection.weight.zero_()
        for layer in transformer.encoder.layers:
            layer.self_attn.out_proj.weight.zero_()
            layer.self_attn.out_proj.bias.zero_()
    inputs = observations()
    deepset_outputs = deepset(**inputs)
    transformer_outputs = transformer(**inputs)
    for actual, expected in zip(deepset_outputs, transformer_outputs, strict=True):
        torch.testing.assert_close(actual, expected)
    deepset_count = sum(parameter.numel() for parameter in deepset.encoder.parameters())
    transformer_count = sum(parameter.numel() for parameter in transformer.encoder.parameters())
    assert abs(deepset_count / transformer_count - 1) < 0.02


@pytest.mark.parametrize("context_in_elements", [False, True])
def test_global_context_placement_is_explicit_and_preserves_global_q_conditioning(context_in_elements: bool) -> None:
    config = aligned_config()
    config = replace(config, context_in_elements=context_in_elements)
    critic = JointTwinCritic(config=config, **critic_dimensions())
    inputs = observations()
    initial_q, _q2, initial_latents = critic(**inputs)
    modified = {**inputs, "global_obs": inputs["global_obs"] + 4.0}
    modified_q, _q2, modified_latents = critic(**modified)
    assert not torch.allclose(initial_q[0], modified_q[0])
    if context_in_elements:
        assert not torch.allclose(initial_latents[0], modified_latents[0])
        assert critic.q1.num_global_features == 0
    else:
        torch.testing.assert_close(initial_latents, modified_latents)
        assert critic.q1.num_global_features == 2


@pytest.mark.parametrize("single_q", [False, True])
@pytest.mark.parametrize("joint_obs_embedding", [False, True])
def test_scalar_global_count_conditions_q_values_and_receives_gradients(
    single_q: bool, joint_obs_embedding: bool,
) -> None:
    torch.manual_seed(42)
    config = aligned_config(joint_obs_embedding=joint_obs_embedding)
    critic_class = JointQNetwork if single_q else JointTwinCritic
    critic = critic_class(config=config, **{**critic_dimensions(), "global_obs_dim": 0})
    counts = torch.tensor([[1.0], [3.0]], requires_grad=True)
    inputs = dict(
        local_obs=torch.ones(2, 3, 2), global_obs=torch.empty(2, 0),
        hidden_local_vars=torch.ones(2, 3, 1), hidden_global_vars=counts,
        actions=torch.ones(2, 3, 1),
        agent_mask=torch.tensor([[True, False, False], [True, True, True]]),
    )

    if single_q:
        q_values = (critic(**inputs),)
    else:
        q1, q2, _latents = critic(**inputs)
        q_values = (q1, q2)

    for values in q_values:
        assert not torch.isclose(values[0], values[1])
    gradient = torch.autograd.grad(sum(values.sum() for values in q_values), counts)[0]
    assert torch.isfinite(gradient).all() and (gradient.abs() > 0).all()


@pytest.mark.parametrize("gated", [False, True])
def test_action_and_observation_gradients_survive_both_communication_and_readout(gated: bool) -> None:
    critic = JointTwinCritic(config=aligned_config(gated=gated), **critic_dimensions())
    inputs = observations()
    feature_names = ("local_obs", "global_obs", "hidden_local_vars", "hidden_global_vars", "actions")
    for name in feature_names:
        inputs[name].requires_grad_()
    q1, q2, latents = critic(**inputs)
    (q1.sum() + q2.sum()).backward()
    for name in feature_names:
        gradients = inputs[name].grad
        assert torch.isfinite(gradients).all()
        assert gradients[0].abs().sum() > 0
        if gradients.ndim == 3:
            assert (gradients[~inputs["agent_mask"]] == 0).all()
    assert latents.shape[-1] == critic.nop_source_latent_dim
