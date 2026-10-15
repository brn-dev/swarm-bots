"""Behavioral contracts for sharing set encoders between clipped double-Q heads."""

from collections.abc import Iterator
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch

from swarmbots.learn.algos.off_policy.joint_critic import JointCriticConfig, JointTwinCritic
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
        patch.object(critic.encoder2, "forward_with_latents", wraps=critic.encoder2.forward_with_latents)
        if independent_encoders else nullcontext()
    )
    with (
        patch.object(critic.encoder, "forward_with_latents", wraps=critic.encoder.forward_with_latents) as encode,
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
