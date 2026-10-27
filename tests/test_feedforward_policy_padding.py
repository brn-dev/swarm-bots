from collections.abc import Iterator
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples
from tests.test_model_scale import make_policy


@pytest.fixture(scope="module", autouse=True)
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("variant", ["ppo", "mappo__deepset_critic", "mat_ind", "mat_dec", "mat_qcx", "mat_orig"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("padding", [torch.nan, torch.inf])
def test_inactive_padding_preserves_active_actions_and_gradients(
    variant: str, training: bool, padding: float,
) -> None:
    torch.manual_seed(42)
    policy = make_policy(variant, "2.5M", use_nop=False).train(training)
    reference = deepcopy(policy)
    active = torch.tensor([[True, True, False], [True, True, True], [False, False, False]])
    local_obs = torch.randn(3, 3, 5)
    clean = local_obs.masked_fill(~active.unsqueeze(-1), 0.0).requires_grad_()
    padded = local_obs.masked_fill(~active.unsqueeze(-1), padding).requires_grad_()
    global_obs = torch.randn(3, 2)
    hidden_local_vars = torch.randn(3, 3, 2)
    clean_hidden = hidden_local_vars.masked_fill(~active.unsqueeze(-1), 0.0).requires_grad_()
    padded_hidden = hidden_local_vars.masked_fill(~active.unsqueeze(-1), padding).requires_grad_()
    hidden_global_vars = torch.randn(3, 1)
    with torch.no_grad():
        expected = reference.act(clean, global_obs, agent_mask=active, deterministic=True)
        actual = policy.act(padded, global_obs, agent_mask=active, deterministic=True)
        torch.testing.assert_close(actual[active], expected[active])
        expected_values = reference.predict_values(clean, global_obs, clean_hidden, hidden_global_vars, active)
        values = policy.predict_values(padded, global_obs, padded_hidden, hidden_global_vars, active)
        torch.testing.assert_close(values, expected_values)
    if training:
        batch = PPOSamples(
            local_obs=clean, global_obs=global_obs, hidden_local_vars=clean_hidden,
            hidden_global_vars=hidden_global_vars, agent_mask=active, previous_actions=None,
            actions=expected, log_probs=torch.zeros(3, 3), values=torch.zeros(3),
            returns=torch.zeros(3), advantages=torch.ones(3),
        )
        expected_log_probs, expected_values, _, _ = reference.evaluate_actions(batch)
        log_probs, values, _, _ = policy.evaluate_actions(
            replace(batch, local_obs=padded, hidden_local_vars=padded_hidden),
        )
        torch.testing.assert_close(log_probs[active], expected_log_probs[active])
        torch.testing.assert_close(values, expected_values)
        (expected_log_probs[active].square().sum() + expected_values.square().sum()).backward()
        (log_probs[active].square().sum() + values.square().sum()).backward()
        torch.testing.assert_close(padded.grad, clean.grad)
        torch.testing.assert_close(padded_hidden.grad, clean_hidden.grad)
        assert not padded.grad[~active].count_nonzero()
        assert not padded_hidden.grad[~active].count_nonzero()
        for (name, parameter), (_, expected_parameter) in zip(
            policy.named_parameters(), reference.named_parameters(), strict=True,
        ):
            torch.testing.assert_close(parameter.grad, expected_parameter.grad, msg=name)


@pytest.mark.parametrize("joint_obs_embedding", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[
    pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
])])
def test_encoder_excludes_nonfinite_padding_before_projection(joint_obs_embedding: bool, device: str) -> None:
    encoder = MATEncoder(
        MATEncoderConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16,
                         joint_obs_embedding=joint_obs_embedding),
        max_agents=3, local_obs_dim=5, global_obs_dim=2,
    ).to(device)
    reference = deepcopy(encoder)
    active = torch.tensor([[True, False, True], [False, False, False]], device=device)
    local_obs = torch.randn(2, 3, 5, device=device)
    clean = local_obs.masked_fill(~active.unsqueeze(-1), 0.0).requires_grad_()
    padded = local_obs.masked_fill(~active.unsqueeze(-1), torch.nan).requires_grad_()
    global_obs = torch.randn(2, 2, device=device)
    expected = reference(clean, global_obs, agent_mask=active)
    actual = encoder(padded, global_obs, agent_mask=active)
    torch.testing.assert_close(actual, expected)
    expected[active].square().sum().backward()
    actual[active].square().sum().backward()
    torch.testing.assert_close(padded.grad, clean.grad)
    for (name, parameter), (_, expected_parameter) in zip(
        encoder.named_parameters(), reference.named_parameters(), strict=True,
    ):
        torch.testing.assert_close(parameter.grad, expected_parameter.grad, msg=name)
