from collections.abc import Iterator
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig
from swarmbots.learn.algos.mat_orig import MATOrigDecoder, MATOrigDecoderConfig, MATOrigEncoder, MATOrigEncoderConfig
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


@pytest.mark.parametrize("padding", [torch.nan, torch.inf])
@pytest.mark.parametrize("global_obs_dim", [0, 2])
def test_original_mat_encoder_padding_preserves_outputs_and_gradients(padding: float, global_obs_dim: int) -> None:
    torch.manual_seed(7)
    encoder = MATOrigEncoder(
        MATOrigEncoderConfig(d_model=8, nhead=2, num_layers=1),
        max_agents=3, local_obs_dim=3, global_obs_dim=global_obs_dim,
    )
    reference = deepcopy(encoder)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    local_obs = torch.randn(2, 3, 3)
    global_obs = torch.randn(2, global_obs_dim)
    clean = local_obs.masked_fill(~mask.unsqueeze(-1), 0.0).requires_grad_()
    padded = local_obs.masked_fill(~mask.unsqueeze(-1), padding).requires_grad_()
    empty_rows = ~mask.any(dim=1, keepdim=True)
    clean_global = global_obs.masked_fill(empty_rows, 0.0).requires_grad_()
    padded_global = global_obs.masked_fill(empty_rows, padding).requires_grad_()
    original_local, original_global = padded.detach().clone(), padded_global.detach().clone()

    expected = reference(clean, clean_global, agent_mask=mask)
    output = encoder(padded, padded_global, agent_mask=mask)
    torch.testing.assert_close(output, expected)
    assert torch.isfinite(output).all() and not output[~mask].count_nonzero()
    loss_weights = torch.randn_like(output)
    (expected * loss_weights).sum().backward()
    (output * loss_weights).sum().backward()

    torch.testing.assert_close(padded.grad, clean.grad)
    assert not padded.grad[~mask].count_nonzero()
    torch.testing.assert_close(padded_global.grad, clean_global.grad)
    _assert_matching_finite_parameter_gradients(encoder, reference)
    torch.testing.assert_close(padded, original_local, equal_nan=True)
    torch.testing.assert_close(padded_global, original_global, equal_nan=True)


@pytest.mark.parametrize("padding", [torch.nan, torch.inf])
@pytest.mark.parametrize("padded_input", ["actions", "observations", "both"])
def test_original_mat_decoder_padding_preserves_outputs_and_gradients(padding: float, padded_input: str) -> None:
    torch.manual_seed(7)
    decoder = MATOrigDecoder(
        MATOrigDecoderConfig(d_model=8, nhead=2, num_layers=1), max_agents=3, action_input_dim=2,
    )
    reference = deepcopy(decoder)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    actions = torch.randn(2, 3, 2)
    observations = torch.randn(2, 3, 8)
    clean_actions = actions.masked_fill(~mask.unsqueeze(-1), 0.0).requires_grad_()
    clean_observations = observations.masked_fill(~mask.unsqueeze(-1), 0.0).requires_grad_()
    padded_actions = actions.masked_fill(
        ~mask.unsqueeze(-1), 0.0 if padded_input == "observations" else padding,
    ).requires_grad_()
    padded_observations = observations.masked_fill(
        ~mask.unsqueeze(-1), 0.0 if padded_input == "actions" else padding,
    ).requires_grad_()
    original_actions = padded_actions.detach().clone()
    original_observations = padded_observations.detach().clone()

    expected = reference(clean_actions, clean_observations, agent_mask=mask)
    output = decoder(padded_actions, padded_observations, agent_mask=mask)
    torch.testing.assert_close(output, expected)
    assert torch.isfinite(output).all() and not output[~mask].count_nonzero()
    loss_weights = torch.randn_like(output)
    (expected * loss_weights).sum().backward()
    (output * loss_weights).sum().backward()

    torch.testing.assert_close(padded_actions.grad, clean_actions.grad)
    torch.testing.assert_close(padded_observations.grad, clean_observations.grad)
    assert not padded_actions.grad[~mask].count_nonzero()
    assert not padded_observations.grad[~mask].count_nonzero()
    _assert_matching_finite_parameter_gradients(decoder, reference)
    torch.testing.assert_close(padded_actions, original_actions, equal_nan=True)
    torch.testing.assert_close(padded_observations, original_observations, equal_nan=True)


def _assert_matching_finite_parameter_gradients(model: torch.nn.Module, reference: torch.nn.Module) -> None:
    for (name, parameter), (_, expected_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, expected_parameter.grad, msg=name)
