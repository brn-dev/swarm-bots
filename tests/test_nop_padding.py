from collections.abc import Iterator
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from swarmbots.learn.algos.off_policy.nop import SACNOPConfig, SACNOPModule, SACNOPSequenceBatch
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import RecurrentTMASACPolicy
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig, PredictDeltaMode
from tests.test_recurrent_tmasac import (
    _DummyContinuousEnv, _actor_state_critic_inputs, _shared_recurrent_policy_config,
)


@pytest.fixture(scope="module", autouse=True)
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _assert_matching_gradients(actual: nn.Module, reference: nn.Module) -> None:
    for (name, parameter), (_, expected) in zip(actual.named_parameters(), reference.named_parameters(), strict=True):
        if expected.grad is None:
            assert parameter.grad is None, name
        else:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, expected.grad, msg=name)


def _nop_module(
    device: str, *, skip_first_transition: bool, target_config: NextObsPredConfig | None = None,
) -> SACNOPModule:
    return SACNOPModule(
        n_agents=3, source_latent_dim=6, action_dim=2, name="test", skip_first_transition=skip_first_transition,
        config=SACNOPConfig(
            enabled=True, nop_latent_dim=8, transition_model_d_model=8, transition_model_nhead=2,
            transition_model_num_layers=1, transition_model_dim_feedforward=16,
            next_obs_pred_config=target_config or NextObsPredConfig(local_scalar_target_indices=[0], predict_delta=False),
        ),
    ).to(device)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("compiled", [False, True])
def test_precomputed_shared_padding_is_safe_through_actual_nop_loss(device: str, compiled: bool) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(13)
    policy = RecurrentTMASACPolicy(_DummyContinuousEnv(), _shared_recurrent_policy_config()).to(device)
    reference = deepcopy(policy)
    inputs = {name: value.to(device) for name, value in _actor_state_critic_inputs(sequence_length=3).items()}
    mask = inputs["agent_mask"]
    mask[..., -1] = False
    train_mask = torch.ones(2, 3, 2, dtype=torch.bool, device=device)
    train_mask[1, -1] = False
    with torch.no_grad():
        latents, state = policy.shared_observation_encoder(inputs["local_obs"], inputs["global_obs"], agent_mask=mask)
    valid_source = mask & train_mask.any(dim=-1, keepdim=True)
    clean = latents.masked_fill(~valid_source.unsqueeze(-1), 0.0).requires_grad_()
    padded = latents.masked_fill(~valid_source.unsqueeze(-1), torch.nan).requires_grad_()
    batch = SACNOPSequenceBatch(
        local_obs=inputs["local_obs"], global_obs=inputs["global_obs"], agent_mask=mask,
        actions=inputs["actions"].unsqueeze(2).expand(-1, -1, 2, -1, -1),
        next_local_obs=(inputs["local_obs"] + 0.1).unsqueeze(2).expand(-1, -1, 2, -1, -1),
        next_global_obs=None, next_agent_mask=mask.unsqueeze(2).expand(-1, -1, 2, -1), train_mask=train_mask,
    )
    if compiled:
        torch.compiler.reset()
        policy.critic_nop._compute_next_obs_pred_loss_fn = torch.compile(
            policy.critic_nop._compute_next_obs_pred_loss_impl, backend="aot_eager", fullgraph=True,
        )
    try:
        expected_q = reference.q_values_sequence(
            **inputs, target=False, time_mask=train_mask.any(dim=-1),
            precomputed_shared_latents=clean, precomputed_shared_state=state,
        )
        actual_q = policy.q_values_sequence(
            **inputs, target=False, time_mask=train_mask.any(dim=-1),
            precomputed_shared_latents=padded, precomputed_shared_state=state,
        )
        assert expected_q[2] is clean and actual_q[2] is padded
        for actual, expected in zip(actual_q[:2], expected_q[:2], strict=True):
            torch.testing.assert_close(actual, expected)
        expected_loss, _ = reference.compute_critic_nop_loss_from_latents(source_latents=expected_q[2], batch=batch)
        actual_loss, _ = policy.compute_critic_nop_loss_from_latents(source_latents=actual_q[2], batch=batch)
        torch.testing.assert_close(actual_loss, expected_loss)
        expected_loss.backward()
        actual_loss.backward()
        _assert_matching_gradients(policy, reference)
        torch.testing.assert_close(padded.grad, clean.grad)
        assert padded.grad[valid_source].abs().sum() > 0
        assert torch.count_nonzero(padded.grad[~valid_source]) == 0
        with torch.no_grad():
            evaluation_loss, _ = policy.compute_critic_nop_loss_from_latents(source_latents=padded, batch=batch)
        torch.testing.assert_close(evaluation_loss, actual_loss.detach())
    finally:
        if compiled:
            torch.compiler.reset()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("skip_first_transition", [False, True])
@pytest.mark.parametrize("case", ["later_episode", "unscored_context", "unscored_window"])
def test_nop_source_mask_tracks_origin_and_retains_unscored_history(
    device: str, skip_first_transition: bool, case: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(81)
    module = _nop_module(device, skip_first_transition=skip_first_transition)
    reference = deepcopy(module)
    masks = {
        "later_episode": [[True, False, False], [True, True, True]],
        "unscored_context": [[True, True, True], [True, False, False]],
        "unscored_window": [[True, True, False], [True, True, False]],
    }
    agent_mask = torch.tensor([[masks[case]]], device=device)
    train_mask = torch.tensor([[{
        "later_episode": [True, False], "unscored_context": [False, True], "unscored_window": [False, False],
    }[case]]], device=device)
    source = torch.randn(1, 1, 3, 6, device=device)
    valid_source = agent_mask[:, :, 0] & train_mask.any(dim=-1, keepdim=True)
    clean = source.masked_fill(~valid_source.unsqueeze(-1), 0.0).requires_grad_()
    padded = source.masked_fill(~valid_source.unsqueeze(-1), torch.nan).requires_grad_()
    batch = SACNOPSequenceBatch(
        local_obs=torch.randn(1, 1, 3, 1, device=device), global_obs=None, agent_mask=agent_mask,
        actions=torch.randn(1, 1, 2, 3, 2, device=device),
        next_local_obs=torch.randn(1, 1, 2, 3, 1, device=device), next_global_obs=None,
        next_agent_mask=agent_mask, train_mask=train_mask,
    )
    expected_loss, _ = reference.compute_loss(source_latents=clean, batch=batch)
    actual_loss, _ = module.compute_loss(source_latents=padded, batch=batch)
    torch.testing.assert_close(actual_loss, expected_loss)
    expected_loss.backward()
    actual_loss.backward()
    _assert_matching_gradients(module, reference)
    torch.testing.assert_close(padded.grad, clean.grad)
    assert torch.count_nonzero(padded.grad[~valid_source]) == 0
    if case == "unscored_context" and not skip_first_transition:
        assert padded.grad[0, 0, 1:].abs().sum() > 0
    with torch.no_grad():
        evaluation_loss, _ = module.compute_loss(source_latents=padded, batch=batch)
    torch.testing.assert_close(evaluation_loss, actual_loss.detach())


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("skip_first_transition", [False, True])
@pytest.mark.parametrize("mask_agents", [False, True])
@pytest.mark.parametrize("time_pattern", ["tail", "context", "unscored"])
def test_nop_ignores_unused_action_tails_and_preserves_unscored_action_context(
    device: str, skip_first_transition: bool, mask_agents: bool, time_pattern: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(82)
    module = _nop_module(device, skip_first_transition=skip_first_transition)
    reference = deepcopy(module)
    train_mask = torch.tensor([[{
        "tail": [True, False, False], "context": [False, False, True], "unscored": [False, False, False],
    }[time_pattern]]], device=device)
    needed_steps = torch.tensor([[{
        "tail": [True, False, False], "context": [True, True, True], "unscored": [False, False, False],
    }[time_pattern]]], device=device)
    actions = torch.randn(1, 1, 3, 3, 2, device=device)
    clean_actions = actions.masked_fill(~needed_steps[..., None, None], 0.0).requires_grad_()
    padded_actions = actions.masked_fill(~needed_steps[..., None, None], torch.nan).requires_grad_()
    agent_mask = torch.ones(1, 1, 3, dtype=torch.bool, device=device) if mask_agents else None
    source = torch.randn(1, 1, 3, 6, device=device, requires_grad=True)
    clean_source = source.detach().clone().requires_grad_()
    batch = SACNOPSequenceBatch(
        local_obs=torch.randn(1, 1, 3, 1, device=device), global_obs=None, agent_mask=agent_mask,
        actions=clean_actions, next_local_obs=torch.randn(1, 1, 3, 3, 1, device=device),
        next_global_obs=None, next_agent_mask=agent_mask, train_mask=train_mask,
    )
    expected_loss, _ = reference.compute_loss(source_latents=clean_source, batch=batch)
    actual_loss, _ = module.compute_loss(source_latents=source, batch=replace(batch, actions=padded_actions))
    torch.testing.assert_close(actual_loss, expected_loss)
    expected_loss.backward()
    actual_loss.backward()
    _assert_matching_gradients(module, reference)
    assert torch.isfinite(source.grad).all()
    torch.testing.assert_close(source.grad, clean_source.grad)
    if clean_actions.grad is not None:
        assert torch.isfinite(padded_actions.grad).all()
        torch.testing.assert_close(padded_actions.grad, clean_actions.grad)
        assert torch.count_nonzero(padded_actions.grad.masked_select(~needed_steps[..., None, None])) == 0
        if time_pattern == "context":
            first_used = int(skip_first_transition)
            assert padded_actions.grad[:, :, first_used].abs().sum() > 0
    with torch.no_grad():
        evaluation_loss, _ = module.compute_loss(source_latents=source, batch=replace(batch, actions=padded_actions))
    torch.testing.assert_close(evaluation_loss, actual_loss.detach())


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("predict_delta", [False, PredictDeltaMode.PER_STEP_DELTA, PredictDeltaMode.INITIAL_DELTA])
def test_all_nop_targets_ignore_padding_without_erasing_delta_predecessors(
    device: str, compiled: bool, predict_delta: bool | PredictDeltaMode,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(83)
    module = _nop_module(device, skip_first_transition=False, target_config=NextObsPredConfig(
        local_scalar_target_indices=[0], local_angle_target_indices=[1], local_rot6d_target_indices=[3],
        local_binary_target_indices=[9], global_scalar_target_indices=[0], global_rot6d_target_indices=[1],
        predict_delta=predict_delta,
    ))
    reference = deepcopy(module)
    agent_mask = torch.tensor([[[True, True, False]] * 2], device=device)
    time_mask = torch.tensor([[[True, False, True, False], [False, False, False, False]]], device=device)
    target_valid = agent_mask.unsqueeze(2) & time_mask.unsqueeze(-1)
    needed_local_targets = target_valid.clone()
    needed_global_targets = time_mask.clone()
    if predict_delta == PredictDeltaMode.PER_STEP_DELTA:
        needed_local_targets[:, :, :-1] |= target_valid[:, :, 1:]
        needed_global_targets[:, :, :-1] |= time_mask[:, :, 1:]
    origin_valid = target_valid[:, :, 0]
    global_origin_valid = time_mask[:, :, 0]
    if predict_delta == PredictDeltaMode.INITIAL_DELTA:
        origin_valid = target_valid.any(dim=2)
        global_origin_valid = time_mask.any(dim=2)
    observations = {
        "local_obs": (torch.randn(1, 2, 3, 10, device=device), origin_valid),
        "global_obs": (torch.randn(1, 2, 7, device=device), global_origin_valid),
        "next_local_obs": (torch.randn(1, 2, 4, 3, 10, device=device), needed_local_targets),
        "next_global_obs": (torch.randn(1, 2, 4, 7, device=device), needed_global_targets),
    }
    observations["next_local_obs"][0][..., 9] = torch.rand(1, 2, 4, 3, device=device)
    clean_obs = {name: value.masked_fill(~valid.unsqueeze(-1), 0.0) for name, (value, valid) in observations.items()}
    padded_obs = {
        name: value.masked_fill(~valid.unsqueeze(-1), torch.nan) for name, (value, valid) in observations.items()
    }
    source = torch.randn(1, 2, 3, 6, device=device, requires_grad=True)
    clean_source = source.detach().clone().requires_grad_()
    actions = torch.randn(1, 2, 4, 3, 2, device=device, requires_grad=True)
    clean_actions = actions.detach().clone().requires_grad_()
    batch = SACNOPSequenceBatch(
        **clean_obs, agent_mask=agent_mask, actions=clean_actions,
        next_agent_mask=agent_mask, train_mask=time_mask,
    )
    if compiled:
        torch.compiler.reset()
        module._compute_next_obs_pred_loss_fn = torch.compile(
            module._compute_next_obs_pred_loss_impl, backend="aot_eager", fullgraph=True,
        )
    try:
        expected_loss, expected_metrics = reference.compute_loss(source_latents=clean_source, batch=batch)
        actual_loss, actual_metrics = module.compute_loss(
            source_latents=source, batch=replace(batch, **padded_obs, actions=actions),
        )
        torch.testing.assert_close(actual_loss, expected_loss)
        for name, expected in expected_metrics.items():
            torch.testing.assert_close(actual_metrics[name], expected, msg=name)
        expected_loss.backward()
        actual_loss.backward()
        _assert_matching_gradients(module, reference)
        for actual, expected in ((source, clean_source), (actions, clean_actions)):
            assert torch.isfinite(actual.grad).all()
            torch.testing.assert_close(actual.grad, expected.grad)
        torch.testing.assert_close(module.binary_target_ema, reference.binary_target_ema)
        module.eval()
        with torch.no_grad():
            evaluation_loss, _ = module.compute_loss(
                source_latents=source, batch=replace(batch, **padded_obs, actions=actions),
            )
        torch.testing.assert_close(evaluation_loss, actual_loss.detach())
    finally:
        if compiled:
            torch.compiler.reset()
