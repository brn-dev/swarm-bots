from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from swarmbots.learn.algos.r_mat.temporal_sequence_model import LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig
from swarmbots.learn.algos.sac import RecurrentTMASACPolicy, SegmentTMASACPolicy, TMASACPolicy
from swarmbots.learn.algos.off_policy.actor_state_critic import ActorStateCriticInputConfig
from tests.test_recurrent_tmasac import (
    _DummyScenarioEnv, _encoder_config, _policy_config, _scenario_encoder_config, _scenario_inputs, _segment_policy_config,
)
from tests.test_tmasac_policy import _make_config


DEVICES = ["cpu", pytest.param("cuda", marks=[
    pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
])]


def _assert_matching_parameter_gradients(policy: torch.nn.Module, reference: torch.nn.Module) -> None:
    for (name, parameter), (_, reference_parameter) in zip(
        policy.named_parameters(), reference.named_parameters(), strict=True,
    ):
        if reference_parameter.grad is None:
            assert parameter.grad is None, name
        else:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, msg=name)


def _scenario_sequence_policy(kind: str, device: str) -> RecurrentTMASACPolicy | SegmentTMASACPolicy:
    if kind == "segment":
        return SegmentTMASACPolicy(_DummyScenarioEnv(), _segment_policy_config(
            scenario_encoder_config=_scenario_encoder_config(),
        )).to(device)
    config = _policy_config(
        _encoder_config(LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig()),
        recurrent_critic=kind == "recurrent", scenario_encoder_config=_scenario_encoder_config(),
    )
    if kind == "actor_state":
        config = replace(config, actor_state_critic_input_config=ActorStateCriticInputConfig(projection_dim=4))
    return RecurrentTMASACPolicy(_DummyScenarioEnv(), config).to(device)


@pytest.mark.parametrize("device", DEVICES)
def test_scenario_actor_single_step_matches_one_step_sequence(device: str) -> None:
    torch.manual_seed(7)
    policy = _scenario_sequence_policy("recurrent", device)
    reference = deepcopy(policy)
    observations = {key: value.to(device) for key, value in _scenario_inputs(sequence_length=None).items()}
    time_mask = torch.tensor([[True], [False]], device=device)
    observations["global_obs"][1] = torch.nan
    actor_inputs = {key: observations[key] for key in ("local_obs", "global_obs", "agent_mask", "scenario_ids")}
    sequence_inputs = {key: value.unsqueeze(1) for key, value in actor_inputs.items()}

    expected, _ = reference.encode_actor_sequence(**sequence_inputs, initial_state=None, time_mask=time_mask)
    actual, _ = policy.encode_actor_sequence(**actor_inputs, initial_state=None, time_mask=time_mask)

    torch.testing.assert_close(actual, expected.squeeze(1))
    expected[time_mask].square().sum().backward()
    actual[time_mask.squeeze(1)].square().sum().backward()
    _assert_matching_parameter_gradients(policy, reference)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("critic_kind", ["recurrent", "feedforward", "actor_state"])
@pytest.mark.parametrize("with_agent_mask", [False, True])
def test_scenario_critic_single_step_matches_one_step_sequence(
    device: str, critic_kind: str, with_agent_mask: bool,
) -> None:
    torch.manual_seed(6)
    policy = _scenario_sequence_policy(critic_kind, device)
    reference = deepcopy(policy)
    observations = {key: value.to(device) for key, value in _scenario_inputs(sequence_length=None).items()}
    time_mask = torch.tensor([[True], [False]], device=device)
    for field in ("global_obs", "hidden_local_vars", "hidden_global_vars"):
        observations[field][1] = torch.nan
    if not with_agent_mask:
        observations["agent_mask"] = None
    if critic_kind == "actor_state":
        observations["actor_state"] = torch.randn(2, _DummyScenarioEnv.n_agents, 16, device=device)
    sequence_inputs = {key: None if value is None else value.unsqueeze(1) for key, value in observations.items()}

    expected = reference.q_values_sequence(**sequence_inputs, target=False, time_mask=time_mask)[:2]
    actual = policy.q_values_sequence(**observations, target=False, time_mask=time_mask)[:2]

    for value, reference_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(value[time_mask.squeeze(1)], reference_value[time_mask])
    sum(value[time_mask].square().sum() for value in expected).backward()
    sum(value[time_mask.squeeze(1)].square().sum() for value in actual).backward()
    _assert_matching_parameter_gradients(policy, reference)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("recurrent", [False, True])
def test_scenario_critics_ignore_inactive_hidden_inputs_in_parameter_gradients(device: str, recurrent: bool) -> None:
    torch.manual_seed(6)
    scenario_config = _scenario_encoder_config()
    if recurrent:
        policy = RecurrentTMASACPolicy(_DummyScenarioEnv(), _policy_config(
            _encoder_config(LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig()),
            recurrent_critic=True, scenario_encoder_config=scenario_config,
        ))
    else:
        policy = TMASACPolicy(_DummyScenarioEnv(), replace(_make_config(), scenario_encoder_config=scenario_config))
    policy.to(device)
    reference = deepcopy(policy)
    observations = {
        key: value.to(device) for key, value in _scenario_inputs(sequence_length=3 if recurrent else None).items()
    }
    observations["agent_mask"][..., -1] = False
    invalid = ~observations["agent_mask"].unsqueeze(-1)
    clean = {**observations, "hidden_local_vars": observations["hidden_local_vars"].masked_fill(invalid, 0)}
    padded = {**observations, "hidden_local_vars": observations["hidden_local_vars"].masked_fill(invalid, torch.nan)}
    if recurrent:
        expected = reference.q_values_sequence(**clean, target=False)[:2]
        actual = policy.q_values_sequence(**padded, target=False)[:2]
    else:
        expected = reference.q_values(**clean)
        actual = policy.q_values(**padded)
    for value, reference_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference_value)
    sum(value.square().sum() for value in expected).backward()
    sum(value.square().sum() for value in actual).backward()
    _assert_matching_parameter_gradients(policy, reference)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("critic_kind", ["recurrent", "feedforward", "actor_state", "segment"])
@pytest.mark.parametrize("field", ["global_obs", "hidden_local_vars", "hidden_global_vars"])
@pytest.mark.parametrize("with_agent_mask", [False, True])
@pytest.mark.parametrize("num_action_samples", [1, 2])
def test_scenario_critics_ignore_timestep_padding_in_parameter_gradients(
    device: str, critic_kind: str, field: str, with_agent_mask: bool, num_action_samples: int,
) -> None:
    torch.manual_seed(6)
    policy = _scenario_sequence_policy(critic_kind, device)
    reference = deepcopy(policy)
    observations = {key: value.to(device) for key, value in _scenario_inputs(sequence_length=4).items()}
    time_mask = torch.tensor([[True, False, True, False], [False, True, True, False]], device=device)
    tensor = observations[field]
    invalid = ~time_mask.reshape(*time_mask.shape, *([1] * (tensor.ndim - time_mask.ndim)))
    clean = {**observations, field: tensor.masked_fill(invalid, 0)}
    padded = {**observations, field: tensor.masked_fill(invalid, torch.nan)}
    kwargs = {"target": False, "time_mask": time_mask}
    if not with_agent_mask:
        clean["agent_mask"] = padded["agent_mask"] = None
    if critic_kind == "actor_state":
        kwargs["actor_state"] = torch.randn(2, 4, _DummyScenarioEnv.n_agents, 16, device=device)
    valid = time_mask
    if num_action_samples > 1:
        actions = observations["actions"].unsqueeze(0).repeat(num_action_samples, 1, 1, 1, 1)
        actions[1] *= 0.5
        clean["actions"] = padded["actions"] = actions
        kwargs["num_action_samples"] = num_action_samples
        valid = time_mask.unsqueeze(0).expand(num_action_samples, -1, -1)
    expected = reference.q_values_sequence(**clean, **kwargs)[:2]
    actual = policy.q_values_sequence(**padded, **kwargs)[:2]
    for value, reference_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(value[valid], reference_value[valid])
    sum(value[valid].square().sum() for value in expected).backward()
    sum(value[valid].square().sum() for value in actual).backward()
    _assert_matching_parameter_gradients(policy, reference)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("with_agent_mask", [False, True])
@pytest.mark.parametrize("kind", ["recurrent", "segment"])
def test_scenario_actor_ignores_timestep_padding_in_parameter_gradients(
    device: str, with_agent_mask: bool, kind: str,
) -> None:
    torch.manual_seed(7)
    policy = _scenario_sequence_policy(kind, device)
    reference = deepcopy(policy)
    observations = {key: value.to(device) for key, value in _scenario_inputs(sequence_length=4).items()}
    time_mask = torch.tensor([[True, False, True, False], [False, True, True, False]], device=device)
    common = {"local_obs": observations["local_obs"], "scenario_ids": observations["scenario_ids"],
              "agent_mask": observations["agent_mask"] if with_agent_mask else None,
              "initial_state": None, "time_mask": time_mask}
    clean = observations["global_obs"].masked_fill(~time_mask.unsqueeze(-1), 0)
    padded = observations["global_obs"].masked_fill(~time_mask.unsqueeze(-1), torch.nan)
    expected, _ = reference.encode_actor_sequence(global_obs=clean, **common)
    actual, _ = policy.encode_actor_sequence(global_obs=padded, **common)
    torch.testing.assert_close(actual[time_mask], expected[time_mask])
    expected[time_mask].square().sum().backward()
    actual[time_mask].square().sum().backward()
    _assert_matching_parameter_gradients(policy, reference)
