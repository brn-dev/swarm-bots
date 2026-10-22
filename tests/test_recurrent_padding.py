from copy import deepcopy

import pytest
import torch
from torch import nn

from swarmbots.learn.algos.off_policy.recurrent_transformer_critic import RecurrentTransformerTwinCritic
from swarmbots.learn.algos.off_policy.transformer_critic import TransformerCriticConfig
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig


def _compare_padding_gradients(
    model: nn.Module,
    clean_inputs: dict[str, torch.Tensor | None],
    padded_inputs: dict[str, torch.Tensor | None],
    *,
    encode_only: bool = False,
) -> None:
    reference = deepcopy(model)
    expected_outputs = reference.encode(**clean_inputs) if encode_only else reference(**clean_inputs)
    actual_outputs = model.encode(**padded_inputs) if encode_only else model(**padded_inputs)
    # Temporal histories need not contribute to the training objective.
    expected = expected_outputs[:-1]
    actual = actual_outputs[:-1]
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    sum(tensor.square().sum() for tensor in expected).backward()
    sum(tensor.square().sum() for tensor in actual).backward()
    for (name, parameter), (_, expected_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True,
    ):
        if expected_parameter.grad is None:
            assert parameter.grad is None, name
        else:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, expected_parameter.grad, msg=name)


@pytest.mark.parametrize("joint_obs_embedding", [False, True])
def test_rmat_padding_does_not_change_outputs_or_parameter_gradients(joint_obs_embedding: bool) -> None:
    torch.manual_seed(5)
    model = RMATEncoder(
        RMATEncoderConfig(
            d_model=8, nhead=2, num_layers=1, dim_feedforward=16,
            joint_obs_embedding=joint_obs_embedding,
        ),
        max_agents=3, local_obs_dim=3, global_obs_dim=2,
    )
    agent_mask = torch.tensor([[[True, True, False]] * 3] * 2)
    time_mask = torch.tensor([[True, True, True], [True, True, False]])
    valid = agent_mask & time_mask.unsqueeze(-1)
    local = torch.randn(2, 3, 3, 3)
    global_obs = torch.randn(2, 3, 2)
    clean_inputs = {
        "local_obs": local.masked_fill(~valid.unsqueeze(-1), 0),
        "global_obs": global_obs.masked_fill(~time_mask.unsqueeze(-1), 0),
        "agent_mask": agent_mask,
        "time_mask": time_mask,
    }
    padded_inputs = {
        **clean_inputs,
        "local_obs": local.masked_fill(~valid.unsqueeze(-1), torch.nan),
        "global_obs": global_obs.masked_fill(~time_mask.unsqueeze(-1), torch.nan),
    }
    _compare_padding_gradients(model, clean_inputs, padded_inputs)


@pytest.mark.parametrize("separate_encoders", [False, True])
@pytest.mark.parametrize("independent_encoders", [False, True])
@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("encode_only", [False, True])
def test_recurrent_critic_padding_keeps_gradients_finite(
    separate_encoders: bool, independent_encoders: bool, sequence: bool, encode_only: bool,
) -> None:
    torch.manual_seed(6)
    critic = RecurrentTransformerTwinCritic(
        n_agents=3, max_agents=3, local_input_dim=3, global_input_dim=2,
        hidden_local_vars_dim=1, hidden_global_vars_dim=1, action_dim=2,
        encoder_config=RMATEncoderConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16),
        critic_config=TransformerCriticConfig(
            separate_observation_action_encoders=separate_encoders,
            independent_encoders=independent_encoders,
            action_coembed_hidden_dims=[8], local_projection_hidden_dims=[8],
            value_regressor_hidden_dims=[8],
        ),
        dropout=0,
    )
    prefix = (2, 3) if sequence else (2,)
    agent_mask = torch.ones((*prefix, 3), dtype=torch.bool)
    agent_mask[..., -1] = False
    time_mask = torch.ones(prefix, dtype=torch.bool)
    time_mask[-1] = False
    valid = agent_mask & time_mask.unsqueeze(-1)
    inputs = {
        "local_inputs" if encode_only else "local_obs": torch.randn(*prefix, 3, 3),
        "global_inputs" if encode_only else "global_obs": torch.randn(*prefix, 2),
        "actions": torch.randn(*prefix, 3, 2),
        "hidden_local_vars": torch.randn(*prefix, 3, 1),
        "hidden_global_vars": torch.randn(*prefix, 1),
    }
    clean_inputs = {
        key: value.masked_fill(~(valid if value.ndim == len(prefix) + 2 else time_mask).unsqueeze(-1), 0)
        for key, value in inputs.items()
    }
    padded_inputs = {
        key: value.masked_fill(~(valid if value.ndim == len(prefix) + 2 else time_mask).unsqueeze(-1), torch.nan)
        for key, value in inputs.items()
    }
    clean_inputs.update(agent_mask=agent_mask, time_mask=time_mask)
    padded_inputs.update(agent_mask=agent_mask, time_mask=time_mask)
    _compare_padding_gradients(critic, clean_inputs, padded_inputs, encode_only=encode_only)
