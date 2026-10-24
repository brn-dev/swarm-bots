from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.utils._pytree import tree_flatten

from swarmbots.learn.algos.off_policy.recurrent_transformer_critic import RecurrentTransformerTwinCritic
from swarmbots.learn.algos.off_policy.transformer_critic import TransformerCriticConfig
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig
from swarmbots.learn.algos.r_mat.temporal_sequence_model import LSTMTemporalSequenceModel, TemporalSequenceModel
from swarmbots.learn.algos.xlstm.mlstm import MLSTMTemporalSequenceModel, MLSTMTemporalSequenceModelConfig
from swarmbots.learn.algos.xlstm.slstm import SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig


def _temporal_model(kind: str) -> TemporalSequenceModel:
    if kind == "lstm":
        return LSTMTemporalSequenceModel(8)
    if kind == "slstm":
        return SLSTMTemporalSequenceModel(8, SLSTMTemporalSequenceModelConfig(num_heads=2))
    return MLSTMTemporalSequenceModel(8, MLSTMTemporalSequenceModelConfig(
        num_heads=2, use_parallel_sequence=kind == "mlstm_parallel",
    ))


@pytest.mark.parametrize("kind", ["lstm", "slstm", "mlstm_parallel", "mlstm_recurrent"])
def test_temporal_inference_does_not_copy_inputs_for_training_only_sanitization(kind: str) -> None:
    model = _temporal_model(kind)
    projection = model.lstm if kind == "lstm" else model.input_projection if kind == "slstm" else model.qkv_projection
    pointers: list[int] = []
    hook = projection.register_forward_pre_hook(lambda _module, args: pointers.append(args[0].data_ptr()))
    inputs = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, False, True], [True, True, False]])
    try:
        with torch.no_grad():
            model(inputs, valid_mask=mask)
        assert pointers[0] == inputs.data_ptr()
    finally:
        hook.remove()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[
    pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
])])
@pytest.mark.parametrize("kind", ["lstm", "slstm", "mlstm_parallel", "mlstm_recurrent"])
def test_temporal_models_ignore_nonfinite_padding_in_outputs_states_and_gradients(device: str, kind: str) -> None:
    torch.manual_seed(8)
    model = _temporal_model(kind).to(device)
    reference = deepcopy(model)
    mask = torch.tensor([[True, False, True, False], [False, True, True, False]], device=device)
    reset = torch.tensor([[True, False, False, False], [False, True, False, False]], device=device)
    inputs = torch.randn(2, 4, 8, device=device)
    clean = inputs.masked_fill(~mask.unsqueeze(-1), 0).requires_grad_()
    padded = inputs.masked_fill(~mask.unsqueeze(-1), torch.nan).requires_grad_()
    kwargs = dict(valid_mask=mask, reset_mask=reset, state_output_indices=torch.tensor([[0, 2], [1, 3]], device=device))
    expected = reference(clean, **kwargs)
    actual = model(padded, **kwargs)
    for tensor, expected_tensor in zip(tree_flatten(actual)[0], tree_flatten(expected)[0], strict=True):
        torch.testing.assert_close(tensor, expected_tensor)
    expected[0].square().sum().backward()
    actual[0].square().sum().backward()
    torch.testing.assert_close(padded.grad, clean.grad)
    assert torch.count_nonzero(padded.grad[~mask]) == 0
    assert padded.grad[mask].abs().sum() > 0
    for (name, parameter), (_, reference_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True,
    ):
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, msg=name)


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
