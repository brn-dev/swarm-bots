from itertools import product
from unittest.mock import patch

import pytest
import torch

from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig
from swarmbots.learn.algos.xlstm.mlstm import MLSTMCell, MLSTMCellConfig
from swarmbots.learn.algos.xlstm.mlstm import (
    MLSTMCellState,
    MLSTMTemporalSequenceModel,
    MLSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.algos.xlstm.temporal_utils import reset_state, select_state
from swarmbots.learn.algos.xlstm.slstm.slstm_cell import SLSTMCell, SLSTMCellConfig
from swarmbots.learn.algos.xlstm.slstm import SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig
from swarmbots.learn.temporal_state import index_temporal_state_batch_time, stack_temporal_states


@pytest.mark.parametrize(
    ("model_cls", "config"),
    [
        (MLSTMTemporalSequenceModel, MLSTMTemporalSequenceModelConfig(num_heads=2)),
        (SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=2)),
    ],
)
def test_xlstm_temporal_sequence_matches_explicit_step_flow(
        model_cls: type[MLSTMTemporalSequenceModel | SLSTMTemporalSequenceModel],
        config: MLSTMTemporalSequenceModelConfig | SLSTMTemporalSequenceModelConfig,
) -> None:
    torch.manual_seed(0)
    model = model_cls(hidden_dim=8, config=config)
    inputs = torch.randn(3, 5, 8)
    reset_mask = torch.tensor([
        [True, False, False, False, False],
        [True, False, True, False, False],
        [True, False, False, False, True],
    ])
    valid_mask = torch.tensor([
        [True, False, True, True, True],
        [True, True, False, True, True],
        [False, True, True, False, True],
    ])

    state_output_indices = torch.tensor([[0, 1], [1, 4], [0, 3]])
    sequence_output, sequence_state, selected_states = model(
        inputs,
        valid_mask=valid_mask,
        reset_mask=reset_mask,
        state_output_indices=state_output_indices,
    )

    step_state = model.initial_state(batch_size=inputs.shape[0], device=inputs.device, dtype=inputs.dtype)
    step_outputs = []
    step_states = []
    for time_idx in range(inputs.shape[1]):
        step_output, step_state = model(
            inputs[:, time_idx:time_idx + 1],
            valid_mask=valid_mask[:, time_idx:time_idx + 1],
            initial_state=step_state,
            reset_mask=reset_mask[:, time_idx:time_idx + 1],
        )
        step_outputs.append(step_output[:, 0])
        step_states.append(step_state)

    torch.testing.assert_close(sequence_output, torch.stack(step_outputs, dim=1))
    for sequence_tensor, step_tensor in zip(sequence_state, step_state, strict=True):
        torch.testing.assert_close(sequence_tensor, step_tensor)
    expected_state_sequence = stack_temporal_states(step_states, dim=1)
    expected_selected_states = index_temporal_state_batch_time(
        expected_state_sequence,
        state_output_indices,
    )
    for selected_tensor, expected_tensor in zip(selected_states, expected_selected_states, strict=True):
        torch.testing.assert_close(selected_tensor, expected_tensor)


def test_slstm_projects_the_full_sequence_once() -> None:
    model = SLSTMTemporalSequenceModel(
        hidden_dim=8,
        config=SLSTMTemporalSequenceModelConfig(num_heads=2),
    )
    inputs = torch.randn(3, 5, 8)

    with patch.object(
            model.input_projection,
            "forward",
            wraps=model.input_projection.forward,
    ) as projection_forward:
        model(inputs)

    projection_forward.assert_called_once()
    assert projection_forward.call_args.args[0].shape == inputs.shape


def test_slstm_can_compile_its_recurrent_step() -> None:
    with patch(
            "swarmbots.learn.algos.xlstm.slstm.slstm_temporal_sequence_model.torch.compile",
            side_effect=lambda function, **_kwargs: function,
    ) as compile_mock:
        model = SLSTMTemporalSequenceModel(
            hidden_dim=8,
            config=SLSTMTemporalSequenceModelConfig(
                num_heads=2,
                compile_step=True,
                compile_mode="default",
            ),
        )
        output, _state = model(torch.randn(3, 5, 8))

    assert output.shape == (3, 5, 8)
    compile_mock.assert_called_once()
    assert compile_mock.call_args.kwargs == {
        "mode": "default",
        "fullgraph": False,
        "dynamic": True,
    }


def test_slstm_clamps_restored_normalizer_before_hidden_division() -> None:
    cell = SLSTMCell(
        hidden_dim=1,
        config=SLSTMCellConfig(
            num_heads=1,
            recurrent_weight_init="zeros",
            bias_init="zeros",
        ),
    )
    gate_inputs = torch.tensor([[-10.0, 10.0, 0.0, 0.0]])
    restored_state = (
        torch.zeros(1, 1),
        torch.full((1, 1), 2.0),
        torch.full((1, 1), 0.25),
        torch.zeros(1, 1),
    )

    output, state = cell(gate_inputs, restored_state)

    torch.testing.assert_close(state[2], torch.ones(1, 1))
    torch.testing.assert_close(output, torch.ones(1, 1), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    ("model_cls", "config"),
    [
        (MLSTMTemporalSequenceModel, MLSTMTemporalSequenceModelConfig(num_heads=2)),
        (SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=2)),
    ],
)
def test_xlstm_valid_mask_zeroes_output_and_preserves_state(
        model_cls: type[MLSTMTemporalSequenceModel | SLSTMTemporalSequenceModel],
        config: MLSTMTemporalSequenceModelConfig | SLSTMTemporalSequenceModelConfig,
) -> None:
    torch.manual_seed(1)
    model = model_cls(hidden_dim=8, config=config)
    inputs = torch.randn(2, 4, 8)
    valid_mask = torch.tensor([
        [True, True, False, True],
        [True, False, False, False],
    ])

    output, state = model(inputs, valid_mask=valid_mask)

    torch.testing.assert_close(output[0, 2], torch.zeros(8))
    torch.testing.assert_close(output[1, 1:], torch.zeros(3, 8))

    _, row_one_expected_state = model(inputs[1:2, 0:1])

    for actual, expected in zip(state, row_one_expected_state, strict=True):
        torch.testing.assert_close(actual[1:2], expected)


@pytest.mark.parametrize(
    ("model_cls", "config"),
    [
        (MLSTMTemporalSequenceModel, MLSTMTemporalSequenceModelConfig(num_heads=2)),
        (SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=2)),
    ],
)
def test_xlstm_reset_zero_input_does_not_emit_bias_signal(
        model_cls: type[MLSTMTemporalSequenceModel | SLSTMTemporalSequenceModel],
        config: MLSTMTemporalSequenceModelConfig | SLSTMTemporalSequenceModelConfig,
) -> None:
    torch.manual_seed(2)
    model = model_cls(hidden_dim=8, config=config)
    inputs = torch.zeros(3, 4, 8)
    reset_mask = torch.ones(3, 4, dtype=torch.bool)

    output, _ = model(inputs, reset_mask=reset_mask)

    torch.testing.assert_close(output, torch.zeros_like(output))


@pytest.mark.parametrize(
    ("model_cls", "config"),
    [
        (MLSTMTemporalSequenceModel, MLSTMTemporalSequenceModelConfig(num_heads=3)),
        (SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=3)),
    ],
)
def test_xlstm_rejects_hidden_dim_not_divisible_by_heads(
        model_cls: type[MLSTMTemporalSequenceModel | SLSTMTemporalSequenceModel],
        config: MLSTMTemporalSequenceModelConfig | SLSTMTemporalSequenceModelConfig,
) -> None:
    with pytest.raises(ValueError, match="divisible"):
        model_cls(hidden_dim=8, config=config)


def test_mlstm_parallel_sequence_matches_recurrent_path_with_resets() -> None:
    torch.manual_seed(2)
    parallel_model = MLSTMTemporalSequenceModel(
        hidden_dim=8,
        config=MLSTMTemporalSequenceModelConfig(num_heads=2, use_parallel_sequence=True),
    )
    recurrent_model = MLSTMTemporalSequenceModel(
        hidden_dim=8,
        config=MLSTMTemporalSequenceModelConfig(num_heads=2, use_parallel_sequence=False),
    )
    recurrent_model.load_state_dict(parallel_model.state_dict())

    inputs = torch.randn(3, 6, 8)
    initial_state: MLSTMCellState = tuple(torch.randn_like(item) for item in parallel_model.initial_state(3))
    reset_mask = torch.tensor([
        [False, False, False, False, False, False],
        [False, False, True, False, False, False],
        [False, False, False, False, True, False],
    ])

    parallel_output, parallel_state = parallel_model(
        inputs,
        initial_state=initial_state,
        reset_mask=reset_mask,
    )
    recurrent_output, recurrent_state = recurrent_model(
        inputs,
        initial_state=initial_state,
        reset_mask=reset_mask,
    )

    torch.testing.assert_close(parallel_output, recurrent_output, atol=1e-5, rtol=1e-5)
    for parallel_tensor, recurrent_tensor in zip(parallel_state, recurrent_state, strict=True):
        torch.testing.assert_close(parallel_tensor, recurrent_tensor, atol=1e-5, rtol=1e-5)


def test_mlstm_parallel_cell_valid_and_reset_masks_exhaustively_match_step_flow() -> None:
    torch.manual_seed(4)
    cell = MLSTMCell(hidden_dim=4, config=MLSTMCellConfig(num_heads=2))
    sequence_length = 4
    queries = torch.randn(1, sequence_length, 4)
    keys = torch.randn(1, sequence_length, 4)
    values = torch.randn(1, sequence_length, 4)
    initial_state: MLSTMCellState = tuple(torch.randn_like(item) for item in cell.initial_state(1))

    for valid_bits in product([False, True], repeat=sequence_length):
        valid_mask = torch.tensor([valid_bits], dtype=torch.bool)
        for reset_bits in product([False, True], repeat=sequence_length):
            reset_mask = torch.tensor([reset_bits], dtype=torch.bool)

            parallel_output, parallel_state = cell.forward_sequence(
                queries,
                keys,
                values,
                initial_state,
                valid_mask=valid_mask,
                reset_mask=reset_mask,
            )
            recurrent_output, recurrent_state = _run_recurrent_mlstm_cell_sequence(
                cell=cell,
                queries=queries,
                keys=keys,
                values=values,
                initial_state=initial_state,
                valid_mask=valid_mask,
                reset_mask=reset_mask,
            )

            torch.testing.assert_close(parallel_output, recurrent_output, atol=1e-5, rtol=1e-5)
            for parallel_tensor, recurrent_tensor in zip(parallel_state, recurrent_state, strict=True):
                torch.testing.assert_close(parallel_tensor, recurrent_tensor, atol=1e-5, rtol=1e-5)


def test_mlstm_parallel_cell_reset_keeps_zero_state_stabilizer() -> None:
    torch.manual_seed(6)
    cell = MLSTMCell(hidden_dim=4, config=MLSTMCellConfig(num_heads=2))
    with torch.no_grad():
        cell.input_gate.weight.zero_()
        cell.input_gate.bias.fill_(-10.0)
        cell.forget_gate.weight.zero_()
        cell.forget_gate.bias.fill_(6.0)

    sequence_length = 2
    queries = torch.randn(1, sequence_length, 4)
    keys = torch.randn(1, sequence_length, 4)
    values = torch.randn(1, sequence_length, 4)
    initial_state: MLSTMCellState = tuple(torch.randn_like(item) for item in cell.initial_state(1))
    valid_mask = torch.ones(1, sequence_length, dtype=torch.bool)
    reset_mask = torch.tensor([[True, False]])

    parallel_output, parallel_state = cell.forward_sequence(
        queries,
        keys,
        values,
        initial_state,
        valid_mask=valid_mask,
        reset_mask=reset_mask,
    )
    recurrent_output, recurrent_state = _run_recurrent_mlstm_cell_sequence(
        cell=cell,
        queries=queries,
        keys=keys,
        values=values,
        initial_state=initial_state,
        valid_mask=valid_mask,
        reset_mask=reset_mask,
    )

    torch.testing.assert_close(parallel_output, recurrent_output, atol=1e-5, rtol=1e-5)
    for parallel_tensor, recurrent_tensor in zip(parallel_state, recurrent_state, strict=True):
        torch.testing.assert_close(parallel_tensor, recurrent_tensor, atol=1e-5, rtol=1e-5)


def test_mlstm_parallel_cell_ignores_invalid_nan_inputs() -> None:
    torch.manual_seed(7)
    cell = MLSTMCell(hidden_dim=4, config=MLSTMCellConfig(num_heads=2))
    sequence_length = 3
    valid_mask = torch.tensor([[True, False, True]])
    reset_mask = torch.zeros(1, sequence_length, dtype=torch.bool)
    queries = torch.randn(1, sequence_length, 4)
    keys = torch.randn(1, sequence_length, 4)
    values = torch.randn(1, sequence_length, 4)
    initial_state = cell.initial_state(1)

    nan_queries = queries.clone()
    nan_keys = keys.clone()
    nan_values = values.clone()
    nan_queries[:, 1] = torch.nan
    nan_keys[:, 1] = torch.nan
    nan_values[:, 1] = torch.nan

    nan_output, nan_state = cell.forward_sequence(
        nan_queries,
        nan_keys,
        nan_values,
        initial_state,
        valid_mask=valid_mask,
        reset_mask=reset_mask,
    )
    clean_queries = queries.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
    clean_keys = keys.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
    clean_values = values.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
    clean_output, clean_state = cell.forward_sequence(
        clean_queries,
        clean_keys,
        clean_values,
        initial_state,
        valid_mask=valid_mask,
        reset_mask=reset_mask,
    )

    torch.testing.assert_close(nan_output, clean_output, atol=1e-5, rtol=1e-5)
    for nan_tensor, clean_tensor in zip(nan_state, clean_state, strict=True):
        torch.testing.assert_close(nan_tensor, clean_tensor, atol=1e-5, rtol=1e-5)


def test_mlstm_parallel_temporal_sequence_matches_recurrent_path_with_invalid_masks() -> None:
    torch.manual_seed(5)
    parallel_model = MLSTMTemporalSequenceModel(
        hidden_dim=8,
        config=MLSTMTemporalSequenceModelConfig(num_heads=2, use_parallel_sequence=True),
    )
    recurrent_model = MLSTMTemporalSequenceModel(
        hidden_dim=8,
        config=MLSTMTemporalSequenceModelConfig(num_heads=2, use_parallel_sequence=False),
    )
    recurrent_model.load_state_dict(parallel_model.state_dict())

    inputs = torch.randn(4, 6, 8)
    initial_state: MLSTMCellState = tuple(torch.randn_like(item) for item in parallel_model.initial_state(4))
    valid_mask = torch.tensor([
        [True, True, False, True, False, True],
        [False, False, False, False, False, False],
        [True, False, True, False, True, False],
        [False, True, True, False, False, True],
    ])
    reset_mask = torch.tensor([
        [False, False, True, False, False, False],
        [False, True, False, False, False, False],
        [True, False, False, True, False, False],
        [False, True, False, False, True, False],
    ])

    parallel_output, parallel_state = parallel_model(
        inputs,
        valid_mask=valid_mask,
        initial_state=initial_state,
        reset_mask=reset_mask,
    )
    recurrent_output, recurrent_state = recurrent_model(
        inputs,
        valid_mask=valid_mask,
        initial_state=initial_state,
        reset_mask=reset_mask,
    )

    torch.testing.assert_close(parallel_output, recurrent_output, atol=1e-5, rtol=1e-5)
    for parallel_tensor, recurrent_tensor in zip(parallel_state, recurrent_state, strict=True):
        torch.testing.assert_close(parallel_tensor, recurrent_tensor, atol=1e-5, rtol=1e-5)


def test_rmat_encoder_accepts_per_layer_xlstm_temporal_specs() -> None:
    torch.manual_seed(3)
    encoder = RMATEncoder(
        RMATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            temporal_model_cls=[SLSTMTemporalSequenceModel, MLSTMTemporalSequenceModel],
            temporal_model_config=[
                SLSTMTemporalSequenceModelConfig(num_heads=2),
                MLSTMTemporalSequenceModelConfig(num_heads=2),
            ],
        ),
        max_agents=3,
        local_obs_dim=4,
        global_obs_dim=2,
    )

    output, state = encoder(
        local_obs=torch.randn(2, 3, 3, 4),
        global_obs=torch.randn(2, 3, 2),
        agent_mask=torch.ones(2, 3, 3, dtype=torch.bool),
        reset_mask=torch.tensor([
            [True, False, False],
            [True, False, True],
        ]),
    )

    assert output.shape == (2, 3, 3, 8)
    assert len(state) == 2


def test_slstm_reset_state_with_large_negative_input_is_finite() -> None:
    cell = SLSTMCell(hidden_dim=4, config=SLSTMCellConfig(num_heads=2))
    state = cell.initial_state(batch_size=1)
    gate_inputs = torch.zeros(1, 16)
    gate_inputs[:, :4] = -1000.0

    output, next_state = cell(gate_inputs, state)

    assert torch.isfinite(output).all()
    for state_tensor in next_state:
        assert torch.isfinite(state_tensor).all()


def _run_recurrent_mlstm_cell_sequence(
        *,
        cell: MLSTMCell,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        initial_state: MLSTMCellState,
        valid_mask: torch.Tensor,
        reset_mask: torch.Tensor,
) -> tuple[torch.Tensor, MLSTMCellState]:
    sequence_length = queries.shape[1]
    state = initial_state
    zero_output = queries.new_zeros((queries.shape[0], cell.hidden_dim))
    outputs: list[torch.Tensor] = []

    for time_idx in range(sequence_length):
        state = reset_state(state, reset_mask[:, time_idx])
        step_output, next_state = cell(
            queries[:, time_idx],
            keys[:, time_idx],
            values[:, time_idx],
            state,
        )
        valid_t = valid_mask[:, time_idx]
        outputs.append(torch.where(valid_t.unsqueeze(-1), step_output, zero_output))
        state = select_state(next_state, state, valid_t)

    return torch.stack(outputs, dim=1), state
