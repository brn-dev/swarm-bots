import pytest
import torch

from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig
from swarmbots.learn.algos.xlstm.mlstm import (
    MLSTMCellState,
    MLSTMTemporalSequenceModel,
    MLSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.algos.xlstm.slstm.slstm_cell import SLSTMCell, SLSTMCellConfig
from swarmbots.learn.algos.xlstm.slstm import SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig


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

    sequence_output, sequence_state = model(inputs, reset_mask=reset_mask)

    step_state = model.initial_state(batch_size=inputs.shape[0], device=inputs.device, dtype=inputs.dtype)
    step_outputs = []
    for time_idx in range(inputs.shape[1]):
        step_output, step_state = model(
            inputs[:, time_idx:time_idx + 1],
            initial_state=step_state,
            reset_mask=reset_mask[:, time_idx:time_idx + 1],
        )
        step_outputs.append(step_output[:, 0])

    torch.testing.assert_close(sequence_output, torch.stack(step_outputs, dim=1))
    for sequence_tensor, step_tensor in zip(sequence_state, step_state, strict=True):
        torch.testing.assert_close(sequence_tensor, step_tensor)


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


def test_mlstm_parallel_sequence_matches_recurrent_path_with_reset_segments() -> None:
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
