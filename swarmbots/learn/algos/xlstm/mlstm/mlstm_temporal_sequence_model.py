from dataclasses import dataclass

import torch
from torch import nn

from swarmbots.learn.algos.r_mat.temporal_sequence_model import TemporalSequenceModel
from swarmbots.learn.algos.xlstm.temporal_utils import check_mask, check_sequence_inputs, reset_state, select_state
from swarmbots.learn.algos.xlstm.mlstm.mlstm_cell import MLSTMCell, MLSTMCellConfig, MLSTMCellState


@dataclass(frozen=True)
class MLSTMTemporalSequenceModelConfig:
    num_heads: int = 4
    bias: bool = False
    eps: float = 1e-6
    use_parallel_sequence: bool = True


class MLSTMTemporalSequenceModel(TemporalSequenceModel):

    def __init__(
            self,
            hidden_dim: int,
            config: MLSTMTemporalSequenceModelConfig = MLSTMTemporalSequenceModelConfig(),
    ) -> None:
        super().__init__(hidden_dim=hidden_dim)
        self.config = config
        self.qkv_projection = nn.Linear(hidden_dim, 3 * hidden_dim, bias=config.bias)
        self.cell = MLSTMCell(
            hidden_dim=hidden_dim,
            config=MLSTMCellConfig(num_heads=config.num_heads, eps=config.eps),
        )
        if self.qkv_projection.bias is not None:
            nn.init.zeros_(self.qkv_projection.bias)

    def initial_state(
            self,
            batch_size: int,
            *,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> MLSTMCellState:
        parameter = next(self.parameters())
        return self.cell.initial_state(
            batch_size=batch_size,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )

    def forward(
            self,
            inputs: torch.Tensor,
            *,
            valid_mask: torch.Tensor | None = None,
            initial_state: MLSTMCellState | None = None,
            reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MLSTMCellState]:
        batch_size, sequence_length, _ = check_sequence_inputs(inputs)
        check_mask(valid_mask, batch_size=batch_size, sequence_length=sequence_length, name="valid_mask")
        check_mask(reset_mask, batch_size=batch_size, sequence_length=sequence_length, name="reset_mask")

        state = (
            self.initial_state(batch_size=batch_size, device=inputs.device, dtype=inputs.dtype)
            if initial_state is None
            else initial_state
        )

        if self.config.use_parallel_sequence and _can_use_parallel_sequence(valid_mask):
            return self._forward_parallel(
                inputs,
                initial_state=state,
                reset_mask=reset_mask,
            )

        return self._forward_recurrent(
            inputs,
            valid_mask=valid_mask,
            initial_state=state,
            reset_mask=reset_mask,
        )

    def _forward_recurrent(
            self,
            inputs: torch.Tensor,
            *,
            valid_mask: torch.Tensor | None,
            initial_state: MLSTMCellState,
            reset_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, MLSTMCellState]:
        batch_size, sequence_length, _ = inputs.shape
        state = initial_state
        zero_output = inputs.new_zeros((batch_size, self.hidden_dim))
        outputs: list[torch.Tensor] = []

        for time_idx in range(sequence_length):
            if reset_mask is not None:
                state = reset_state(state, reset_mask[:, time_idx])

            q, k, v = self.qkv_projection(inputs[:, time_idx]).chunk(3, dim=-1)
            step_output, next_state = self.cell(q, k, v, state)

            if valid_mask is None:
                outputs.append(step_output)
                state = next_state
                continue

            valid_t = valid_mask[:, time_idx]
            outputs.append(torch.where(valid_t.unsqueeze(-1), step_output, zero_output))
            state = select_state(next_state, state, valid_t)

        return torch.stack(outputs, dim=1).contiguous(), state

    def _forward_parallel(
            self,
            inputs: torch.Tensor,
            *,
            initial_state: MLSTMCellState,
            reset_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, MLSTMCellState]:
        q, k, v = self.qkv_projection(inputs).chunk(3, dim=-1)
        sequence_length = inputs.shape[1]
        state = initial_state
        outputs: list[torch.Tensor] = []
        start_idx = 0

        for end_idx in _parallel_segment_end_indices(reset_mask=reset_mask, sequence_length=sequence_length):
            if reset_mask is not None:
                state = reset_state(state, reset_mask[:, start_idx])
            segment_output, state = self.cell.forward_sequence(
                q[:, start_idx:end_idx],
                k[:, start_idx:end_idx],
                v[:, start_idx:end_idx],
                state,
            )
            outputs.append(segment_output)
            start_idx = end_idx

        return torch.cat(outputs, dim=1).contiguous(), state


def _can_use_parallel_sequence(valid_mask: torch.Tensor | None) -> bool:
    return valid_mask is None or bool(torch.all(valid_mask))


def _parallel_segment_end_indices(*, reset_mask: torch.Tensor | None, sequence_length: int) -> list[int]:
    if reset_mask is None:
        return [sequence_length]

    reset_times = torch.any(reset_mask, dim=0).nonzero(as_tuple=False).flatten().tolist()
    segment_starts = [time_idx for time_idx in reset_times if time_idx > 0]
    return [*segment_starts, sequence_length]
