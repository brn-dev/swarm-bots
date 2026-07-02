from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from swarmbots.learn.algos.r_mat.temporal_sequence_model import TemporalSequenceModel
from swarmbots.learn.algos.xlstm.head_utils import MultiHeadLayerNorm
from swarmbots.learn.algos.xlstm.temporal_utils import check_mask, check_sequence_inputs, reset_state, select_state
from swarmbots.learn.algos.xlstm.slstm.slstm_cell import SLSTMCell, SLSTMCellConfig, SLSTMCellState


@dataclass(frozen=True)
class SLSTMTemporalSequenceModelConfig:
    num_heads: int = 4
    bias: bool = False
    recurrent_weight_init: Literal["zeros", "standard"] = "zeros"
    bias_init: Literal["powerlaw_blockdependent", "small_init", "zeros", "standard"] = "powerlaw_blockdependent"
    eps: float = 1e-6
    output_norm: bool = True


class SLSTMTemporalSequenceModel(TemporalSequenceModel):

    def __init__(
            self,
            hidden_dim: int,
            config: SLSTMTemporalSequenceModelConfig = SLSTMTemporalSequenceModelConfig(),
    ) -> None:
        super().__init__(hidden_dim=hidden_dim)
        self.config = config
        self.input_projection = nn.Linear(hidden_dim, 4 * hidden_dim, bias=config.bias)
        self.cell = SLSTMCell(
            hidden_dim=hidden_dim,
            config=SLSTMCellConfig(
                num_heads=config.num_heads,
                recurrent_weight_init=config.recurrent_weight_init,
                bias_init=config.bias_init,
                eps=config.eps,
            ),
        )
        self.output_norm = (
            MultiHeadLayerNorm(hidden_dim=hidden_dim, num_heads=config.num_heads, bias=False)
            if config.output_norm
            else nn.Identity()
        )
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)

    def initial_state(
            self,
            batch_size: int,
            *,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> SLSTMCellState:
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
            initial_state: SLSTMCellState | None = None,
            reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SLSTMCellState]:
        batch_size, sequence_length, _ = check_sequence_inputs(inputs)
        check_mask(valid_mask, batch_size=batch_size, sequence_length=sequence_length, name="valid_mask")
        check_mask(reset_mask, batch_size=batch_size, sequence_length=sequence_length, name="reset_mask")

        state = (
            self.initial_state(batch_size=batch_size, device=inputs.device, dtype=inputs.dtype)
            if initial_state is None
            else initial_state
        )
        zero_output = inputs.new_zeros((batch_size, self.hidden_dim))
        outputs: list[torch.Tensor] = []

        for time_idx in range(sequence_length):
            if reset_mask is not None:
                state = reset_state(state, reset_mask[:, time_idx])

            step_output, next_state = self.cell(self.input_projection(inputs[:, time_idx]), state)
            step_output = self._normalize_output(step_output)

            if valid_mask is None:
                outputs.append(step_output)
                state = next_state
                continue

            valid_t = valid_mask[:, time_idx]
            outputs.append(torch.where(valid_t.unsqueeze(-1), step_output, zero_output))
            state = select_state(next_state, state, valid_t)

        return torch.stack(outputs, dim=1).contiguous(), state

    def _normalize_output(self, output: torch.Tensor) -> torch.Tensor:
        batch_size = output.shape[0]
        output_heads = output.reshape(batch_size, self.config.num_heads, 1, self.cell.head_dim)
        return self.output_norm(output_heads).reshape(batch_size, self.hidden_dim)
