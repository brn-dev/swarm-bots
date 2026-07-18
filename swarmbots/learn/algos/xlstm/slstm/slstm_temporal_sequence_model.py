from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import nn

from swarmbots.learn.algos.r_mat.temporal_sequence_model import TemporalSequenceModel
from swarmbots.learn.algos.xlstm.head_utils import MultiHeadLayerNorm
from swarmbots.learn.algos.xlstm.temporal_utils import check_mask, check_sequence_inputs, reset_state, select_state
from swarmbots.learn.algos.xlstm.slstm.slstm_cell import SLSTMCell, SLSTMCellConfig, SLSTMCellState
from swarmbots.learn.temporal_state import index_temporal_state_batch_time, stack_temporal_states


@dataclass(frozen=True)
class SLSTMTemporalSequenceModelConfig:
    num_heads: int = 4
    bias: bool = False
    recurrent_weight_init: Literal["zeros", "standard"] = "zeros"
    bias_init: Literal["powerlaw", "small_init", "zeros", "standard"] = "powerlaw"
    output_norm: bool = True
    compile_step: bool = False
    compile_mode: str = "default"


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
            ),
        )
        self.output_norm = (
            MultiHeadLayerNorm(hidden_dim=hidden_dim, num_heads=config.num_heads, bias=False)
            if config.output_norm
            else nn.Identity()
        )
        self._step_fn: Callable[
            [torch.Tensor, SLSTMCellState],
            tuple[torch.Tensor, SLSTMCellState],
        ] = self._step
        if config.compile_step:
            if not hasattr(torch, "compile") or not callable(torch.compile):
                raise RuntimeError("SLSTMTemporalSequenceModelConfig.compile_step=True requires torch.compile support.")
            if not config.compile_mode:
                raise ValueError("compile_mode must be non-empty when compile_step=True.")
            self._step_fn = torch.compile(
                self._step,
                mode=config.compile_mode,
                fullgraph=False,
                dynamic=True,
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
            state_output_indices: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, SLSTMCellState]
        | tuple[torch.Tensor, SLSTMCellState, SLSTMCellState]
    ):
        batch_size, sequence_length, _ = check_sequence_inputs(inputs)
        check_mask(valid_mask, batch_size=batch_size, sequence_length=sequence_length, name="valid_mask")
        check_mask(reset_mask, batch_size=batch_size, sequence_length=sequence_length, name="reset_mask")

        state = (
            self.initial_state(batch_size=batch_size, device=inputs.device, dtype=inputs.dtype)
            if initial_state is None
            else initial_state
        )
        zero_output = inputs.new_zeros((batch_size, self.hidden_dim))
        projected_inputs = self.input_projection(inputs)
        outputs: list[torch.Tensor] = []
        states: list[SLSTMCellState] = []

        for time_idx in range(sequence_length):
            if reset_mask is not None:
                state = reset_state(state, reset_mask[:, time_idx])

            step_output, next_state = self._step_fn(projected_inputs[:, time_idx], state)

            if valid_mask is None:
                outputs.append(step_output)
                state = next_state
                if state_output_indices is not None:
                    states.append(state)
                continue

            valid_t = valid_mask[:, time_idx]
            outputs.append(torch.where(valid_t.unsqueeze(-1), step_output, zero_output))
            state = select_state(next_state, state, valid_t)
            if state_output_indices is not None:
                states.append(state)

        outputs_tensor = torch.stack(outputs, dim=1)
        output_sequence = self._normalize_outputs(outputs_tensor).contiguous()
        if state_output_indices is not None:
            state_sequence = stack_temporal_states(states, dim=1)
            selected_states = index_temporal_state_batch_time(state_sequence, state_output_indices)
            return output_sequence, state, selected_states
        return output_sequence, state

    def _step(
            self,
            projected_input: torch.Tensor,
            state: SLSTMCellState,
    ) -> tuple[torch.Tensor, SLSTMCellState]:
        return self.cell(projected_input, state)

    def _normalize_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = outputs.shape
        output_heads = outputs.reshape(
            batch_size,
            sequence_length,
            self.config.num_heads,
            self.cell.head_dim,
        ).transpose(1, 2)
        return self.output_norm(output_heads).transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            self.hidden_dim,
        )
