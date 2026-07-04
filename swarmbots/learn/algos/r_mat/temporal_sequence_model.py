import abc
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

TemporalModelState = Any


class TemporalSequenceModel(nn.Module, abc.ABC):

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        self.hidden_dim = hidden_dim

    @abc.abstractmethod
    def initial_state(
            self,
            batch_size: int,
            *,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> TemporalModelState:
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
            self,
            inputs: torch.Tensor,
            *,
            valid_mask: torch.Tensor | None = None,
            initial_state: TemporalModelState | None = None,
            reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TemporalModelState]:
        raise NotImplementedError


LSTMTemporalModelState = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class LSTMTemporalSequenceModelConfig:
    num_layers: int = 1
    bias: bool = False
    dropout: float = 0.0


class LSTMTemporalSequenceModel(TemporalSequenceModel):

    def __init__(
            self,
            hidden_dim: int,
            config: LSTMTemporalSequenceModelConfig = LSTMTemporalSequenceModelConfig(),
    ) -> None:
        super().__init__(hidden_dim=hidden_dim)
        if config.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {config.num_layers}")
        if config.dropout < 0:
            raise ValueError(f"dropout must be >= 0, got {config.dropout}")

        self.config = config
        lstm_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=config.num_layers,
            bias=config.bias,
            dropout=lstm_dropout,
            batch_first=True,
        )

    def initial_state(
            self,
            batch_size: int,
            *,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> LSTMTemporalModelState:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        shape = (batch_size, self.config.num_layers, self.hidden_dim)
        hidden_state = torch.zeros(shape, device=device, dtype=dtype)
        cell_state = torch.zeros(shape, device=device, dtype=dtype)
        return hidden_state, cell_state

    def forward(
            self,
            inputs: torch.Tensor,
            *,
            valid_mask: torch.Tensor | None = None,
            initial_state: LSTMTemporalModelState | None = None,
            reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LSTMTemporalModelState]:
        if inputs.ndim != 3:
            raise ValueError(f"Expected inputs shape (B, T, H), got {tuple(inputs.shape)}")

        batch_size, sequence_length, hidden_dim = inputs.shape

        valid_mask = self._normalize_mask(valid_mask, batch_size=batch_size, sequence_length=sequence_length, name="valid_mask")
        reset_mask = self._normalize_mask(reset_mask, batch_size=batch_size, sequence_length=sequence_length, name="reset_mask")

        if initial_state is None:
            hidden_state, cell_state = self.initial_state(
                batch_size=batch_size,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        else:
            hidden_state, cell_state = initial_state

        outputs: list[torch.Tensor] = []
        zero_output = inputs.new_zeros((batch_size, 1, self.hidden_dim))

        for time_idx in range(sequence_length):
            if reset_mask is not None:
                reset_t = reset_mask[:, time_idx]
                if torch.any(reset_t):
                    keep_state_mask = (~reset_t).view(batch_size, 1, 1)
                    hidden_state = hidden_state * keep_state_mask
                    cell_state = cell_state * keep_state_mask

            step_output, (next_hidden_state, next_cell_state) = self.lstm(
                inputs[:, time_idx:time_idx + 1, :],
                (
                    hidden_state.transpose(0, 1).contiguous(),
                    cell_state.transpose(0, 1).contiguous(),
                ),
            )
            next_hidden_state = next_hidden_state.transpose(0, 1).contiguous()
            next_cell_state = next_cell_state.transpose(0, 1).contiguous()

            if valid_mask is None:
                outputs.append(step_output)
                hidden_state = next_hidden_state
                cell_state = next_cell_state
                continue

            valid_t = valid_mask[:, time_idx]
            valid_output_mask = valid_t.view(batch_size, 1, 1)
            valid_state_mask = valid_t.view(batch_size, 1, 1)
            outputs.append(torch.where(valid_output_mask, step_output, zero_output))
            hidden_state = torch.where(valid_state_mask, next_hidden_state, hidden_state)
            cell_state = torch.where(valid_state_mask, next_cell_state, cell_state)

        return torch.cat(outputs, dim=1).contiguous(), (hidden_state, cell_state)

    @staticmethod
    def _normalize_mask(
            mask: torch.Tensor | None,
            *,
            batch_size: int,
            sequence_length: int,
            name: str,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.shape != (batch_size, sequence_length):
            raise ValueError(
                f"Expected {name} shape ({batch_size}, {sequence_length}), got {tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            raise ValueError(f"Expected {name} dtype torch.bool, got {mask.dtype}")
        return mask
