from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.r_mat.temporal_sequence_model import (
    LSTMTemporalSequenceModel,
    LSTMTemporalSequenceModelConfig,
    TemporalModelState,
    TemporalSequenceModel,
)
from swarmbots.learn.nn_components.mlp import MLP
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal

RMATEncoderState = list[TemporalModelState]


@dataclass(frozen=True)
class RMATEncoderConfig(MATEncoderConfig):
    temporal_model_cls: type[TemporalSequenceModel] = LSTMTemporalSequenceModel
    temporal_model_config: Any = field(default_factory=LSTMTemporalSequenceModelConfig)


class _RMATBlock(nn.Module):

    def __init__(
            self,
            config: RMATEncoderConfig,
    ) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.inter_agent_attention_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.act_fn_cls(),
                layer_norm_eps=config.layer_norm_eps,
                batch_first=True,
                norm_first=config.norm_first,
                bias=config.bias,
            ),
            num_layers=1,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=not config.norm_first,
        )
        self.temporal_model = config.temporal_model_cls(
            hidden_dim=config.d_model,
            config=config.temporal_model_config,
        )

    def forward(
            self,
            embeddings: torch.Tensor,
            *,
            agent_mask: torch.Tensor | None,
            time_mask: torch.Tensor | None,
            initial_state: TemporalModelState | None,
            reset_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, TemporalModelState]:
        batch_size, sequence_length, n_agents, hidden_dim = embeddings.shape
        flat_inter_agent_attention_inputs = embeddings.reshape(batch_size * sequence_length, n_agents, hidden_dim)
        flat_agent_mask = (
            None
            if agent_mask is None
            else agent_mask.reshape(batch_size * sequence_length, n_agents)
        )
        inter_agent_attention_outputs = self.inter_agent_attention_encoder(
            flat_inter_agent_attention_inputs,
            src_key_padding_mask=None if flat_agent_mask is None else ~flat_agent_mask,
        ).reshape(batch_size, sequence_length, n_agents, hidden_dim)

        valid_agent_time_mask = _combine_agent_time_mask(
            agent_mask=agent_mask,
            time_mask=time_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            n_agents=n_agents,
            device=embeddings.device,
        )
        if valid_agent_time_mask is not None:
            inter_agent_attention_outputs = inter_agent_attention_outputs.masked_fill(
                ~valid_agent_time_mask.unsqueeze(-1),
                0.0,
            )

        temporal_inputs = inter_agent_attention_outputs.permute(0, 2, 1, 3).reshape(
            batch_size * n_agents,
            sequence_length,
            hidden_dim,
        )
        temporal_valid_mask = None
        if valid_agent_time_mask is not None:
            temporal_valid_mask = valid_agent_time_mask.permute(0, 2, 1).reshape(batch_size * n_agents, sequence_length)

        temporal_reset_mask = None
        if reset_mask is not None:
            temporal_reset_mask = reset_mask.unsqueeze(1).expand(batch_size, n_agents, sequence_length)
            temporal_reset_mask = temporal_reset_mask.reshape(batch_size * n_agents, sequence_length)

        temporal_outputs, next_state = self.temporal_model(
            temporal_inputs,
            valid_mask=temporal_valid_mask,
            initial_state=initial_state,
            reset_mask=temporal_reset_mask,
        )
        outputs = temporal_outputs.reshape(batch_size, n_agents, sequence_length, hidden_dim).permute(0, 2, 1, 3)
        if valid_agent_time_mask is not None:
            outputs = outputs.masked_fill(~valid_agent_time_mask.unsqueeze(-1), 0.0)
        return outputs.contiguous(), next_state


class RMATEncoder(nn.Module):

    def __init__(
            self,
            config: RMATEncoderConfig,
            *,
            max_agents: int,
            local_obs_dim: int,
            global_obs_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.has_global_obs = global_obs_dim > 0
        self.add_agent_embeddings = config.add_agent_embeddings
        self.max_agents = max_agents
        self.d_model = config.d_model

        if config.local_obs_encoder_hidden_dims:
            self.local_obs_encoder = MLP(
                input_dim=self.local_obs_dim,
                hidden_dims=[*config.local_obs_encoder_hidden_dims, config.d_model],
                end_with_act_fn=False,
                linear_init=init_linear_orthogonal,
                act_fn_cls=config.act_fn_cls,
            )
        else:
            self.local_obs_encoder = nn.Linear(self.local_obs_dim, config.d_model)
            init_linear_orthogonal(self.local_obs_encoder)

        if self.has_global_obs:
            if config.global_obs_encoder_hidden_dims:
                self.global_obs_encoder = MLP(
                    input_dim=self.global_obs_dim,
                    hidden_dims=[*config.global_obs_encoder_hidden_dims, config.d_model],
                    end_with_act_fn=False,
                    linear_init=init_linear_orthogonal,
                    act_fn_cls=config.act_fn_cls,
                )
            else:
                self.global_obs_encoder = nn.Linear(self.global_obs_dim, config.d_model)
                init_linear_orthogonal(self.global_obs_encoder)
        else:
            self.global_obs_encoder = None

        # noinspection PyTypeChecker
        self.layers: list[_RMATBlock] = nn.ModuleList([
            _RMATBlock(config=config)
            for _ in range(config.num_layers)
        ])
        self.output_norm = nn.LayerNorm(config.d_model)

        self.agent_embeddings: nn.Parameter | None = None
        if self.add_agent_embeddings:
            self.agent_embeddings = nn.Parameter(
                torch.zeros(1, self.max_agents, config.d_model),
                requires_grad=True,
            )
            nn.init.orthogonal_(self.agent_embeddings)

    def initial_state(
            self,
            batch_size: int,
            *,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> RMATEncoderState:
        return [
            layer.temporal_model.initial_state(batch_size=batch_size, device=device, dtype=dtype)
            for layer in self.layers
        ]

    def forward(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None = None,
            *,
            time_mask: torch.Tensor | None = None,
            initial_state: RMATEncoderState | None = None,
            reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RMATEncoderState]:
        local_obs, global_obs, agent_mask, time_mask, reset_mask, squeeze_time = self._normalize_inputs(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            time_mask=time_mask,
            reset_mask=reset_mask,
        )
        batch_size, sequence_length, n_agents, _ = local_obs.shape

        embeddings = self.local_obs_encoder(local_obs)
        if self.agent_embeddings is not None:
            embeddings = embeddings + self.agent_embeddings[:, :n_agents, :].unsqueeze(1)

        if self.has_global_obs:
            global_embeddings = self.global_obs_encoder(global_obs)
            embeddings = embeddings + global_embeddings.unsqueeze(2)

        valid_agent_time_mask = _combine_agent_time_mask(
            agent_mask=agent_mask,
            time_mask=time_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            n_agents=n_agents,
            device=local_obs.device,
        )
        if valid_agent_time_mask is not None:
            embeddings = embeddings.masked_fill(~valid_agent_time_mask.unsqueeze(-1), 0.0)

        layer_states = self._normalize_initial_state(
            initial_state=initial_state,
            batch_size=batch_size * n_agents,
            device=local_obs.device,
            dtype=embeddings.dtype,
        )

        next_states: RMATEncoderState = []
        hidden = embeddings
        for layer, layer_state in zip(self.layers, layer_states, strict=True):
            hidden, next_state = layer(
                hidden,
                agent_mask=agent_mask,
                time_mask=time_mask,
                initial_state=layer_state,
                reset_mask=reset_mask,
            )
            next_states.append(next_state)

        hidden = self.output_norm(hidden)
        if valid_agent_time_mask is not None:
            hidden = hidden.masked_fill(~valid_agent_time_mask.unsqueeze(-1), 0.0)

        if squeeze_time:
            return hidden[:, 0].contiguous(), next_states
        return hidden.contiguous(), next_states

    def _normalize_initial_state(
            self,
            *,
            initial_state: RMATEncoderState | None,
            batch_size: int,
            device: torch.device,
            dtype: torch.dtype,
    ) -> RMATEncoderState:
        if initial_state is None:
            return [
                layer.temporal_model.initial_state(batch_size=batch_size, device=device, dtype=dtype)
                for layer in self.layers
            ]
        return list(initial_state)

    @staticmethod
    def _normalize_inputs(
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            time_mask: torch.Tensor | None,
            reset_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, bool]:
        if local_obs.ndim == 3:
            batch_size = local_obs.shape[0]
            if time_mask is not None:
                if time_mask.ndim == 1:
                    time_mask = time_mask.unsqueeze(1)
            else:
                time_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=local_obs.device)

            if reset_mask is not None:
                if reset_mask.ndim == 1:
                    reset_mask = reset_mask.unsqueeze(1)

            normalized_agent_mask = None if agent_mask is None else agent_mask.unsqueeze(1)
            return (
                local_obs.unsqueeze(1),
                global_obs.unsqueeze(1),
                normalized_agent_mask,
                time_mask,
                reset_mask,
                True,
            )

        batch_size, sequence_length, _n_agents, _obs_dim = local_obs.shape
        if time_mask is None:
            time_mask = torch.ones((batch_size, sequence_length), dtype=torch.bool, device=local_obs.device)
        return local_obs, global_obs, agent_mask, time_mask, reset_mask, False


def _combine_agent_time_mask(
        *,
        agent_mask: torch.Tensor | None,
        time_mask: torch.Tensor | None,
        batch_size: int,
        sequence_length: int,
        n_agents: int,
        device: torch.device,
) -> torch.Tensor | None:
    valid_mask = None
    if agent_mask is not None:
        if agent_mask.dtype != torch.bool:
            raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
        if agent_mask.shape != (batch_size, sequence_length, n_agents):
            raise ValueError(
                f"Expected agent_mask shape ({batch_size}, {sequence_length}, {n_agents}), got {tuple(agent_mask.shape)}"
            )
        valid_mask = agent_mask

    if time_mask is not None:
        if time_mask.dtype != torch.bool:
            raise ValueError(f"Expected time_mask dtype bool, got {time_mask.dtype}")
        if time_mask.shape != (batch_size, sequence_length):
            raise ValueError(
                f"Expected time_mask shape ({batch_size}, {sequence_length}), got {tuple(time_mask.shape)}"
            )
        time_valid_mask = time_mask.unsqueeze(-1).expand(batch_size, sequence_length, n_agents)
        valid_mask = time_valid_mask if valid_mask is None else valid_mask & time_valid_mask

    if valid_mask is None:
        return torch.ones((batch_size, sequence_length, n_agents), dtype=torch.bool, device=device)
    return valid_mask
