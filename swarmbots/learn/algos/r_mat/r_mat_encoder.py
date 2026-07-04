from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

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
from swarmbots.learn.nn_components.nn_init import (
    make_init_linear_orthogonal,
    reinitialize_multihead_attention,
)
from swarmbots.learn.temporal_state import (
    flatten_temporal_state_batch_agents,
    unflatten_temporal_state_batch_agents,
)

RMATEncoderState = list[TemporalModelState]


@dataclass(frozen=True)
class RMATEncoderConfig(MATEncoderConfig):
    temporal_model_cls: type[TemporalSequenceModel] | Sequence[type[TemporalSequenceModel]] = LSTMTemporalSequenceModel
    temporal_model_config: Any = field(default_factory=LSTMTemporalSequenceModelConfig)
    temporal_model_order: Literal["temporal_first", "inter_agent_attention_first"] = "inter_agent_attention_first"
    inter_module_mlp: bool = False
    temporal_residual: bool = True
    temporal_layer_norm: bool = True


class _RMATTransformerEncoderLayer(nn.Module):

    def __init__(
            self,
            config: RMATEncoderConfig,
            *,
            layer_idx: int,
    ) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.temporal_model_order = config.temporal_model_order
        self.self_attn = nn.MultiheadAttention(
            config.d_model,
            config.nhead,
            dropout=config.dropout,
            bias=config.bias,
            batch_first=True,
        )
        temporal_model_cls = _resolve_per_layer_value(
            config.temporal_model_cls,
            layer_idx=layer_idx,
            num_layers=config.num_layers,
            name="temporal_model_cls",
        )
        temporal_model_config = _resolve_per_layer_value(
            config.temporal_model_config,
            layer_idx=layer_idx,
            num_layers=config.num_layers,
            name="temporal_model_config",
        )
        self.temporal_model = temporal_model_cls(
            hidden_dim=config.d_model,
            config=temporal_model_config,
        )
        temporal_output_projection_gain = (
            config.linear_init_gain
            if config.linear_projection_init_gain is None
            else config.linear_projection_init_gain
        )
        self.temporal_output_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        nn.init.orthogonal_(self.temporal_output_projection.weight, gain=temporal_output_projection_gain)
        self.temporal_norm: nn.Module = (
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
            if config.temporal_layer_norm
            else nn.Identity()
        )
        self.attention_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.feedforward_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        feedforward_linear_init = (
            make_init_linear_orthogonal(config.transformer_ff_init_gain)
            if config.transformer_ff_init_gain is not None
            else make_init_linear_orthogonal(config.linear_init_gain)
        )
        feedforward_projection_init = make_init_linear_orthogonal(1.0)
        self.inter_module_feedforward: MLP | None = (
            MLP(
                input_dim=config.d_model,
                hidden_dims=[config.dim_feedforward, config.d_model],
                end_with_act_fn=False,
                linear_init=feedforward_linear_init,
                final_linear_init=feedforward_projection_init,
                act_fn_cls=config.act_fn_cls,
                bias=config.bias,
                dropout=config.dropout,
            )
            if config.inter_module_mlp
            else None
        )
        self.inter_module_feedforward_norm: nn.LayerNorm | None = (
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
            if config.inter_module_mlp
            else None
        )
        self.feedforward = MLP(
            input_dim=config.d_model,
            hidden_dims=[config.dim_feedforward, config.d_model],
            end_with_act_fn=False,
            linear_init=feedforward_linear_init,
            final_linear_init=feedforward_projection_init,
            act_fn_cls=config.act_fn_cls,
            bias=config.bias,
            dropout=config.dropout,
        )
        self.temporal_dropout = nn.Dropout(config.dropout)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.feedforward_dropout = nn.Dropout(config.dropout)
        self.inter_module_feedforward_dropout = nn.Dropout(config.dropout)
        self.temporal_residual = config.temporal_residual
        self.temporal_layer_norm = config.temporal_layer_norm
        self.norm_first = config.norm_first
        if config.transformer_ff_init_gain is not None:
            reinitialize_multihead_attention(self.self_attn)

        if self.temporal_model_order not in {"temporal_first", "inter_agent_attention_first"}:
            raise ValueError(f"Unknown temporal_model_order: {self.temporal_model_order}")

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
        valid_agent_time_mask = _combine_agent_time_mask(
            agent_mask=agent_mask,
            time_mask=time_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            n_agents=n_agents,
            device=embeddings.device,
        )

        hidden = embeddings
        if self.temporal_model_order == "temporal_first":
            hidden, next_state = self._temporal_block(
                hidden,
                valid_agent_time_mask=valid_agent_time_mask,
                initial_state=initial_state,
                reset_mask=reset_mask,
            )
            hidden = self._apply_inter_module_feedforward(hidden, valid_agent_time_mask=valid_agent_time_mask)
            hidden = self._inter_agent_attention_block(
                hidden,
                agent_mask=agent_mask,
                valid_agent_time_mask=valid_agent_time_mask,
            )
        else:
            hidden = self._inter_agent_attention_block(
                hidden,
                agent_mask=agent_mask,
                valid_agent_time_mask=valid_agent_time_mask,
            )
            hidden = self._apply_inter_module_feedforward(hidden, valid_agent_time_mask=valid_agent_time_mask)
            hidden, next_state = self._temporal_block(
                hidden,
                valid_agent_time_mask=valid_agent_time_mask,
                initial_state=initial_state,
                reset_mask=reset_mask,
            )

        hidden = self._feedforward_block(
            hidden,
            feedforward=self.feedforward,
            norm=self.feedforward_norm,
            dropout=self.feedforward_dropout,
        )
        hidden = _mask_invalid_agent_time(hidden, valid_agent_time_mask)
        return hidden.contiguous(), next_state

    def _inter_agent_attention_block(
            self,
            embeddings: torch.Tensor,
            *,
            agent_mask: torch.Tensor | None,
            valid_agent_time_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attention_inputs = self.attention_norm(embeddings) if self.norm_first else embeddings
        batch_size, sequence_length, n_agents, hidden_dim = attention_inputs.shape
        flat_attention_inputs = attention_inputs.reshape(batch_size * sequence_length, n_agents, hidden_dim)
        flat_agent_mask = (
            None
            if agent_mask is None
            else agent_mask.reshape(batch_size * sequence_length, n_agents)
        )
        attention_outputs = self.self_attn(
            flat_attention_inputs,
            flat_attention_inputs,
            flat_attention_inputs,
            key_padding_mask=None if flat_agent_mask is None else ~flat_agent_mask,
            need_weights=False,
        )[0].reshape(batch_size, sequence_length, n_agents, hidden_dim)

        if self.norm_first:
            outputs = embeddings + self.attention_dropout(attention_outputs)
        else:
            outputs = self.attention_norm(embeddings + self.attention_dropout(attention_outputs))
        return _mask_invalid_agent_time(outputs, valid_agent_time_mask)

    def _temporal_block(
            self,
            embeddings: torch.Tensor,
            *,
            valid_agent_time_mask: torch.Tensor | None,
            initial_state: TemporalModelState | None,
            reset_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, TemporalModelState]:
        batch_size, sequence_length, n_agents, hidden_dim = embeddings.shape
        temporal_inputs = self.temporal_norm(embeddings) if self.temporal_layer_norm and self.norm_first else embeddings

        flat_temporal_inputs = temporal_inputs.permute(0, 2, 1, 3).reshape(
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

        temporal_model_outputs, next_state = self.temporal_model(
            flat_temporal_inputs,
            valid_mask=temporal_valid_mask,
            initial_state=initial_state,
            reset_mask=temporal_reset_mask,
        )
        temporal_model_outputs = self.temporal_output_projection(temporal_model_outputs)
        temporal_model_outputs = temporal_model_outputs.reshape(
            batch_size,
            n_agents,
            sequence_length,
            hidden_dim,
        ).permute(0, 2, 1, 3)
        if self.temporal_residual:
            outputs = embeddings + self.temporal_dropout(temporal_model_outputs)
        else:
            outputs = temporal_model_outputs
        if self.temporal_layer_norm and not self.norm_first:
            outputs = self.temporal_norm(outputs)
        return _mask_invalid_agent_time(outputs, valid_agent_time_mask), next_state

    def _apply_inter_module_feedforward(
            self,
            embeddings: torch.Tensor,
            *,
            valid_agent_time_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.inter_module_feedforward is None:
            return embeddings
        assert self.inter_module_feedforward_norm is not None
        outputs = self._feedforward_block(
            embeddings,
            feedforward=self.inter_module_feedforward,
            norm=self.inter_module_feedforward_norm,
            dropout=self.inter_module_feedforward_dropout,
        )
        return _mask_invalid_agent_time(outputs, valid_agent_time_mask)

    def _feedforward_block(
            self,
            embeddings: torch.Tensor,
            *,
            feedforward: MLP,
            norm: nn.LayerNorm,
            dropout: nn.Dropout,
    ) -> torch.Tensor:
        if self.norm_first:
            return embeddings + dropout(feedforward(norm(embeddings)))
        return norm(embeddings + dropout(feedforward(embeddings)))


class _RMATTransformerEncoder(nn.Module):

    def __init__(
            self,
            config: RMATEncoderConfig,
    ) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList([
            _RMATTransformerEncoderLayer(config=config, layer_idx=layer_idx)
            for layer_idx in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
            self,
            embeddings: torch.Tensor,
            *,
            agent_mask: torch.Tensor | None,
            time_mask: torch.Tensor | None,
            initial_states: RMATEncoderState,
            reset_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, RMATEncoderState]:
        if len(initial_states) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} initial states, got {len(initial_states)}")

        next_states: RMATEncoderState = []
        hidden = embeddings
        for layer, layer_state in zip(self.layers, initial_states, strict=True):
            hidden, next_state = layer(
                hidden,
                agent_mask=agent_mask,
                time_mask=time_mask,
                initial_state=layer_state,
                reset_mask=reset_mask,
            )
            next_states.append(next_state)

        return self.norm(hidden), next_states


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
        self.local_obs_input_norm = nn.LayerNorm(local_obs_dim) if config.normalize_obs_inputs else nn.Identity()
        self.global_obs_input_norm = (
            nn.LayerNorm(global_obs_dim)
            if config.normalize_obs_inputs and self.has_global_obs
            else nn.Identity()
        )
        self.token_norm = nn.LayerNorm(config.d_model) if config.normalize_tokens else nn.Identity()
        linear_init = make_init_linear_orthogonal(config.linear_init_gain)
        projection_linear_init = (
            linear_init
            if config.linear_projection_init_gain is None
            else make_init_linear_orthogonal(config.linear_projection_init_gain)
        )

        if config.local_obs_encoder_hidden_dims:
            self.local_obs_encoder = MLP(
                input_dim=self.local_obs_dim,
                hidden_dims=[*config.local_obs_encoder_hidden_dims, config.d_model],
                end_with_act_fn=False,
                linear_init=linear_init,
                final_linear_init=projection_linear_init,
                act_fn_cls=config.act_fn_cls,
                bias=config.bias,
            )
        else:
            self.local_obs_encoder = nn.Linear(self.local_obs_dim, config.d_model, bias=config.bias)
            projection_linear_init(self.local_obs_encoder)

        if self.has_global_obs:
            if config.global_obs_encoder_hidden_dims:
                self.global_obs_encoder = MLP(
                    input_dim=self.global_obs_dim,
                    hidden_dims=[*config.global_obs_encoder_hidden_dims, config.d_model],
                    end_with_act_fn=False,
                    linear_init=linear_init,
                    final_linear_init=projection_linear_init,
                    act_fn_cls=config.act_fn_cls,
                    bias=config.bias,
                )
            else:
                self.global_obs_encoder = nn.Linear(self.global_obs_dim, config.d_model, bias=config.bias)
                projection_linear_init(self.global_obs_encoder)
        else:
            self.global_obs_encoder = None

        self.encoder = _RMATTransformerEncoder(config=config)

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
            n_agents: int,
            *,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> RMATEncoderState:
        return [
            unflatten_temporal_state_batch_agents(
                layer.temporal_model.initial_state(
                    batch_size=batch_size * n_agents,
                    device=device,
                    dtype=dtype,
                ),
                batch_size=batch_size,
                n_agents=n_agents,
            )
            for layer in self.layers
        ]

    @property
    def layers(self) -> nn.ModuleList:
        return self.encoder.layers

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
        if n_agents > self.max_agents:
            raise ValueError(f"Expected local_obs agent dim <= {self.max_agents}, got {n_agents}")

        local_obs = self.local_obs_input_norm(local_obs)
        embeddings = self.local_obs_encoder(local_obs)
        if self.agent_embeddings is not None:
            embeddings = embeddings + self.agent_embeddings[:, :n_agents, :].unsqueeze(1)

        if self.has_global_obs:
            global_obs = self.global_obs_input_norm(global_obs)
            global_embeddings = self.global_obs_encoder(global_obs)
            embeddings = embeddings + global_embeddings.unsqueeze(2)
        embeddings = self.token_norm(embeddings)

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
            batch_size=batch_size,
            n_agents=n_agents,
            device=local_obs.device,
            dtype=embeddings.dtype,
        )

        hidden, flat_next_states = self.encoder(
            embeddings,
            agent_mask=agent_mask,
            time_mask=time_mask,
            initial_states=layer_states,
            reset_mask=reset_mask,
        )
        next_states = [
            unflatten_temporal_state_batch_agents(
                next_state,
                batch_size=batch_size,
                n_agents=n_agents,
            )
            for next_state in flat_next_states
        ]

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
            n_agents: int,
            device: torch.device,
            dtype: torch.dtype,
    ) -> RMATEncoderState:
        if initial_state is None:
            return [
                layer.temporal_model.initial_state(
                    batch_size=batch_size * n_agents,
                    device=device,
                    dtype=dtype,
                )
                for layer in self.layers
            ]
        return [
            flatten_temporal_state_batch_agents(layer_state)
            for layer_state in initial_state
        ]

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


def _mask_invalid_agent_time(
        values: torch.Tensor,
        valid_agent_time_mask: torch.Tensor | None,
) -> torch.Tensor:
    if valid_agent_time_mask is None:
        return values
    return values.masked_fill(~valid_agent_time_mask.unsqueeze(-1), 0.0)


def _resolve_per_layer_value(
        value: Any,
        *,
        layer_idx: int,
        num_layers: int,
        name: str,
) -> Any:
    if not _is_per_layer_sequence(value):
        return value
    if len(value) != num_layers:
        raise ValueError(f"Expected {name} to have {num_layers} entries, got {len(value)}")
    return value[layer_idx]


def _is_per_layer_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
