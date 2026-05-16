from dataclasses import dataclass

import torch
from torch import nn

from swarmbots.learn.nn_components.activations import ActivationFactory, make_activation
from swarmbots.learn.nn_components.nn_init import (
    DEFAULT_ORTHOGONAL_GAIN,
    make_init_linear_orthogonal,
    reinitialize_transformer_stack,
)
from swarmbots.learn.nn_components.mlp import MLP


@dataclass(frozen=True)
class MATEncoderConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.0
    act_fn_cls: ActivationFactory = nn.ReLU
    norm_first: bool = True
    layer_norm_eps: float = 1e-5
    bias: bool = True
    add_agent_embeddings: bool = True
    linear_init_gain: float = DEFAULT_ORTHOGONAL_GAIN
    linear_projection_init_gain: float | None = None
    transformer_ff_init_gain: float | None = None
    local_obs_encoder_hidden_dims: list[int] | None = None
    global_obs_encoder_hidden_dims: list[int] | None = None


class MATEncoder(nn.Module):

    def __init__(
            self,
            config: MATEncoderConfig,
            *,
            max_agents: int,
            local_obs_dim: int,
            global_obs_dim: int,
    ) -> None:
        super().__init__()
        self.local_obs_dim: int = local_obs_dim
        self.global_obs_dim: int = global_obs_dim
        self.has_global_obs: bool = global_obs_dim > 0
        self.add_agent_embeddings = config.add_agent_embeddings
        self.max_agents = max_agents
        linear_init = make_init_linear_orthogonal(config.linear_init_gain)
        projection_linear_init = (
            linear_init
            if config.linear_projection_init_gain is None
            else make_init_linear_orthogonal(config.linear_projection_init_gain)
        )

        if config.local_obs_encoder_hidden_dims is None or len(config.local_obs_encoder_hidden_dims) == 0:
            self.local_obs_encoder = nn.Linear(self.local_obs_dim, config.d_model)
            projection_linear_init(self.local_obs_encoder)
        else:
            self.local_obs_encoder = MLP(
                input_dim=self.local_obs_dim,
                hidden_dims=[*config.local_obs_encoder_hidden_dims, config.d_model],
                end_with_act_fn=False,
                linear_init=linear_init,
                final_linear_init=projection_linear_init,
                act_fn_cls=config.act_fn_cls,
            )

        if self.has_global_obs:
            if config.global_obs_encoder_hidden_dims is None or len(config.global_obs_encoder_hidden_dims) == 0:
                self.global_obs_encoder = nn.Linear(self.global_obs_dim, config.d_model)
                projection_linear_init(self.global_obs_encoder)
            else:
                self.global_obs_encoder = MLP(
                    input_dim=self.global_obs_dim,
                    hidden_dims=[*config.global_obs_encoder_hidden_dims, config.d_model],
                    end_with_act_fn=False,
                    linear_init=linear_init,
                    final_linear_init=projection_linear_init,
                    act_fn_cls=config.act_fn_cls,
                )
        else:
            self.global_obs_encoder = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=make_activation(config.act_fn_cls, num_features=config.dim_feedforward),
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
            norm_first=config.norm_first,
            bias=config.bias,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=not config.norm_first,
        )
        if config.transformer_ff_init_gain is not None:
            reinitialize_transformer_stack(
                self.encoder,
                feedforward_init_gain=config.transformer_ff_init_gain,
            )

        self.agent_embeddings: nn.Parameter | None = None
        if self.add_agent_embeddings:
            self.agent_embeddings = nn.Parameter(
                torch.zeros(1, self.max_agents, config.d_model), requires_grad=True
            )
            nn.init.orthogonal_(self.agent_embeddings)

    def forward(
        self,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_agents = local_obs.shape[1]
        if n_agents > self.max_agents:
            raise ValueError(f"Expected local_obs second dim <= {self.max_agents}, got {n_agents}")
        local_embeddings = self.local_obs_encoder(local_obs)
        if self.agent_embeddings is not None:
            local_embeddings = local_embeddings + self.agent_embeddings[:, :n_agents, :]

        if self.has_global_obs:
            global_embeddings = self.global_obs_encoder(global_obs)
            expanded_global_embeddings = global_embeddings.unsqueeze(1).expand(-1, n_agents, -1)
            local_embeddings = local_embeddings + expanded_global_embeddings

        src_key_padding_mask = None
        if agent_mask is not None:
            src_key_padding_mask = ~agent_mask

        augmented_observations = self.encoder(local_embeddings, src_key_padding_mask=src_key_padding_mask)
        return augmented_observations
