from typing import Callable

import torch
from torch import nn

from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal
from swarmbots.learn.nn_components.mlp import MLP


class MATEncoder(nn.Module):

    def __init__(
            self,
            n_agents: int,
            local_obs_dim: int,
            global_obs_dim: int,
            d_model: int,
            num_layers: int,
            bias: bool,
            norm_first: bool,
            layer_norm_eps: float,
            activation: Callable[[torch.Tensor], torch.Tensor],
            act_fn_cls: type[nn.Module],
            dropout: float,
            dim_feedforward: int,
            nhead: int,
            output_norm: nn.Module,
            enable_nested_tensor: bool = True,
            add_agent_embeddings: bool = True,
            local_obs_encoder_hidden_dims: list[int] | None = None,
            global_obs_encoder_hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.n_agents: int = n_agents
        self.local_obs_dim: int = local_obs_dim
        self.global_obs_dim: int = global_obs_dim
        self.has_global_obs: bool = global_obs_dim > 0
        self.add_agent_embeddings = add_agent_embeddings

        if local_obs_encoder_hidden_dims is None or len(local_obs_encoder_hidden_dims) == 0:
            self.local_obs_encoder = nn.Linear(self.local_obs_dim, d_model)
            init_linear_orthogonal(self.local_obs_encoder)
        else:
            self.local_obs_encoder = MLP(
                input_dim=self.local_obs_dim,
                hidden_dims=[*local_obs_encoder_hidden_dims, d_model],
                end_with_act_fn=False,
                linear_init=init_linear_orthogonal,
                act_fn_cls=act_fn_cls,
            )

        if self.has_global_obs:
            if global_obs_encoder_hidden_dims is None or len(global_obs_encoder_hidden_dims) == 0:
                self.global_obs_encoder = nn.Linear(self.global_obs_dim, d_model)
                init_linear_orthogonal(self.global_obs_encoder)
            else:
                self.global_obs_encoder = MLP(
                    input_dim=self.global_obs_dim,
                    hidden_dims=[*global_obs_encoder_hidden_dims, d_model],
                    end_with_act_fn=False,
                    linear_init=init_linear_orthogonal,
                    act_fn_cls=act_fn_cls,
                )
        else:
            self.global_obs_encoder = None

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=True,
                norm_first=norm_first,
                bias=bias,
            ),
            num_layers=num_layers,
            norm=output_norm,
            enable_nested_tensor=enable_nested_tensor,
        )

        self.agent_embeddings: nn.Parameter | None = None
        if self.add_agent_embeddings:
            self.agent_embeddings = nn.Parameter(
                torch.zeros(1, n_agents, d_model), requires_grad=True
            )
            nn.init.orthogonal_(self.agent_embeddings)

    def forward(
        self,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_embeddings = self.local_obs_encoder(local_obs)
        if self.agent_embeddings is not None:
            local_embeddings = local_embeddings + self.agent_embeddings

        if self.has_global_obs:
            global_embeddings = self.global_obs_encoder(global_obs)
            expanded_global_embeddings = global_embeddings.unsqueeze(1).expand(-1, self.n_agents, -1)
            local_embeddings = local_embeddings + expanded_global_embeddings

        src_key_padding_mask = None
        if agent_mask is not None:
            if agent_mask.shape != local_obs.shape[:2]:
                raise ValueError(
                    f"Expected agent_mask shape {tuple(local_obs.shape[:2])}, got {tuple(agent_mask.shape)}"
                )
            src_key_padding_mask = ~agent_mask

        augmented_observations = self.encoder(local_embeddings, src_key_padding_mask=src_key_padding_mask)
        return augmented_observations
