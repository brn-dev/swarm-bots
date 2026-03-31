from dataclasses import dataclass

import torch
from torch import nn

from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal
from swarmbots.learn.nn_components.mlp import MLP


@dataclass(frozen=True)
class MATEncoderConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.0
    act_fn_cls: type[nn.Module] = nn.ReLU
    norm_first: bool = True
    layer_norm_eps: float = 1e-5
    bias: bool = True
    add_agent_embeddings: bool = True
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

        if config.local_obs_encoder_hidden_dims is None or len(config.local_obs_encoder_hidden_dims) == 0:
            self.local_obs_encoder = nn.Linear(self.local_obs_dim, config.d_model)
            init_linear_orthogonal(self.local_obs_encoder)
        else:
            self.local_obs_encoder = MLP(
                input_dim=self.local_obs_dim,
                hidden_dims=[*config.local_obs_encoder_hidden_dims, config.d_model],
                end_with_act_fn=False,
                linear_init=init_linear_orthogonal,
                act_fn_cls=config.act_fn_cls,
            )

        if self.has_global_obs:
            if config.global_obs_encoder_hidden_dims is None or len(config.global_obs_encoder_hidden_dims) == 0:
                self.global_obs_encoder = nn.Linear(self.global_obs_dim, config.d_model)
                init_linear_orthogonal(self.global_obs_encoder)
            else:
                self.global_obs_encoder = MLP(
                    input_dim=self.global_obs_dim,
                    hidden_dims=[*config.global_obs_encoder_hidden_dims, config.d_model],
                    end_with_act_fn=False,
                    linear_init=init_linear_orthogonal,
                    act_fn_cls=config.act_fn_cls,
                )
        else:
            self.global_obs_encoder = None

        self.encoder = nn.TransformerEncoder(
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
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=not config.norm_first,
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
