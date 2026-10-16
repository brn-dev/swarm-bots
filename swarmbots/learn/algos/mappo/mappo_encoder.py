from dataclasses import dataclass, field

import torch
from torch import nn

from swarmbots.learn.nn_components.feed_forward import MLP
from swarmbots.learn.nn_components.nn_init import LinearInitialization, init_linear_orthogonal


@dataclass(frozen=True)
class MAPPOEncoderConfig:
    hidden_dims: list[int] = field(default_factory=list)
    latent_dim: int = 64
    act_fun_class: type[nn.Module] = nn.Tanh


class MAPPOEncoder(nn.Module):
    def __init__(
        self, n_agents: int, local_obs_dim: int, global_obs_dim: int, config: MAPPOEncoderConfig,
        linear_init: LinearInitialization = init_linear_orthogonal,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.has_global_obs = global_obs_dim > 0
        self.mlp = MLP(
            input_dim=local_obs_dim + global_obs_dim,
            hidden_dims=[*config.hidden_dims, config.latent_dim],
            end_with_act_fn=True,
            linear_init=linear_init,
            act_fn_cls=config.act_fun_class,
        )

    def forward(self, local_obs: torch.Tensor, global_obs: torch.Tensor) -> torch.Tensor:
        if self.has_global_obs:
            global_obs = global_obs.unsqueeze(1).expand(-1, self.n_agents, -1)
            local_obs = torch.cat((local_obs, global_obs), dim=-1)
        return self.mlp(local_obs)
