from dataclasses import dataclass, field

import torch
from torch import nn

from swarmbots.learn.nn_components.mlp import MLP
from swarmbots.learn.nn_components.nn_init import LinearInitialization, init_linear_orthogonal


@dataclass(frozen=True)
class MAPPOActorConfig:
    hidden_dims: list[int] = field(default_factory=list)
    latent_pi_dim: int = 64
    act_fun_class: type[nn.Module] = nn.Tanh


class MAPPOActor(nn.Module):

    def __init__(
            self,
            n_agents: int,
            local_obs_dim: int,
            global_obs_dim: int,
            config: MAPPOActorConfig,
            linear_init: LinearInitialization = init_linear_orthogonal,
            act_fun_class: type[nn.Module] | None = None,
    ):
        super().__init__()
        actor_act_fun_class = config.act_fun_class if act_fun_class is None else act_fun_class
        self.n_agents = n_agents
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.has_global_obs = global_obs_dim > 0
        self.hidden_dims = config.hidden_dims
        self.latent_pi_dim = config.latent_pi_dim

        self.mlp = MLP(
            input_dim=local_obs_dim + global_obs_dim,
            hidden_dims=[*config.hidden_dims, config.latent_pi_dim],
            end_with_act_fn=True,
            linear_init=linear_init,
            act_fn_cls=actor_act_fun_class
        )

    def forward(self, local_obs: torch.Tensor, global_obs: torch.Tensor) -> torch.Tensor:
        actor_input = local_obs
        if self.has_global_obs:
            expanded_global_obs = global_obs.unsqueeze(1).expand(-1, self.n_agents, -1)
            actor_input = torch.cat((actor_input, expanded_global_obs), dim=-1)

        return self.mlp(actor_input)
