from typing import TypedDict

import torch
from torch import nn

from swarmbots.learn.algos.ppo.ppo import AGENTS_DIM
from swarmbots.learn.nn_components.mlp import MLP
from swarmbots.learn.nn_components.nn_init import LinearInitialization, init_linear_orthogonal


class MAPPODeepSetCriticHiddenDims(TypedDict):
    local_projection_hidden_dims: list[int]
    value_regressor_hidden_dims: list[int]


class MAPPODeepSetCritic(nn.Module):

    def __init__(
            self,
            n_agents: int,
            local_obs_dim: int,
            global_obs_dim: int,
            local_projection_hidden_dims: list[int],
            value_regressor_hidden_dims: list[int],
            linear_init: LinearInitialization = init_linear_orthogonal,
            act_fun_class = nn.Tanh
    ):
        super().__init__()
        self.n_agents = n_agents
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.has_global_obs = global_obs_dim > 0
        self.local_projection_hidden_dims = local_projection_hidden_dims
        self.value_regressor_hidden_dims = value_regressor_hidden_dims

        self.local_projection = MLP(
            input_dim=local_obs_dim,
            hidden_dims=local_projection_hidden_dims,
            end_with_act_fn=False,
            linear_init=linear_init,
            act_fn_cls=act_fun_class
        )

        value_regressor_input_dim = local_projection_hidden_dims[-1] + global_obs_dim
        self.value_regressor = MLP(
            input_dim=value_regressor_input_dim,
            hidden_dims=value_regressor_hidden_dims + [1],
            end_with_act_fn=False,
            linear_init=linear_init,
            act_fn_cls=act_fun_class
        )

    def forward(self, local_obs: torch.Tensor, global_obs: torch.Tensor) -> torch.Tensor:
        local_projections = self.local_projection(local_obs)
        pooled = local_projections.mean(AGENTS_DIM)

        if self.has_global_obs:
            pooled = torch.cat((pooled, global_obs), dim=-1)

        return self.value_regressor(pooled).squeeze(dim=-1)
