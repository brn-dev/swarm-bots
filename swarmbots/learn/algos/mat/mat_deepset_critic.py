import torch
from torch import nn

from swarmbots.learn.algos.ppo.ppo import AGENTS_DIM
from swarmbots.learn.nn_components.mlp import MLP
from swarmbots.learn.nn_components.nn_init import LinearInitialization, init_linear_orthogonal


class MATDeepSetCritic(nn.Module):

    def __init__(
            self,
            n_agents: int,
            augmented_observations_dim: int,
            local_projection_hidden_dims: list[int],
            value_regressor_hidden_dims: list[int],
            linear_init: LinearInitialization = init_linear_orthogonal,
            act_fn_class=nn.ReLU
    ):
        super().__init__()
        self.n_agents = n_agents
        self.augmented_observations_dim = augmented_observations_dim
        self.local_projection_hidden_dims = local_projection_hidden_dims
        self.value_regressor_hidden_dims = value_regressor_hidden_dims

        if local_projection_hidden_dims:
            self.local_projection = MLP(
                input_dim=augmented_observations_dim,
                hidden_dims=local_projection_hidden_dims,
                end_with_act_fn=False,
                linear_init=linear_init,
                act_fn_cls=act_fn_class
            )
            value_regressor_input_dim = local_projection_hidden_dims[-1]
        else:
            self.local_projection = nn.Identity()
            value_regressor_input_dim = augmented_observations_dim


        self.value_regressor = MLP(
            input_dim=value_regressor_input_dim,
            hidden_dims=value_regressor_hidden_dims + [1],
            end_with_act_fn=False,
            linear_init=linear_init,
            act_fn_cls=act_fn_class
        )

    def forward(self, augmented_observations: torch.Tensor) -> torch.Tensor:
        local_projections = self.local_projection(augmented_observations)
        pooled = local_projections.mean(AGENTS_DIM)

        return self.value_regressor(pooled).squeeze(dim=-1)
