from typing import TypedDict

import torch
from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution
from swarmbots.learn.algos.ppo.ppo_policy import PPOCritic, PPOPolicy
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.nn_components.mlp import MLP
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal, LinearInitialization


AGENTS_DIM = 1

class MAPPOActor(nn.Module):

    def __init__(
            self,
            n_agents: int,
            local_obs_dim: int,
            global_obs_dim: int,
            hidden_dims: list[int],
            latent_pi_dim: int,
            linear_init: LinearInitialization = init_linear_orthogonal,
            act_fun_class = nn.Tanh
    ):
        super().__init__()
        self.n_agents = n_agents
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.has_global_obs = global_obs_dim > 0
        self.hidden_dims = hidden_dims
        self.latent_pi_dim = latent_pi_dim

        self.mlp = MLP(
            input_dim=local_obs_dim + global_obs_dim,
            hidden_dims=hidden_dims + [latent_pi_dim],
            end_with_act_fn=True,
            linear_init=linear_init,
            act_fn_cls=act_fun_class
        )

    def forward(self, local_obs: torch.Tensor, global_obs: torch.Tensor) -> torch.Tensor:
        actor_input = local_obs
        if self.has_global_obs:
            expanded_global_obs = global_obs.unsqueeze(1).expand(-1, self.n_agents, -1)
            actor_input = torch.cat((actor_input, expanded_global_obs), dim=-1)
        
        return self.mlp(actor_input)


MAPPOCritic = PPOCritic

class MAPPODeepSetCriticHiddenDims(TypedDict):
    local_embedding_hidden_dims: list[int]
    aggregator_hidden_dims: list[int]

class MAPPODeepSetCritic(nn.Module):

    def __init__(
            self,
            n_agents: int,
            local_obs_dim: int,
            global_obs_dim: int,
            local_embedding_hidden_dims: list[int],
            aggregator_hidden_dims: list[int],
            linear_init: LinearInitialization = init_linear_orthogonal,
            act_fun_class = nn.Tanh
    ):
        super().__init__()
        self.n_agents = n_agents
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.has_global_obs = global_obs_dim > 0
        self.local_embedding_hidden_dims = local_embedding_hidden_dims
        self.aggregator_hidden_dims = aggregator_hidden_dims

        self.local_embedding = MLP(
            input_dim=local_obs_dim,
            hidden_dims=local_embedding_hidden_dims,
            end_with_act_fn=False,
            linear_init=linear_init,
            act_fn_cls=act_fun_class
        )

        aggregator_input_dim = local_embedding_hidden_dims[-1] + global_obs_dim
        self.aggregator = MLP(
            input_dim=aggregator_input_dim,
            hidden_dims=aggregator_hidden_dims + [1],
            end_with_act_fn=False,
            linear_init=linear_init,
            act_fn_cls=act_fun_class
        )



    def forward(self, local_obs: torch.Tensor, global_obs: torch.Tensor) -> torch.Tensor:
        local_embeddings = self.local_embedding(local_obs)
        pooled = local_embeddings.sum(AGENTS_DIM)

        if self.has_global_obs:
            pooled = torch.cat((pooled, global_obs), dim=-1)

        return self.aggregator(pooled).squeeze(dim=-1)


class MAPPOPolicy(PPOPolicy):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            actor_hidden_dims: list[int],
            latent_pi_dim_per_agent: int,
            critic_hidden_dims: list[int] | MAPPODeepSetCriticHiddenDims,
            act_fun_class = nn.Tanh,
            base_std: float = 1.0
    ):
        BasePolicy.__init__(self)

        self.n_agents = env.n_agents
        self.local_obs_dim = env.local_obs_dim
        self.global_obs_dim = env.global_obs_dim
        self.latent_pi_dim = latent_pi_dim_per_agent

        self.actor = MAPPOActor(
            n_agents=env.n_agents,
            local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim,
            hidden_dims=actor_hidden_dims,
            latent_pi_dim=latent_pi_dim_per_agent,
            act_fun_class=act_fun_class
        )
        self.action_dist = HybridActionDistribution(
            latent_dim=latent_pi_dim_per_agent,
            action_space=env.action_space,
            base_std=base_std,
        )

        if isinstance(critic_hidden_dims, list):
            self.critic = MAPPOCritic(
                n_agents=env.n_agents,
                local_obs_dim=env.local_obs_dim,
                global_obs_dim=env.global_obs_dim,
                hidden_dims=critic_hidden_dims,
                act_fun_class=act_fun_class
            )
        else:
            self.critic = MAPPODeepSetCritic(
                n_agents=env.n_agents,
                local_obs_dim=env.local_obs_dim,
                global_obs_dim=env.global_obs_dim,
                local_embedding_hidden_dims=critic_hidden_dims['local_embedding_hidden_dims'],
                aggregator_hidden_dims=critic_hidden_dims['aggregator_hidden_dims'],
                act_fun_class=act_fun_class
            )
