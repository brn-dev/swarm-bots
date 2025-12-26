import torch
from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.nn_components.mlp import MLP
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal, LinearInitialization


class PPOActor(nn.Module):

    def __init__(
            self,
            n_agents: int,
            local_obs_dim: int,
            global_obs_dim: int,
            hidden_dims: list[int],
            latent_pi_dim_per_agent: int,
            linear_init: LinearInitialization = init_linear_orthogonal,
            act_fun_class = nn.Tanh
    ):
        super().__init__()
        self.n_agents = n_agents
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.has_global_obs = global_obs_dim > 0
        self.hidden_dims = hidden_dims
        self.latent_pi_dim = latent_pi_dim_per_agent

        self.mlp = MLP(
            input_dim=local_obs_dim * n_agents + global_obs_dim,
            hidden_dims=hidden_dims + [latent_pi_dim_per_agent * n_agents],
            end_with_act_fn=True,
            linear_init=linear_init,
            act_fn_cls=act_fun_class
        )

    def forward(self, local_obs: torch.Tensor, global_obs: torch.Tensor) -> torch.Tensor:
        actor_input = torch.flatten(local_obs, start_dim=1)
        if self.has_global_obs:
            actor_input = torch.cat((actor_input, global_obs), dim=-1)
        return self.mlp(actor_input).view((local_obs.shape[0], self.n_agents, self.latent_pi_dim))


class PPOCritic(nn.Module):

    def __init__(
            self,
            n_agents: int,
            local_obs_dim: int,
            global_obs_dim: int,
            hidden_dims: list[int],
            linear_init: LinearInitialization = init_linear_orthogonal,
            act_fun_class = nn.Tanh
    ):
        super().__init__()
        self.n_agents = n_agents
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.has_global_obs = global_obs_dim > 0
        self.hidden_dims = hidden_dims

        self.mlp = MLP(
            input_dim=local_obs_dim * n_agents + global_obs_dim,
            hidden_dims=hidden_dims + [1],
            end_with_act_fn=False,
            linear_init=linear_init,
            act_fn_cls=act_fun_class
        )

    def forward(self, local_obs: torch.Tensor, global_obs: torch.Tensor) -> torch.Tensor:
        critic_input = torch.flatten(local_obs, start_dim=1)
        if self.has_global_obs:
            critic_input = torch.cat((critic_input, global_obs), dim=-1)
        return self.mlp(critic_input).squeeze(dim=-1)


class PPOPolicy(BasePolicy):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            actor_hidden_dims: list[int],
            latent_pi_dim_per_agent: int,
            critic_hidden_dims: list[int],
            act_fun_class = nn.Tanh,
            base_std: float = 1.0
    ):
        super().__init__()

        self.n_agents = env.n_agents
        self.local_obs_dim = env.local_obs_dim
        self.global_obs_dim = env.global_obs_dim
        self.latent_pi_dim = latent_pi_dim_per_agent

        self.actor = PPOActor(
            n_agents=env.n_agents,
            local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim,
            hidden_dims=actor_hidden_dims,
            latent_pi_dim_per_agent=latent_pi_dim_per_agent,
            act_fun_class=act_fun_class
        )
        self.action_dist = HybridActionDistribution(
            latent_dim=latent_pi_dim_per_agent,
            action_space=env.action_space,
            base_std=base_std,
        )

        self.critic = PPOCritic(
            n_agents=env.n_agents,
            local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim,
            hidden_dims=critic_hidden_dims,
            act_fun_class=act_fun_class
        )

    def forward(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            deterministic: bool = False
    ):
        latent_pi = self.actor(local_obs, global_obs)
        actions, log_probs = self.action_dist.get_actions_with_log_probs(latent_pi, deterministic)

        values = self.critic(local_obs, global_obs)
        return actions, log_probs, values

    def evaluate_actions(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
    ):
        latent_pi = self.actor(local_obs, global_obs)
        self.action_dist.update_latent_features(latent_pi)
        log_probs = self.action_dist.log_prob(actions)
        entropies = self.action_dist.entropy()

        values = self.critic(local_obs, global_obs)
        return log_probs, entropies, values

    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            deterministic: bool = False
    ) -> torch.Tensor:
        latent_pi = self.actor(local_obs, global_obs)
        actions = self.action_dist.update_latent_features(latent_pi).get_actions(deterministic)
        return actions