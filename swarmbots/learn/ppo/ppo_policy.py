import torch
from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution
from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper


class PPOActor(nn.Module):

    def __init__(
            self,
            local_obs_dim: int,
            global_obs_dim: int,
            hidden_dims: list[int],
            act_fun_class = nn.Tanh
    ):
        super().__init__()
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.has_global_obs = global_obs_dim > 0
        self.hidden_dims = hidden_dims
        dims = [local_obs_dim + global_obs_dim, *hidden_dims]

        modules = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i+1]))
            modules.append(act_fun_class())

        self.mlp = nn.Sequential(*modules)

    def forward(self, local_obs: torch.Tensor, global_obs: torch.Tensor) -> torch.Tensor:
        if self.has_global_obs:
            n_agents = local_obs.shape[1]
            global_obs_expanded = global_obs.unsqueeze(1).expand(-1, n_agents, -1)
            return self.mlp(torch.concatenate(
                (local_obs, global_obs_expanded),
                dim=-1
            ))
        else:
            return self.mlp(local_obs)


class PPOCritic(nn.Module):

    def __init__(
            self,
            local_obs_dim: int,
            global_obs_dim: int,
            hidden_dims: list[int],
            act_fun_class = nn.Tanh
    ):
        super().__init__()
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.has_global_obs = global_obs_dim > 0
        self.hidden_dims = hidden_dims
        dims = [local_obs_dim + global_obs_dim, *hidden_dims, 1]

        modules = []
        for i in range(len(dims) - 2):
            modules.append(nn.Linear(dims[i], dims[i+1]))
            modules.append(act_fun_class())
        modules.append(nn.Linear(dims[-2], dims[-1]))

        self.mlp = nn.Sequential(*modules)

    def forward(self, local_obs: torch.Tensor, global_obs: torch.Tensor) -> torch.Tensor:
        if self.has_global_obs:
            n_agents = local_obs.shape[1]
            global_obs_expanded = global_obs.unsqueeze(1).expand(-1, n_agents, -1)
            return self.mlp(torch.concatenate(
                (local_obs, global_obs_expanded),
                dim=-1
            )).squeeze(dim=-1).mean(dim=-1)
        else:
            return self.mlp(local_obs).squeeze(dim=-1).mean(dim=-1)


class PPOPolicy(nn.Module):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            actor_hidden_dims: list[int],
            critic_hidden_dims: list[int],
            act_fun_class = nn.Tanh
    ):
        super().__init__()

        self.local_obs_dim = env.local_obs_dim
        self.global_obs_dim = env.global_obs_dim

        self.actor = PPOActor(
            local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim,
            hidden_dims=actor_hidden_dims,
            act_fun_class=act_fun_class
        )
        self.critic = PPOCritic(
            local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim,
            hidden_dims=critic_hidden_dims,
            act_fun_class=act_fun_class
        )

        self.action_dist = HybridActionDistribution(
            latent_dim=actor_hidden_dims[-1],
            action_space=env.action_space,
            action_net_initialization=None,  # todo
            base_std=1.0,  # todo
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