import abc
from typing import Any
import torch
from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import (
    HybridActionDistribution,
    ContinuousActionDistConfig,
    serialize_continuous_action_dist_configs,
)
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.nn_components.mlp import MLP
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal, LinearInitialization


class BasePPOPolicy(BasePolicy, abc.ABC):
    action_dist: HybridActionDistribution

    @property
    def gsde_enabled(self) -> bool:
        return self.action_dist.has_gsde

    @abc.abstractmethod
    def forward(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :return: return actions, log_probs, values
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate_actions(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        :return: log_probs, entropies, values
        """
        raise NotImplementedError()


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

    def forward(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if agent_mask is not None:
            if agent_mask.dtype != torch.bool:
                raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
            if agent_mask.shape != local_obs.shape[:2]:
                raise ValueError(
                    f"Expected agent_mask shape {tuple(local_obs.shape[:2])}, got {tuple(agent_mask.shape)}"
                )
            local_obs = local_obs * agent_mask.to(dtype=local_obs.dtype).unsqueeze(-1)
        critic_input = torch.flatten(local_obs, start_dim=1)
        if self.has_global_obs:
            critic_input = torch.cat((critic_input, global_obs), dim=-1)
        return self.mlp(critic_input).squeeze(dim=-1)


class PPOPolicy(BasePPOPolicy):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            actor_hidden_dims: list[int],
            latent_pi_dim_per_agent: int,
            critic_hidden_dims: list[int],
            act_fun_class = nn.Tanh,
            continuous_config: ContinuousActionDistConfig | list[ContinuousActionDistConfig | None] | None = None,
            bernoulli_initial_prob: float | None = None,
    ):
        super().__init__()

        self.n_agents = env.n_agents
        self.local_obs_dim = env.local_obs_dim
        self.global_obs_dim = env.global_obs_dim
        self.hidden_vars_dim = env.hidden_vars_dim
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
            continuous_config=continuous_config,
            bernoulli_initial_prob=bernoulli_initial_prob,
        )

        self.critic = PPOCritic(
            n_agents=env.n_agents,
            local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim + self.hidden_vars_dim,
            hidden_dims=critic_hidden_dims,
            act_fun_class=act_fun_class
        )

        self.hyper_parameters = {
            "actor_hidden_dims": actor_hidden_dims,
            "critic_hidden_dims": critic_hidden_dims,
            "latent_pi_dim_per_agent": latent_pi_dim_per_agent,
            "act_fun_class": act_fun_class.__name__,
            "continuous_config": serialize_continuous_action_dist_configs(continuous_config),
            "bernoulli_initial_prob": bernoulli_initial_prob,
        }

    def get_hyper_parameters(self) -> dict[str, Any]:
        return self.hyper_parameters

    def forward(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_obs = self._mask_local_obs(local_obs, agent_mask)
        latent_pi = self.actor(local_obs, global_obs)
        actions, log_probs = self.action_dist.get_actions_with_log_probs(latent_pi, deterministic)

        critic_global_obs = self._build_critic_global_obs(global_obs, hidden_vars)
        values = self.critic(local_obs, critic_global_obs, agent_mask=agent_mask)
        return actions, log_probs, values

    def evaluate_actions(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        local_obs = self._mask_local_obs(local_obs, agent_mask)
        latent_pi = self.actor(local_obs, global_obs)
        self.action_dist.update_latent_features(latent_pi)
        log_probs = self.action_dist.log_prob(actions)
        entropies = self.action_dist.entropy()

        critic_global_obs = self._build_critic_global_obs(global_obs, hidden_vars)
        values = self.critic(local_obs, critic_global_obs, agent_mask=agent_mask)
        return log_probs, entropies, values

    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            deterministic: bool = False
    ) -> torch.Tensor:
        local_obs = self._mask_local_obs(local_obs, agent_mask)
        latent_pi = self.actor(local_obs, global_obs)
        actions = self.action_dist.update_latent_features(latent_pi).get_actions(deterministic)
        return actions

    @staticmethod
    def _mask_local_obs(
            local_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if agent_mask is None:
            return local_obs
        if agent_mask.dtype != torch.bool:
            raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
        if agent_mask.shape != local_obs.shape[:2]:
            raise ValueError(
                f"Expected agent_mask shape {tuple(local_obs.shape[:2])}, got {tuple(agent_mask.shape)}"
            )
        return local_obs * agent_mask.to(dtype=local_obs.dtype).unsqueeze(-1)

    @staticmethod
    def _build_critic_global_obs(
            global_obs: torch.Tensor,
            hidden_vars: torch.Tensor | None
    ) -> torch.Tensor:
        if hidden_vars is None or hidden_vars.shape[-1] == 0:
            return global_obs
        if global_obs.shape[-1] == 0:
            return hidden_vars
        return torch.cat((global_obs, hidden_vars), dim=-1)
