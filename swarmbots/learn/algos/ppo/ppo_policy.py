import abc
import re
from dataclasses import dataclass, field
from typing import Any
import torch
from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import (
    HybridActionDistribution,
    ContinuousActionDistConfigInput,
    serialize_continuous_action_dist_configs,
    serialize_bernoulli_config,
)
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.config_serialization import serialize_dataclass_config
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.nn_components.mlp import MLP
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal, LinearInitialization
from swarmbots.learn.nn_components.popart import PopArtLinear


@dataclass(frozen=True)
class PopArtConfig:
    beta: float = 3e-4
    eps: float = 1e-5
    min_std: float = 1e-4
    init_sigma: float = 1.0


@dataclass(frozen=True)
class PPOActorConfig:
    hidden_dims: list[int] = field(default_factory=list)
    latent_pi_dim_per_agent: int = 64
    act_fun_class: type[nn.Module] = nn.Tanh


@dataclass(frozen=True)
class PPOCriticConfig:
    hidden_dims: list[int] = field(default_factory=list)
    act_fun_class: type[nn.Module] = nn.Tanh
    use_popart: bool = False
    popart_config: PopArtConfig = field(default_factory=PopArtConfig)


@dataclass(frozen=True)
class PPOPolicyConfig:
    actor_config: PPOActorConfig = field(default_factory=PPOActorConfig)
    critic_config: PPOCriticConfig = field(default_factory=PPOCriticConfig)
    continuous_config: ContinuousActionDistConfigInput = None
    bernoulli_config: BernoulliConfig | None = None


def serialize_popart_config(config: PopArtConfig) -> dict[str, float]:
    return serialize_dataclass_config(config)


def serialize_ppo_policy_config(config: PPOPolicyConfig) -> dict[str, Any]:
    data = serialize_dataclass_config(config)
    data["continuous_config"] = serialize_continuous_action_dist_configs(config.continuous_config)
    data["bernoulli_config"] = serialize_bernoulli_config(config.bernoulli_config)
    return data


class BasePPOPolicy(BasePolicy, abc.ABC):
    action_dist: HybridActionDistribution
    _PER_ACTION_ENTROPY_WEIGHT_PATTERN = re.compile(r"^act(?P<idx>\d+)_(?P<alias>ent_loss_coef|entropy|ent)$")

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
    ) -> tuple[torch.Tensor, torch.Tensor, LossDict, LossMetrics]:
        """
        :return: log_probs, values, extra_losses, extra_loss_metrics
        """
        raise NotImplementedError()

    @property
    def has_popart(self) -> bool:
        return False

    def update_value_normalizer(self, targets: torch.Tensor) -> None:
        _ = targets

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        return values

    def get_value_normalizer_metrics(self) -> dict[str, float]:
        return {}

    def update_loss_weights(self, **weights: float) -> None:
        if weights:
            raise ValueError(f'Unknown weights given: {weights}')

    @staticmethod
    def _pop_loss_weight_alias(
            weights: dict[str, float],
            *,
            aliases: tuple[str, ...],
    ) -> tuple[str, float] | None:
        matching_aliases = [alias for alias in aliases if alias in weights]
        if not matching_aliases:
            return None
        if len(matching_aliases) > 1:
            raise ValueError(f"Multiple aliases for the same loss weight are not allowed: {matching_aliases}")
        alias = matching_aliases[0]
        value = float(weights.pop(alias))
        return alias, value

    def _pop_per_action_entropy_weights(self, weights: dict[str, float]) -> dict[int, float]:
        updates: dict[int, float] = {}
        for key in list(weights):
            match = self._PER_ACTION_ENTROPY_WEIGHT_PATTERN.match(key)
            if match is None:
                continue

            idx = int(match.group("idx"))
            if idx in updates:
                raise ValueError(f"Multiple entropy aliases for action index {idx} are not allowed")
            updates[idx] = float(weights.pop(key))
        return updates


class PPOActor(nn.Module):

    def __init__(
            self,
            n_agents: int,
            local_obs_dim: int,
            global_obs_dim: int,
            config: PPOActorConfig,
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
        self.latent_pi_dim = config.latent_pi_dim_per_agent

        self.mlp = MLP(
            input_dim=local_obs_dim * n_agents + global_obs_dim,
            hidden_dims=[*config.hidden_dims, config.latent_pi_dim_per_agent * n_agents],
            end_with_act_fn=True,
            linear_init=linear_init,
            act_fn_cls=actor_act_fun_class
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
            config: PPOCriticConfig,
            linear_init: LinearInitialization = init_linear_orthogonal,
            act_fun_class: type[nn.Module] | None = None,
    ):
        super().__init__()
        critic_act_fun_class = config.act_fun_class if act_fun_class is None else act_fun_class
        self.n_agents = n_agents
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.has_global_obs = global_obs_dim > 0
        self.hidden_dims = config.hidden_dims
        self.use_popart = bool(config.use_popart)

        value_head_input_dim = local_obs_dim * n_agents + global_obs_dim
        if config.hidden_dims:
            self.value_features: nn.Module = MLP(
                input_dim=value_head_input_dim,
                hidden_dims=config.hidden_dims,
                end_with_act_fn=True,
                linear_init=linear_init,
                act_fn_cls=critic_act_fun_class
            )
            value_head_input_dim = int(config.hidden_dims[-1])
        else:
            self.value_features = nn.Identity()

        if self.use_popart:
            self.value_head: nn.Module = PopArtLinear(
                in_features=value_head_input_dim,
                out_features=1,
                beta=config.popart_config.beta,
                eps=config.popart_config.eps,
                min_std=config.popart_config.min_std,
                init_sigma=config.popart_config.init_sigma,
            )
        else:
            self.value_head = nn.Linear(value_head_input_dim, 1)
            linear_init(self.value_head)

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
        critic_features = self.value_features(critic_input)
        return self.value_head(critic_features).squeeze(dim=-1)

    @property
    def has_popart(self) -> bool:
        return isinstance(self.value_head, PopArtLinear)

    def update_popart(self, targets: torch.Tensor) -> None:
        if not isinstance(self.value_head, PopArtLinear):
            raise RuntimeError("PopArt is not enabled for this PPOCritic")
        self.value_head.update(targets)

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.value_head, PopArtLinear):
            return values
        return self.value_head.normalize(values)

    def get_popart_metrics(self) -> dict[str, float]:
        if not isinstance(self.value_head, PopArtLinear):
            return {}
        sigma = self.value_head.sigma()
        return {
            "popart_mu": self.value_head.mu.mean().item(),
            "popart_sigma": sigma.mean().item(),
        }


class PPOPolicy(BasePPOPolicy):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: PPOPolicyConfig = PPOPolicyConfig(),
    ):
        super().__init__()
        self.config = config

        self.n_agents = env.n_agents
        self.local_obs_dim = env.local_obs_dim
        self.global_obs_dim = env.global_obs_dim
        self.hidden_vars_dim = env.hidden_vars_dim
        self.latent_pi_dim = config.actor_config.latent_pi_dim_per_agent

        self.actor = PPOActor(
            n_agents=env.n_agents,
            local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim,
            config=config.actor_config,
        )
        self.action_dist = HybridActionDistribution(
            latent_dim=config.actor_config.latent_pi_dim_per_agent,
            action_space=env.action_space,
            continuous_config=config.continuous_config,
            bernoulli_config=config.bernoulli_config,
        )

        self.critic = PPOCritic(
            n_agents=env.n_agents,
            local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim + self.hidden_vars_dim,
            config=config.critic_config,
        )

        self.hyper_parameters = {
            "ppo_policy_config": serialize_ppo_policy_config(config),
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
    ) -> tuple[torch.Tensor, torch.Tensor, LossDict, LossMetrics]:
        local_obs = self._mask_local_obs(local_obs, agent_mask)
        latent_pi = self.actor(local_obs, global_obs)

        self.action_dist.update_latent_features(latent_pi)
        log_probs = self.action_dist.log_prob(actions)

        critic_global_obs = self._build_critic_global_obs(global_obs, hidden_vars)
        values = self.critic(local_obs, critic_global_obs, agent_mask=agent_mask)

        extra_losses, extra_loss_metrics = self.action_dist.compute_extra_losses(agent_mask=agent_mask)
        return log_probs, values, extra_losses, extra_loss_metrics

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

    @property
    def has_popart(self) -> bool:
        return getattr(self.critic, "has_popart", False)

    def update_value_normalizer(self, targets: torch.Tensor) -> None:
        if not self.has_popart:
            return
        self.critic.update_popart(targets)

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        return self.critic.normalize_values(values)

    def get_value_normalizer_metrics(self) -> dict[str, float]:
        return self.critic.get_popart_metrics()

    def get_grad_norms(self) -> dict[str, float]:
        return {
            "actor": self._module_grad_norm(self.actor),
            "action_dist": self._module_grad_norm(self.action_dist),
            "critic": self._module_grad_norm(self.critic),
            "total": self._module_grad_norm(self),
        }

    def update_loss_weights(self, **weights: float) -> None:
        if not weights:
            return

        remaining_weights = dict(weights)
        per_action_entropy_weights = self._pop_per_action_entropy_weights(remaining_weights)
        for idx, value in per_action_entropy_weights.items():
            if value < 0:
                raise ValueError(f"act{idx}_ent_loss_coef must be >= 0, got {value}")
            self.action_dist.set_sub_ent_loss_coef(idx, value)
            self._set_serialized_continuous_config()

        action_magnitude_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("action_magnitude_loss_coef", "action_magnitude"),
        )
        if action_magnitude_weight is not None:
            alias, value = action_magnitude_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.action_dist.set_action_magnitude_loss_coef(value)
            self._set_serialized_continuous_config()

        entropy_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("ent_loss_coef", "entropy", "ent"),
        )
        if entropy_weight is not None:
            alias, value = entropy_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.action_dist.set_all_ent_loss_coefs(value)
            self._set_serialized_continuous_config()

        super().update_loss_weights(**remaining_weights)

    def _set_serialized_continuous_config(self) -> None:
        serialized = serialize_continuous_action_dist_configs(self.action_dist.continuous_configs)
        if "ppo_policy_config" in self.hyper_parameters:
            self.hyper_parameters["ppo_policy_config"]["continuous_config"] = serialized
            return
        if "mappo_policy_config" in self.hyper_parameters:
            self.hyper_parameters["mappo_policy_config"]["continuous_config"] = serialized
            return
        self.hyper_parameters["continuous_config"] = serialized
