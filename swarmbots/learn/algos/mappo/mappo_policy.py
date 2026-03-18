from dataclasses import dataclass, field
from typing import Any
from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import (
    HybridActionDistribution,
    ContinuousActionDistConfigInput,
    serialize_continuous_action_dist_configs,
    serialize_bernoulli_config,
)
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.algos.mappo.mappo_actor import MAPPOActor, MAPPOActorConfig
from swarmbots.learn.algos.ppo.ppo import AGENTS_DIM
from swarmbots.learn.algos.ppo.ppo_policy import PPOCritic, PPOPolicy, PPOCriticConfig
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.config_serialization import serialize_dataclass_config
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.nn_components.deep_set import DeepSetCritic, DeepSetCriticConfig

MAPPOCritic = PPOCritic


@dataclass(frozen=True)
class MAPPOCriticConfig:
    mlp_hidden_dims: list[int] | None = field(default_factory=list)
    deep_set_config: DeepSetCriticConfig | None = None
    act_fun_class: type[nn.Module] = nn.Tanh


@dataclass(frozen=True)
class MAPPOPolicyConfig:
    actor_config: MAPPOActorConfig = field(default_factory=MAPPOActorConfig)
    critic_config: MAPPOCriticConfig = field(default_factory=MAPPOCriticConfig)
    continuous_config: ContinuousActionDistConfigInput = None
    bernoulli_config: BernoulliConfig | None = None


def serialize_mappo_policy_config(config: MAPPOPolicyConfig) -> dict[str, Any]:
    data = serialize_dataclass_config(config)
    data["continuous_config"] = serialize_continuous_action_dist_configs(config.continuous_config)
    data["bernoulli_config"] = serialize_bernoulli_config(config.bernoulli_config)
    return data


class MAPPOPolicy(PPOPolicy):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: MAPPOPolicyConfig = MAPPOPolicyConfig(),
    ):
        BasePolicy.__init__(self)
        self.config = config

        self.n_agents = env.n_agents
        self.local_obs_dim = env.local_obs_dim
        self.global_obs_dim = env.global_obs_dim
        self.hidden_vars_dim = env.hidden_vars_dim
        self.latent_pi_dim = config.actor_config.latent_pi_dim

        self.actor = MAPPOActor(
            n_agents=env.n_agents,
            local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim,
            config=config.actor_config,
        )
        self.action_dist = HybridActionDistribution(
            latent_dim=config.actor_config.latent_pi_dim,
            action_space=env.action_space,
            continuous_config=config.continuous_config,
            bernoulli_config=config.bernoulli_config,
        )

        if config.critic_config.deep_set_config is None:
            self.critic = MAPPOCritic(
                n_agents=env.n_agents,
                local_obs_dim=env.local_obs_dim,
                global_obs_dim=env.global_obs_dim + self.hidden_vars_dim,
                config=PPOCriticConfig(
                    hidden_dims=config.critic_config.mlp_hidden_dims or [],
                    act_fun_class=config.critic_config.act_fun_class,
                ),
            )
        else:
            self.critic = DeepSetCritic(
                num_local_features=env.local_obs_dim,
                local_projection_hidden_dims=config.critic_config.deep_set_config.local_projection_hidden_dims,
                value_regressor_hidden_dims=config.critic_config.deep_set_config.value_regressor_hidden_dims,
                num_global_features=env.global_obs_dim + self.hidden_vars_dim,
                set_dim=AGENTS_DIM,
                pool_mode="mean",
                act_fn_cls=config.critic_config.act_fun_class,
            )

        self.hyper_parameters = {
            "mappo_policy_config": serialize_mappo_policy_config(config),
        }

    def get_hyper_parameters(self) -> dict[str, Any]:
        return self.hyper_parameters
