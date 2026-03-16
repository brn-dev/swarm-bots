from typing import Any
from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import (
    HybridActionDistribution,
    ContinuousActionDistConfig,
    serialize_continuous_action_dist_configs,
    serialize_bernoulli_config,
)
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.algos.mappo.mappo_actor import MAPPOActor
from swarmbots.learn.algos.ppo.ppo import AGENTS_DIM
from swarmbots.learn.algos.ppo.ppo_policy import PPOCritic, PPOPolicy
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.nn_components.deep_set import DeepSetCritic, DeepSetCriticHiddenDims

MAPPOCritic = PPOCritic


class MAPPOPolicy(PPOPolicy):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            actor_hidden_dims: list[int],
            latent_pi_dim_per_agent: int,
            critic_hidden_dims: list[int] | DeepSetCriticHiddenDims,
            act_fun_class = nn.Tanh,
            continuous_config: ContinuousActionDistConfig | list[ContinuousActionDistConfig | None] | None = None,
            bernoulli_config: BernoulliConfig | None = None,
    ):
        BasePolicy.__init__(self)

        self.n_agents = env.n_agents
        self.local_obs_dim = env.local_obs_dim
        self.global_obs_dim = env.global_obs_dim
        self.hidden_vars_dim = env.hidden_vars_dim
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
            continuous_config=continuous_config,
            bernoulli_config=bernoulli_config,
        )

        if isinstance(critic_hidden_dims, list):
            self.critic = MAPPOCritic(
                n_agents=env.n_agents,
                local_obs_dim=env.local_obs_dim,
                global_obs_dim=env.global_obs_dim + self.hidden_vars_dim,
                hidden_dims=critic_hidden_dims,
                act_fun_class=act_fun_class
            )
        else:
            self.critic = DeepSetCritic(
                num_local_features=env.local_obs_dim,
                local_projection_hidden_dims=critic_hidden_dims["local_projection_hidden_dims"],
                value_regressor_hidden_dims=critic_hidden_dims["value_regressor_hidden_dims"],
                num_global_features=env.global_obs_dim + self.hidden_vars_dim,
                set_dim=AGENTS_DIM,
                pool_mode="mean",
                act_fn_cls=act_fun_class,
            )

        self.hyper_parameters = {
            "actor_hidden_dims": actor_hidden_dims,
            "critic_hidden_dims": critic_hidden_dims,
            "latent_pi_dim_per_agent": latent_pi_dim_per_agent,
            "act_fun_class": act_fun_class.__name__,
            "continuous_config": serialize_continuous_action_dist_configs(continuous_config),
            "bernoulli_config": serialize_bernoulli_config(bernoulli_config),
        }

    def get_hyper_parameters(self) -> dict[str, Any]:
        return self.hyper_parameters
