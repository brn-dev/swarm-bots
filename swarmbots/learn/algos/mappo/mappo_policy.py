from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution
from swarmbots.learn.algos.mappo.mappo_actor import MAPPOActor
from swarmbots.learn.algos.mappo.mappo_deep_set_critic import MAPPODeepSetCriticHiddenDims, MAPPODeepSetCritic
from swarmbots.learn.algos.ppo.ppo_policy import PPOCritic, PPOPolicy
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper

MAPPOCritic = PPOCritic


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
                local_projection_hidden_dims=critic_hidden_dims['local_projection_hidden_dims'],
                value_regressor_hidden_dims=critic_hidden_dims['value_regressor_hidden_dims'],
                act_fun_class=act_fun_class
            )
