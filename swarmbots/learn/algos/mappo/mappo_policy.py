from dataclasses import dataclass, field, replace
from typing import Any

import torch

from swarmbots.learn.action_dists.hybrid_action_dist import (
    HybridActionDistribution,
    ContinuousActionDistConfigInput,
    continuous_config_to_dicts,
    bernoulli_config_to_dict,
)
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.algos.mappo.mappo_actor import MAPPOActor, MAPPOActorConfig
from swarmbots.learn.algos.mappo.mappo_encoder import MAPPOEncoder
from swarmbots.learn.algos.mat.deep_set_encoder import DeepSetEncoder
from swarmbots.learn.algos.mat.independent_encoders import encode_critic_observations, make_value_head
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat_qc_base_policy import MATCriticConfig
from swarmbots.learn.algos.ppo.ppo_policy import PPOPolicy
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.nn_components.deep_set import DeepSetInteractionConfig
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal
from swarmbots.learn.serialization_utils import serialize_dataclass


@dataclass(frozen=True)
class MAPPOPolicyConfig:
    actor_config: MAPPOActorConfig = field(default_factory=MAPPOActorConfig)
    critic_encoder_config: MATEncoderConfig = field(default_factory=MATEncoderConfig)
    critic_interaction_config: DeepSetInteractionConfig = field(default_factory=DeepSetInteractionConfig)
    critic_config: MATCriticConfig = field(default_factory=MATCriticConfig)
    continuous_config: ContinuousActionDistConfigInput = None
    bernoulli_config: BernoulliConfig | None = None
    compile_modules: bool = False
    compile_mode: str = "default"
    action_net_init_gain: float = 0.01


class MAPPOPolicy(PPOPolicy):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: MAPPOPolicyConfig = MAPPOPolicyConfig(),
    ) -> None:
        BasePolicy.__init__(self)
        self.config = config

        self.n_agents = env.n_agents
        self.local_obs_dim = env.local_obs_dim
        self.global_obs_dim = env.global_obs_dim
        self.hidden_local_vars_dim = env.hidden_local_vars_dim
        self.hidden_global_vars_dim = env.hidden_global_vars_dim
        self.latent_pi_dim = config.actor_config.latent_pi_dim
        self.local_latent_dim = config.actor_config.latent_dim
        self.critic_encoder_config = replace(config.critic_encoder_config, use_agent_attention=False)
        self.critic_latent_dim = self.critic_encoder_config.d_model
        self.nop_latent_dim = self.critic_latent_dim
        self.nop_latent_source = "critic_encoder"
        self.actor_encoder = MAPPOEncoder(
            n_agents=env.n_agents,
            local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim,
            config=config.actor_config,
        )
        self.critic_encoder = DeepSetEncoder(
            config=self.critic_encoder_config,
            interaction_config=config.critic_interaction_config,
            max_agents=env.n_agents,
            local_obs_dim=env.local_obs_dim + env.hidden_local_vars_dim,
            global_obs_dim=env.global_obs_dim + env.hidden_global_vars_dim,
        )
        self.actor = MAPPOActor(
            local_latent_dim=self.local_latent_dim,
            config=config.actor_config,
        )
        self.action_dist = HybridActionDistribution(
            latent_dim=config.actor_config.latent_pi_dim,
            action_space=env.action_space,
            continuous_config=config.continuous_config,
            bernoulli_config=config.bernoulli_config,
            action_net_initialization=make_init_linear_orthogonal(config.action_net_init_gain),
        )

        self.critic = make_value_head(
            config.critic_config, self.critic_latent_dim, self.critic_encoder_config.act_fn_cls,
        )
        self._apply_optional_compile()

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            "mappo_policy_config": {
                "actor_config": serialize_dataclass(self.config.actor_config),
                "critic_encoder_config": serialize_dataclass(self.critic_encoder_config),
                "critic_interaction_config": serialize_dataclass(self.config.critic_interaction_config),
                "critic_config": serialize_dataclass(self.config.critic_config),
                "continuous_config": continuous_config_to_dicts(self.action_dist.continuous_configs),
                "bernoulli_config": bernoulli_config_to_dict(self.action_dist.bernoulli_config),
                "compile_modules": self.config.compile_modules,
                "compile_mode": self.config.compile_mode,
                "action_net_init_gain": self.config.action_net_init_gain,
            }
        }

    def _encode_actor_observations(
        self, local_obs: torch.Tensor, global_obs: torch.Tensor,
    ) -> torch.Tensor:
        return self.actor_encoder(local_obs, global_obs)

    def _evaluate_critic(
        self, local_obs: torch.Tensor, global_obs: torch.Tensor,
        hidden_local_vars: torch.Tensor | None, hidden_global_vars: torch.Tensor | None,
        agent_mask: torch.Tensor | None, *, actor_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        critic_latents = encode_critic_observations(
            self.critic_encoder, local_obs, global_obs, hidden_local_vars, hidden_global_vars, agent_mask,
        )
        return self.critic(critic_latents, agent_mask=agent_mask), critic_latents

    def get_grad_norms(self) -> dict[str, float]:
        return {
            "actor_encoder": self._module_grad_norm(self.actor_encoder),
            "critic_encoder": self._module_grad_norm(self.critic_encoder),
            "actor": self._module_grad_norm(self.actor),
            "action_dist": self._module_grad_norm(self.action_dist),
            "critic": self._module_grad_norm(self.critic),
            "total": self._module_grad_norm(self),
        }

    def _apply_optional_compile(self) -> None:
        if self.config.compile_modules:
            self._ensure_torch_compile_available()
            self.actor_encoder = self._compile_module(self.actor_encoder)
            self.critic_encoder = self._compile_module(self.critic_encoder)
            self.actor = self._compile_module(self.actor)
            self.critic = self._compile_module(self.critic)
