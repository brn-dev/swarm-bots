from dataclasses import dataclass, field, replace
from typing import Any, Optional

import torch
from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import (
    HybridActionDistribution,
    ContinuousActionDistConfigInput,
    serialize_continuous_action_dist_configs,
    serialize_bernoulli_config,
)
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.algos.mat.mat_decoder import MATDecoder
from swarmbots.learn.algos.mat.mat_decoder import MATDecoderConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoder
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.ppo.ppo import AGENTS_DIM
from swarmbots.learn.algos.ppo.ppo_policy import BasePPOPolicy, PopArtConfig, serialize_popart_config
from swarmbots.learn.config_serialization import serialize_dataclass_config
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.nn_components.deep_set import DeepSetCritic
from swarmbots.learn.nn_components.mlp import MLP
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


@dataclass(frozen=True)
class MATCriticConfig:
    n_local_projection_hidden_layers: int = 1
    n_value_regressor_hidden_layers: int = 2
    use_popart: bool = False
    popart_config: PopArtConfig = field(default_factory=PopArtConfig)


@dataclass(frozen=True)
class MATPolicyConfig:
    encoder_config: MATEncoderConfig = field(default_factory=MATEncoderConfig)
    decoder_config: MATDecoderConfig = field(default_factory=MATDecoderConfig)
    critic_config: MATCriticConfig = field(default_factory=MATCriticConfig)
    act_fn_cls: type[nn.Module] = nn.ReLU
    dropout: float = 0.0
    continuous_config: ContinuousActionDistConfigInput = None
    bernoulli_config: BernoulliConfig | None = None
    max_agents: int | None = None


def serialize_mat_policy_config(config: MATPolicyConfig) -> dict[str, Any]:
    data = serialize_dataclass_config(config)
    data["critic_config"]["popart_config"] = serialize_popart_config(config.critic_config.popart_config)
    data["continuous_config"] = serialize_continuous_action_dist_configs(config.continuous_config)
    data["bernoulli_config"] = serialize_bernoulli_config(config.bernoulli_config)
    return data


class MATPolicy(BasePPOPolicy):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: MATPolicyConfig = MATPolicyConfig(),
    ) -> None:
        super().__init__()

        self.n_agents: int = env.n_agents
        self.max_agents = self.n_agents if config.max_agents is None else config.max_agents
        if self.max_agents < self.n_agents:
            raise ValueError(
                f"max_agents must be >= env.n_agents ({self.n_agents}), got {self.max_agents}"
            )
        self.local_obs_dim: int = env.local_obs_dim
        self.global_obs_dim: int = env.global_obs_dim
        self.has_global_obs = env.global_obs_dim > 0
        self.hidden_vars_dim: int = env.hidden_vars_dim

        self.d_model_encoder = config.encoder_config.d_model
        self.d_model_decoder = (
            config.encoder_config.d_model
            if config.decoder_config.d_model is None
            else config.decoder_config.d_model
        )

        encoder_config = replace(
            config.encoder_config,
            d_model=self.d_model_encoder,
            act_fn_cls=config.act_fn_cls,
            dropout=config.dropout,
        )
        self.encoder = MATEncoder(
            config=encoder_config,
            max_agents=self.max_agents,
            local_obs_dim=self.local_obs_dim,
            global_obs_dim=self.global_obs_dim,
        )

        self.agent_embeddings_decoder: nn.Parameter | None = None
        if config.decoder_config.add_agent_embeddings:
            self.agent_embeddings_decoder = nn.Parameter(
                torch.zeros(1, self.max_agents - 1, self.d_model_decoder), requires_grad=True
            )
            nn.init.orthogonal_(self.agent_embeddings_decoder)

        if (
            config.decoder_config.action_encoder_hidden_dims is None
            or len(config.decoder_config.action_encoder_hidden_dims) == 0
        ):
            self.action_encoder = nn.Linear(env.action_space.total_agent_action_dim, self.d_model_decoder)
            init_linear_orthogonal(self.action_encoder)
        else:
            self.action_encoder = MLP(
                input_dim=env.action_space.total_agent_action_dim,
                hidden_dims=[*config.decoder_config.action_encoder_hidden_dims, self.d_model_decoder],
                end_with_act_fn=False,
                linear_init=init_linear_orthogonal,
                act_fn_cls=config.act_fn_cls,
            )

        self.sos_token = nn.Parameter(torch.zeros(1, 1, self.d_model_decoder), requires_grad=True)
        nn.init.orthogonal_(self.sos_token)

        decoder_config = replace(
            config.decoder_config,
            d_model=self.d_model_decoder,
            act_fn_cls=config.act_fn_cls,
            dropout=config.dropout,
        )
        self.decoder = MATDecoder(
            config=decoder_config,
            max_agents=self.max_agents,
            d_model=self.d_model_decoder,
            memory_d_model=self.d_model_encoder,
        )

        if (
            config.decoder_config.actor_head_hidden_dims is not None
            and len(config.decoder_config.actor_head_hidden_dims) > 0
        ):
            self.actor_head = MLP(
                input_dim=self.d_model_decoder,
                hidden_dims=config.decoder_config.actor_head_hidden_dims,
                end_with_act_fn=True,
                linear_init=init_linear_orthogonal,
                act_fn_cls=config.act_fn_cls
            )
            latent_pi_dim = config.decoder_config.actor_head_hidden_dims[-1]
        else:
            self.actor_head = nn.Identity()
            latent_pi_dim = self.d_model_decoder

        self.action_dist = HybridActionDistribution(
            latent_dim=latent_pi_dim,
            action_space=env.action_space,
            continuous_config=config.continuous_config,
            bernoulli_config=config.bernoulli_config,
        )

        self.critic = DeepSetCritic(
            num_local_features=self.d_model_encoder,
            local_projection_hidden_dims=[self.d_model_encoder] * config.critic_config.n_local_projection_hidden_layers,
            value_regressor_hidden_dims=[self.d_model_encoder] * config.critic_config.n_value_regressor_hidden_layers,
            num_global_features=self.hidden_vars_dim,
            act_fn_cls=config.act_fn_cls,
            context_in_elements=True,
            use_popart=config.critic_config.use_popart,
            popart_beta=config.critic_config.popart_config.beta,
            popart_eps=config.critic_config.popart_config.eps,
            popart_min_std=config.critic_config.popart_config.min_std,
            popart_init_sigma=config.critic_config.popart_config.init_sigma,
        )

        self.hyper_parameters = {
            "mat_policy_config": serialize_mat_policy_config(config),
        }

    def get_hyper_parameters(self) -> dict[str, Any]:
        return self.hyper_parameters

    def _generate_actions(
        self,
        augmented_observations: torch.Tensor,
        batch_size: int,
        agent_mask: torch.Tensor | None = None,
        deterministic: bool = False,
        return_log_probs: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Autoregressively generate actions based on augmented observations (encoder output)
        :return: (actions, Optional[log_probs])
        """
        self._validate_agent_mask(agent_mask, batch_size=batch_size)
        actions_list = []
        log_probs_list = []

        decoder_input = self.sos_token.expand(batch_size, 1, -1)

        for i in range(self.n_agents):
            out = self.decoder(
                action_embeddings=decoder_input,
                augmented_observations=augmented_observations,
                agent_mask=agent_mask,
            )

            latent_pi = self.actor_head(out[:, -1:, :])

            if return_log_probs:
                action, log_prob = self.action_dist.get_actions_with_log_probs(latent_pi, deterministic, agent=i)
                log_probs_list.append(log_prob)
            else:
                action = self.action_dist.update_latent_features(latent_pi).get_actions(deterministic, agent=i)

            actions_list.append(action)

            if i < self.n_agents - 1:
                action_emb = self.action_encoder(action)
                if self.agent_embeddings_decoder is not None:
                    action_emb = action_emb + self.agent_embeddings_decoder[:, i, :]
                decoder_input = torch.cat([decoder_input, action_emb], dim=AGENTS_DIM)

        actions = torch.cat(actions_list, dim=AGENTS_DIM)
        if return_log_probs:
            return actions, torch.cat(log_probs_list, dim=AGENTS_DIM)
        return actions, None

    def forward(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        self._validate_agent_mask(agent_mask, batch_size=local_obs.shape[0])
        augmented_observations = self.encoder(local_obs, global_obs, agent_mask=agent_mask)
        actions, log_probs = self._generate_actions(
            augmented_observations=augmented_observations,
            batch_size=local_obs.shape[0],
            agent_mask=agent_mask,
            deterministic=deterministic,
            return_log_probs=True,
        )
        values = self._critic_with_hidden_vars(augmented_observations, hidden_vars, agent_mask=agent_mask)

        return actions, log_probs, values

    def evaluate_actions(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, LossDict, LossMetrics]:
        _, log_probs, values, extra_losses, extra_loss_metrics = self._evaluate_actions(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_vars=hidden_vars,
            agent_mask=agent_mask,
        )

        return log_probs, values, extra_losses, extra_loss_metrics

    def _evaluate_actions(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, LossDict, LossMetrics]:
        self._validate_agent_mask(agent_mask, batch_size=local_obs.shape[0])
        augmented_observations = self.encoder(local_obs, global_obs, agent_mask=agent_mask)
        
        action_embeddings = self.action_encoder(actions[:, :-1, :])
        if self.agent_embeddings_decoder is not None:
            if action_embeddings.shape[1] > self.agent_embeddings_decoder.shape[1]:
                raise ValueError(
                    "Expected actions second dim to be <= "
                    f"{self.agent_embeddings_decoder.shape[1] + 1}, got {actions.shape[1]}"
                )
            action_embeddings = action_embeddings + self.agent_embeddings_decoder[:, :action_embeddings.shape[1], :]
        sos_expanded = self.sos_token.expand(actions.shape[0], 1, -1)
        
        shifted_actions = torch.cat([sos_expanded, action_embeddings], dim=1)
        
        latent_pi = self.actor_head(self.decoder(
            action_embeddings=shifted_actions,
            augmented_observations=augmented_observations,
            agent_mask=agent_mask
        ))

        self.action_dist.update_latent_features(latent_pi)
        log_probs = self.action_dist.log_prob(actions)

        values = self._critic_with_hidden_vars(augmented_observations, hidden_vars, agent_mask=agent_mask)
        extra_losses, extra_loss_metrics = self.action_dist.compute_extra_losses(agent_mask=agent_mask)
        return augmented_observations, log_probs, values, extra_losses, extra_loss_metrics

    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            deterministic: bool = False
    ) -> torch.Tensor:
        self._validate_agent_mask(agent_mask, batch_size=local_obs.shape[0])
        augmented_observations = self.encoder(local_obs, global_obs, agent_mask=agent_mask)
        actions, _ = self._generate_actions(
            augmented_observations=augmented_observations,
            batch_size=local_obs.shape[0],
            agent_mask=agent_mask,
            deterministic=deterministic,
            return_log_probs=False,
        )
        return actions

    def _validate_agent_mask(
            self,
            agent_mask: torch.Tensor | None,
            *,
            batch_size: int,
    ) -> None:
        if agent_mask is None:
            return
        if agent_mask.dtype != torch.bool:
            raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
        if agent_mask.ndim != 2:
            raise ValueError(f"Expected agent_mask shape (B, N), got {tuple(agent_mask.shape)}")
        expected_shape = (batch_size, self.n_agents)
        if agent_mask.shape != expected_shape:
            raise ValueError(f"Expected agent_mask shape {expected_shape}, got {tuple(agent_mask.shape)}")
        # MAT decoder uses agent embeddings as identity + autoregressive position, so active agents must be contiguous.
        # This is intentional for this version of MAT.
        if not agent_mask[:, 0].all():
            raise ValueError("agent_mask must start with True for every batch entry")
        mask_int = agent_mask.to(torch.int8)
        if not torch.all(mask_int[:, 1:] <= mask_int[:, :-1]):
            raise ValueError("agent_mask must be a True-prefix/False-suffix for every batch entry")

    def _critic_with_hidden_vars(
            self,
            augmented_observations: torch.Tensor,
            hidden_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.hidden_vars_dim <= 0:
            return self.critic(augmented_observations, agent_mask=agent_mask)
        if hidden_vars is None:
            raise ValueError("hidden_vars must be provided when hidden_vars_dim > 0")
        return self.critic(augmented_observations, hidden_vars, agent_mask=agent_mask)

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
            "encoder": self._module_grad_norm(self.encoder),
            "action_encoder": self._module_grad_norm(self.action_encoder),
            "agent_embeddings_decoder": self._parameter_grad_norm(self.agent_embeddings_decoder),
            "sos_token": self._parameter_grad_norm(self.sos_token),
            "decoder": self._module_grad_norm(self.decoder),
            "actor_head": self._module_grad_norm(self.actor_head),
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
            self.hyper_parameters["mat_policy_config"]["continuous_config"] = serialize_continuous_action_dist_configs(
                self.action_dist.continuous_configs
            )

        action_magnitude_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("action_magnitude_loss_coef", "action_magnitude"),
        )
        if action_magnitude_weight is not None:
            alias, value = action_magnitude_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.action_dist.set_action_magnitude_loss_coef(value)
            self.hyper_parameters["mat_policy_config"]["continuous_config"] = serialize_continuous_action_dist_configs(
                self.action_dist.continuous_configs
            )

        entropy_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("ent_loss_coef", "entropy", "ent"),
        )
        if entropy_weight is not None:
            alias, value = entropy_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.action_dist.set_all_ent_loss_coefs(value)
            self.hyper_parameters["mat_policy_config"]["continuous_config"] = serialize_continuous_action_dist_configs(
                self.action_dist.continuous_configs
            )

        super().update_loss_weights(**remaining_weights)
