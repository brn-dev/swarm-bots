from collections.abc import Callable
from dataclasses import dataclass, field, replace
import shutil
import sys
from typing import Any, Optional

import torch
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionMetricsSplitterInput
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.hybrid_action_dist import (
    HybridActionDistribution,
    ContinuousActionDistConfigInput,
    continuous_config_to_dicts,
    bernoulli_config_to_dict,
)
from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig
from swarmbots.learn.algos.mat.mat_decoder import MATDecoder, MATDecoderConfig
from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo import AGENTS_DIM
from swarmbots.learn.algos.ppo.ppo_policy import PopArtConfig
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples, PPOSampler, PPOSamplerConfig
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.deep_set import DeepSetCritic
from swarmbots.learn.nn_components.mlp import MLP
from swarmbots.learn.nn_components.nn_init import DEFAULT_ORTHOGONAL_GAIN, make_init_linear_orthogonal
from swarmbots.learn.serialization_utils import serialize_dataclass, serialize_value


@dataclass(frozen=True)
class MATCriticConfig:
    n_local_projection_hidden_layers: int = 1
    n_value_regressor_hidden_layers: int = 2
    use_popart: bool = False
    popart_config: PopArtConfig = field(default_factory=PopArtConfig)
    local_projection_init_gain: float = DEFAULT_ORTHOGONAL_GAIN
    value_regressor_init_gain: float = DEFAULT_ORTHOGONAL_GAIN
    value_head_init_gain: float = DEFAULT_ORTHOGONAL_GAIN

@dataclass(frozen=True)
class MATPolicyConfig:
    encoder_config: MATEncoderConfig = field(default_factory=MATEncoderConfig)
    decoder_config: MATDecoderConfig = field(default_factory=MATDecoderConfig)
    critic_config: MATCriticConfig = field(default_factory=MATCriticConfig)
    act_fn_cls: ActivationFactory = nn.ReLU
    dropout: float = 0.0
    continuous_config: ContinuousActionDistConfigInput = None
    bernoulli_config: BernoulliConfig | None = None
    max_agents: int | None = None
    compile_modules: bool = False
    compile_mode: str = "default"
    action_net_init_gain: float = DEFAULT_ORTHOGONAL_GAIN


def _ensure_torch_compile_available(*, compile_mode: str) -> None:
    if not hasattr(torch, "compile"):
        raise RuntimeError("MATPolicyConfig.compile_modules=True requires torch.compile support.")
    if sys.platform == "win32" and shutil.which("cl") is None:
        raise RuntimeError(
            "MATPolicyConfig.compile_modules=True on this Windows setup requires cl.exe on PATH for torch.compile."
        )
    if not compile_mode:
        raise ValueError("MATPolicyConfig.compile_mode must be a non-empty string when compile_modules=True.")


class MATPolicy(BasePPOPolicy[PPOSamples, PPOSamplerConfig]):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: MATPolicyConfig = MATPolicyConfig(),
    ) -> None:
        super().__init__()
        self.config = config

        self.n_agents: int = env.n_agents
        self.max_agents = self.n_agents if config.max_agents is None else config.max_agents
        if self.max_agents < self.n_agents:
            raise ValueError(
                f"max_agents must be >= env.n_agents ({self.n_agents}), got {self.max_agents}"
            )

        self.local_obs_dim: int = env.local_obs_dim
        self.global_obs_dim: int = env.global_obs_dim
        self.hidden_local_vars_dim: int = env.hidden_local_vars_dim
        self.hidden_global_vars_dim: int = env.hidden_global_vars_dim
        self.act_fn_cls = config.act_fn_cls
        self.dropout = config.dropout
        self.agent_action_dim = env.action_space.total_agent_action_dim

        self.d_model_encoder = config.encoder_config.d_model
        self.d_model_decoder = (
            config.encoder_config.d_model
            if config.decoder_config.d_model is None
            else config.decoder_config.d_model
        )

        self.encoder_config = self._build_encoder_config()
        self.encoder = self._build_encoder()

        self.agent_embeddings_decoder: nn.Parameter | None = None
        if config.decoder_config.add_agent_embeddings:
            self.agent_embeddings_decoder = nn.Parameter(
                torch.zeros(1, self.max_agents, self.d_model_decoder), requires_grad=True
            )
            nn.init.orthogonal_(self.agent_embeddings_decoder)

        self.query_encoder = self._build_token_encoder(
            input_dim=self.d_model_encoder,
            output_dim=self.d_model_decoder,
            hidden_dims=config.decoder_config.query_encoder_hidden_dims,
            act_fn_cls=config.act_fn_cls,
            linear_init_gain=config.decoder_config.token_encoder_init_gain,
            projection_init_gain=config.decoder_config.token_encoder_projection_init_gain,
        )

        self.context_encoder = self._build_token_encoder(
            input_dim=self.d_model_encoder + self.agent_action_dim,
            output_dim=self.d_model_decoder,
            hidden_dims=config.decoder_config.context_encoder_hidden_dims,
            act_fn_cls=config.act_fn_cls,
            linear_init_gain=config.decoder_config.token_encoder_init_gain,
            projection_init_gain=config.decoder_config.token_encoder_projection_init_gain,
        )
        memory_dims = config.decoder_config.memory_dims
        if memory_dims is None:
            self.memory_encoder = nn.Identity()
            self.memory_d_model = self.d_model_encoder
        else:
            if len(memory_dims) == 0:
                raise ValueError("decoder_config.memory_dims must be None or contain at least one dimension")
            self.memory_encoder = self._build_encoder_from_dims(
                input_dim=self.d_model_encoder,
                dims=memory_dims,
                act_fn_cls=config.act_fn_cls,
                linear_init_gain=config.decoder_config.token_encoder_init_gain,
                projection_init_gain=config.decoder_config.token_encoder_projection_init_gain,
            )
            self.memory_d_model = memory_dims[-1]

        decoder_config = replace(
            config.decoder_config,
            d_model=self.d_model_decoder,
            act_fn_cls=config.act_fn_cls,
            dropout=config.dropout,
        )
        self.decoder_config = decoder_config
        self.decoder = MATDecoder(
            config=decoder_config,
            max_agents=self.max_agents,
            memory_d_model=self.memory_d_model,
        )

        if (
            config.decoder_config.actor_head_hidden_dims is not None
            and len(config.decoder_config.actor_head_hidden_dims) > 0
        ):
            self.actor_head = MLP(
                input_dim=self.d_model_decoder,
                hidden_dims=config.decoder_config.actor_head_hidden_dims,
                end_with_act_fn=True,
                linear_init=make_init_linear_orthogonal(config.decoder_config.actor_head_init_gain),
                act_fn_cls=config.act_fn_cls,
            )
            latent_pi_dim = config.decoder_config.actor_head_hidden_dims[-1]
        else:
            self.actor_head = nn.Identity()
            latent_pi_dim = self.d_model_decoder

        self.action_dist = self._build_action_dist(env=env, latent_pi_dim=latent_pi_dim)

        self.critic = DeepSetCritic(
            num_local_features=self.d_model_encoder + self.hidden_local_vars_dim,
            local_projection_hidden_dims=[self.d_model_encoder] * config.critic_config.n_local_projection_hidden_layers,
            value_regressor_hidden_dims=[self.d_model_encoder] * config.critic_config.n_value_regressor_hidden_layers,
            num_global_features=self.hidden_global_vars_dim,
            act_fn_cls=config.act_fn_cls,
            context_in_elements=self.hidden_global_vars_dim > 0,
            local_projection_linear_init_gain=config.critic_config.local_projection_init_gain,
            value_regressor_linear_init_gain=config.critic_config.value_regressor_init_gain,
            value_head_linear_init_gain=config.critic_config.value_head_init_gain,
            use_popart=config.critic_config.use_popart,
            popart_beta=config.critic_config.popart_config.beta,
            popart_eps=config.critic_config.popart_config.eps,
            popart_min_std=config.critic_config.popart_config.min_std,
            popart_init_sigma=config.critic_config.popart_config.init_sigma,
        )
        self._generate_actions_fn: Callable[..., tuple[torch.Tensor, Optional[torch.Tensor]]] = self._generate_actions_impl
        self._evaluate_latent_and_values_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
            self._evaluate_latent_and_values_impl
        )
        self._evaluate_actions_core_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
            self._evaluate_actions_core_impl
        )
        self._apply_optional_compile()

    def _build_encoder_config(self) -> MATEncoderConfig:
        return replace(
            self.config.encoder_config,
            d_model=self.d_model_encoder,
            act_fn_cls=self.config.act_fn_cls,
            dropout=self.config.dropout,
        )

    def _build_encoder(self) -> nn.Module:
        return MATEncoder(
            config=self.encoder_config,
            max_agents=self.max_agents,
            local_obs_dim=self.local_obs_dim,
            global_obs_dim=self.global_obs_dim,
        )

    def _build_action_dist(
            self,
            *,
            env: BaseLearnEnvWrapper,
            latent_pi_dim: int,
    ) -> HybridActionDistribution:
        return HybridActionDistribution(
            latent_dim=latent_pi_dim,
            action_space=env.action_space,
            continuous_config=self.config.continuous_config,
            bernoulli_config=self.config.bernoulli_config,
            action_net_initialization=make_init_linear_orthogonal(self.config.action_net_init_gain),
        )

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            "mat_policy_config": {
                "encoder_config": serialize_dataclass(self.encoder_config),
                "decoder_config": serialize_dataclass(self.decoder_config),
                "critic_config": serialize_dataclass(self.config.critic_config),
                "act_fn_cls": serialize_value(self.act_fn_cls),
                "dropout": self.dropout,
                "continuous_config": continuous_config_to_dicts(self.action_dist.continuous_configs),
                "bernoulli_config": bernoulli_config_to_dict(self.action_dist.bernoulli_config),
                "max_agents": self.max_agents,
                "compile_modules": self.config.compile_modules,
                "compile_mode": self.config.compile_mode,
                "action_net_init_gain": self.config.action_net_init_gain,
                "action_dist_compile_friendly": self.action_dist.compile_friendly,
            }
        }

    def requires_previous_actions(self) -> bool:
        return self.action_dist.requires_previous_actions()

    def _encode_query_tokens(
            self,
            augmented_observations: torch.Tensor,
            *,
            agent_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.query_encoder(augmented_observations)
        if agent_embeddings is not None:
            return tokens + agent_embeddings
        if self.agent_embeddings_decoder is None:
            return tokens

        return tokens + self.agent_embeddings_decoder[:, :tokens.shape[1], :]

    def _encode_context_tokens(
            self,
            augmented_observations: torch.Tensor,
            actions: torch.Tensor,
            *,
            agent_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context_input = torch.cat((augmented_observations, actions), dim=-1)
        tokens = self.context_encoder(context_input)
        if agent_embeddings is not None:
            return tokens + agent_embeddings
        if self.agent_embeddings_decoder is None:
            return tokens

        return tokens + self.agent_embeddings_decoder[:, :tokens.shape[1], :]

    def _generate_actions(
            self,
            augmented_observations: torch.Tensor,
            *,
            batch_size: int,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
            return_log_probs: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self._generate_actions_fn(
            augmented_observations=augmented_observations,
            batch_size=batch_size,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            return_log_probs=return_log_probs,
        )

    def _generate_actions_impl(
            self,
            augmented_observations: torch.Tensor,
            *,
            batch_size: int,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
            return_log_probs: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        query_tokens = self._encode_query_tokens(augmented_observations)
        memory_tokens = self._encode_memory_tokens(augmented_observations)

        actions_list: list[torch.Tensor] = []
        log_probs_list: list[torch.Tensor] = []
        context_tokens = query_tokens.new_zeros((batch_size, 0, self.d_model_decoder))
        context_mask: torch.Tensor | None
        if agent_mask is None:
            context_mask = None
        else:
            context_mask = agent_mask[:, :0]

        for i in range(self.n_agents):
            query_token_i = query_tokens[:, i:i + 1, :]
            query_mask_i = None if agent_mask is None else agent_mask[:, i]
            out = self.decoder.forward_step(
                context_tokens=context_tokens,
                query_token=query_token_i,
                memory_tokens=memory_tokens,
                query_prefix_tokens=query_tokens[:, :i, :],
                context_mask=context_mask,
                query_prefix_mask=None if agent_mask is None else agent_mask[:, :i],
                query_mask=query_mask_i,
                memory_mask=agent_mask,
            )
            latent_pi = self.actor_head(out).contiguous()
            sample_agent_idx = i if self.action_dist.sampling_depends_on_agent else None

            previous_action_i = None if previous_actions is None else previous_actions[:, i:i + 1, :]
            if return_log_probs:
                action, log_prob = self.action_dist.get_actions_with_log_probs(
                    latent_pi,
                    deterministic,
                    agent=sample_agent_idx,
                    previous_actions=previous_action_i,
                )
                log_probs_list.append(log_prob)
            else:
                action = self.action_dist.update_latent_features(latent_pi).get_actions(
                    deterministic=deterministic,
                    agent=sample_agent_idx,
                    previous_actions=previous_action_i,
                )
            actions_list.append(action)

            agent_embeddings_i = (
                None
                if self.agent_embeddings_decoder is None
                else self.agent_embeddings_decoder[:, i:i + 1, :]
            )
            context_token_i = self._encode_context_tokens(
                augmented_observations[:, i:i + 1, :],
                action,
                agent_embeddings=agent_embeddings_i,
            )
            context_tokens = torch.cat((context_tokens, context_token_i), dim=AGENTS_DIM)
            if context_mask is not None:
                context_mask = torch.cat((context_mask, agent_mask[:, i:i + 1]), dim=AGENTS_DIM)

        actions = torch.cat(actions_list, dim=AGENTS_DIM)
        if return_log_probs:
            return actions, torch.cat(log_probs_list, dim=AGENTS_DIM)
        return actions, None

    def forward(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        augmented_observations = self.encoder(local_obs, global_obs, agent_mask=agent_mask)
        actions, log_probs = self._generate_actions(
            augmented_observations=augmented_observations,
            batch_size=local_obs.shape[0],
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            return_log_probs=True,
        )
        values = self._critic_with_hidden_vars(
            augmented_observations,
            hidden_local_vars,
            hidden_global_vars,
            agent_mask=agent_mask,
        )
        return actions, log_probs, values

    def _evaluate_actions(
            self,
            batch: PPOSamples,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[torch.Tensor, torch.Tensor, LossDict, LossMetrics, torch.Tensor]:
        actions = self._policy_actions(batch.actions)
        if self.action_dist.compile_friendly:
            log_probs, values, augmented_observations = self._evaluate_actions_core(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                hidden_local_vars=batch.hidden_local_vars,
                hidden_global_vars=batch.hidden_global_vars,
                agent_mask=batch.agent_mask,
                previous_actions=batch.previous_actions,
                actions=actions,
            )
        else:
            latent_pi, values, augmented_observations = self._evaluate_latent_and_values(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                hidden_local_vars=batch.hidden_local_vars,
                hidden_global_vars=batch.hidden_global_vars,
                agent_mask=batch.agent_mask,
                actions=actions,
            )
            self.action_dist.update_latent_features(latent_pi)
            log_probs = self.action_dist.log_prob(actions, previous_actions=batch.previous_actions)
        extra_losses, extra_loss_metrics = self.action_dist.compute_extra_losses(
            agent_mask=batch.agent_mask,
            action_splitter=action_splitter,
        )
        return log_probs, values, extra_losses, extra_loss_metrics, augmented_observations

    def predict_values(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = previous_actions
        augmented_observations = self.encoder(local_obs, global_obs, agent_mask=agent_mask)
        return self._critic_with_hidden_vars(
            augmented_observations,
            hidden_local_vars,
            hidden_global_vars,
            agent_mask=agent_mask,
        )

    def _evaluate_latent_and_values(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._evaluate_latent_and_values_fn(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=actions,
        )

    def _evaluate_latent_and_values_impl(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        augmented_observations = self.encoder(local_obs, global_obs, agent_mask=agent_mask)
        query_tokens = self._encode_query_tokens(augmented_observations)
        memory_tokens = self._encode_memory_tokens(augmented_observations)
        context_tokens = self._encode_context_tokens(augmented_observations, actions)
        latent_pi = self.actor_head(
            self.decoder(
                query_tokens=query_tokens,
                context_tokens=context_tokens,
                memory_tokens=memory_tokens,
                agent_mask=agent_mask,
                memory_mask=agent_mask,
            )
        ).contiguous()
        values = self._critic_with_hidden_vars(
            augmented_observations,
            hidden_local_vars,
            hidden_global_vars,
            agent_mask=agent_mask,
        )
        return latent_pi, values, augmented_observations

    def _evaluate_actions_core(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._evaluate_actions_core_fn(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            actions=actions,
        )

    def _evaluate_actions_core_impl(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_pi, values, augmented_observations = self._evaluate_latent_and_values_impl(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=actions,
        )
        self.action_dist.update_latent_features(latent_pi)
        log_probs = self.action_dist.log_prob(actions, previous_actions=previous_actions)
        return log_probs, values, augmented_observations

    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> torch.Tensor:
        _ = hidden_local_vars
        _ = hidden_global_vars
        augmented_observations = self.encoder(local_obs, global_obs, agent_mask=agent_mask)
        actions, _ = self._generate_actions(
            augmented_observations=augmented_observations,
            batch_size=local_obs.shape[0],
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            return_log_probs=False,
        )
        return actions

    def make_sampler(
            self,
            episodes: list[PPOEpisodeSegment],
            config: PPOSamplerConfig,
    ) -> PPOSampler:
        return PPOSampler(
            episodes=episodes,
            config=config,
            requires_previous_actions=self.requires_previous_actions(),
        )

    def _critic_with_hidden_vars(
            self,
            augmented_observations: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        critic_local_obs = augmented_observations
        if self.hidden_local_vars_dim > 0:
            if hidden_local_vars is None:
                raise ValueError("hidden_local_vars must be provided when hidden_local_vars_dim > 0")
            critic_local_obs = torch.cat((augmented_observations, hidden_local_vars), dim=-1)

        if self.hidden_global_vars_dim <= 0:
            return self.critic(critic_local_obs, agent_mask=agent_mask)
        if hidden_global_vars is None:
            raise ValueError("hidden_global_vars must be provided when hidden_global_vars_dim > 0")
        return self.critic(critic_local_obs, hidden_global_vars, agent_mask=agent_mask)

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
            "query_encoder": self._module_grad_norm(self.query_encoder),
            "context_encoder": self._module_grad_norm(self.context_encoder),
            "memory_encoder": self._module_grad_norm(self.memory_encoder),
            "agent_embeddings_decoder": self._parameter_grad_norm(self.agent_embeddings_decoder),
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

        action_magnitude_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("action_magnitude_loss_coef", "action_magnitude"),
        )
        if action_magnitude_weight is not None:
            alias, value = action_magnitude_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.action_dist.set_action_magnitude_loss_coef(value)

        entropy_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("ent_loss_coef", "entropy", "ent"),
        )
        if entropy_weight is not None:
            alias, value = entropy_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.action_dist.set_all_ent_loss_coefs(value)

        super().update_loss_weights(**remaining_weights)

    @staticmethod
    def _build_token_encoder(
            *,
            input_dim: int,
            output_dim: int,
            hidden_dims: list[int] | None,
            act_fn_cls: ActivationFactory,
            linear_init_gain: float,
            projection_init_gain: float | None,
    ) -> nn.Module:
        linear_init = make_init_linear_orthogonal(linear_init_gain)
        projection_linear_init = (
            linear_init
            if projection_init_gain is None
            else make_init_linear_orthogonal(projection_init_gain)
        )
        if hidden_dims is None or len(hidden_dims) == 0:
            linear = nn.Linear(input_dim, output_dim)
            projection_linear_init(linear)
            return linear
        return MATPolicy._build_encoder_from_dims(
            input_dim=input_dim,
            dims=[*hidden_dims, output_dim],
            act_fn_cls=act_fn_cls,
            linear_init_gain=linear_init_gain,
            projection_init_gain=projection_init_gain,
        )

    @staticmethod
    def _build_encoder_from_dims(
            *,
            input_dim: int,
            dims: list[int],
            act_fn_cls: ActivationFactory,
            linear_init_gain: float,
            projection_init_gain: float | None,
    ) -> nn.Module:
        linear_init = make_init_linear_orthogonal(linear_init_gain)
        projection_linear_init = (
            linear_init
            if projection_init_gain is None
            else make_init_linear_orthogonal(projection_init_gain)
        )
        return MLP(
            input_dim=input_dim,
            hidden_dims=dims,
            end_with_act_fn=False,
            linear_init=linear_init,
            final_linear_init=projection_linear_init,
            act_fn_cls=act_fn_cls,
        )

    def _encode_memory_tokens(
            self,
            augmented_observations: torch.Tensor,
    ) -> torch.Tensor:
        return self.memory_encoder(augmented_observations)

    def _apply_optional_compile(self) -> None:
        if not self.config.compile_modules:
            return

        _ensure_torch_compile_available(compile_mode=self.config.compile_mode)

        self.encoder = self._compile_module(self.encoder)
        self.query_encoder = self._compile_module(self.query_encoder)
        self.context_encoder = self._compile_module(self.context_encoder)
        self.memory_encoder = self._compile_module(self.memory_encoder)
        self.decoder = self._compile_module(self.decoder)
        self.actor_head = self._compile_module(self.actor_head)
        self.critic = self._compile_module(self.critic)
        self._evaluate_latent_and_values_fn = self._compile_callable(self._evaluate_latent_and_values_impl)
        if self.action_dist.compile_friendly:
            self._generate_actions_fn = self._compile_callable(self._generate_actions_impl)
            self._evaluate_actions_core_fn = self._compile_callable(self._evaluate_actions_core_impl)

    def _compile_module(
            self,
            module: nn.Module,
    ) -> nn.Module:
        if isinstance(module, nn.Identity):
            return module
        return torch.compile(
            module,
            mode=self.config.compile_mode,
            fullgraph=False,
            dynamic=False,
        )

    def _compile_callable(
            self,
            fn: Callable[..., Any],
    ) -> Callable[..., Any]:
        return torch.compile(
            fn,
            mode=self.config.compile_mode,
            fullgraph=False,
            dynamic=False,
        )
