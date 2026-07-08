from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import torch
from gymnasium import spaces
from torch import nn

from swarmbots.learn.action_dists.beta_action_dist import BetaConfig
from swarmbots.learn.action_dists.gsde_action_dist import GSDEConfig
from swarmbots.learn.action_dists.hybrid_action_dist import (
    ContinuousActionDistConfigInput,
    HybridActionDistribution,
    continuous_config_to_dicts,
)
from swarmbots.learn.action_dists.predicted_std_action_dist import PredictedStdConfig
from swarmbots.learn.action_dists.reparameterized_sign_magnitude_kumaraswamy_action_dist import (
    ReparameterizedSignMagnitudeKumaraswamyConfig,
)
from swarmbots.learn.action_dists.reparameterized_squashed_gaussian_mixture_action_dist import (
    ReparameterizedSquashedGaussianMixtureConfig,
)
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import SquashedDiagGaussianConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig
from swarmbots.learn.algos.off_policy.replay_buffer import OffPolicyReplayBatch, OffPolicyReplayEpisodeSegmentBatch
from swarmbots.learn.algos.ppo.ppo_policy import PopArtConfig
from swarmbots.learn.algos.sac.base_sac_policy import BaseSACPolicy
from swarmbots.learn.algos.sac.sac_nop import (
    SACNOPConfig,
    SACNOPLatentSource,
    SACNOPModule,
    normalize_nop_latent_source,
)
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.deep_set import DeepSetCritic
from swarmbots.learn.nn_components.mlp import MLP
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal
from swarmbots.learn.polyak_update import polyak_update
from swarmbots.learn.serialization_utils import serialize_dataclass, serialize_value


class TMASACActorHeadKind(Enum):
    DECENTRALIZED = "decentralized"


@dataclass(frozen=True)
class TMASACActorHeadConfig:
    kind: TMASACActorHeadKind | str = TMASACActorHeadKind.DECENTRALIZED
    hidden_dims: list[int] | None = None
    normalize_input: bool = False
    init_gain: float = 1.0


@dataclass(frozen=True)
class TMASACCriticConfig:
    n_local_projection_hidden_layers: int = 1
    n_value_regressor_hidden_layers: int = 2
    use_popart: bool = False
    popart_config: PopArtConfig = field(default_factory=PopArtConfig)
    local_projection_init_gain: float = 1.0
    value_regressor_init_gain: float = 1.0
    value_head_init_gain: float = 0.01


@dataclass(frozen=True)
class TMASACPolicyConfig:
    actor_encoder_config: MATEncoderConfig = field(default_factory=MATEncoderConfig)
    critic_encoder_config: MATEncoderConfig = field(default_factory=MATEncoderConfig)
    share_observation_encoder: bool = False
    actor_head_config: TMASACActorHeadConfig = field(default_factory=TMASACActorHeadConfig)
    critic_config: TMASACCriticConfig = field(default_factory=TMASACCriticConfig)
    act_fn_cls: ActivationFactory = nn.GELU
    dropout: float = 0.0
    continuous_config: ContinuousActionDistConfigInput = field(
        default_factory=lambda: PredictedStdConfig(base_std=1.0)
    )
    max_agents: int | None = None
    nop_config: SACNOPConfig = field(default_factory=SACNOPConfig)
    compile_modules: bool = False
    compile_mode: str = "default"
    action_net_init_gain: float = 0.01


class TMASACDecentralizedActorHead(nn.Module):
    def __init__(
            self,
            *,
            d_model: int,
            config: TMASACActorHeadConfig,
            act_fn_cls: ActivationFactory,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.config = config
        self.input_norm = nn.LayerNorm(d_model) if config.normalize_input else nn.Identity()
        if config.hidden_dims:
            self.head = MLP(
                input_dim=d_model,
                hidden_dims=[*config.hidden_dims],
                end_with_act_fn=True,
                linear_init=make_init_linear_orthogonal(config.init_gain),
                act_fn_cls=act_fn_cls,
            )
            self.latent_dim = int(config.hidden_dims[-1])
        else:
            self.head = nn.Identity()
            self.latent_dim = int(d_model)

    def forward(self, actor_latents: torch.Tensor, agent_mask: torch.Tensor | None = None) -> torch.Tensor:
        _ = agent_mask
        return self.head(self.input_norm(actor_latents)).contiguous()


class TMASACTwinCritic(nn.Module):
    def __init__(
            self,
            *,
            n_agents: int,
            max_agents: int,
            local_obs_dim: int,
            global_obs_dim: int,
            hidden_local_vars_dim: int,
            hidden_global_vars_dim: int,
            action_dim: int,
            encoder_config: MATEncoderConfig,
            critic_config: TMASACCriticConfig,
            act_fn_cls: ActivationFactory,
            dropout: float,
            encoder: MATEncoder | None = None,
    ) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.hidden_local_vars_dim = int(hidden_local_vars_dim)
        self.hidden_global_vars_dim = int(hidden_global_vars_dim)
        self.action_dim = int(action_dim)
        self.d_model = int(encoder_config.d_model)
        self.encoder_config = replace(
            encoder_config,
            act_fn_cls=act_fn_cls,
            dropout=dropout,
        )
        self.encoder = encoder if encoder is not None else MATEncoder(
            config=self.encoder_config,
            max_agents=max_agents,
            local_obs_dim=local_obs_dim,
            global_obs_dim=global_obs_dim,
        )
        q_local_dim = self.d_model + self.hidden_local_vars_dim + self.action_dim
        self.q1 = self._build_q_network(
            q_local_dim=q_local_dim,
            hidden_global_vars_dim=self.hidden_global_vars_dim,
            critic_config=critic_config,
            act_fn_cls=act_fn_cls,
        )
        self.q2 = self._build_q_network(
            q_local_dim=q_local_dim,
            hidden_global_vars_dim=self.hidden_global_vars_dim,
            critic_config=critic_config,
            act_fn_cls=act_fn_cls,
        )

    def encode(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.encoder(local_obs, global_obs, agent_mask=agent_mask)

    def forward(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        critic_latents = self.encode(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
        )
        q_features = self._build_q_local_features(
            critic_latents=critic_latents,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
        )
        q_global_features = self._build_q_global_features(hidden_global_vars)
        return (
            self.q1(q_features, q_global_features, agent_mask=agent_mask),
            self.q2(q_features, q_global_features, agent_mask=agent_mask),
        )

    def _build_q_local_features(
            self,
            *,
            critic_latents: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = [critic_latents]
        if self.hidden_local_vars_dim > 0:
            if hidden_local_vars is None:
                raise ValueError("hidden_local_vars must be provided when hidden_local_vars_dim > 0")
            parts.append(hidden_local_vars)
        parts.append(actions)
        return torch.cat(parts, dim=-1)

    def _build_q_global_features(self, hidden_global_vars: torch.Tensor | None) -> torch.Tensor | None:
        if self.hidden_global_vars_dim <= 0:
            return None
        if hidden_global_vars is None:
            raise ValueError("hidden_global_vars must be provided when hidden_global_vars_dim > 0")
        return hidden_global_vars

    @staticmethod
    def _build_q_network(
            *,
            q_local_dim: int,
            hidden_global_vars_dim: int,
            critic_config: TMASACCriticConfig,
            act_fn_cls: ActivationFactory,
    ) -> DeepSetCritic:
        return DeepSetCritic(
            num_local_features=q_local_dim,
            local_projection_hidden_dims=[
                q_local_dim
            ] * critic_config.n_local_projection_hidden_layers,
            value_regressor_hidden_dims=[
                q_local_dim
            ] * critic_config.n_value_regressor_hidden_layers,
            num_global_features=hidden_global_vars_dim,
            act_fn_cls=act_fn_cls,
            context_in_elements=hidden_global_vars_dim > 0,
            local_projection_linear_init_gain=critic_config.local_projection_init_gain,
            value_regressor_linear_init_gain=critic_config.value_regressor_init_gain,
            value_head_linear_init_gain=critic_config.value_head_init_gain,
            use_popart=critic_config.use_popart,
            popart_beta=critic_config.popart_config.beta,
            popart_eps=critic_config.popart_config.eps,
            popart_min_std=critic_config.popart_config.min_std,
            popart_init_sigma=critic_config.popart_config.init_sigma,
        )


class TMASACPolicy(BaseSACPolicy):
    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: TMASACPolicyConfig = TMASACPolicyConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self._validate_continuous_action_space(env)
        self._validate_config(config)

        self.n_agents = int(env.n_agents)
        self.max_agents = self.n_agents if config.max_agents is None else int(config.max_agents)
        if self.max_agents < self.n_agents:
            raise ValueError(f"max_agents must be >= env.n_agents ({self.n_agents}), got {self.max_agents}")
        self.local_obs_dim = int(env.local_obs_dim)
        self.global_obs_dim = int(env.global_obs_dim)
        self.hidden_local_vars_dim = int(env.hidden_local_vars_dim)
        self.hidden_global_vars_dim = int(env.hidden_global_vars_dim)
        self.agent_action_dim = int(env.action_space.total_agent_action_dim)
        self.act_fn_cls = config.act_fn_cls
        self.dropout = float(config.dropout)
        self.actor_head_kind = self._normalize_actor_head_kind(config.actor_head_config.kind)
        self.nop_latent_source = normalize_nop_latent_source(config.nop_config.latent_source)
        self.share_observation_encoder = bool(config.share_observation_encoder)

        self.actor_encoder_config = replace(
            config.actor_encoder_config,
            act_fn_cls=config.act_fn_cls,
            dropout=config.dropout,
        )
        self.critic_encoder_config = replace(
            config.critic_encoder_config,
            act_fn_cls=config.act_fn_cls,
            dropout=config.dropout,
        )
        if self.share_observation_encoder and self.actor_encoder_config != self.critic_encoder_config:
            raise ValueError(
                "share_observation_encoder=True requires matching actor_encoder_config and critic_encoder_config "
                "after applying TMASACPolicyConfig.act_fn_cls/dropout."
            )

        shared_encoder = (
            self._build_observation_encoder(self.actor_encoder_config)
            if self.share_observation_encoder
            else None
        )
        self._actor_encoder = None if self.share_observation_encoder else self._build_observation_encoder(
            self.actor_encoder_config
        )
        self.actor_head = self._build_actor_head()
        self.action_dist = HybridActionDistribution(
            latent_dim=self.actor_head.latent_dim,
            action_space=env.action_space,
            continuous_config=self._validated_continuous_config(env),
            bernoulli_config=None,
            action_net_initialization=make_init_linear_orthogonal(config.action_net_init_gain),
        )

        self.critic = TMASACTwinCritic(
            n_agents=self.n_agents,
            max_agents=self.max_agents,
            local_obs_dim=self.local_obs_dim,
            global_obs_dim=self.global_obs_dim,
            hidden_local_vars_dim=self.hidden_local_vars_dim,
            hidden_global_vars_dim=self.hidden_global_vars_dim,
            action_dim=self.agent_action_dim,
            encoder_config=self.critic_encoder_config,
            critic_config=config.critic_config,
            act_fn_cls=config.act_fn_cls,
            dropout=config.dropout,
            encoder=shared_encoder,
        )
        self.critic_target = TMASACTwinCritic(
            n_agents=self.n_agents,
            max_agents=self.max_agents,
            local_obs_dim=self.local_obs_dim,
            global_obs_dim=self.global_obs_dim,
            hidden_local_vars_dim=self.hidden_local_vars_dim,
            hidden_global_vars_dim=self.hidden_global_vars_dim,
            action_dim=self.agent_action_dim,
            encoder_config=self.critic_encoder_config,
            critic_config=config.critic_config,
            act_fn_cls=config.act_fn_cls,
            dropout=config.dropout,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)

        self.actor_nop = self._build_nop_module("actor")
        self.critic_nop = self._build_nop_module("critic")
        self._apply_optional_compile()

    @property
    def actor_encoder(self) -> MATEncoder:
        if self.share_observation_encoder:
            return self._critic_module().encoder
        assert self._actor_encoder is not None
        return self._actor_encoder

    def action_log_prob(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            deterministic: bool = False,
            use_rsample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = hidden_local_vars
        _ = hidden_global_vars
        actor_latents = self.encode_actor(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
        )
        latent_pi = self.actor_head(actor_latents, agent_mask=agent_mask)
        actions, log_probs = self.action_dist.get_actions_with_log_probs(
            latent_pi,
            deterministic=deterministic,
            use_rsample=use_rsample,
        )
        return self._mask_actions(actions, agent_mask), self._mask_log_probs(log_probs, agent_mask)

    def q_values(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.critic(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )

    def target_q_values(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.critic_target(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )

    def encode_critic(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._critic_module().encode(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
        )

    def encode_actor(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        actor_latents = self.actor_encoder(local_obs, global_obs, agent_mask=agent_mask)
        if self.share_observation_encoder:
            return actor_latents.detach()
        return actor_latents

    def actor_parameters(self) -> list[nn.Parameter]:
        actor_encoder = None if self.share_observation_encoder else self.actor_encoder
        return list(self._iter_parameters(actor_encoder, self.actor_head, self.action_dist, self.actor_nop))

    def critic_parameters(self) -> list[nn.Parameter]:
        return list(self._iter_parameters(self.critic, self.critic_nop))

    def compute_actor_nop_loss(
            self,
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if self.actor_nop is None:
            return None, {}
        local_obs, global_obs, agent_mask = self._nop_initial_obs(batch)
        actor_latents = self.encode_actor(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
        )
        return self.actor_nop.compute_loss(source_latents=actor_latents, batch=batch)

    def compute_critic_nop_loss(
            self,
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if self.critic_nop is None:
            return None, {}
        local_obs, global_obs, agent_mask = self._nop_initial_obs(batch)
        critic_latents = self.encode_critic(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
        )
        return self.critic_nop.compute_loss(source_latents=critic_latents, batch=batch)

    def has_nop_loss(self) -> bool:
        return self.actor_nop is not None or self.critic_nop is not None

    def polyak_update_targets(self, tau: float) -> None:
        polyak_update(self.critic, self.critic_target, tau)

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            "tmasac_policy_config": {
                "actor_encoder_config": serialize_dataclass(self.actor_encoder_config),
                "critic_encoder_config": serialize_dataclass(self.critic_encoder_config),
                    "share_observation_encoder": self.share_observation_encoder,
                    "actor_head_config": serialize_dataclass(self.config.actor_head_config),
                "critic_config": serialize_dataclass(self.config.critic_config),
                "act_fn_cls": serialize_value(self.act_fn_cls),
                "dropout": self.dropout,
                "continuous_config": continuous_config_to_dicts(self.action_dist.continuous_configs),
                "max_agents": self.max_agents,
                "nop_config": serialize_dataclass(self.config.nop_config),
                "nop_latent_source": self.nop_latent_source.value,
                "actor_nop": None if self.actor_nop is None else self.actor_nop.get_hyper_parameters(),
                "critic_nop": None if self.critic_nop is None else self.critic_nop.get_hyper_parameters(),
                "compile_modules": self.config.compile_modules,
                "compile_mode": self.config.compile_mode,
                "action_net_init_gain": self.config.action_net_init_gain,
                "action_dist_compile_friendly": self.action_dist.compile_friendly,
            }
        }

    def get_grad_norms(self) -> dict[str, float]:
        return {
            "actor_encoder": self._module_grad_norm(self.actor_encoder),
            "actor_head": self._module_grad_norm(self.actor_head),
            "action_dist": self._module_grad_norm(self.action_dist),
            "critic": self._module_grad_norm(self.critic),
            "critic_target": self._module_grad_norm(self.critic_target),
            "actor_nop": self._module_grad_norm(self.actor_nop),
            "critic_nop": self._module_grad_norm(self.critic_nop),
            "total": self._module_grad_norm(self),
        }

    def update_loss_weights(self, **weights: float) -> None:
        if not weights:
            return
        remaining_weights = dict(weights)
        nop_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("nop_loss_coef", "world_model_loss_coef", "wm_loss_coef"),
        )
        if nop_weight is not None:
            alias, value = nop_weight
            self._set_all_nop_loss_coefs(alias=alias, value=value)

        actor_nop_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("actor_nop_loss_coef", "actor_world_model_loss_coef"),
        )
        if actor_nop_weight is not None:
            alias, value = actor_nop_weight
            self._set_nop_loss_coef(self.actor_nop, alias=alias, value=value)

        critic_nop_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("critic_nop_loss_coef", "critic_world_model_loss_coef"),
        )
        if critic_nop_weight is not None:
            alias, value = critic_nop_weight
            self._set_nop_loss_coef(self.critic_nop, alias=alias, value=value)

        if remaining_weights:
            raise ValueError(f"Unknown weights given: {remaining_weights}")

    def _build_actor_head(self) -> TMASACDecentralizedActorHead:
        if self.actor_head_kind is not TMASACActorHeadKind.DECENTRALIZED:
            raise NotImplementedError(f"Unsupported TMASAC actor head kind: {self.actor_head_kind}")
        return TMASACDecentralizedActorHead(
            d_model=self.actor_encoder_config.d_model,
            config=self.config.actor_head_config,
            act_fn_cls=self.config.act_fn_cls,
        )

    def _build_observation_encoder(self, config: MATEncoderConfig) -> MATEncoder:
        return MATEncoder(
            config=config,
            max_agents=self.max_agents,
            local_obs_dim=self.local_obs_dim,
            global_obs_dim=self.global_obs_dim,
        )

    def _build_nop_module(self, source_name: str) -> SACNOPModule | None:
        nop_config = self.config.nop_config
        if not nop_config.enabled:
            return None
        source = self.nop_latent_source
        if source_name == "actor":
            if source not in (SACNOPLatentSource.ACTOR, SACNOPLatentSource.BOTH):
                return None
            source_latent_dim = self.actor_encoder_config.d_model
        elif source_name == "critic":
            if source not in (SACNOPLatentSource.CRITIC, SACNOPLatentSource.BOTH):
                return None
            source_latent_dim = self.critic_encoder_config.d_model
        else:
            raise ValueError(f"Unknown NOP source {source_name!r}")
        return SACNOPModule(
            n_agents=self.n_agents,
            source_latent_dim=source_latent_dim,
            action_dim=self.agent_action_dim,
            config=nop_config,
            name=source_name,
        )

    def _validated_continuous_config(self, env: BaseLearnEnvWrapper) -> ContinuousActionDistConfigInput:
        config = self.config.continuous_config
        configs = config if isinstance(config, list) else [config] * env.action_space.n_spaces
        if len(configs) != env.action_space.n_spaces:
            raise ValueError(
                f"Expected {env.action_space.n_spaces} continuous configs, got {len(configs)}"
            )
        supported_reparameterized_configs = (
            BetaConfig,
            GSDEConfig,
            PredictedStdConfig,
            ReparameterizedSignMagnitudeKumaraswamyConfig,
            ReparameterizedSquashedGaussianMixtureConfig,
            SquashedDiagGaussianConfig,
        )
        for idx, sub_config in enumerate(configs):
            if not isinstance(sub_config, supported_reparameterized_configs):
                raise ValueError(
                    "TMASACPolicy currently supports only reparameterized continuous action configs for SAC "
                    f"({', '.join(cls.__name__ for cls in supported_reparameterized_configs)}); "
                    f"action sub-space {idx} got {type(sub_config).__name__}."
                )
        return config

    def _apply_optional_compile(self) -> None:
        if not self.config.compile_modules:
            return
        if not hasattr(torch, "compile") or not callable(torch.compile):
            raise RuntimeError("TMASACPolicyConfig.compile_modules=True requires torch.compile support.")
        if not self.config.compile_mode:
            raise ValueError("TMASACPolicyConfig.compile_mode must be a non-empty string when compile_modules=True.")

        if self.share_observation_encoder:
            self.critic.encoder = self._compile_module(self.critic.encoder)
        else:
            assert self._actor_encoder is not None
            self._actor_encoder = self._compile_module(self._actor_encoder)
        self.actor_head = self._compile_module(self.actor_head)
        self.critic = self._compile_module(self.critic)
        self.critic_target = self._compile_module(self.critic_target)

    def _compile_module(self, module: nn.Module) -> nn.Module:
        return torch.compile(
            module,
            mode=self.config.compile_mode,
            fullgraph=False,
            dynamic=False,
        )

    def _critic_module(self) -> TMASACTwinCritic:
        if isinstance(self.critic, TMASACTwinCritic):
            return self.critic
        orig_mod = getattr(self.critic, "_orig_mod", None)
        if isinstance(orig_mod, TMASACTwinCritic):
            return orig_mod
        raise RuntimeError(f"Cannot resolve TMASACTwinCritic from compiled module {type(self.critic).__name__}")

    @staticmethod
    def _nop_initial_obs(
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if isinstance(batch, OffPolicyReplayEpisodeSegmentBatch):
            return (
                batch.local_obs[:, 0],
                batch.global_obs[:, 0],
                None if batch.agent_mask is None else batch.agent_mask[:, 0],
            )
        return batch.local_obs, batch.global_obs, batch.agent_mask

    @staticmethod
    def _normalize_actor_head_kind(kind: TMASACActorHeadKind | str) -> TMASACActorHeadKind:
        if isinstance(kind, TMASACActorHeadKind):
            return kind
        try:
            return TMASACActorHeadKind(kind.lower())
        except ValueError as exc:
            valid = [item.value for item in TMASACActorHeadKind]
            raise ValueError(f"Unknown TMASAC actor head kind {kind!r}; expected one of {valid}") from exc

    @staticmethod
    def _validate_continuous_action_space(env: BaseLearnEnvWrapper) -> None:
        for key, sub_space in env.action_space.items():
            if not isinstance(sub_space, spaces.Box):
                raise ValueError(
                    "SAC/TMASAC currently supports continuous Box action sub-spaces only; "
                    f"sub-space {key!r} is {type(sub_space).__name__}."
                )

    @staticmethod
    def _validate_config(config: TMASACPolicyConfig) -> None:
        if config.critic_config.use_popart:
            raise NotImplementedError(
                "TMASACPolicy does not support PopArt critics yet because SAC must update PopArt target "
                "statistics during critic training. Set critic_config.use_popart=False."
            )

    @staticmethod
    def _mask_actions(actions: torch.Tensor, agent_mask: torch.Tensor | None) -> torch.Tensor:
        if agent_mask is None:
            return actions
        return actions.masked_fill(~agent_mask.unsqueeze(-1), 0.0)

    @staticmethod
    def _mask_log_probs(log_probs: torch.Tensor, agent_mask: torch.Tensor | None) -> torch.Tensor:
        if agent_mask is None:
            return log_probs
        return log_probs.masked_fill(~agent_mask, 0.0)

    @staticmethod
    def _iter_parameters(*modules: nn.Module | None) -> Iterable[nn.Parameter]:
        for module in modules:
            if module is None:
                continue
            yield from module.parameters()

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

    def _set_all_nop_loss_coefs(self, *, alias: str, value: float) -> None:
        self._validate_nop_loss_coef(alias=alias, value=value)
        if self.actor_nop is not None:
            self.actor_nop.nop_loss_coef = value
        if self.critic_nop is not None:
            self.critic_nop.nop_loss_coef = value

    def _set_nop_loss_coef(self, module: SACNOPModule | None, *, alias: str, value: float) -> None:
        self._validate_nop_loss_coef(alias=alias, value=value)
        if module is None:
            raise ValueError(f"{alias} was provided but the corresponding NOP module is disabled.")
        module.nop_loss_coef = value

    @staticmethod
    def _validate_nop_loss_coef(*, alias: str, value: float) -> None:
        if value < 0:
            raise ValueError(f"{alias} must be >= 0, got {value}")
