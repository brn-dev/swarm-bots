"""Deterministic actors and centralized critics for DDPG, MATD3 and TMATD3."""

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Self

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig
from swarmbots.learn.algos.off_policy.joint_critic import JointCriticConfig, JointQNetwork
from swarmbots.learn.algos.off_policy.action_noise import add_normalized_action_noise
from swarmbots.learn.algos.off_policy.replay_buffer import OffPolicyReplayBatch, OffPolicyReplayEpisodeSegmentBatch
from swarmbots.learn.algos.off_policy.nop import SACNOPConfig, SACNOPLatentSource, SACNOPModule, normalize_nop_latent_source
from swarmbots.learn.algos.off_policy.actor_heads import (
    TMASACActorHeadConfig,
    TMASACActorHeadKind,
    TMASACIndependentActorHead,
)
from swarmbots.learn.algos.off_policy.transformer_critic import (
    TransformerCriticConfig as TMASACCriticConfig,
    TransformerTwinCritic as TMASACTwinCritic,
)
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal
from swarmbots.learn.polyak_update import polyak_update
from swarmbots.learn.serialization_utils import serialize_dataclass


@dataclass(frozen=True)
class TD3PolicyConfig:
    actor_encoder_config: MATEncoderConfig = field(default_factory=lambda: MATEncoderConfig(use_agent_attention=False))
    actor_head_config: TMASACActorHeadConfig = field(default_factory=TMASACActorHeadConfig)
    critic_encoder_config: MATEncoderConfig = field(default_factory=MATEncoderConfig)
    transformer_critic_config: TMASACCriticConfig = field(
        default_factory=TMASACCriticConfig
    )
    critic_kind: Literal["mlp", "deepset", "transformer"] = "mlp"
    joint_critic_config: JointCriticConfig = field(default_factory=JointCriticConfig)
    n_critics: int = 2
    nop_config: SACNOPConfig = field(default_factory=SACNOPConfig)
    action_net_init_gain: float = 0.01
    compile_modules: bool = False
    compile_mode: str = "default"


class DeterministicActor(nn.Module):
    def __init__(self, env, config: TD3PolicyConfig) -> None:
        super().__init__()
        self.encoder = self._build_encoder(env, config)
        self.head = TMASACIndependentActorHead(
            d_model=config.actor_encoder_config.d_model,
            config=config.actor_head_config,
            act_fn_cls=config.actor_encoder_config.act_fn_cls,
        )
        self.action_net = nn.Linear(self.head.latent_dim, env.action_space.total_agent_action_dim)
        make_init_linear_orthogonal(config.action_net_init_gain)(self.action_net)
        lows, highs = [], []
        for space in env.action_space.sub_spaces:
            if not isinstance(space, spaces.Box):
                raise ValueError("DDPG/TD3 require continuous Box action spaces")
            if not np.isfinite(space.low).all() or not np.isfinite(space.high).all():
                raise ValueError("DDPG/TD3 require finite action bounds")
            if not (space.high > space.low).all():
                raise ValueError("DDPG/TD3 action upper bounds must exceed lower bounds")
            low, high = space.low, space.high
            if low.ndim == 3:
                if not ((low == low[0]).all() and (high == high[0]).all()):
                    raise ValueError("DDPG/TD3 require identical action bounds across environments")
                low, high = low[0], high[0]
            lows.append(torch.as_tensor(low, dtype=torch.float32))
            highs.append(torch.as_tensor(high, dtype=torch.float32))
        low, high = torch.cat(lows, -1), torch.cat(highs, -1)
        self.register_buffer("action_center", (high + low) / 2)
        self.register_buffer("action_scale", (high - low) / 2)

    def _build_encoder(self, env, config: TD3PolicyConfig) -> nn.Module:
        return MATEncoder(
            config.actor_encoder_config,
            max_agents=env.n_agents,
            local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim,
        )

    def forward(
        self,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if agent_mask is not None:
            local_obs = local_obs.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
        latent = self.encoder(local_obs, global_obs, agent_mask=agent_mask)
        actions = self.action_net(self.head(latent, agent_mask=agent_mask)).tanh()
        actions = self.action_center + self.action_scale * actions
        return actions if agent_mask is None else actions.masked_fill(~agent_mask.unsqueeze(-1), 0.0)


class TD3Policy(BasePolicy):
    def __init__(self, env, config: TD3PolicyConfig = TD3PolicyConfig()) -> None:
        super().__init__()
        if config.n_critics not in (1, 2):
            raise ValueError("DDPG/TD3 use one/two critics respectively")
        if config.critic_kind not in {"mlp", "deepset", "transformer"}:
            raise ValueError(f"Unknown critic kind: {config.critic_kind!r}")
        if config.critic_kind == "transformer" and config.n_critics != 2:
            raise ValueError("The transformer critic requires two Q networks")
        if config.nop_config.enabled and config.critic_kind == "mlp":
            raise ValueError("MLP critics do not support NOP; use a Deep Set or transformer critic")
        self.nop_latent_source = normalize_nop_latent_source(config.nop_config.latent_source)
        if config.nop_config.enabled and self.nop_latent_source is SACNOPLatentSource.SHARED_ENCODER:
            raise ValueError("DDPG/TD3 do not share actor and critic observation encoders")
        actor_kind = TMASACActorHeadKind(config.actor_head_config.kind)
        if actor_kind is TMASACActorHeadKind.QCX:
            raise ValueError("DDPG/TD3 use independent action heads, not a QCX decoder")
        if actor_kind is TMASACActorHeadKind.DECENTRALIZED:
            config = replace(
                config, actor_encoder_config=replace(config.actor_encoder_config, use_agent_attention=False)
            )
        if config.critic_kind != "transformer":
            config = replace(config, joint_critic_config=replace(config.joint_critic_config, kind=config.critic_kind))
        if config.transformer_critic_config.use_popart:
            raise ValueError("DDPG/TD3 do not support PopArt")
        if getattr(env, "has_scenario_id", False):
            raise ValueError("DDPG/TD3 currently require a single scenario observation layout")
        self.config = config
        self.n_agents = int(env.n_agents)
        self.local_obs_dim = int(env.local_obs_dim)
        self.global_obs_dim = int(env.global_obs_dim)
        self.agent_action_dim = int(env.action_space.total_agent_action_dim)
        self.exploration_noise = 0.1
        self.actor = self._build_actor(env, config)
        critic_kwargs = dict(
            hidden_local_vars_dim=env.hidden_local_vars_dim,
            hidden_global_vars_dim=env.hidden_global_vars_dim,
            action_dim=self.agent_action_dim,
            act_fn_cls=config.critic_encoder_config.act_fn_cls,
        )
        if config.critic_kind == "transformer":
            self.critic = self._build_transformer_critic(
                n_agents=env.n_agents,
                max_agents=env.n_agents,
                local_input_dim=env.local_obs_dim,
                global_input_dim=env.global_obs_dim,
                encoder_config=config.critic_encoder_config,
                critic_config=config.transformer_critic_config,
                dropout=0.0,
                **critic_kwargs,
            )
        else:
            self.critic = nn.ModuleList(
                [
                    self._build_joint_critic(
                        n_agents=env.n_agents,
                        local_obs_dim=env.local_obs_dim,
                        global_obs_dim=env.global_obs_dim,
                        config=config.joint_critic_config,
                        **critic_kwargs,
                    )
                    for _ in range(config.n_critics)
                ]
            )
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False).eval()
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False).eval()
        self.actor_nop = self._build_nop_module("actor")
        self.critic_nop = self._build_nop_module("critic")
        # Keep state_dict names identical between eager and compiled policies.
        object.__setattr__(self, "_actor_forward", None)
        object.__setattr__(self, "_target_actor_forward", None)
        object.__setattr__(self, "_critic_forward", None)
        object.__setattr__(self, "_target_critic_forward", None)
        object.__setattr__(self, "_critic_with_latents_forward", None)
        if config.compile_modules:
            for name, function in (
                ("_actor_forward", self.actor),
                ("_target_actor_forward", self.actor_target),
                ("_critic_forward", self._q_values),
                ("_target_critic_forward", self._target_q_values),
                ("_critic_with_latents_forward", self._q_values_with_nop_latents),
            ):
                object.__setattr__(self, name, torch.compile(function, mode=config.compile_mode, dynamic=False))

    def _build_actor(self, env, config: TD3PolicyConfig) -> DeterministicActor:
        return DeterministicActor(env, config)

    def _build_transformer_critic(self, **kwargs) -> TMASACTwinCritic:
        return TMASACTwinCritic(**kwargs)

    def _build_joint_critic(self, **kwargs) -> JointQNetwork:
        return JointQNetwork(**kwargs)

    @property
    def gsde_enabled(self) -> bool:
        return False

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.actor_target.eval()
        self.critic_target.eval()
        return self

    def actor_actions(
        self,
        *,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
        target: bool = False,
    ) -> torch.Tensor:
        compiled = self._target_actor_forward if target else self._actor_forward
        actor = self.actor_target if target else self.actor
        if compiled is not None:
            actor = compiled
        return actor(local_obs, global_obs, agent_mask)

    def add_action_noise(
        self,
        actions: torch.Tensor,
        *,
        std: float,
        clip: float | None = None,
        agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return add_normalized_action_noise(
            actions, center=self.actor.action_center, scale=self.actor.action_scale,
            std=std, clip=clip, agent_mask=agent_mask,
        )

    def act(
        self,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        hidden_local_vars: torch.Tensor | None = None,
        hidden_global_vars: torch.Tensor | None = None,
        agent_mask: torch.Tensor | None = None,
        scenario_ids: torch.Tensor | None = None,
        previous_actions: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        actions = self.actor_actions(local_obs=local_obs, global_obs=global_obs, agent_mask=agent_mask)
        if not deterministic and self.exploration_noise:
            actions = self.add_action_noise(actions, std=self.exploration_noise, agent_mask=agent_mask)
        return actions

    def _evaluate_critic(self, critic, **observations):
        for key in ("hidden_local_vars", "hidden_global_vars", "agent_mask"):
            observations.setdefault(key, None)
        if self.config.critic_kind == "transformer":
            mask = observations["agent_mask"]
            if mask is not None:
                for key in ("local_obs", "hidden_local_vars", "actions"):
                    if observations[key] is not None:
                        observations[key] = observations[key].masked_fill(~mask.unsqueeze(-1), 0.0)
            return critic(**observations)[:2]
        return tuple(network(**observations) for network in critic)

    def _q_values(self, **observations):
        return self._evaluate_critic(self.critic, **observations)

    def _target_q_values(self, **observations):
        return self._evaluate_critic(self.critic_target, **observations)

    def q_values(self, **observations: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        forward = self._q_values if self._critic_forward is None else self._critic_forward
        return forward(**observations)

    def target_q_values(self, **observations: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
        forward = self._target_q_values if self._target_critic_forward is None else self._target_critic_forward
        return forward(**observations)

    def _q_values_with_nop_latents(self, **observations):
        for key in ("hidden_local_vars", "hidden_global_vars", "agent_mask"):
            observations.setdefault(key, None)
        if self.config.critic_kind == "transformer":
            mask = observations["agent_mask"]
            if mask is not None:
                for key in ("local_obs", "hidden_local_vars", "actions"):
                    if observations[key] is not None:
                        observations[key] = observations[key].masked_fill(~mask.unsqueeze(-1), 0.0)
            q1, q2, latents = self.critic(**observations)
            if mask is not None:
                latents = latents.masked_fill(~mask.unsqueeze(-1), 0.0)
            return (q1, q2), latents
        results = [network.forward_with_latents(**observations) for network in self.critic]
        return tuple(value for value, _ in results), torch.cat([latents for _, latents in results], dim=-1)

    def q_values_with_nop_latents(self, **observations):
        if self.critic_nop is None:
            return self.q_values(**observations), None
        forward = self._q_values_with_nop_latents
        if self._critic_with_latents_forward is not None:
            forward = self._critic_with_latents_forward
        return forward(**observations)

    def _build_nop_module(self, source_name: str) -> SACNOPModule | None:
        config = self.config.nop_config
        source = self.nop_latent_source
        if not config.enabled or source not in (SACNOPLatentSource(source_name), SACNOPLatentSource.BOTH):
            return None
        latent_dim = self.config.actor_encoder_config.d_model if source_name == "actor" else (
            self.critic.nop_source_latent_dim if self.config.critic_kind == "transformer" else
            sum(network.nop_source_latent_dim for network in self.critic)
        )
        return SACNOPModule(
            n_agents=self.n_agents, source_latent_dim=latent_dim, action_dim=self.agent_action_dim,
            config=config, name=source_name,
            skip_first_transition=config.skip_first_transition_for_critic and source_name == "critic",
        )

    def compute_actor_nop_loss(
        self, batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if self.actor_nop is None:
            return None, {}
        origin = batch.origin_batch if isinstance(batch, OffPolicyReplayEpisodeSegmentBatch) else batch
        local = origin.local_obs
        if origin.agent_mask is not None:
            local = local.masked_fill(~origin.agent_mask.unsqueeze(-1), 0.0)
        latents = self.actor.encoder(local, origin.global_obs, agent_mask=origin.agent_mask)
        return self.actor_nop.compute_loss(source_latents=latents, batch=batch)

    def compute_critic_nop_loss(
        self, batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
        *, source_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if self.critic_nop is None:
            return None, {}
        if source_latents is None:
            origin = batch.origin_batch if isinstance(batch, OffPolicyReplayEpisodeSegmentBatch) else batch
            _, source_latents = self.q_values_with_nop_latents(
                local_obs=origin.local_obs, global_obs=origin.global_obs, actions=origin.actions,
                hidden_local_vars=origin.hidden_local_vars, hidden_global_vars=origin.hidden_global_vars,
                agent_mask=origin.agent_mask,
            )
        return self.critic_nop.compute_loss(source_latents=source_latents, batch=batch)

    def actor_parameters(self) -> list[nn.Parameter]:
        return list(self.actor.parameters()) + ([] if self.actor_nop is None else list(self.actor_nop.parameters()))

    def critic_parameters(self) -> list[nn.Parameter]:
        return list(self.critic.parameters()) + ([] if self.critic_nop is None else list(self.critic_nop.parameters()))

    def polyak_update_targets(self, tau: float) -> None:
        polyak_update(self.actor, self.actor_target, tau)
        polyak_update(self.critic, self.critic_target, tau)

    def requires_previous_actions(self) -> bool:
        return False

    def requires_recurrent_training(self) -> bool:
        return False

    def has_nop_loss(self) -> bool:
        return self.actor_nop is not None or self.critic_nop is not None

    def get_nop_num_next_steps(self) -> int:
        return self.config.nop_config.num_next_steps

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            "td3_policy_config": serialize_dataclass(self.config), "exploration_noise": self.exploration_noise,
            "actor_nop": None if self.actor_nop is None else self.actor_nop.get_hyper_parameters(),
            "critic_nop": None if self.critic_nop is None else self.critic_nop.get_hyper_parameters(),
        }

    def get_grad_norms(self) -> dict[str, float]:
        return {
            "actor": self._module_grad_norm(self.actor), "critic": self._module_grad_norm(self.critic),
            "actor_nop": self._module_grad_norm(self.actor_nop), "critic_nop": self._module_grad_norm(self.critic_nop),
        }

    def update_loss_weights(self, **weights: float) -> None:
        remaining = dict(weights)
        for aliases, modules in (
            (("nop_loss_coef", "world_model_loss_coef", "wm_loss_coef"), (self.actor_nop, self.critic_nop)),
            (("actor_nop_loss_coef", "actor_world_model_loss_coef"), (self.actor_nop,)),
            (("critic_nop_loss_coef", "critic_world_model_loss_coef"), (self.critic_nop,)),
        ):
            present = [alias for alias in aliases if alias in remaining]
            if len(present) > 1:
                raise ValueError(f"Use one loss weight alias: {present}")
            if present:
                alias = present[0]
                value = remaining.pop(alias)
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"{alias} must be finite and nonnegative")
                active = [module for module in modules if module is not None]
                if not active:
                    raise ValueError(f"{alias} requires an enabled NOP module")
                for module in active:
                    module.nop_loss_coef = value
        if remaining:
            raise ValueError(f"Unsupported DDPG/TD3 loss weights: {tuple(remaining)}")
