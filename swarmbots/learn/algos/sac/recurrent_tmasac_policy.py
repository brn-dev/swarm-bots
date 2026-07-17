from dataclasses import dataclass, field, replace
from typing import Any, cast

import torch
from torch import nn

from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig, RMATEncoderState
from swarmbots.learn.algos.sac.sac_nop import SACNOPSequenceBatch
from swarmbots.learn.algos.sac.tmasac_policy import (
    TMASACCriticConfig,
    TMASACPolicy,
    TMASACPolicyConfig,
    TMASACTwinCritic,
)
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper


RecurrentCriticState = RMATEncoderState | tuple[RMATEncoderState, RMATEncoderState]


@dataclass(frozen=True)
class RecurrentTMASACPolicyConfig(TMASACPolicyConfig):
    actor_encoder_config: RMATEncoderConfig = field(default_factory=RMATEncoderConfig)
    recurrent_critic: bool = False


class RecurrentTMASACTwinCritic(TMASACTwinCritic):
    def __init__(
            self,
            *,
            n_agents: int,
            max_agents: int,
            local_input_dim: int,
            global_input_dim: int,
            hidden_local_vars_dim: int,
            hidden_global_vars_dim: int,
            action_dim: int,
            encoder_config: RMATEncoderConfig,
            critic_config: TMASACCriticConfig,
            dropout: float,
    ) -> None:
        nn.Module.__init__(self)
        self.n_agents = int(n_agents)
        self.hidden_local_vars_dim = int(hidden_local_vars_dim)
        self.hidden_global_vars_dim = int(hidden_global_vars_dim)
        self.action_dim = int(action_dim)
        self.d_model = int(encoder_config.d_model)
        self.encoder_config = replace(
            encoder_config,
            dropout=dropout,
            local_obs_encoder_hidden_dims=(
                [self.d_model]
                if critic_config.action_coembed_hidden_dims is None
                else [*critic_config.action_coembed_hidden_dims]
            ),
            linear_init_gain=critic_config.action_coembed_init_gain,
            linear_projection_init_gain=critic_config.action_coembed_output_init_gain,
        )
        local_encoder_input_dim = local_input_dim + self.hidden_local_vars_dim + self.action_dim
        global_encoder_input_dim = global_input_dim + self.hidden_global_vars_dim

        def build_encoder() -> RMATEncoder:
            return RMATEncoder(
                config=self.encoder_config,
                max_agents=max_agents,
                local_obs_dim=local_encoder_input_dim,
                global_obs_dim=global_encoder_input_dim,
            )

        self.encoder = build_encoder()
        self.encoder2 = build_encoder() if critic_config.independent_encoders else None
        self.nop_source_latent_dim = self.d_model * (2 if self.encoder2 is not None else 1)
        self.q1 = self._build_q_network(
            q_local_dim=self.d_model,
            critic_config=critic_config,
            act_fn_cls=encoder_config.act_fn_cls,
        )
        self.q2 = self._build_q_network(
            q_local_dim=self.d_model,
            critic_config=critic_config,
            act_fn_cls=encoder_config.act_fn_cls,
        )

    def initial_state(
            self,
            batch_size: int,
            n_agents: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
    ) -> RecurrentCriticState:
        primary_state = self.encoder.initial_state(
            batch_size=batch_size,
            n_agents=n_agents,
            device=device,
            dtype=dtype,
        )
        if self.encoder2 is None:
            return primary_state
        return primary_state, self.encoder2.initial_state(
            batch_size=batch_size,
            n_agents=n_agents,
            device=device,
            dtype=dtype,
        )

    def encode(
            self,
            *,
            local_inputs: torch.Tensor,
            global_inputs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            time_mask: torch.Tensor | None = None,
            initial_state: RecurrentCriticState | None = None,
            reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RecurrentCriticState]:
        primary_state, secondary_state = self._split_initial_state(initial_state)
        local_encoder_inputs, global_encoder_inputs = self._encoder_inputs(
            local_inputs=local_inputs,
            global_inputs=global_inputs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
        )
        primary_latents, next_primary_state = self.encoder(
            local_encoder_inputs,
            global_encoder_inputs,
            agent_mask=agent_mask,
            time_mask=time_mask,
            initial_state=primary_state,
            reset_mask=reset_mask,
        )
        if self.encoder2 is None:
            return primary_latents, next_primary_state
        secondary_latents, next_secondary_state = self.encoder2(
            local_encoder_inputs,
            global_encoder_inputs,
            agent_mask=agent_mask,
            time_mask=time_mask,
            initial_state=secondary_state,
            reset_mask=reset_mask,
        )
        return torch.cat((primary_latents, secondary_latents), dim=-1), (
            next_primary_state,
            next_secondary_state,
        )

    def forward(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            time_mask: torch.Tensor | None = None,
            initial_state: RecurrentCriticState | None = None,
            reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RecurrentCriticState]:
        primary_state, secondary_state = self._split_initial_state(initial_state)
        local_encoder_inputs, global_encoder_inputs = self._encoder_inputs(
            local_inputs=local_obs,
            global_inputs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
        )
        q1_latents, next_primary_state = self.encoder(
            local_encoder_inputs,
            global_encoder_inputs,
            agent_mask=agent_mask,
            time_mask=time_mask,
            initial_state=primary_state,
            reset_mask=reset_mask,
        )
        if self.encoder2 is None:
            q2_latents = q1_latents
            next_state: RecurrentCriticState = next_primary_state
        else:
            q2_latents, next_secondary_state = self.encoder2(
                local_encoder_inputs,
                global_encoder_inputs,
                agent_mask=agent_mask,
                time_mask=time_mask,
                initial_state=secondary_state,
                reset_mask=reset_mask,
            )
            next_state = next_primary_state, next_secondary_state
        return (
            self._evaluate_q_head(self.q1, q1_latents, agent_mask),
            self._evaluate_q_head(self.q2, q2_latents, agent_mask),
            q1_latents if self.encoder2 is None else torch.cat((q1_latents, q2_latents), dim=-1),
            next_state,
        )

    def _encoder_inputs(
            self,
            *,
            local_inputs: torch.Tensor,
            global_inputs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_parts = [local_inputs]
        if self.hidden_local_vars_dim > 0:
            if hidden_local_vars is None:
                raise ValueError("hidden_local_vars must be provided when hidden_local_vars_dim > 0")
            local_parts.append(hidden_local_vars)
        local_parts.append(actions)

        global_parts = [global_inputs] if global_inputs.shape[-1] > 0 else []
        if self.hidden_global_vars_dim > 0:
            if hidden_global_vars is None:
                raise ValueError("hidden_global_vars must be provided when hidden_global_vars_dim > 0")
            global_parts.append(hidden_global_vars)
        if global_parts:
            global_encoder_inputs = global_parts[0] if len(global_parts) == 1 else torch.cat(global_parts, dim=-1)
        else:
            global_encoder_inputs = global_inputs
        return torch.cat(local_parts, dim=-1), global_encoder_inputs

    def _split_initial_state(
            self,
        state: RecurrentCriticState | None,
    ) -> tuple[RMATEncoderState | None, RMATEncoderState | None]:
        if self.encoder2 is None:
            return cast(RMATEncoderState | None, state), None
        if state is None:
            return None, None
        if not isinstance(state, tuple) or len(state) != 2:
            raise ValueError("Independent recurrent critic encoders require a pair of temporal states.")
        return state

    @staticmethod
    def _evaluate_q_head(
            q_head: nn.Module,
            latents: torch.Tensor,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if latents.ndim == 3:
            return q_head(latents, agent_mask=agent_mask)
        batch_size, sequence_length, n_agents, latent_dim = latents.shape
        flat_mask = None if agent_mask is None else agent_mask.reshape(batch_size * sequence_length, n_agents)
        values = q_head(
            latents.reshape(batch_size * sequence_length, n_agents, latent_dim),
            agent_mask=flat_mask,
        )
        return values.reshape(batch_size, sequence_length)


class RecurrentTMASACPolicy(TMASACPolicy):
    config: RecurrentTMASACPolicyConfig

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: RecurrentTMASACPolicyConfig = RecurrentTMASACPolicyConfig(),
    ) -> None:
        if not isinstance(config.actor_encoder_config, RMATEncoderConfig):
            raise TypeError("RecurrentTMASACPolicy requires an RMATEncoderConfig for actor_encoder_config.")
        if config.shared_encoder_config is not None or config.share_observation_encoder:
            raise ValueError("RecurrentTMASACPolicy does not support a separate shared observation encoder.")
        if config.compile_modules:
            raise ValueError("RecurrentTMASACPolicy does not currently support compile_modules=True.")
        if config.recurrent_critic and not isinstance(config.critic_encoder_config, RMATEncoderConfig):
            raise TypeError("recurrent_critic=True requires an RMATEncoderConfig for critic_encoder_config.")
        super().__init__(env=env, config=config)

    @property
    def recurrent_critic(self) -> bool:
        return self.config.recurrent_critic

    def requires_recurrent_training(self) -> bool:
        return True

    def initial_temporal_state(
            self,
            batch_size: int,
            n_agents: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
    ) -> RMATEncoderState:
        return self._actor_encoder.initial_state(
            batch_size=batch_size,
            n_agents=n_agents,
            device=device,
            dtype=dtype,
        )

    def act_with_temporal_state(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
            *,
            temporal_state: RMATEncoderState | None = None,
            episode_start_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RMATEncoderState]:
        _ = hidden_local_vars
        _ = hidden_global_vars
        actions, _log_probs, _latents, next_state = self.action_log_prob_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            use_rsample=False,
            initial_state=temporal_state,
            reset_mask=episode_start_mask,
        )
        return actions, next_state

    def encode_actor(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        latents, _state = self._actor_encoder(local_obs, global_obs, agent_mask=agent_mask)
        return latents

    def action_log_prob_sequence(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            use_rsample: bool,
            initial_state: RMATEncoderState | None,
            time_mask: torch.Tensor | None = None,
            reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RMATEncoderState]:
        actor_latents, next_state = self.encode_actor_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            time_mask=time_mask,
            initial_state=initial_state,
            reset_mask=reset_mask,
        )
        latent_pi = self.actor_head(actor_latents, agent_mask=agent_mask)
        actions, log_probs = self.action_dist.get_actions_with_log_probs(
            latent_pi,
            deterministic=deterministic,
            previous_actions=previous_actions,
            use_rsample=use_rsample,
        )
        return (
            self._mask_actions(actions, agent_mask),
            self._mask_log_probs(log_probs, agent_mask),
            actor_latents,
            next_state,
        )

    def encode_actor_sequence(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            initial_state: RMATEncoderState | None,
            time_mask: torch.Tensor | None = None,
            reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RMATEncoderState]:
        return self._actor_encoder(
            local_obs,
            global_obs,
            agent_mask=agent_mask,
            time_mask=time_mask,
            initial_state=initial_state,
            reset_mask=reset_mask,
        )

    def q_values_sequence(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            target: bool,
            initial_state: RecurrentCriticState | None = None,
            time_mask: torch.Tensor | None = None,
            reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, RecurrentCriticState | None]:
        if self.recurrent_critic:
            critic = self.critic_target if target else self.critic
            assert isinstance(critic, RecurrentTMASACTwinCritic)
            q1, q2, latents, next_state = critic(
                local_obs=local_obs,
                global_obs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                time_mask=time_mask,
                initial_state=initial_state,
                reset_mask=reset_mask,
            )
            return q1, q2, None if target or self.critic_nop is None else latents, next_state

        flat_inputs, batch_size, sequence_length = self._flatten_sequence_inputs(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )
        if target:
            q1, q2 = self.target_q_values(**flat_inputs)
            latents = None
        else:
            q1, q2, latents = self.q_values_with_nop_latents(**flat_inputs)
        if local_obs.ndim == 3:
            return q1, q2, latents, None
        q1 = q1.reshape(batch_size, sequence_length)
        q2 = q2.reshape(batch_size, sequence_length)
        if latents is not None:
            latents = latents.reshape(batch_size, sequence_length, *latents.shape[1:])
        return q1, q2, latents, None

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
        if not self.recurrent_critic:
            return super().q_values(
                local_obs=local_obs,
                global_obs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
            )
        q1, q2, _latents, _state = self.q_values_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            target=False,
        )
        return q1, q2

    def q_values_with_nop_latents(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not self.recurrent_critic:
            return super().q_values_with_nop_latents(
                local_obs=local_obs,
                global_obs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
            )
        q1, q2, latents, _state = self.q_values_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            target=False,
        )
        return q1, q2, latents

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
        if not self.recurrent_critic:
            return super().target_q_values(
                local_obs=local_obs,
                global_obs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
            )
        q1, q2, _latents, _state = self.q_values_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            target=True,
        )
        return q1, q2

    def encode_critic(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.recurrent_critic:
            return super().encode_critic(
                local_obs=local_obs,
                global_obs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
            )
        critic = self._critic_module()
        assert isinstance(critic, RecurrentTMASACTwinCritic)
        latents, _state = critic.encode(
            local_inputs=local_obs,
            global_inputs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )
        return latents

    def initial_critic_state(
            self,
            batch_size: int,
            n_agents: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
            target: bool,
    ) -> RecurrentCriticState | None:
        if not self.recurrent_critic:
            return None
        critic = self.critic_target if target else self.critic
        assert isinstance(critic, RecurrentTMASACTwinCritic)
        return critic.initial_state(
            batch_size=batch_size,
            n_agents=n_agents,
            device=device,
            dtype=dtype,
        )

    def compute_actor_nop_loss_from_latents(
            self,
            *,
            source_latents: torch.Tensor,
            batch: SACNOPSequenceBatch,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if self.actor_nop is None:
            return None, {}
        return self.actor_nop.compute_loss(source_latents=source_latents, batch=batch)

    def compute_critic_nop_loss_from_latents(
            self,
            *,
            source_latents: torch.Tensor | None,
            batch: SACNOPSequenceBatch,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if self.critic_nop is None:
            return None, {}
        if source_latents is None:
            raise ValueError("Critic NOP requires critic source latents.")
        return self.critic_nop.compute_loss(source_latents=source_latents, batch=batch)

    def get_hyper_parameters(self) -> dict[str, Any]:
        hyper_parameters = super().get_hyper_parameters()
        hyper_parameters["tmasac_policy_config"]["recurrent_critic"] = self.recurrent_critic
        return hyper_parameters

    def _build_actor_encoder(
            self,
            *,
            local_obs_dim: int,
            global_obs_dim: int,
    ) -> RMATEncoder:
        return RMATEncoder(
            config=self.actor_encoder_config,
            max_agents=self.max_agents,
            local_obs_dim=local_obs_dim,
            global_obs_dim=global_obs_dim,
        )

    def _build_critic(
            self,
            *,
            local_input_dim: int,
            global_input_dim: int,
    ) -> TMASACTwinCritic:
        if not self.config.recurrent_critic:
            return super()._build_critic(
                local_input_dim=local_input_dim,
                global_input_dim=global_input_dim,
            )
        encoder_config = cast(RMATEncoderConfig, self.critic_encoder_config)
        return RecurrentTMASACTwinCritic(
            n_agents=self.n_agents,
            max_agents=self.max_agents,
            local_input_dim=local_input_dim,
            global_input_dim=global_input_dim,
            hidden_local_vars_dim=self.hidden_local_vars_dim,
            hidden_global_vars_dim=self.hidden_global_vars_dim,
            action_dim=self.agent_action_dim,
            encoder_config=encoder_config,
            critic_config=self.config.critic_config,
            dropout=self.dropout,
        )

    def _critic_module(self) -> TMASACTwinCritic:
        if isinstance(self.critic, TMASACTwinCritic):
            return self.critic
        raise RuntimeError(f"Cannot resolve TMASAC critic from {type(self.critic).__name__}")

    @staticmethod
    def _flatten_sequence_inputs(
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor | None], int, int]:
        if local_obs.ndim == 3:
            return {
                "local_obs": local_obs,
                "global_obs": global_obs,
                "actions": actions,
                "hidden_local_vars": hidden_local_vars,
                "hidden_global_vars": hidden_global_vars,
                "agent_mask": agent_mask,
            }, local_obs.shape[0], 1
        batch_size, sequence_length = local_obs.shape[:2]

        def flatten(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return tensor.reshape(batch_size * sequence_length, *tensor.shape[2:])

        return {
            "local_obs": flatten(local_obs),
            "global_obs": flatten(global_obs),
            "actions": flatten(actions),
            "hidden_local_vars": flatten(hidden_local_vars),
            "hidden_global_vars": flatten(hidden_global_vars),
            "agent_mask": flatten(agent_mask),
        }, batch_size, sequence_length
