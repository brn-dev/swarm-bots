"""Recurrent action-conditioned transformer team critics."""

from dataclasses import replace
from typing import Any, cast
import torch
from torch import nn
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig, RMATEncoderState
from swarmbots.learn.algos.off_policy.transformer_critic import (
    TransformerCriticConfig,
    TransformerTwinCritic,
    ObservationActionEncoder,
)
from swarmbots.learn.nn_components.feed_forward import MLPConfig

RecurrentCriticState = RMATEncoderState | tuple[RMATEncoderState, RMATEncoderState]


class RecurrentTransformerTwinCritic(TransformerTwinCritic):
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
        critic_config: TransformerCriticConfig,
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
            local_obs_encoder_config=MLPConfig(
                hidden_dims=(
                    [self.d_model]
                    if critic_config.action_coembed_hidden_dims is None
                    else [*critic_config.action_coembed_hidden_dims]
                )
            ),
            linear_init_gain=critic_config.action_coembed_init_gain,
            linear_projection_init_gain=critic_config.action_coembed_output_init_gain,
        )
        local_observation_input_dim = local_input_dim + self.hidden_local_vars_dim
        global_encoder_input_dim = global_input_dim + self.hidden_global_vars_dim

        def build_observation_action_encoder() -> ObservationActionEncoder:
            return ObservationActionEncoder(
                observation_input_dim=local_observation_input_dim,
                action_dim=self.action_dim,
                d_model=self.d_model,
                action_encoder_dim=critic_config.action_encoder_dim,
                linear_init_gain=critic_config.action_coembed_init_gain,
                act_fn_cls=encoder_config.act_fn_cls,
            )

        self.observation_action_encoder = (
            build_observation_action_encoder() if critic_config.separate_observation_action_encoders else None
        )
        local_encoder_input_dim = (
            local_observation_input_dim + self.action_dim
            if self.observation_action_encoder is None
            else self.observation_action_encoder.output_dim
        )

        def build_encoder() -> RMATEncoder:
            return RMATEncoder(
                config=self.encoder_config,
                max_agents=max_agents,
                local_obs_dim=local_encoder_input_dim,
                global_obs_dim=global_encoder_input_dim,
            )

        self.encoder = build_encoder()
        self.encoder2 = build_encoder() if critic_config.independent_encoders else None
        self.observation_action_encoder2 = (
            build_observation_action_encoder()
            if self.observation_action_encoder is not None and self.encoder2 is not None
            else None
        )
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
            observation_action_encoder=self.observation_action_encoder,
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
        secondary_local_encoder_inputs = local_encoder_inputs
        if self.observation_action_encoder2 is not None:
            secondary_local_encoder_inputs, _secondary_global_encoder_inputs = self._encoder_inputs(
                local_inputs=local_inputs,
                global_inputs=global_inputs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                observation_action_encoder=self.observation_action_encoder2,
            )
        secondary_latents, next_secondary_state = self.encoder2(
            secondary_local_encoder_inputs,
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
            observation_action_encoder=self.observation_action_encoder,
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
            secondary_local_encoder_inputs = local_encoder_inputs
            if self.observation_action_encoder2 is not None:
                secondary_local_encoder_inputs, _secondary_global_encoder_inputs = self._encoder_inputs(
                    local_inputs=local_obs,
                    global_inputs=global_obs,
                    actions=actions,
                    hidden_local_vars=hidden_local_vars,
                    hidden_global_vars=hidden_global_vars,
                    observation_action_encoder=self.observation_action_encoder2,
                )
            q2_latents, next_secondary_state = self.encoder2(
                secondary_local_encoder_inputs,
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
        observation_action_encoder: ObservationActionEncoder | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_observation_parts = [local_inputs]
        if self.hidden_local_vars_dim > 0:
            if hidden_local_vars is None:
                raise ValueError("hidden_local_vars must be provided when hidden_local_vars_dim > 0")
            local_observation_parts.append(hidden_local_vars)
        local_observation_inputs = (
            local_observation_parts[0]
            if len(local_observation_parts) == 1
            else torch.cat(local_observation_parts, dim=-1)
        )
        local_encoder_inputs = (
            torch.cat((local_observation_inputs, actions), dim=-1)
            if observation_action_encoder is None
            else observation_action_encoder(local_observation_inputs, actions)
        )

        global_parts = [global_inputs] if global_inputs.shape[-1] > 0 else []
        if self.hidden_global_vars_dim > 0:
            if hidden_global_vars is None:
                raise ValueError("hidden_global_vars must be provided when hidden_global_vars_dim > 0")
            global_parts.append(hidden_global_vars)
        if global_parts:
            global_encoder_inputs = global_parts[0] if len(global_parts) == 1 else torch.cat(global_parts, dim=-1)
        else:
            global_encoder_inputs = global_inputs
        return local_encoder_inputs, global_encoder_inputs

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self.observation_action_encoder2 is not None:
            primary_prefix = f"{prefix}observation_action_encoder."
            secondary_prefix = f"{prefix}observation_action_encoder2."
            has_primary_encoder = any(key.startswith(primary_prefix) for key in state_dict)
            has_secondary_encoder = any(key.startswith(secondary_prefix) for key in state_dict)
            if has_primary_encoder and not has_secondary_encoder:
                for key, value in list(state_dict.items()):
                    if key.startswith(primary_prefix):
                        state_dict[f"{secondary_prefix}{key.removeprefix(primary_prefix)}"] = value
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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
