"""TD3 policies with independent online and target actor history states."""

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import torch

from swarmbots.learn.algos.off_policy.nop import SACNOPSequenceBatch
from swarmbots.learn.algos.off_policy.actor_state_critic import (
    ActorStateCriticInputConfig,
    ActorStateJointQNetwork,
    ActorStateTransformerTwinCritic,
    actor_state_input_dim,
    actor_state_critic_input,
)
from swarmbots.learn.algos.off_policy.recurrent_transformer_critic import RecurrentTransformerTwinCritic
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig
from swarmbots.learn.algos.td3.td3_policy import DeterministicActor, TD3Policy, TD3PolicyConfig


@dataclass(frozen=True)
class RecurrentTD3PolicyConfig(TD3PolicyConfig):
    actor_encoder_config: RMATEncoderConfig = field(default_factory=RMATEncoderConfig)
    recurrent_critic: bool = False
    actor_state_critic_input_config: ActorStateCriticInputConfig | Literal["auto"] | None = "auto"


class RecurrentDeterministicActor(DeterministicActor):
    def _build_encoder(self, env, config: RecurrentTD3PolicyConfig) -> RMATEncoder:
        return RMATEncoder(
            config.actor_encoder_config,
            max_agents=env.n_agents,
            local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim,
        )

    def encode(self, local_obs, global_obs, agent_mask=None, **kwargs):
        if agent_mask is not None:
            local_obs = local_obs.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
        return self.encoder(local_obs, global_obs, agent_mask=agent_mask, **kwargs)

    def actions_from_latents(self, latents, agent_mask=None):
        actions = self.action_net(self.head(latents, agent_mask=agent_mask)).tanh()
        actions = self.action_center + self.action_scale * actions
        return actions if agent_mask is None else actions.masked_fill(~agent_mask.unsqueeze(-1), 0.0)

    def forward(self, local_obs, global_obs, agent_mask=None):
        latents, _state = self.encode(local_obs, global_obs, agent_mask)
        return self.actions_from_latents(latents, agent_mask)


class RecurrentTD3Policy(TD3Policy):
    config: RecurrentTD3PolicyConfig

    def __init__(self, env, config: RecurrentTD3PolicyConfig = RecurrentTD3PolicyConfig()) -> None:
        if config.actor_state_critic_input_config == "auto":
            config = replace(
                config,
                actor_state_critic_input_config=None if config.recurrent_critic else ActorStateCriticInputConfig(),
            )
        state_config = config.actor_state_critic_input_config
        if state_config is not None:
            if not isinstance(state_config, ActorStateCriticInputConfig):
                raise TypeError("actor_state_critic_input_config must be 'auto', None or ActorStateCriticInputConfig")
            if config.recurrent_critic:
                raise ValueError("Actor-state critic input is not supported with recurrent_critic=True")
            if state_config.projection_dim is not None and state_config.projection_dim < 1:
                raise ValueError("actor-state critic projection_dim must be >= 1")
            if state_config.projection_hidden_dims is not None and any(
                dim < 1 for dim in state_config.projection_hidden_dims
            ):
                raise ValueError("actor-state critic projection_hidden_dims must all be >= 1")
        if not isinstance(config.actor_encoder_config, RMATEncoderConfig):
            raise TypeError("Recurrent TD3 requires RMATEncoderConfig for the actor")
        if config.recurrent_critic and (
            config.critic_kind != "transformer" or not isinstance(config.critic_encoder_config, RMATEncoderConfig)
        ):
            raise ValueError("A recurrent TD3 critic requires a transformer critic with RMATEncoderConfig")
        if config.n_critics != 2:
            raise ValueError("Recurrent TD3 requires two critics")
        super().__init__(env, config)
        object.__setattr__(self, "_actor_encoder_sequence_forward", None)
        object.__setattr__(self, "_target_actor_encoder_sequence_forward", None)
        object.__setattr__(self, "_critic_sequence_forward", None)
        object.__setattr__(self, "_target_critic_sequence_forward", None)
        if config.compile_modules:
            for name, function in (
                ("_actor_encoder_sequence_forward", self.actor.encode),
                ("_target_actor_encoder_sequence_forward", self.actor_target.encode),
            ):
                object.__setattr__(self, name, torch.compile(function, mode=config.compile_mode, dynamic=False))
            for name, critic in (
                ("_critic_sequence_forward", self.critic),
                ("_target_critic_sequence_forward", self.critic_target),
            ):
                forward = (
                    torch.compile(critic, mode=config.compile_mode, dynamic=False)
                    if config.critic_kind == "transformer"
                    else tuple(
                        torch.compile(network.forward_with_latents, mode=config.compile_mode, dynamic=False)
                        for network in critic
                    )
                )
                object.__setattr__(self, name, forward)

    def _build_actor(self, env, config):
        return RecurrentDeterministicActor(env, config)

    def _build_transformer_critic(self, **kwargs):
        if self.uses_actor_state_critic_input:
            return ActorStateTransformerTwinCritic(**kwargs, **self._actor_state_critic_kwargs())
        if not self.config.recurrent_critic:
            return super()._build_transformer_critic(**kwargs)
        kwargs.pop("act_fn_cls")
        return RecurrentTransformerTwinCritic(**kwargs)

    def _build_joint_critic(self, **kwargs):
        if self.uses_actor_state_critic_input:
            return ActorStateJointQNetwork(**kwargs, **self._actor_state_critic_kwargs())
        return super()._build_joint_critic(**kwargs)

    def _actor_state_critic_kwargs(self):
        config = self.config.actor_state_critic_input_config
        return dict(
            actor_state_input_dim=actor_state_input_dim(
                self.actor.encoder.layers[-1].temporal_model,
                self.config.actor_encoder_config.d_model,
                config,
            ),
            actor_state_config=config,
            actor_state_default_projection_dim=self.config.actor_encoder_config.d_model,
        )

    @property
    def uses_actor_state_critic_input(self) -> bool:
        return self.config.actor_state_critic_input_config is not None

    def actor_last_layer_state_critic_input(self, state, *, target=False):
        if not self.uses_actor_state_critic_input:
            raise RuntimeError("Actor-state critic input is not configured")
        actor = self.actor_target if target else self.actor
        return actor_state_critic_input(
            actor.encoder.layers[-1].temporal_model,
            state,
            self.config.actor_state_critic_input_config,
        )

    def actor_state_critic_input(self, state, *, target=False):
        return self.actor_last_layer_state_critic_input(state[-1], target=target)

    def _stateless_actor_state_critic_input(self, observations, *, target=False):
        with torch.no_grad():
            _latents, state = self.encode_actor_sequence(
                local_obs=observations["local_obs"],
                global_obs=observations["global_obs"],
                agent_mask=observations.get("agent_mask"),
                target=target,
            )
        return self.actor_state_critic_input(state, target=target)

    @property
    def recurrent_critic(self) -> bool:
        return self.config.recurrent_critic

    def requires_recurrent_training(self) -> bool:
        return True

    def initial_temporal_state(self, batch_size, n_agents, *, device, dtype):
        return {
            name: actor.encoder.initial_state(batch_size, n_agents, device=device, dtype=dtype)
            for name, actor in (("actor", self.actor), ("target_actor", self.actor_target))
        }

    def _evaluate_critic(self, critic, **observations):
        if self.recurrent_critic:
            q1, q2, _latents, _state = self.q_values_sequence(
                **observations,
                target=critic is self.critic_target,
            )
            return q1, q2
        if self.uses_actor_state_critic_input:
            if observations.get("actor_state") is None:
                observations["actor_state"] = self._stateless_actor_state_critic_input(
                    observations,
                    target=critic is self.critic_target,
                )
            mask = observations.get("agent_mask")
            if mask is not None:
                observations["actor_state"] = observations["actor_state"].masked_fill(~mask.unsqueeze(-1), 0.0)
        return super()._evaluate_critic(critic, **observations)

    def _q_values_with_nop_latents(self, **observations):
        if self.recurrent_critic or self.uses_actor_state_critic_input:
            if self.uses_actor_state_critic_input and observations.get("actor_state") is None:
                observations["actor_state"] = self._stateless_actor_state_critic_input(observations)
            q1, q2, latents, _state = self.q_values_sequence(**observations)
            return (q1, q2), latents
        return super()._q_values_with_nop_latents(**observations)

    def encode_actor_sequence(
        self,
        *,
        local_obs,
        global_obs,
        agent_mask=None,
        target=False,
        initial_state=None,
        time_mask=None,
        reset_mask=None,
        scenario_ids=None,
        state_output_indices=None,
        return_last_layer_state_sequence=False,
    ):
        if scenario_ids is not None:
            raise ValueError("TD3 currently requires a single scenario observation layout")
        actor = self.actor_target if target else self.actor
        forward = self._target_actor_encoder_sequence_forward if target else self._actor_encoder_sequence_forward
        if forward is None:
            forward = actor.encode
        return forward(
            local_obs,
            global_obs,
            agent_mask,
            initial_state=initial_state,
            time_mask=time_mask,
            reset_mask=reset_mask,
            state_output_indices=state_output_indices,
            return_last_layer_state_sequence=return_last_layer_state_sequence,
        )

    def actor_actions_sequence(self, *, target=False, **kwargs):
        result = self.encode_actor_sequence(target=target, **kwargs)
        latents, state, *selected_states = result
        actor = self.actor_target if target else self.actor
        actions = actor.actions_from_latents(latents, kwargs.get("agent_mask"))
        return actions, latents, state, *selected_states

    def act_with_temporal_state(
        self,
        local_obs,
        global_obs,
        hidden_local_vars=None,
        hidden_global_vars=None,
        agent_mask=None,
        scenario_ids=None,
        previous_actions=None,
        deterministic=False,
        *,
        temporal_state=None,
        episode_start_mask=None,
    ):
        if temporal_state is None:
            temporal_state = self.initial_temporal_state(
                local_obs.shape[0],
                local_obs.shape[1],
                device=local_obs.device,
                dtype=local_obs.dtype,
            )
        inputs = dict(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            reset_mask=episode_start_mask,
        )
        actions, _latents, actor_state = self.actor_actions_sequence(
            **inputs,
            initial_state=temporal_state["actor"],
        )
        with torch.no_grad():
            _target_latents, target_state = self.encode_actor_sequence(
                **inputs,
                target=True,
                initial_state=temporal_state["target_actor"],
            )
        if not deterministic and self.exploration_noise:
            actions = self.add_action_noise(actions, std=self.exploration_noise, agent_mask=agent_mask)
        return actions, {"actor": actor_state, "target_actor": target_state}

    def initial_critic_state(self, batch_size, n_agents, *, device, dtype, target=False):
        if not self.recurrent_critic:
            return None
        critic = self.critic_target if target else self.critic
        return critic.initial_state(batch_size, n_agents, device=device, dtype=dtype)

    def q_values_sequence(
        self,
        *,
        local_obs,
        global_obs,
        actions,
        hidden_local_vars=None,
        hidden_global_vars=None,
        agent_mask=None,
        target=False,
        initial_state=None,
        time_mask=None,
        reset_mask=None,
        scenario_ids=None,
        num_action_samples=1,
        actor_state=None,
    ):
        if scenario_ids is not None or num_action_samples != 1:
            raise ValueError("Recurrent TD3 uses a single scenario and one action candidate")
        observations = dict(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )
        if self.uses_actor_state_critic_input:
            if actor_state is None:
                raise ValueError("Actor-state critic input requires explicit actor_state for sequence evaluation")
            observations["actor_state"] = actor_state.detach()
        if agent_mask is not None:
            for name in ("local_obs", "actions", "hidden_local_vars", "actor_state"):
                if observations.get(name) is not None:
                    observations[name] = observations[name].masked_fill(~agent_mask.unsqueeze(-1), 0.0)
        critic = self.critic_target if target else self.critic
        if self.recurrent_critic:
            forward = self._target_critic_sequence_forward if target else self._critic_sequence_forward
            if forward is None:
                forward = critic
            return forward(**observations, initial_state=initial_state, time_mask=time_mask, reset_mask=reset_mask)
        sequence = local_obs.ndim == 4
        if sequence:
            batch_size, length = local_obs.shape[:2]
            observations = {
                name: None if value is None else value.reshape(batch_size * length, *value.shape[2:])
                for name, value in observations.items()
            }
        if self.config.critic_kind == "transformer":
            forward = self._target_critic_sequence_forward if target else self._critic_sequence_forward
            q1, q2, latents = (critic if forward is None else forward)(**observations)
        else:
            forward = self._target_critic_sequence_forward if target else self._critic_sequence_forward
            results = (
                [network.forward_with_latents(**observations) for network in critic]
                if forward is None
                else [function(**observations) for function in forward]
            )
            (q1, q2) = tuple(value for value, _latents in results)
            latents = None if results[0][1] is None else torch.cat([latent for _value, latent in results], dim=-1)
        if sequence:
            q1, q2 = q1.reshape(batch_size, length), q2.reshape(batch_size, length)
            if latents is not None:
                latents = latents.reshape(batch_size, length, *latents.shape[1:])
        if agent_mask is not None and latents is not None:
            latents = latents.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
        return q1, q2, latents, None

    def compute_actor_nop_loss_from_latents(self, *, source_latents, batch: SACNOPSequenceBatch):
        if self.actor_nop is None:
            return None, {}
        return self.actor_nop.compute_loss(source_latents=source_latents, batch=batch)

    def compute_critic_nop_loss_from_latents(self, *, source_latents, batch: SACNOPSequenceBatch):
        if self.critic_nop is None or source_latents is None:
            return None, {}
        return self.critic_nop.compute_loss(source_latents=source_latents, batch=batch)

    def get_hyper_parameters(self) -> dict[str, Any]:
        settings = super().get_hyper_parameters()
        settings["recurrent_td3_policy_config"] = settings.pop("td3_policy_config")
        return settings
