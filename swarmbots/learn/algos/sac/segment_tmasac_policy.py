from collections.abc import Collection
from typing import Any, cast

import torch

from swarmbots.learn.algos.sac.sac_nop import SACNOPSequenceBatch
from swarmbots.learn.algos.sac.tmasac_policy import TMASACPolicy


class SegmentTMASACPolicy(TMASACPolicy):
    """Feed-forward TMASAC adapted to RecurrentSAC's segment-training interface.

    The scalar temporal state is deliberately meaningless. It only makes recurrent replay
    record the same checkpoint-aligned candidate starts used by genuinely recurrent actors.
    """

    @property
    def recurrent_critic(self) -> bool:
        return False

    @property
    def uses_temporal_actor_state(self) -> bool:
        return False

    @property
    def compiled_actor_sequence_lengths(self) -> frozenset[int]:
        return frozenset()

    @property
    def compiled_actor_selected_state_sequence_lengths(self) -> frozenset[int]:
        return frozenset()

    @property
    def compiled_actor_encoder_sequence_lengths(self) -> frozenset[int]:
        return frozenset()

    def configure_actor_compilation(
            self,
            *,
            encoder_only_sequence_lengths: Collection[int],
            action_sequence_lengths: Collection[int],
            action_sequence_with_selected_states_lengths: Collection[int],
    ) -> None:
        _ = (
            encoder_only_sequence_lengths,
            action_sequence_lengths,
            action_sequence_with_selected_states_lengths,
        )

    def requires_recurrent_training(self) -> bool:
        return True

    def initial_temporal_state(
            self,
            batch_size: int,
            n_agents: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
    ) -> torch.Tensor:
        _ = n_agents
        return torch.zeros((batch_size, 1), device=device, dtype=dtype)

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
            temporal_state: torch.Tensor | None = None,
            episode_start_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = temporal_state, episode_start_mask
        actions = self.act(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
        )
        return actions, self._dummy_state(local_obs)

    def encode_actor_sequence(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            initial_state: torch.Tensor | None,
            time_mask: torch.Tensor | None = None,
            reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = initial_state, time_mask, reset_mask
        flat_inputs, batch_size, sequence_length = self._flatten_sequence_inputs(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
        )
        actor_latents = self.encode_actor(**flat_inputs)
        return (
            self._restore_sequence(actor_latents, batch_size, sequence_length),
            self._dummy_state(local_obs),
        )

    def action_log_prob_sequence(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            use_rsample: bool,
            initial_state: torch.Tensor | None,
            time_mask: torch.Tensor | None = None,
            reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _ = initial_state, time_mask, reset_mask
        flat_inputs, batch_size, sequence_length = self._flatten_sequence_inputs(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
        )
        flat_agent_mask = flat_inputs["agent_mask"]
        flat_previous_actions = flat_inputs["previous_actions"]
        actor_latents = self.encode_actor(
            local_obs=cast(torch.Tensor, flat_inputs["local_obs"]),
            global_obs=cast(torch.Tensor, flat_inputs["global_obs"]),
            agent_mask=flat_agent_mask,
        )
        latent_pi = self.actor_head(actor_latents, agent_mask=flat_agent_mask)
        actions, log_probs = self.action_dist.get_actions_with_log_probs(
            latent_pi,
            deterministic=deterministic,
            previous_actions=flat_previous_actions,
            use_rsample=use_rsample,
        )
        actions = self._mask_actions(actions, flat_agent_mask)
        log_probs = self._mask_log_probs(log_probs, flat_agent_mask)
        return (
            self._restore_sequence(actions, batch_size, sequence_length),
            self._restore_sequence(log_probs, batch_size, sequence_length),
            self._restore_sequence(actor_latents, batch_size, sequence_length),
            self._dummy_state(local_obs),
        )

    def action_log_prob_sequence_with_selected_states(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            use_rsample: bool,
            initial_state: torch.Tensor | None,
            state_output_indices: torch.Tensor,
            time_mask: torch.Tensor | None = None,
            reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, log_probs, actor_latents, next_state = self.action_log_prob_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            use_rsample=use_rsample,
            initial_state=initial_state,
            time_mask=time_mask,
            reset_mask=reset_mask,
        )
        selected_states = torch.zeros(
            (state_output_indices.shape[0], 1),
            device=local_obs.device,
            dtype=local_obs.dtype,
        )
        return actions, log_probs, actor_latents, next_state, selected_states

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
            initial_state: Any = None,
            time_mask: torch.Tensor | None = None,
            reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None]:
        _ = initial_state, time_mask, reset_mask
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
        return (
            self._restore_sequence(q1, batch_size, sequence_length),
            self._restore_sequence(q2, batch_size, sequence_length),
            None if latents is None else self._restore_sequence(latents, batch_size, sequence_length),
            None,
        )

    def initial_critic_state(
            self,
            batch_size: int,
            n_agents: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
            target: bool,
    ) -> None:
        _ = batch_size, n_agents, device, dtype, target
        return None

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
        hyper_parameters["tmasac_policy_config"]["segment_sampling_adapter"] = True
        return hyper_parameters

    @staticmethod
    def _dummy_state(reference: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (reference.shape[0], 1),
            device=reference.device,
            dtype=reference.dtype,
        )

    @staticmethod
    def _restore_sequence(
            tensor: torch.Tensor,
            batch_size: int,
            sequence_length: int,
    ) -> torch.Tensor:
        if sequence_length == 1:
            return tensor
        return tensor.reshape(batch_size, sequence_length, *tensor.shape[1:])

    @staticmethod
    def _flatten_sequence_inputs(
            **inputs: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor | None], int, int]:
        local_obs = inputs["local_obs"]
        local_obs = cast(torch.Tensor, local_obs)
        if local_obs.ndim == 3:
            return inputs, local_obs.shape[0], 1
        batch_size, sequence_length = local_obs.shape[:2]
        return {
            name: (
                None
                if tensor is None
                else tensor.reshape(batch_size * sequence_length, *tensor.shape[2:])
            )
            for name, tensor in inputs.items()
        }, batch_size, sequence_length
