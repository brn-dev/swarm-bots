"""Sequence replay and history branching shared by recurrent off-policy learners."""

from dataclasses import dataclass, replace
from typing import Any
import torch
from loguru import logger
from swarmbots.learn.algos.off_policy.replay_buffer import (
    NoEpisodeSegmentCandidatesError,
    OffPolicyReplayBatch,
    OffPolicyReplayBuffer,
    OffPolicyReplayEpisodeSegmentBatch,
)
from swarmbots.learn.algos.off_policy.recurrent_transformer_critic import RecurrentCriticState
from swarmbots.learn.algos.off_policy.nop import SACNOPSequenceBatch
from swarmbots.learn.temporal_state import concatenate_temporal_states, detach_temporal_state


class RecurrentOffPolicyMixin:
    supports_recurrent_training = True

    def __init__(
        self,
        policy,
        env,
        *,
        burn_in_steps: int = 32,
        learning_steps: int = 64,
        temporal_state_store_interval: int = 32,
        temporal_state_storage_dtype: torch.dtype | None = None,
        max_truncations_per_segment: int = 1,
        **kwargs: Any,
    ) -> None:
        self.burn_in_steps = int(burn_in_steps)
        self.learning_steps = int(learning_steps)
        self.temporal_state_store_interval = int(temporal_state_store_interval)
        self.temporal_state_storage_dtype = temporal_state_storage_dtype
        self.max_truncations_per_segment = int(max_truncations_per_segment)
        super().__init__(policy=policy, env=env, **kwargs)

    @property
    def replay_fill_target(self) -> int:
        sequence_fill_target = (self.burn_in_steps + self.learning_steps) * self.env.action_space.n_envs
        return max(super().replay_fill_target, sequence_fill_target)

    @property
    def _critic_uses_temporal_state(self) -> bool:
        return self.policy.recurrent_critic or getattr(
            self.policy,
            "uses_recurrent_shared_encoder",
            False,
        )

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "burn_in_steps": self.burn_in_steps,
            "learning_steps": self.learning_steps,
            "compiled_actor_encoder_sequence_lengths": sorted(
                getattr(self.policy, "compiled_actor_encoder_sequence_lengths", ())
            ),
            "compiled_actor_sequence_lengths": sorted(getattr(self.policy, "compiled_actor_sequence_lengths", ())),
            "compiled_actor_selected_state_sequence_lengths": sorted(
                getattr(self.policy, "compiled_actor_selected_state_sequence_lengths", ())
            ),
            "max_truncations_per_segment": self.max_truncations_per_segment,
            "temporal_state_store_interval": self.temporal_state_store_interval,
            "temporal_state_storage_dtype": str(self.replay_buffer.temporal_state_storage_dtype),
        }

    def train(self, *, gradient_steps: int) -> dict[str, Any]:
        try:
            return super().train(gradient_steps=gradient_steps)
        except NoEpisodeSegmentCandidatesError:
            logger.warning(
                f"Skipping {type(self).__name__} updates because replay has no temporal-state-anchored "
                "stream segment of the requested length."
            )
            return {
                "updates": 0,
                "total_updates": self.n_total_updates,
                "replay_size": len(self.replay_buffer),
                "training_skipped": True,
            }

    def _build_replay_buffer(self) -> OffPolicyReplayBuffer:
        return OffPolicyReplayBuffer(
            capacity_per_env=self.buffer_capacity_per_env,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            store_previous_actions=self.policy.requires_previous_actions(),
            temporal_state_store_interval=self.temporal_state_store_interval,
            temporal_state_storage_dtype=self.temporal_state_storage_dtype,
            storage_device=self.replay_storage_device,
            storage_dtype=torch.float32,
            storage_pin_memory=self.replay_storage_pin_memory,
            train_device=self.train_device,
            train_dtype=torch.float32,
            compile_tensor_operations=self.replay_compile_tensor_operations,
        )

    def _sample_training_batches(
        self,
    ) -> tuple[
        OffPolicyReplayEpisodeSegmentBatch,
        OffPolicyReplayEpisodeSegmentBatch | None,
        bool,
    ]:
        batch = self.replay_buffer.sample_episode_segments(
            self.batch_size,
            segment_length=self.learning_steps,
            burn_in_steps=self.burn_in_steps,
            require_initial_temporal_state=True,
            allow_episode_boundaries=True,
            max_train_truncations=self.max_truncations_per_segment,
        )
        return batch, None, False

    def _burn_in_critic_state(
        self,
        batch: OffPolicyReplayEpisodeSegmentBatch,
        *,
        target: bool,
    ) -> RecurrentCriticState | None:
        if getattr(self.policy, "uses_recurrent_shared_encoder", False):
            critic_state = batch.initial_temporal_state
        else:
            critic_state = self.policy.initial_critic_state(
                batch_size=batch.actions.shape[0],
                n_agents=batch.actions.shape[2],
                device=batch.actions.device,
                dtype=batch.actions.dtype,
                target=target,
            )
        if not self._critic_uses_temporal_state or self.burn_in_steps == 0:
            return critic_state
        burn_in_batch = _slice_segment(batch, 0, self.burn_in_steps)
        with torch.no_grad():
            _q1, _q2, _latents, critic_state = self.policy.q_values_sequence(
                local_obs=burn_in_batch.local_obs,
                global_obs=burn_in_batch.global_obs,
                actions=burn_in_batch.actions,
                hidden_local_vars=burn_in_batch.hidden_local_vars,
                hidden_global_vars=burn_in_batch.hidden_global_vars,
                agent_mask=burn_in_batch.agent_mask,
                scenario_ids=burn_in_batch.scenario_ids,
                target=target,
                initial_state=critic_state,
                time_mask=torch.ones_like(burn_in_batch.train_mask),
                reset_mask=burn_in_batch.episode_start_mask,
            )
        return detach_temporal_state(critic_state)

    def _padded_truncation_indices(
        self,
        truncations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = truncations.shape[0]
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=truncations.device)
        time_indices = truncations.to(dtype=torch.long).argmax(dim=1)
        indices = torch.stack((batch_indices, time_indices), dim=1)
        return indices, truncations.any(dim=1)

    def _target_next_q_values(
        self,
        *,
        batch: OffPolicyReplayEpisodeSegmentBatch,
        next_actions: torch.Tensor,
        initial_state: RecurrentCriticState | None,
        actor_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_action_samples = next_actions.shape[0] if next_actions.ndim == batch.actions.ndim + 1 else 1
        if not self._critic_uses_temporal_state:
            q1, q2, _latents, _state = self.policy.q_values_sequence(
                num_action_samples=num_action_samples,
                local_obs=batch.next_local_obs,
                global_obs=batch.next_global_obs,
                actions=next_actions,
                hidden_local_vars=batch.next_hidden_local_vars,
                hidden_global_vars=batch.next_hidden_global_vars,
                agent_mask=batch.next_agent_mask,
                scenario_ids=batch.next_scenario_ids,
                target=True,
                **({} if actor_state is None else {"actor_state": actor_state}),
            )
            return q1, q2

        history_state = initial_state
        q1_values: list[torch.Tensor] = []
        q2_values: list[torch.Tensor] = []
        for time_idx in range(batch.sequence_length):
            _q1, _q2, _latents, history_state = self.policy.q_values_sequence(
                local_obs=batch.local_obs[:, time_idx],
                global_obs=batch.global_obs[:, time_idx],
                actions=batch.actions[:, time_idx],
                hidden_local_vars=batch.hidden_local_vars[:, time_idx],
                hidden_global_vars=batch.hidden_global_vars[:, time_idx],
                agent_mask=None if batch.agent_mask is None else batch.agent_mask[:, time_idx],
                scenario_ids=(None if batch.scenario_ids is None else batch.scenario_ids[:, time_idx]),
                target=True,
                initial_state=history_state,
                reset_mask=(None if batch.episode_start_mask is None else batch.episode_start_mask[:, time_idx]),
            )
            q1, q2, _latents, _branch_state = self.policy.q_values_sequence(
                num_action_samples=num_action_samples,
                local_obs=batch.next_local_obs[:, time_idx],
                global_obs=batch.next_global_obs[:, time_idx],
                actions=next_actions[:, :, time_idx] if num_action_samples > 1 else next_actions[:, time_idx],
                hidden_local_vars=batch.next_hidden_local_vars[:, time_idx],
                hidden_global_vars=batch.next_hidden_global_vars[:, time_idx],
                agent_mask=(None if batch.next_agent_mask is None else batch.next_agent_mask[:, time_idx]),
                scenario_ids=(None if batch.next_scenario_ids is None else batch.next_scenario_ids[:, time_idx]),
                target=True,
                initial_state=history_state,
            )
            q1_values.append(q1)
            q2_values.append(q2)
        time_dim = 2 if num_action_samples > 1 else 1
        return torch.stack(q1_values, dim=time_dim), torch.stack(q2_values, dim=time_dim)

    def _actor_q_values(
        self,
        *,
        batch: OffPolicyReplayEpisodeSegmentBatch,
        actions_pi: torch.Tensor,
        initial_state: RecurrentCriticState | None,
        actor_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_action_samples = actions_pi.shape[0] if actions_pi.ndim == batch.actions.ndim + 1 else 1
        if not self._critic_uses_temporal_state:
            q1, q2, _latents, _state = self.policy.q_values_sequence(
                num_action_samples=num_action_samples,
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                actions=actions_pi,
                hidden_local_vars=batch.hidden_local_vars,
                hidden_global_vars=batch.hidden_global_vars,
                agent_mask=batch.agent_mask,
                scenario_ids=batch.scenario_ids,
                target=False,
                **({} if actor_state is None else {"actor_state": actor_state}),
            )
            return q1, q2

        history_state = initial_state
        q1_values: list[torch.Tensor] = []
        q2_values: list[torch.Tensor] = []
        for time_idx in range(batch.sequence_length):
            reset_mask = None if batch.episode_start_mask is None else batch.episode_start_mask[:, time_idx]
            q1, q2, _latents, branch_state = self.policy.q_values_sequence(
                num_action_samples=num_action_samples,
                local_obs=batch.local_obs[:, time_idx],
                global_obs=batch.global_obs[:, time_idx],
                actions=actions_pi[:, :, time_idx] if num_action_samples > 1 else actions_pi[:, time_idx],
                hidden_local_vars=batch.hidden_local_vars[:, time_idx],
                hidden_global_vars=batch.hidden_global_vars[:, time_idx],
                agent_mask=None if batch.agent_mask is None else batch.agent_mask[:, time_idx],
                scenario_ids=(None if batch.scenario_ids is None else batch.scenario_ids[:, time_idx]),
                target=False,
                initial_state=history_state,
                reset_mask=reset_mask,
            )
            if getattr(self.policy, "uses_recurrent_shared_encoder", False):
                history_state = branch_state
            else:
                _q1, _q2, _latents, history_state = self.policy.q_values_sequence(
                    local_obs=batch.local_obs[:, time_idx],
                    global_obs=batch.global_obs[:, time_idx],
                    actions=batch.actions[:, time_idx],
                    hidden_local_vars=batch.hidden_local_vars[:, time_idx],
                    hidden_global_vars=batch.hidden_global_vars[:, time_idx],
                    agent_mask=None if batch.agent_mask is None else batch.agent_mask[:, time_idx],
                    scenario_ids=(None if batch.scenario_ids is None else batch.scenario_ids[:, time_idx]),
                    target=False,
                    initial_state=history_state,
                    reset_mask=reset_mask,
                )
            q1_values.append(q1)
            q2_values.append(q2)
        time_dim = 2 if num_action_samples > 1 else 1
        return torch.stack(q1_values, dim=time_dim), torch.stack(q2_values, dim=time_dim)

    def _mask_terminal_next_observations(
        self,
        batch: OffPolicyReplayEpisodeSegmentBatch,
    ) -> OffPolicyReplayEpisodeSegmentBatch:
        (
            next_local_obs,
            next_global_obs,
            next_hidden_local_vars,
            next_hidden_global_vars,
            next_agent_mask,
        ) = self._tensor_operations.mask_terminal_observations(
            batch.terminal_mask,
            batch.next_local_obs,
            batch.next_global_obs,
            batch.next_hidden_local_vars,
            batch.next_hidden_global_vars,
            batch.next_agent_mask,
        )
        return replace(
            batch,
            next_local_obs=next_local_obs,
            next_global_obs=next_global_obs,
            next_hidden_local_vars=next_hidden_local_vars,
            next_hidden_global_vars=next_hidden_global_vars,
            next_agent_mask=next_agent_mask,
        )

    def _validate_hyper_parameters(self) -> None:
        super()._validate_hyper_parameters()
        if self.burn_in_steps < 0:
            raise ValueError(f"burn_in_steps must be >= 0, got {self.burn_in_steps}")
        if self.learning_steps <= 0:
            raise ValueError(f"learning_steps must be > 0, got {self.learning_steps}")
        if self.temporal_state_store_interval <= 0:
            raise ValueError(f"temporal_state_store_interval must be > 0, got {self.temporal_state_store_interval}")
        if self.learning_steps < self.temporal_state_store_interval:
            raise ValueError(
                "learning_steps must be >= temporal_state_store_interval so checkpoint-anchored "
                "segments cover every replay transition; "
                f"got learning_steps={self.learning_steps}, "
                f"temporal_state_store_interval={self.temporal_state_store_interval}"
            )
        if self.max_truncations_per_segment != 1:
            raise ValueError(
                "Only one truncation per recurrent segment is supported; "
                "max_truncations_per_segment must be 1, got "
                f"{self.max_truncations_per_segment}"
            )
        minimum_buffer_capacity_per_env = (
            self.burn_in_steps + self.learning_steps + self.temporal_state_store_interval - 1
        )
        if self.buffer_capacity_per_env < minimum_buffer_capacity_per_env:
            raise ValueError(
                "buffer_capacity_per_env must be at least burn_in_steps + learning_steps + "
                "temporal_state_store_interval - 1 so a full replay ring always contains a "
                "checkpoint-anchored training segment; got "
                f"{self.buffer_capacity_per_env}, need at least {minimum_buffer_capacity_per_env}"
            )
        if self.independent_nop_sampling:
            raise ValueError(f"{type(self).__name__} does not support independent_nop_sampling=True.")
        if self.policy.has_nop_loss() and self.policy.get_nop_num_next_steps() > self.learning_steps:
            raise ValueError(
                "Recurrent NOP num_next_steps cannot exceed learning_steps; NOP reuses the sampled learning sequence."
            )


def _final_and_truncation_rows(
    tensor: torch.Tensor | None,
    truncation_indices: torch.Tensor,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    rows, times = truncation_indices.unbind(-1)
    return torch.cat((tensor[:, -1], tensor[rows, times]))


def _final_and_truncation_actor_inputs(
    *,
    batch: OffPolicyReplayEpisodeSegmentBatch,
    next_actor_state: Any,
    truncation_actor_states: Any,
    truncation_indices: torch.Tensor,
) -> dict[str, Any]:
    """Pack the final successor and one padded truncation successor per replay row."""
    return {
        "local_obs": _final_and_truncation_rows(batch.next_local_obs, truncation_indices),
        "global_obs": _final_and_truncation_rows(batch.next_global_obs, truncation_indices),
        "agent_mask": _final_and_truncation_rows(batch.next_agent_mask, truncation_indices),
        "scenario_ids": _final_and_truncation_rows(batch.next_scenario_ids, truncation_indices),
        "initial_state": concatenate_temporal_states((next_actor_state, truncation_actor_states)),
    }


def _successor_sequence(
    current: torch.Tensor,
    final_and_truncation: torch.Tensor,
    truncation_indices: torch.Tensor,
    truncation_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size = current.shape[0]
    successor = torch.cat((current[:, 1:], final_and_truncation[:batch_size].unsqueeze(1)), dim=1)
    rows, times = truncation_indices.unbind(-1)
    mask = truncation_mask.reshape(batch_size, *((1,) * (current.ndim - 2)))
    successor[rows, times] = torch.where(mask, final_and_truncation[batch_size:], successor[rows, times])
    return successor


def _slice_segment(
    batch: OffPolicyReplayEpisodeSegmentBatch,
    start: int,
    end: int,
) -> OffPolicyReplayEpisodeSegmentBatch:
    def slice_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return None if tensor is None else tensor[:, start:end]

    return OffPolicyReplayEpisodeSegmentBatch(
        local_obs=batch.local_obs[:, start:end],
        global_obs=batch.global_obs[:, start:end],
        hidden_local_vars=batch.hidden_local_vars[:, start:end],
        hidden_global_vars=batch.hidden_global_vars[:, start:end],
        agent_mask=slice_tensor(batch.agent_mask),
        actions=batch.actions[:, start:end],
        rewards=batch.rewards[:, start:end],
        terminations=batch.terminations[:, start:end],
        truncations=batch.truncations[:, start:end],
        previous_actions=slice_tensor(batch.previous_actions),
        next_local_obs=batch.next_local_obs[:, start:end],
        next_global_obs=batch.next_global_obs[:, start:end],
        next_hidden_local_vars=batch.next_hidden_local_vars[:, start:end],
        next_hidden_global_vars=batch.next_hidden_global_vars[:, start:end],
        next_agent_mask=slice_tensor(batch.next_agent_mask),
        episode_start_mask=slice_tensor(batch.episode_start_mask),
        train_mask=batch.train_mask[:, start:end],
        initial_temporal_state=batch.initial_temporal_state,
        burn_in_steps=0,
        scenario_ids=slice_tensor(batch.scenario_ids),
        next_scenario_ids=slice_tensor(batch.next_scenario_ids),
    )


@dataclass(frozen=True, slots=True)
class _RecurrentNOPTrainingBatch:
    batch: SACNOPSequenceBatch
    actor_source_latents: torch.Tensor | None
    critic_source_latents: torch.Tensor | None


def _parallel_nop_windows(
    batch: OffPolicyReplayEpisodeSegmentBatch,
    *,
    window_length: int,
    include_global_obs: bool,
) -> SACNOPSequenceBatch:
    sequence_length = batch.sequence_length
    num_origins = sequence_length - window_length + 1

    def parallel_windows(tensor: torch.Tensor) -> torch.Tensor:
        unfolded = tensor.unfold(dimension=1, size=window_length, step=1)
        time_dimension = unfolded.ndim - 1
        dimension_order = (0, 1, time_dimension, *range(2, time_dimension))
        return unfolded.permute(dimension_order)

    def optional_parallel_windows(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return None if tensor is None else parallel_windows(tensor)

    episode_end_windows = parallel_windows(batch.episode_ends)
    prior_episode_ends = torch.cat(
        (
            torch.zeros_like(episode_end_windows[..., :1]),
            episode_end_windows[..., :-1],
        ),
        dim=-1,
    )
    within_origin_episode = prior_episode_ends.to(dtype=torch.long).cumsum(dim=-1) == 0

    return SACNOPSequenceBatch(
        local_obs=batch.local_obs[:, :num_origins],
        global_obs=(batch.global_obs[:, :num_origins] if include_global_obs else None),
        agent_mask=optional_parallel_windows(batch.agent_mask),
        actions=parallel_windows(batch.actions),
        next_local_obs=parallel_windows(batch.next_local_obs),
        next_global_obs=(parallel_windows(batch.next_global_obs) if include_global_obs else None),
        next_agent_mask=optional_parallel_windows(batch.next_agent_mask),
        train_mask=parallel_windows(batch.train_mask) & within_origin_episode,
    )


def _flatten_segment(batch: OffPolicyReplayEpisodeSegmentBatch) -> OffPolicyReplayBatch:
    batch_size, sequence_length = batch.actions.shape[:2]

    def flatten(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.reshape(batch_size * sequence_length, *tensor.shape[2:])

    return OffPolicyReplayBatch(
        local_obs=flatten(batch.local_obs),
        global_obs=flatten(batch.global_obs),
        hidden_local_vars=flatten(batch.hidden_local_vars),
        hidden_global_vars=flatten(batch.hidden_global_vars),
        agent_mask=flatten(batch.agent_mask),
        actions=flatten(batch.actions),
        rewards=flatten(batch.rewards),
        terminations=flatten(batch.terminations),
        truncations=flatten(batch.truncations),
        previous_actions=flatten(batch.previous_actions),
        next_local_obs=flatten(batch.next_local_obs),
        next_global_obs=flatten(batch.next_global_obs),
        next_hidden_local_vars=flatten(batch.next_hidden_local_vars),
        next_hidden_global_vars=flatten(batch.next_hidden_global_vars),
        next_agent_mask=flatten(batch.next_agent_mask),
        episode_start_mask=flatten(batch.episode_start_mask),
        scenario_ids=flatten(batch.scenario_ids),
        next_scenario_ids=flatten(batch.next_scenario_ids),
    )


def _flatten_sequence_tensor(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
) -> torch.Tensor:
    if tensor.ndim < 2 or tensor.shape[:2] != (batch_size, sequence_length):
        return tensor
    return tensor.reshape(batch_size * sequence_length, *tensor.shape[2:])
