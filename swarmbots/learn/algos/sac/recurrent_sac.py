from dataclasses import dataclass, replace
from typing import Any

import torch
from loguru import logger
from torch.nn import functional as F

from swarmbots.learn.algos.off_policy.replay_buffer import (
    NoEpisodeSegmentCandidatesError,
    OffPolicyReplayBatch,
    OffPolicyReplayBuffer,
    OffPolicyReplayEpisodeSegmentBatch,
)
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import (
    RecurrentCriticState,
    RecurrentTMASACPolicy,
)
from swarmbots.learn.algos.sac.sac import SAC
from swarmbots.learn.algos.sac.sac_nop import SACNOPSequenceBatch
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.temporal_state import concatenate_temporal_states, detach_temporal_state


class RecurrentSAC(SAC):
    policy: RecurrentTMASACPolicy
    supports_recurrent_training = True

    def __init__(
            self,
            policy: RecurrentTMASACPolicy,
            env: BaseLearnEnvWrapper,
            *,
            burn_in_steps: int = 32,
            learning_steps: int = 64,
            temporal_state_store_interval: int = 32,
            temporal_state_storage_dtype: torch.dtype | None = None,
            max_truncations_per_segment: int = 1,
            **kwargs: Any,
    ) -> None:
        if not isinstance(policy, RecurrentTMASACPolicy):
            raise TypeError(
                "RecurrentSAC requires RecurrentTMASACPolicy, got "
                f"{type(policy).__name__}"
            )
        self.burn_in_steps = int(burn_in_steps)
        self.learning_steps = int(learning_steps)
        self.temporal_state_store_interval = int(temporal_state_store_interval)
        self.temporal_state_storage_dtype = temporal_state_storage_dtype
        self.max_truncations_per_segment = int(max_truncations_per_segment)
        super().__init__(policy=policy, env=env, **kwargs)
        compiled_sequence_lengths = {1, self.learning_steps}
        if self.burn_in_steps > 0:
            compiled_sequence_lengths.add(self.burn_in_steps)
        self.policy.configure_actor_encoder_compilation(compiled_sequence_lengths)

    @property
    def replay_fill_target(self) -> int:
        sequence_fill_target = (
            (self.burn_in_steps + self.learning_steps)
            * self.env.action_space.n_envs
        )
        return max(super().replay_fill_target, sequence_fill_target)

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "burn_in_steps": self.burn_in_steps,
            "learning_steps": self.learning_steps,
            "compiled_actor_sequence_lengths": sorted(self.policy.compiled_actor_sequence_lengths),
            "max_truncations_per_segment": self.max_truncations_per_segment,
            "temporal_state_store_interval": self.temporal_state_store_interval,
            "temporal_state_storage_dtype": str(
                self.replay_buffer.temporal_state_storage_dtype
            ),
        }

    def train(self, *, gradient_steps: int) -> dict[str, Any]:
        try:
            return super().train(gradient_steps=gradient_steps)
        except NoEpisodeSegmentCandidatesError:
            logger.warning(
                "Skipping recurrent SAC updates because replay has no temporal-state-anchored "
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
        )
        return batch, None, False

    def _train_step(
            self,
            batch: OffPolicyReplayEpisodeSegmentBatch,
            *,
            nop_batch: OffPolicyReplayEpisodeSegmentBatch | None = None,
            reuse_critic_nop_latents: bool = False,
            global_update_idx: int,
    ) -> tuple[dict[str, float], float, float]:
        _ = nop_batch
        _ = reuse_critic_nop_latents
        actor_critic_lr = self._apply_actor_critic_learning_rate_for_update(global_update_idx)
        learning_batch = _slice_segment(batch, self.burn_in_steps, batch.sequence_length)
        flat_batch = _flatten_segment(learning_batch)
        actor_state, critic_state, target_critic_state = self._burn_in_states(batch)
        state_output_indices, state_output_mask = self._padded_truncation_indices(
            learning_batch.truncations,
        )

        self._reset_train_gsde_noise(learning_batch.local_obs)
        (
            actions_pi,
            log_prob_pi,
            actor_latents,
            next_actor_state,
            truncation_actor_states,
        ) = self.policy.action_log_prob_sequence_with_selected_states(
            local_obs=learning_batch.local_obs,
            global_obs=learning_batch.global_obs,
            agent_mask=learning_batch.agent_mask,
            previous_actions=learning_batch.previous_actions,
            deterministic=False,
            use_rsample=True,
            initial_state=actor_state,
            state_output_indices=state_output_indices,
            time_mask=learning_batch.train_mask,
            reset_mask=learning_batch.episode_start_mask,
        )
        actor_action_dist_losses, actor_action_dist_metrics = self._compute_actor_action_dist_extra_losses(
            agent_mask=learning_batch.agent_mask,
        )
        actor_action_dist_losses = {
            name: _flatten_sequence_tensor(value)
            for name, value in actor_action_dist_losses.items()
        }
        reduced_actor_action_dist_losses = self._reduce_actor_action_dist_extra_losses(
            batch=flat_batch,
            extra_losses=actor_action_dist_losses,
        )

        flat_log_prob_pi = log_prob_pi.reshape(-1, log_prob_pi.shape[-1])
        flat_agent_mask = (
            None
            if learning_batch.agent_mask is None
            else learning_batch.agent_mask.reshape(-1, learning_batch.agent_mask.shape[-1])
        )
        log_prob_pi_mean = self._mean_agent_log_probs(flat_log_prob_pi, flat_agent_mask)
        ent_coef, ent_coef_loss = self._update_entropy_coefficient(
            log_prob_mean=log_prob_pi_mean,
            batch=flat_batch,
        )
        target_entropy = self._target_entropy(
            batch=flat_batch,
            dtype=log_prob_pi_mean.dtype,
            device=log_prob_pi_mean.device,
        )

        with torch.no_grad():
            bootstrap_batch = _mask_terminal_next_observations(learning_batch)
            next_actions, next_log_probs = self._next_policy_actions(
                batch=bootstrap_batch,
                actions_pi=actions_pi,
                log_prob_pi=log_prob_pi,
                next_actor_state=detach_temporal_state(next_actor_state),
                truncation_actor_states=detach_temporal_state(truncation_actor_states),
                truncation_indices=state_output_indices,
                truncation_mask=state_output_mask,
            )
            next_log_prob_mean = self._mean_agent_log_probs(
                next_log_probs.reshape(-1, next_log_probs.shape[-1]),
                None if bootstrap_batch.next_agent_mask is None else bootstrap_batch.next_agent_mask.reshape(
                    -1,
                    bootstrap_batch.next_agent_mask.shape[-1],
                ),
            ).reshape_as(learning_batch.rewards)
            target_q1, target_q2 = self._target_next_q_values(
                batch=bootstrap_batch,
                next_actions=next_actions,
                initial_state=target_critic_state,
            )
            next_q = torch.minimum(target_q1, target_q2) - ent_coef * next_log_prob_mean
            target_q = torch.where(
                learning_batch.terminal_mask,
                learning_batch.rewards,
                learning_batch.rewards + self.gamma * next_q,
            )

        current_q1, current_q2, critic_nop_latents, _next_critic_state = self.policy.q_values_sequence(
            local_obs=learning_batch.local_obs,
            global_obs=learning_batch.global_obs,
            actions=learning_batch.actions,
            hidden_local_vars=learning_batch.hidden_local_vars,
            hidden_global_vars=learning_batch.hidden_global_vars,
            agent_mask=learning_batch.agent_mask,
            target=False,
            initial_state=critic_state,
            time_mask=learning_batch.train_mask,
            reset_mask=learning_batch.episode_start_mask,
        )
        critic_loss = 0.5 * (
            F.mse_loss(current_q1, target_q)
            + F.mse_loss(current_q2, target_q)
        )
        nop_training_batch = self._build_nop_training_batch(
            batch=learning_batch,
            actor_latents=actor_latents,
            critic_latents=critic_nop_latents,
        )
        if nop_training_batch is None:
            critic_nop_loss, critic_nop_metrics = None, {}
        else:
            critic_nop_loss, critic_nop_metrics = self.policy.compute_critic_nop_loss_from_latents(
                source_latents=nop_training_batch.critic_source_latents,
                batch=nop_training_batch.batch,
            )
        critic_total_loss = critic_loss if critic_nop_loss is None else critic_loss + critic_nop_loss

        self.critic_optimizer.zero_grad()
        critic_total_loss.backward()
        critic_grad_norm = self._clip_grad_norm(self.policy.critic_parameters())
        self.critic_optimizer.step()

        actor_critic_state = critic_state
        if self.policy.recurrent_critic:
            actor_critic_state = self._burn_in_critic_state(batch, target=False)

        critic_parameters = self.policy.critic_parameters()
        self._set_requires_grad(critic_parameters, False)
        try:
            q1_pi, q2_pi = self._actor_q_values(
                batch=learning_batch,
                actions_pi=actions_pi,
                initial_state=actor_critic_state,
            )
            actor_loss = (
                ent_coef * log_prob_pi_mean.reshape_as(q1_pi)
                - torch.minimum(q1_pi, q2_pi)
            ).mean()
        finally:
            self._set_requires_grad(critic_parameters, True)

        if (
                nop_training_batch is None
                or nop_training_batch.actor_source_latents is None
        ):
            actor_nop_loss, actor_nop_metrics = None, {}
        else:
            actor_nop_loss, actor_nop_metrics = self.policy.compute_actor_nop_loss_from_latents(
                source_latents=nop_training_batch.actor_source_latents,
                batch=nop_training_batch.batch,
            )
        actor_total_loss = actor_loss if actor_nop_loss is None else actor_loss + actor_nop_loss
        if reduced_actor_action_dist_losses:
            actor_total_loss = actor_total_loss + torch.stack(
                tuple(reduced_actor_action_dist_losses.values())
            ).sum()

        self.actor_optimizer.zero_grad()
        actor_total_loss.backward()
        actor_grad_norm = self._clip_grad_norm(self.policy.actor_parameters())
        self.actor_optimizer.step()

        if global_update_idx % self.target_update_interval == 0:
            self.policy.polyak_update_targets(self.tau)
        self.policy.after_optimizer_step()

        metrics = {
            "critic_loss": critic_loss.item(),
            "critic_total_loss": critic_total_loss.item(),
            "actor_loss": actor_loss.item(),
            "actor_total_loss": actor_total_loss.item(),
            "target_q": target_q.mean().item(),
            "current_q1": current_q1.mean().item(),
            "current_q2": current_q2.mean().item(),
            "q_pi": torch.minimum(q1_pi, q2_pi).mean().item(),
            "log_prob": log_prob_pi_mean.mean().item(),
            "entropy": (-log_prob_pi_mean).mean().item(),
            "target_entropy": target_entropy.mean().item(),
            "ent_coef": ent_coef.item(),
            "actor_critic_learning_rate": actor_critic_lr,
        }
        if ent_coef_loss is not None:
            metrics["ent_coef_loss"] = ent_coef_loss.item()
            metrics["ent_coef_learning_rate"] = self._resolved_ent_coef_learning_rate()
        metrics.update({
            f"actor_action_dist_{name}_loss_scaled": value.item()
            for name, value in reduced_actor_action_dist_losses.items()
        })
        metrics.update({
            f"actor_action_dist_{name}": value
            for name, value in actor_action_dist_metrics.items()
        })
        metrics.update(actor_nop_metrics)
        metrics.update(critic_nop_metrics)
        return metrics, actor_grad_norm, critic_grad_norm

    def _burn_in_states(
            self,
            batch: OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[Any, RecurrentCriticState | None, RecurrentCriticState | None]:
        actor_state = batch.initial_temporal_state
        critic_state = None
        target_critic_state = None
        if self.policy.recurrent_critic:
            critic_state = self._burn_in_critic_state(batch, target=False)
            target_critic_state = self._burn_in_critic_state(batch, target=True)
        if self.burn_in_steps == 0:
            return actor_state, critic_state, target_critic_state

        burn_in_batch = _slice_segment(batch, 0, self.burn_in_steps)
        with torch.no_grad():
            _latents, actor_state = self.policy.encode_actor_sequence(
                local_obs=burn_in_batch.local_obs,
                global_obs=burn_in_batch.global_obs,
                agent_mask=burn_in_batch.agent_mask,
                initial_state=actor_state,
                time_mask=torch.ones_like(burn_in_batch.train_mask),
                reset_mask=burn_in_batch.episode_start_mask,
            )
        return (
            detach_temporal_state(actor_state),
            critic_state,
            target_critic_state,
        )

    def _burn_in_critic_state(
            self,
            batch: OffPolicyReplayEpisodeSegmentBatch,
            *,
            target: bool,
    ) -> RecurrentCriticState | None:
        critic_state = self.policy.initial_critic_state(
            batch_size=batch.actions.shape[0],
            n_agents=batch.actions.shape[2],
            device=batch.actions.device,
            dtype=batch.actions.dtype,
            target=target,
        )
        if not self.policy.recurrent_critic or self.burn_in_steps == 0:
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
                target=target,
                initial_state=critic_state,
                time_mask=torch.ones_like(burn_in_batch.train_mask),
                reset_mask=burn_in_batch.episode_start_mask,
            )
        return detach_temporal_state(critic_state)

    def _next_policy_actions(
            self,
            *,
            batch: OffPolicyReplayEpisodeSegmentBatch,
            actions_pi: torch.Tensor,
            log_prob_pi: torch.Tensor,
            next_actor_state: Any,
            truncation_actor_states: Any,
            truncation_indices: torch.Tensor,
            truncation_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch.actions.shape[0]
        selected_batch_indices = truncation_indices[:, 0]
        selected_time_indices = truncation_indices[:, 1]
        target_local_obs = torch.cat((
            batch.next_local_obs[:, -1],
            batch.next_local_obs[selected_batch_indices, selected_time_indices],
        ))
        target_global_obs = torch.cat((
            batch.next_global_obs[:, -1],
            batch.next_global_obs[selected_batch_indices, selected_time_indices],
        ))
        target_agent_mask = (
            None
            if batch.next_agent_mask is None
            else torch.cat((
                batch.next_agent_mask[:, -1],
                batch.next_agent_mask[selected_batch_indices, selected_time_indices],
            ))
        )
        target_previous_actions = torch.cat((
            batch.actions[:, -1],
            batch.actions[selected_batch_indices, selected_time_indices],
        ))
        target_initial_state = concatenate_temporal_states((
            next_actor_state,
            truncation_actor_states,
        ))

        self._reset_train_gsde_noise(target_local_obs)
        target_actions, target_log_probs, _latents, _state = self.policy.action_log_prob_sequence(
            local_obs=target_local_obs,
            global_obs=target_global_obs,
            agent_mask=target_agent_mask,
            previous_actions=target_previous_actions,
            deterministic=False,
            use_rsample=False,
            initial_state=target_initial_state,
        )
        last_actions = target_actions[:batch_size]
        last_log_probs = target_log_probs[:batch_size]
        truncation_actions = target_actions[batch_size:]
        truncation_log_probs = target_log_probs[batch_size:]

        next_actions = torch.cat((actions_pi[:, 1:].detach(), last_actions.unsqueeze(1)), dim=1)
        next_log_probs = torch.cat((log_prob_pi[:, 1:].detach(), last_log_probs.unsqueeze(1)), dim=1)
        valid_indices = truncation_indices[truncation_mask]
        next_actions[valid_indices[:, 0], valid_indices[:, 1]] = truncation_actions[truncation_mask]
        next_log_probs[valid_indices[:, 0], valid_indices[:, 1]] = truncation_log_probs[truncation_mask]
        return next_actions, next_log_probs

    def _padded_truncation_indices(
            self,
            truncations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        truncation_indices = torch.nonzero(truncations, as_tuple=False)
        capacity = truncations.shape[0] * self.max_truncations_per_segment
        if truncation_indices.shape[0] > capacity:
            raise ValueError(
                f"Sampled recurrent segment batch contains {truncation_indices.shape[0]} truncations, "
                f"exceeding configured capacity {capacity}; increase max_truncations_per_segment."
            )
        padded_indices = torch.zeros((capacity, 2), dtype=torch.long, device=truncations.device)
        padded_indices[:truncation_indices.shape[0]] = truncation_indices
        valid_mask = torch.arange(capacity, device=truncations.device) < truncation_indices.shape[0]
        return padded_indices, valid_mask

    def _target_next_q_values(
            self,
            *,
            batch: OffPolicyReplayEpisodeSegmentBatch,
            next_actions: torch.Tensor,
            initial_state: RecurrentCriticState | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.policy.recurrent_critic:
            q1, q2, _latents, _state = self.policy.q_values_sequence(
                local_obs=batch.next_local_obs,
                global_obs=batch.next_global_obs,
                actions=next_actions,
                hidden_local_vars=batch.next_hidden_local_vars,
                hidden_global_vars=batch.next_hidden_global_vars,
                agent_mask=batch.next_agent_mask,
                target=True,
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
                target=True,
                initial_state=history_state,
                reset_mask=(
                    None
                    if batch.episode_start_mask is None
                    else batch.episode_start_mask[:, time_idx]
                ),
            )
            q1, q2, _latents, _branch_state = self.policy.q_values_sequence(
                local_obs=batch.next_local_obs[:, time_idx],
                global_obs=batch.next_global_obs[:, time_idx],
                actions=next_actions[:, time_idx],
                hidden_local_vars=batch.next_hidden_local_vars[:, time_idx],
                hidden_global_vars=batch.next_hidden_global_vars[:, time_idx],
                agent_mask=(
                    None
                    if batch.next_agent_mask is None
                    else batch.next_agent_mask[:, time_idx]
                ),
                target=True,
                initial_state=history_state,
            )
            q1_values.append(q1)
            q2_values.append(q2)
        return torch.stack(q1_values, dim=1), torch.stack(q2_values, dim=1)

    def _actor_q_values(
            self,
            *,
            batch: OffPolicyReplayEpisodeSegmentBatch,
            actions_pi: torch.Tensor,
            initial_state: RecurrentCriticState | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.policy.recurrent_critic:
            q1, q2, _latents, _state = self.policy.q_values_sequence(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                actions=actions_pi,
                hidden_local_vars=batch.hidden_local_vars,
                hidden_global_vars=batch.hidden_global_vars,
                agent_mask=batch.agent_mask,
                target=False,
            )
            return q1, q2

        history_state = initial_state
        q1_values: list[torch.Tensor] = []
        q2_values: list[torch.Tensor] = []
        for time_idx in range(batch.sequence_length):
            reset_mask = (
                None
                if batch.episode_start_mask is None
                else batch.episode_start_mask[:, time_idx]
            )
            q1, q2, _latents, _branch_state = self.policy.q_values_sequence(
                local_obs=batch.local_obs[:, time_idx],
                global_obs=batch.global_obs[:, time_idx],
                actions=actions_pi[:, time_idx],
                hidden_local_vars=batch.hidden_local_vars[:, time_idx],
                hidden_global_vars=batch.hidden_global_vars[:, time_idx],
                agent_mask=None if batch.agent_mask is None else batch.agent_mask[:, time_idx],
                target=False,
                initial_state=history_state,
                reset_mask=reset_mask,
            )
            _q1, _q2, _latents, history_state = self.policy.q_values_sequence(
                local_obs=batch.local_obs[:, time_idx],
                global_obs=batch.global_obs[:, time_idx],
                actions=batch.actions[:, time_idx],
                hidden_local_vars=batch.hidden_local_vars[:, time_idx],
                hidden_global_vars=batch.hidden_global_vars[:, time_idx],
                agent_mask=None if batch.agent_mask is None else batch.agent_mask[:, time_idx],
                target=False,
                initial_state=history_state,
                reset_mask=reset_mask,
            )
            q1_values.append(q1)
            q2_values.append(q2)
        return torch.stack(q1_values, dim=1), torch.stack(q2_values, dim=1)

    def _build_nop_training_batch(
            self,
            *,
            batch: OffPolicyReplayEpisodeSegmentBatch,
            actor_latents: torch.Tensor,
            critic_latents: torch.Tensor | None,
    ) -> "_RecurrentNOPTrainingBatch | None":
        if not self.policy.has_nop_loss():
            return None
        num_next_steps = self.policy.get_nop_num_next_steps()
        num_origins = batch.sequence_length - num_next_steps + 1
        nop_modules = (self.policy.actor_nop, self.policy.critic_nop)
        nop_batch = _parallel_nop_windows(
            batch,
            window_length=num_next_steps,
            include_global_obs=any(
                module is not None and module.has_global_next_obs_pred_targets
                for module in nop_modules
            ),
        )
        return _RecurrentNOPTrainingBatch(
            batch=nop_batch,
            actor_source_latents=(
                None
                if self.policy.actor_nop is None
                else actor_latents[:, :num_origins]
            ),
            critic_source_latents=(
                None
                if critic_latents is None
                else critic_latents[:, :num_origins]
            ),
        )

    def _validate_hyper_parameters(self) -> None:
        super()._validate_hyper_parameters()
        if self.burn_in_steps < 0:
            raise ValueError(f"burn_in_steps must be >= 0, got {self.burn_in_steps}")
        if self.learning_steps <= 0:
            raise ValueError(f"learning_steps must be > 0, got {self.learning_steps}")
        if self.temporal_state_store_interval <= 0:
            raise ValueError(
                "temporal_state_store_interval must be > 0, got "
                f"{self.temporal_state_store_interval}"
            )
        if self.max_truncations_per_segment <= 0:
            raise ValueError(
                "max_truncations_per_segment must be > 0, got "
                f"{self.max_truncations_per_segment}"
            )
        minimum_buffer_capacity_per_env = (
            self.burn_in_steps
            + self.learning_steps
            + self.temporal_state_store_interval
            - 1
        )
        if self.buffer_capacity_per_env < minimum_buffer_capacity_per_env:
            raise ValueError(
                "buffer_capacity_per_env must be at least burn_in_steps + learning_steps + "
                "temporal_state_store_interval - 1 so a full replay ring always contains a "
                "checkpoint-anchored training segment; got "
                f"{self.buffer_capacity_per_env}, need at least {minimum_buffer_capacity_per_env}"
            )
        if self.independent_nop_sampling:
            raise ValueError("RecurrentSAC does not support independent_nop_sampling=True.")
        if (
                self.policy.has_nop_loss()
                and self.policy.get_nop_num_next_steps() > self.learning_steps
        ):
            raise ValueError(
                "Recurrent TMASAC NOP num_next_steps cannot exceed learning_steps; "
                "NOP reuses the sampled learning sequence."
            )


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
    prior_episode_ends = torch.cat((
        torch.zeros_like(episode_end_windows[..., :1]),
        episode_end_windows[..., :-1],
    ), dim=-1)
    within_origin_episode = prior_episode_ends.to(dtype=torch.long).cumsum(dim=-1) == 0

    return SACNOPSequenceBatch(
        local_obs=batch.local_obs[:, :num_origins],
        global_obs=(
            batch.global_obs[:, :num_origins]
            if include_global_obs
            else None
        ),
        agent_mask=optional_parallel_windows(batch.agent_mask),
        actions=parallel_windows(batch.actions),
        next_local_obs=parallel_windows(batch.next_local_obs),
        next_global_obs=(
            parallel_windows(batch.next_global_obs)
            if include_global_obs
            else None
        ),
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
    )


def _flatten_sequence_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 2:
        return tensor
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


def _mask_terminal_next_observations(
        batch: OffPolicyReplayEpisodeSegmentBatch,
) -> OffPolicyReplayEpisodeSegmentBatch:
    def mask_tensor(tensor: torch.Tensor, value: bool | float) -> torch.Tensor:
        row_mask = batch.terminal_mask.reshape(
            *batch.terminal_mask.shape,
            *((1,) * (tensor.ndim - batch.terminal_mask.ndim)),
        )
        return tensor.masked_fill(row_mask, value)

    return replace(
        batch,
        next_local_obs=mask_tensor(batch.next_local_obs, 0.0),
        next_global_obs=mask_tensor(batch.next_global_obs, 0.0),
        next_hidden_local_vars=mask_tensor(batch.next_hidden_local_vars, 0.0),
        next_hidden_global_vars=mask_tensor(batch.next_hidden_global_vars, 0.0),
        next_agent_mask=(
            None
            if batch.next_agent_mask is None
            else mask_tensor(batch.next_agent_mask, True)
        ),
    )
