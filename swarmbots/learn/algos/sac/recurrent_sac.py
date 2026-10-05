from typing import Any

import torch

from swarmbots.learn.algos.off_policy.replay_buffer import (
    OffPolicyReplayEpisodeSegmentBatch,
)
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import (
    RecurrentCriticState,
    RecurrentTMASACPolicy,
)
from swarmbots.learn.algos.sac.sac import SAC, TrainStepResult
from swarmbots.learn.algos.sac.segment_tmasac_policy import SegmentTMASACPolicy
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.temporal_state import (
    detach_temporal_state,
)


from swarmbots.learn.algos.off_policy.recurrent import (
    RecurrentOffPolicyMixin,
    _final_and_truncation_rows as _final_and_truncation_rows,
    _final_and_truncation_actor_inputs as _final_and_truncation_actor_inputs,
    _successor_sequence as _successor_sequence,
    _slice_segment as _slice_segment,
    _RecurrentNOPTrainingBatch as _RecurrentNOPTrainingBatch,
    _parallel_nop_windows as _parallel_nop_windows,
    _flatten_segment as _flatten_segment,
    _flatten_sequence_tensor as _flatten_sequence_tensor,
)


class RecurrentSAC(RecurrentOffPolicyMixin, SAC):
    policy: RecurrentTMASACPolicy | SegmentTMASACPolicy
    supports_recurrent_training = True

    def __init__(
        self,
        policy: RecurrentTMASACPolicy | SegmentTMASACPolicy,
        env: BaseLearnEnvWrapper,
        *,
        burn_in_steps: int = 32,
        learning_steps: int = 64,
        temporal_state_store_interval: int = 32,
        temporal_state_storage_dtype: torch.dtype | None = None,
        max_truncations_per_segment: int = 1,
        **kwargs: Any,
    ) -> None:
        if not isinstance(policy, (RecurrentTMASACPolicy, SegmentTMASACPolicy)):
            raise TypeError(
                f"RecurrentSAC requires RecurrentTMASACPolicy or SegmentTMASACPolicy, got {type(policy).__name__}"
            )
        super().__init__(
            policy=policy,
            env=env,
            burn_in_steps=burn_in_steps,
            learning_steps=learning_steps,
            temporal_state_store_interval=temporal_state_store_interval,
            temporal_state_storage_dtype=temporal_state_storage_dtype,
            max_truncations_per_segment=max_truncations_per_segment,
            **kwargs,
        )
        self.policy.configure_actor_compilation(
            encoder_only_sequence_lengths=(self.burn_in_steps,) if self.burn_in_steps > 0 else (),
            action_sequence_lengths=(1,),
            action_sequence_with_selected_states_lengths=(self.learning_steps,),
        )

    def _train_step(
        self,
        batch: OffPolicyReplayEpisodeSegmentBatch,
        *,
        nop_batch: OffPolicyReplayEpisodeSegmentBatch | None = None,
        reuse_critic_nop_latents: bool = False,
        global_update_idx: int,
        materialize_metrics: bool = True,
    ) -> TrainStepResult:
        self._mark_cuda_graph_train_step_begin()
        _ = nop_batch
        _ = reuse_critic_nop_latents
        actor_critic_lr = self._apply_actor_critic_learning_rate_for_update(global_update_idx)
        update_actor = self._should_update_actor(global_update_idx)
        learning_batch = _slice_segment(batch, self.burn_in_steps, batch.sequence_length)
        flat_batch = _flatten_segment(learning_batch)
        actor_state, critic_state, target_critic_state = self._burn_in_states(batch)
        truncation_indices, state_output_mask = self._padded_truncation_indices(
            learning_batch.truncations,
        )

        self._reset_train_gsde_noise(learning_batch.local_obs)
        (
            actions_pi,
            log_prob_pi,
            actor_latents,
            learning_next_actor_state,
            truncation_actor_states,
            last_layer_actor_state_sequence,
            current_shared_latents,
        ) = self._learning_sequence_actions(
            batch=learning_batch,
            initial_state=actor_state,
            state_output_indices=truncation_indices,
        )
        current_actor_state = None
        if getattr(self.policy, "uses_actor_state_critic_input", False):
            current_actor_state = self.policy.actor_last_layer_state_critic_input(
                last_layer_actor_state_sequence,
            )
        actor_action_dist_losses, actor_action_dist_metrics = self._compute_actor_action_dist_extra_losses(
            agent_mask=learning_batch.agent_mask,
        )
        actor_action_dist_losses = {
            name: _flatten_sequence_tensor(
                value,
                batch_size=learning_batch.actions.shape[0],
                sequence_length=learning_batch.actions.shape[1],
            )
            for name, value in actor_action_dist_losses.items()
        }
        reduced_actor_action_dist_losses = self._reduce_actor_action_dist_extra_losses(
            batch=flat_batch,
            extra_losses=actor_action_dist_losses,
        )

        log_prob_pi_mean = self._mean_agent_log_probs(log_prob_pi, learning_batch.agent_mask)
        ent_coef, ent_coef_loss = self._update_entropy_coefficient(
            log_prob_mean=self._mean_action_samples(
                log_prob_pi_mean,
                self.actor_action_samples,
            ).reshape(-1),
            batch=flat_batch,
            update=update_actor,
        )
        target_entropy = self._target_entropy(
            batch=flat_batch,
            dtype=log_prob_pi_mean.dtype,
            device=log_prob_pi_mean.device,
        )

        with torch.no_grad():
            bootstrap_batch = self._mask_terminal_next_observations(learning_batch)
            next_actions, next_log_probs, next_actor_state = self._next_policy_actions(
                batch=bootstrap_batch,
                actions_pi=actions_pi,
                log_prob_pi=log_prob_pi,
                current_actor_state=current_actor_state,
                next_actor_state=detach_temporal_state(learning_next_actor_state),
                truncation_actor_states=detach_temporal_state(truncation_actor_states),
                truncation_indices=truncation_indices,
                truncation_mask=state_output_mask,
                return_actor_state=True,
                actor_latents=(
                    actor_latents.detach() if max(self.actor_action_samples, self.target_action_samples) > 1 else None
                ),
            )
            next_log_prob_mean = self._mean_agent_log_probs(
                next_log_probs,
                bootstrap_batch.next_agent_mask,
            )
            next_actions = self._smooth_target_actions(next_actions, bootstrap_batch.next_agent_mask)
            target_q1, target_q2 = self._target_next_q_values(
                batch=bootstrap_batch,
                next_actions=next_actions,
                initial_state=target_critic_state,
                actor_state=next_actor_state,
            )
            target_q = self._tensor_operations.bellman_target(
                learning_batch.rewards,
                learning_batch.terminal_mask,
                target_q1,
                target_q2,
                next_log_prob_mean,
                ent_coef,
                self.gamma,
            )
            target_q = self._mean_action_samples(target_q, self.target_action_samples)

        current_q1, current_q2, critic_nop_latents, _next_critic_state = self.policy.q_values_sequence(
            local_obs=learning_batch.local_obs,
            global_obs=learning_batch.global_obs,
            actions=learning_batch.actions,
            hidden_local_vars=learning_batch.hidden_local_vars,
            hidden_global_vars=learning_batch.hidden_global_vars,
            agent_mask=learning_batch.agent_mask,
            scenario_ids=learning_batch.scenario_ids,
            target=False,
            initial_state=critic_state,
            time_mask=learning_batch.train_mask,
            reset_mask=learning_batch.episode_start_mask,
            **({} if current_actor_state is None else {"actor_state": current_actor_state}),
            **(
                {}
                if current_shared_latents is None
                else {
                    "precomputed_shared_latents": current_shared_latents,
                    "precomputed_shared_state": learning_next_actor_state,
                }
            ),
        )
        critic_loss = self._tensor_operations.critic_loss(
            current_q1,
            current_q2,
            target_q,
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
        self._step_actor_or_critic_optimizer(
            optimizer=self.critic_optimizer,
            compiled_step=self._critic_optimizer_step,
            global_update_idx=global_update_idx,
        )

        actor_grad_norm = 0.0
        actor_nop_metrics = {}
        if update_actor:
            actor_critic_state = critic_state
            if self._critic_uses_temporal_state:
                actor_critic_state = self._burn_in_critic_state(batch, target=False)

            critic_parameters = self.policy.critic_parameters()
            self._set_requires_grad(critic_parameters, False)
            try:
                q1_pi, q2_pi = self._actor_q_values(
                    batch=learning_batch,
                    actions_pi=actions_pi,
                    initial_state=actor_critic_state,
                    actor_state=current_actor_state,
                )
                actor_loss = self._tensor_operations.actor_loss(
                    q1_pi,
                    q2_pi,
                    log_prob_pi_mean.reshape_as(q1_pi),
                    ent_coef,
                )
            finally:
                self._set_requires_grad(critic_parameters, True)

            if nop_training_batch is None or nop_training_batch.actor_source_latents is None:
                actor_nop_loss, actor_nop_metrics = None, {}
            else:
                actor_nop_loss, actor_nop_metrics = self.policy.compute_actor_nop_loss_from_latents(
                    source_latents=nop_training_batch.actor_source_latents,
                    batch=nop_training_batch.batch,
                )
            actor_total_loss = actor_loss if actor_nop_loss is None else actor_loss + actor_nop_loss
            if reduced_actor_action_dist_losses:
                actor_total_loss = (
                    actor_total_loss + torch.stack(tuple(reduced_actor_action_dist_losses.values())).sum()
                )

            self.actor_optimizer.zero_grad()
            actor_total_loss.backward()
            actor_grad_norm = self._clip_grad_norm(self.policy.actor_parameters())
            self._step_actor_or_critic_optimizer(
                optimizer=self.actor_optimizer,
                compiled_step=self._actor_optimizer_step,
                global_update_idx=global_update_idx,
            )

        if global_update_idx % self.target_update_interval == 0:
            self.policy.polyak_update_targets(self.tau)
        self.policy.after_optimizer_step()

        metrics = {
            "critic_loss": critic_loss.detach(),
            "critic_total_loss": critic_total_loss.detach(),
            "target_q": target_q.mean().detach(),
            "current_q1": current_q1.mean().detach(),
            "current_q2": current_q2.mean().detach(),
            "log_prob": log_prob_pi_mean.mean().detach(),
            "entropy": (-log_prob_pi_mean).mean().detach(),
            "target_entropy": target_entropy.mean().detach(),
            "ent_coef": ent_coef.detach(),
            "actor_critic_learning_rate": actor_critic_lr,
        }
        if update_actor:
            metrics.update(
                actor_loss=actor_loss.detach(),
                actor_total_loss=actor_total_loss.detach(),
                q_pi=torch.minimum(q1_pi, q2_pi).mean().detach(),
            )
        if self.policy_delay > 1:
            metrics["actor_updated"] = float(update_actor)
        if ent_coef_loss is not None:
            metrics["ent_coef_loss"] = ent_coef_loss.detach()
            metrics["ent_coef_learning_rate"] = self._resolved_ent_coef_learning_rate()
        metrics.update(
            {
                f"actor_action_dist_{name}_loss_scaled": value.detach()
                for name, value in reduced_actor_action_dist_losses.items()
            }
        )
        metrics.update({f"actor_action_dist_{name}": value for name, value in actor_action_dist_metrics.items()})
        metrics.update(actor_nop_metrics)
        metrics.update(critic_nop_metrics)
        result = (metrics, actor_grad_norm, critic_grad_norm)
        if not materialize_metrics:
            return self._preserve_train_step_result(result)
        return self._materialize_train_step_results([result])[0]

    def _learning_sequence_actions(
        self,
        *,
        batch: OffPolicyReplayEpisodeSegmentBatch,
        initial_state: Any,
        state_output_indices: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Any,
        Any,
        Any | None,
        torch.Tensor | None,
    ]:
        if getattr(self.policy, "uses_recurrent_shared_encoder", False):
            assert isinstance(self.policy, RecurrentTMASACPolicy)
            (
                actions,
                log_probs,
                actor_latents,
                next_state,
                selected_states,
                shared_latents,
            ) = self.policy.action_log_prob_sequence_with_selected_states_and_shared_latents(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                agent_mask=batch.agent_mask,
                scenario_ids=batch.scenario_ids,
                previous_actions=batch.previous_actions,
                deterministic=False,
                use_rsample=True,
                num_action_samples=self.actor_action_samples,
                initial_state=initial_state,
                state_output_indices=state_output_indices,
                time_mask=batch.train_mask,
                reset_mask=batch.episode_start_mask,
            )
            return (
                actions,
                log_probs,
                actor_latents,
                next_state,
                selected_states,
                None,
                shared_latents,
            )

        result = self.policy.action_log_prob_sequence_with_selected_states(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            agent_mask=batch.agent_mask,
            scenario_ids=batch.scenario_ids,
            previous_actions=batch.previous_actions,
            deterministic=False,
            use_rsample=True,
            num_action_samples=self.actor_action_samples,
            initial_state=initial_state,
            state_output_indices=state_output_indices,
            time_mask=batch.train_mask,
            reset_mask=batch.episode_start_mask,
        )
        return (*result, None)

    def _burn_in_states(
        self,
        batch: OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[Any, RecurrentCriticState | None, RecurrentCriticState | None]:
        actor_state = batch.initial_temporal_state
        critic_state = None
        target_critic_state = None
        if self._critic_uses_temporal_state:
            critic_state = self._burn_in_critic_state(batch, target=False)
            target_critic_state = self._burn_in_critic_state(batch, target=True)
        if self.burn_in_steps == 0:
            return actor_state, critic_state, target_critic_state

        if not self.policy.uses_temporal_actor_state:
            return actor_state, critic_state, target_critic_state

        burn_in_batch = _slice_segment(batch, 0, self.burn_in_steps)
        with torch.no_grad():
            _latents, actor_state = self.policy.encode_actor_sequence(
                local_obs=burn_in_batch.local_obs,
                global_obs=burn_in_batch.global_obs,
                agent_mask=burn_in_batch.agent_mask,
                scenario_ids=burn_in_batch.scenario_ids,
                initial_state=actor_state,
                time_mask=torch.ones_like(burn_in_batch.train_mask),
                reset_mask=burn_in_batch.episode_start_mask,
            )
        actor_state = detach_temporal_state(actor_state)
        if getattr(self.policy, "uses_recurrent_shared_encoder", False):
            critic_state = actor_state
        return actor_state, critic_state, target_critic_state

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
        current_actor_state: torch.Tensor | None = None,
        return_actor_state: bool = False,
        actor_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        target_inputs = _final_and_truncation_actor_inputs(
            batch=batch,
            next_actor_state=next_actor_state,
            truncation_actor_states=truncation_actor_states,
            truncation_indices=truncation_indices,
        )

        self._reset_train_gsde_noise(target_inputs["local_obs"])
        if actor_latents is not None:
            target_latents, target_next_actor_state = self.policy.encode_actor_sequence(**target_inputs)
            next_latents = _successor_sequence(
                actor_latents,
                target_latents,
                truncation_indices,
                truncation_mask,
            )
            next_actions, next_log_probs = self.policy.action_log_prob_from_latents(
                actor_latents=next_latents,
                agent_mask=batch.next_agent_mask,
                previous_actions=batch.actions,
                use_rsample=False,
                num_action_samples=self.target_action_samples,
            )
        else:
            target_actions, target_log_probs, _latents, target_next_actor_state = self.policy.action_log_prob_sequence(
                **target_inputs,
                previous_actions=_final_and_truncation_rows(batch.actions, truncation_indices),
                deterministic=False,
                use_rsample=False,
            )
            next_actions = _successor_sequence(
                actions_pi.detach(),
                target_actions,
                truncation_indices,
                truncation_mask,
            )
            next_log_probs = _successor_sequence(
                log_prob_pi.detach(),
                target_log_probs,
                truncation_indices,
                truncation_mask,
            )
        if not return_actor_state:
            return next_actions, next_log_probs

        next_actor_state_input = None
        if current_actor_state is not None:
            next_actor_state_input = _successor_sequence(
                current_actor_state,
                self.policy.actor_state_critic_input(target_next_actor_state),
                truncation_indices,
                truncation_mask,
            )
        return next_actions, next_log_probs, next_actor_state_input

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
                module is not None and module.has_global_next_obs_pred_targets for module in nop_modules
            ),
        )
        return _RecurrentNOPTrainingBatch(
            batch=nop_batch,
            actor_source_latents=(None if self.policy.actor_nop is None else actor_latents[:, :num_origins]),
            critic_source_latents=(None if critic_latents is None else critic_latents[:, :num_origins]),
        )
