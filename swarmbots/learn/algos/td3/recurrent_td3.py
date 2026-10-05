"""Recurrent TD3 objectives on the shared off-policy sequence replay lifecycle."""

from typing import Any

import torch
from torch.nn import functional as F

from swarmbots.learn.algos.off_policy.recurrent import (
    RecurrentOffPolicyMixin,
    _RecurrentNOPTrainingBatch,
    _final_and_truncation_actor_inputs,
    _parallel_nop_windows,
    _slice_segment,
    _successor_sequence,
)
from swarmbots.learn.algos.td3.recurrent_td3_policy import RecurrentTD3Policy
from swarmbots.learn.algos.td3.td3 import TD3
from swarmbots.learn.temporal_state import detach_temporal_state


class RecurrentTD3(RecurrentOffPolicyMixin, TD3):
    policy: RecurrentTD3Policy

    def __init__(self, policy: RecurrentTD3Policy, env, **kwargs: Any) -> None:
        if not isinstance(policy, RecurrentTD3Policy):
            raise TypeError("RecurrentTD3 requires RecurrentTD3Policy")
        super().__init__(policy, env, **kwargs)

    def _burn_in_actor_state(self, batch, *, target):
        state = (
            None
            if batch.initial_temporal_state is None
            else batch.initial_temporal_state["target_actor" if target else "actor"]
        )
        if self.burn_in_steps:
            burn_in = _slice_segment(batch, 0, self.burn_in_steps)
            with torch.no_grad():
                _latents, state = self.policy.encode_actor_sequence(
                    local_obs=burn_in.local_obs,
                    global_obs=burn_in.global_obs,
                    agent_mask=burn_in.agent_mask,
                    target=target,
                    initial_state=state,
                    time_mask=torch.ones_like(burn_in.train_mask),
                    reset_mask=burn_in.episode_start_mask,
                )
        return detach_temporal_state(state)

    def _train_step(
        self, batch, *, nop_batch=None, reuse_critic_nop_latents=False, global_update_idx: int, materialize_metrics=True
    ):
        self._mark_cuda_graph_train_step_begin()
        learning_rate = self._apply_actor_critic_learning_rate_for_update(global_update_idx)
        update_actor = self._should_update_actor(global_update_idx)
        learning = _slice_segment(batch, self.burn_in_steps, batch.sequence_length)
        actor_state = self._burn_in_actor_state(batch, target=False)
        target_actor_state = self._burn_in_actor_state(batch, target=True)
        critic_state = self._burn_in_critic_state(batch, target=False)
        target_critic_state = self._burn_in_critic_state(batch, target=True)
        indices, truncation_mask = self._padded_truncation_indices(learning.truncations)
        learning_actor_outputs = None
        current_actor_state = None
        if self.policy.uses_actor_state_critic_input:
            with torch.set_grad_enabled(update_actor):
                actions, actor_latents, _state, _selected, state_sequence = self.policy.actor_actions_sequence(
                    local_obs=learning.local_obs,
                    global_obs=learning.global_obs,
                    agent_mask=learning.agent_mask,
                    initial_state=actor_state,
                    time_mask=learning.train_mask,
                    reset_mask=learning.episode_start_mask,
                    return_last_layer_state_sequence=True,
                )
            learning_actor_outputs = actions, actor_latents
            current_actor_state = self.policy.actor_last_layer_state_critic_input(state_sequence)
        with torch.no_grad():
            bootstrap = self._mask_terminal_next_observations(learning)
            target_result = self.policy.actor_actions_sequence(
                local_obs=learning.local_obs,
                global_obs=learning.global_obs,
                agent_mask=learning.agent_mask,
                target=True,
                initial_state=target_actor_state,
                reset_mask=learning.episode_start_mask,
                state_output_indices=indices,
                return_last_layer_state_sequence=self.policy.uses_actor_state_critic_input,
            )
            target_actions, _latents, next_state, selected_states = target_result[:4]
            target_inputs = _final_and_truncation_actor_inputs(
                batch=bootstrap,
                next_actor_state=next_state,
                truncation_actor_states=selected_states,
                truncation_indices=indices,
            )
            final_actions, _latents, _state = self.policy.actor_actions_sequence(target=True, **target_inputs)
            next_actions = _successor_sequence(target_actions, final_actions, indices, truncation_mask)
            next_actor_state = None
            if self.policy.uses_actor_state_critic_input:
                next_actor_state = _successor_sequence(
                    self.policy.actor_last_layer_state_critic_input(target_result[4], target=True),
                    self.policy.actor_state_critic_input(_state, target=True),
                    indices,
                    truncation_mask,
                )
            next_actions = self._smooth_target_actions(next_actions, bootstrap.next_agent_mask)
            q1, q2 = self._target_next_q_values(
                batch=bootstrap,
                next_actions=next_actions,
                initial_state=target_critic_state,
                actor_state=next_actor_state,
            )
            target_q = torch.where(
                learning.terminal_mask, learning.rewards, learning.rewards + self.gamma * torch.minimum(q1, q2)
            )
        q1, q2, critic_latents, _state = self.policy.q_values_sequence(
            local_obs=learning.local_obs,
            global_obs=learning.global_obs,
            actions=learning.actions,
            hidden_local_vars=learning.hidden_local_vars,
            hidden_global_vars=learning.hidden_global_vars,
            agent_mask=learning.agent_mask,
            initial_state=critic_state,
            time_mask=learning.train_mask,
            reset_mask=learning.episode_start_mask,
            actor_state=current_actor_state,
        )
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        nop_training = None
        critic_nop_loss, critic_nop_metrics = None, {}
        if self.policy.has_nop_loss():
            origins = learning.sequence_length - self.policy.get_nop_num_next_steps() + 1
            nop_training = _RecurrentNOPTrainingBatch(
                batch=_parallel_nop_windows(
                    learning,
                    window_length=self.policy.get_nop_num_next_steps(),
                    include_global_obs=any(
                        module is not None and module.has_global_next_obs_pred_targets
                        for module in (self.policy.actor_nop, self.policy.critic_nop)
                    ),
                ),
                actor_source_latents=None,
                critic_source_latents=None if critic_latents is None else critic_latents[:, :origins],
            )
            critic_nop_loss, critic_nop_metrics = self.policy.compute_critic_nop_loss_from_latents(
                source_latents=nop_training.critic_source_latents,
                batch=nop_training.batch,
            )
        critic_total_loss = critic_loss if critic_nop_loss is None else critic_loss + critic_nop_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_total_loss.backward()
        critic_grad_norm = self._clip_grad_norm(self.policy.critic_parameters())
        self._step_actor_or_critic_optimizer(
            optimizer=self.critic_optimizer,
            compiled_step=self._critic_optimizer_step,
            global_update_idx=global_update_idx,
        )
        actor_grad_norm = 0.0
        metrics = dict(
            critic_loss=critic_loss.detach(),
            critic_total_loss=critic_total_loss.detach(),
            target_q=target_q.mean().detach(),
            current_q1=q1.mean().detach(),
            current_q2=q2.mean().detach(),
            actor_updated=float(update_actor),
            actor_critic_learning_rate=learning_rate,
            **critic_nop_metrics,
        )
        if update_actor:
            actor_critic_state = self._burn_in_critic_state(batch, target=False)
            critic_parameters = self.policy.critic_parameters()
            self._set_requires_grad(critic_parameters, False)
            try:
                if learning_actor_outputs is None:
                    actions, actor_latents, _state = self.policy.actor_actions_sequence(
                        local_obs=learning.local_obs,
                        global_obs=learning.global_obs,
                        agent_mask=learning.agent_mask,
                        initial_state=actor_state,
                        reset_mask=learning.episode_start_mask,
                    )
                else:
                    actions, actor_latents = learning_actor_outputs
                q1_pi, _q2_pi = self._actor_q_values(
                    batch=learning,
                    actions_pi=actions,
                    initial_state=actor_critic_state,
                    actor_state=current_actor_state,
                )
                actor_loss = -q1_pi.mean()
                actor_nop_loss, actor_nop_metrics = None, {}
                if nop_training is not None and self.policy.actor_nop is not None:
                    actor_nop_loss, actor_nop_metrics = self.policy.compute_actor_nop_loss_from_latents(
                        source_latents=actor_latents[:, :origins],
                        batch=nop_training.batch,
                    )
                actor_total_loss = actor_loss if actor_nop_loss is None else actor_loss + actor_nop_loss
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_total_loss.backward()
                actor_grad_norm = self._clip_grad_norm(self.policy.actor_parameters())
                self._step_actor_or_critic_optimizer(
                    optimizer=self.actor_optimizer,
                    compiled_step=self._actor_optimizer_step,
                    global_update_idx=global_update_idx,
                )
            finally:
                self._set_requires_grad(critic_parameters, True)
            self.policy.polyak_update_targets(self.tau)
            metrics.update(
                actor_loss=actor_loss.detach(),
                actor_total_loss=actor_total_loss.detach(),
                q_pi=q1_pi.mean().detach(),
                **actor_nop_metrics,
            )
        result = metrics, actor_grad_norm, critic_grad_norm
        return (
            self._materialize_train_step_results([result])[0]
            if materialize_metrics
            else self._preserve_train_step_result(result)
        )
