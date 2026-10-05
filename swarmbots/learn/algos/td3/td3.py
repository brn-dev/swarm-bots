"""DDPG/TD3 updates using the existing off-policy rollout and checkpoint lifecycle."""

import math
from typing import Any

import torch
from torch.nn import functional as F

from swarmbots.learn.action_dists.action_dist import (
    BOUNDED_ACTION_HISTOGRAM,
    compute_action_metrics,
    resolve_action_metrics_splitter,
)
from swarmbots.learn.algos.off_policy.replay_buffer import OffPolicyReplayBatch, OffPolicyReplayEpisodeSegmentBatch
from swarmbots.learn.algos.off_policy.base_algorithm import OffPolicyAlgorithm, TrainStepResult
from swarmbots.learn.algos.td3.td3_policy import TD3Policy


class TD3(OffPolicyAlgorithm):
    """Shared cooperative actor, twin team critics, target smoothing and delayed updates.

    The common off-policy base supplies replay, rollout, scheduling and persistence.
    The learning step is entirely deterministic and has no entropy objective.
    """

    policy: TD3Policy
    expected_n_critics = 2

    def __init__(
        self,
        policy: TD3Policy,
        env,
        *,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
        **kwargs,
    ) -> None:
        if policy.config.n_critics != self.expected_n_critics:
            raise ValueError(f"{type(self).__name__} requires {self.expected_n_critics} critics")
        if not isinstance(policy_delay, int) or isinstance(policy_delay, bool) or policy_delay < 1:
            raise ValueError("policy_delay must be a positive integer")
        for name, value in (
            ("target_policy_noise", target_policy_noise),
            ("target_noise_clip", target_noise_clip),
            ("exploration_noise", exploration_noise),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        unsupported = {
            "ent_coef",
            "ent_coef_learning_rate",
            "target_entropy",
            "target_update_interval",
            "actor_action_samples",
            "target_action_samples",
        } & kwargs.keys()
        if unsupported:
            raise ValueError(f"SAC-only options are not supported by DDPG/TD3: {sorted(unsupported)}")
        if not policy.has_nop_loss() and {"nop_batch_size", "independent_nop_sampling"} & kwargs.keys():
            raise ValueError("NOP sampling options require an enabled NOP module")
        self.policy_delay = policy_delay
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.exploration_noise = float(exploration_noise)
        policy.exploration_noise = self.exploration_noise
        # Retain the historical inspection attributes; no entropy optimizer exists.
        self.log_ent_coef = None
        self.ent_coef_tensor = None
        self.ent_coef_optimizer = None
        super().__init__(
            policy, env,
            policy_delay=policy_delay, target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip, **kwargs,
        )

    def _compute_action_metrics(self, actions: torch.Tensor, *, prefix: str) -> dict[str, Any]:
        metrics = compute_action_metrics(
            actions,
            resolve_action_metrics_splitter(self.metrics_action_splitters),
            histogram=BOUNDED_ACTION_HISTOGRAM,
        )
        return {f"{prefix}_{name}": value for name, value in metrics.items()}

    @torch.no_grad()
    def _bellman_target(self, batch: OffPolicyReplayBatch) -> torch.Tensor:
        local, global_obs, hidden_local, hidden_global, mask = self._tensor_operations.mask_terminal_observations(
            batch.terminal_mask,
            batch.next_local_obs,
            batch.next_global_obs,
            batch.next_hidden_local_vars,
            batch.next_hidden_global_vars,
            batch.next_agent_mask,
        )
        actions = self.policy.actor_actions(local_obs=local, global_obs=global_obs, agent_mask=mask, target=True)
        if self.target_policy_noise:
            actions = self.policy.add_action_noise(
                actions,
                std=self.target_policy_noise,
                clip=self.target_noise_clip,
                agent_mask=mask,
            )
        values = self.policy.target_q_values(
            local_obs=local,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local,
            hidden_global_vars=hidden_global,
            agent_mask=mask,
        )
        next_q = torch.stack(values).amin(dim=0)
        return torch.where(batch.terminal_mask, batch.rewards, batch.rewards + self.gamma * next_q)

    def _train_step(
        self,
        batch: OffPolicyReplayBatch,
        *,
        nop_batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch | None = None,
        reuse_critic_nop_latents: bool = False,
        global_update_idx: int,
        materialize_metrics: bool = True,
    ) -> TrainStepResult:
        self._mark_cuda_graph_train_step_begin()
        learning_rate = self._apply_actor_critic_learning_rate_for_update(global_update_idx)
        target_q = self._bellman_target(batch)
        observations = dict(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            hidden_local_vars=batch.hidden_local_vars,
            hidden_global_vars=batch.hidden_global_vars,
            agent_mask=batch.agent_mask,
        )
        values, critic_latents = self.policy.q_values_with_nop_latents(actions=batch.actions, **observations)
        critic_loss = sum(F.mse_loss(value, target_q) for value in values)
        nop_loss_batch = batch if nop_batch is None else nop_batch
        skip_nop = self._uses_multi_step_nop() and nop_batch is None
        critic_nop_loss, critic_nop_metrics = (None, {"nop_loss_skipped": 1.0}) if skip_nop else (
            self.policy.compute_critic_nop_loss(
                nop_loss_batch, source_latents=critic_latents if reuse_critic_nop_latents else None,
            )
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
        update_actor = self._should_update_actor(global_update_idx)
        metrics = {
            "critic_loss": critic_loss.detach(),
            "critic_total_loss": critic_total_loss.detach(),
            "target_q": target_q.mean().detach(),
            "actor_updated": float(update_actor),
            "actor_critic_learning_rate": learning_rate,
            **{f"current_q{i + 1}": value.mean().detach() for i, value in enumerate(values)},
            **critic_nop_metrics,
        }
        actor_grad_norm = 0.0
        if update_actor:
            critic_parameters = self.policy.critic_parameters()
            self._set_requires_grad(critic_parameters, False)
            try:
                actions = self.policy.actor_actions(
                    local_obs=batch.local_obs,
                    global_obs=batch.global_obs,
                    agent_mask=batch.agent_mask,
                )
                # The shared team objective differentiates through every active agent's action.
                q_pi = self.policy.q_values(actions=actions, **observations)[0]
                actor_loss = -q_pi.mean()
                actor_nop_loss, actor_nop_metrics = (None, {}) if skip_nop else (
                    self.policy.compute_actor_nop_loss(nop_loss_batch)
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
                actor_loss=actor_loss.detach(), actor_total_loss=actor_total_loss.detach(), q_pi=q_pi.mean().detach(),
                **actor_nop_metrics,
            )
        result = metrics, actor_grad_norm, critic_grad_norm
        if materialize_metrics:
            return self._materialize_train_step_results([result])[0]
        return self._preserve_train_step_result(result)

    def get_hyper_parameters(self) -> dict[str, Any]:
        settings = super().get_hyper_parameters()
        for key in tuple(settings):
            if (
                "ent_coef" in key
                or "entropy" in key
                or ("nop" in key and not self.policy.has_nop_loss())
                or key
                in {
                    "actor_action_samples",
                    "target_action_samples",
                    "target_update_interval",
                }
            ):
                settings.pop(key)
        settings.update(
            policy_delay=self.policy_delay,
            target_policy_noise=self.target_policy_noise,
            target_noise_clip=self.target_noise_clip,
            exploration_noise=self.exploration_noise,
        )
        return settings

    def _get_optimizer_state_dict(self) -> dict[str, Any]:
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def _apply_optimizer_state_dict(
        self,
        state_dict: dict[str, Any],
        missing_keys: list[str],
        unexpected_keys: list[str],
    ) -> None:
        if missing_keys or unexpected_keys:
            raise ValueError("Cannot restore DDPG/TD3 optimizers with mismatched policy parameters")
        for name in ("actor_optimizer", "critic_optimizer"):
            optimizer = getattr(self, name)
            optimizer.load_state_dict(state_dict[name])
            self._move_optimizer_state_to_device(optimizer, self.train_device)
        self._rebuild_optimizer_steps()

    def _execute_command(self, cmd: str, params: str, extra_run_metadata: dict[str, Any] | None) -> bool:
        if cmd in {"set_ent_coef", "ent_coef"}:
            raise ValueError("DDPG/TD3 do not have an entropy coefficient")
        return super()._execute_command(cmd, params, extra_run_metadata)


class DDPG(TD3):
    """Single-critic deterministic policy gradient, without TD3 delay or smoothing."""

    expected_n_critics = 1

    def __init__(self, policy: TD3Policy, env, **kwargs):
        for name, required in (("policy_delay", 1), ("target_policy_noise", 0.0), ("target_noise_clip", 0.0)):
            if name in kwargs and kwargs[name] != required:
                raise ValueError(f"DDPG requires {name}={required}")
            kwargs[name] = required
        super().__init__(policy, env, **kwargs)
