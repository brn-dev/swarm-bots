from typing import Any

import torch
from loguru import logger

from swarmbots.learn.action_dists.action_dist import ActionMetricsSplitterInput
from swarmbots.learn.action_dists.action_sampling import (
    expand_action_samples,
    validate_action_sampling,
)
from swarmbots.learn.algos.base_algorithm import (
    LearningRate,
)
from swarmbots.learn.algos.off_policy.replay_buffer import (
    OffPolicyReplayBatch,
    OffPolicyReplayEpisodeSegmentBatch,
)
from swarmbots.learn.algos.sac.base_sac_policy import BaseSACPolicy
from swarmbots.learn.algos.sac.sac_tensor_ops import (
    build_optimizer_step,
)
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.gsde_reset import (
    GSDEResetMode,
)


from swarmbots.learn.algos.off_policy.base_algorithm import (
    DEFAULT_TOTAL_REPLAY_CAPACITY as DEFAULT_TOTAL_REPLAY_CAPACITY,
    OffPolicyAlgorithm,
    TrainMetric as TrainMetric,
    TrainStepResult as TrainStepResult,
)

ENTROPY_AGENT_REDUCTION = "mean"


class SAC(OffPolicyAlgorithm):
    """SAC objectives with optional delayed policy updates and target Q smoothing."""

    policy: BaseSACPolicy

    def __init__(
        self,
        policy: BaseSACPolicy,
        env: BaseLearnEnvWrapper,
        learning_rate: float = 3e-4,
        learning_rate_warmup_updates: int = 500,
        learning_rate_warmup_start_factor: float = 0.01,
        buffer_capacity_per_env: int | None = None,
        learning_starts: int = 10_000,
        batch_size: int = 256,
        rollout_steps_per_iteration: int | None = None,
        rollout_warmup_steps_per_env: int = 0,
        gradient_steps: int = 1,
        gamma: float = 0.99,
        tau: float = 0.005,
        ent_coef: float | str = "auto",
        ent_coef_learning_rate: float | None = None,
        target_entropy: float | str = "auto",
        target_update_interval: int = 1,
        max_grad_norm: float | None = 2.0,
        nop_batch_size: int | None = None,
        independent_nop_sampling: bool = False,
        gsde_reset_mode: GSDEResetMode | None = None,
        train_device: str | torch.device = "auto",
        rollout_device: str | torch.device = "cpu",
        record_device: str | torch.device | None = None,
        replay_storage_device: str | torch.device = "cuda",
        replay_storage_pin_memory: bool = False,
        replay_compile_tensor_operations: bool | None = None,
        sac_compile_tensor_operations: bool | None = None,
        sac_compile_optimizer_steps: bool | None = None,
        sac_compile_mode: str = "default",
        metrics_action_splitters: ActionMetricsSplitterInput = None,
        actor_action_samples: int = 1,
        target_action_samples: int = 1,
        policy_delay: int = 1,
        target_policy_noise: float = 0.0,
        target_noise_clip: float = 0.0,
    ) -> None:
        self.ent_coef = ent_coef
        self.ent_coef_learning_rate = None if ent_coef_learning_rate is None else float(ent_coef_learning_rate)
        self.target_entropy = target_entropy
        self.actor_action_samples = actor_action_samples
        self.target_action_samples = target_action_samples
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            learning_rate_warmup_updates=learning_rate_warmup_updates,
            learning_rate_warmup_start_factor=learning_rate_warmup_start_factor,
            buffer_capacity_per_env=buffer_capacity_per_env,
            learning_starts=learning_starts,
            batch_size=batch_size,
            rollout_steps_per_iteration=rollout_steps_per_iteration,
            rollout_warmup_steps_per_env=rollout_warmup_steps_per_env,
            gradient_steps=gradient_steps,
            gamma=gamma,
            tau=tau,
            target_update_interval=target_update_interval,
            max_grad_norm=max_grad_norm,
            nop_batch_size=nop_batch_size,
            independent_nop_sampling=independent_nop_sampling,
            gsde_reset_mode=gsde_reset_mode,
            train_device=train_device,
            rollout_device=rollout_device,
            record_device=record_device,
            replay_storage_device=replay_storage_device,
            replay_storage_pin_memory=replay_storage_pin_memory,
            replay_compile_tensor_operations=replay_compile_tensor_operations,
            sac_compile_tensor_operations=sac_compile_tensor_operations,
            sac_compile_optimizer_steps=sac_compile_optimizer_steps,
            sac_compile_mode=sac_compile_mode,
            metrics_action_splitters=metrics_action_splitters,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
        )

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "actor_action_samples": self.actor_action_samples,
            "target_action_samples": self.target_action_samples,
            "ent_coef": self.ent_coef,
            "ent_coef_learning_rate": self.ent_coef_learning_rate,
            "resolved_ent_coef_learning_rate": self._resolved_ent_coef_learning_rate(),
            "target_entropy": self.target_entropy,
            "entropy_agent_reduction": ENTROPY_AGENT_REDUCTION,
        }

    def _compute_action_metrics(self, actions: torch.Tensor, *, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_{name}": value
            for name, value in self.policy.action_dist.get_metrics(
                actions,
                action_splitter=self.metrics_action_splitters,
            ).items()
        }

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
        actor_critic_lr = self._apply_actor_critic_learning_rate_for_update(global_update_idx)
        update_actor = self._should_update_actor(global_update_idx)
        nop_loss_batch = batch if nop_batch is None else nop_batch
        skip_multi_step_nop_loss = self._uses_multi_step_nop() and nop_batch is None
        self._reset_train_gsde_noise(batch.local_obs)
        actions_pi, log_prob_pi = self.policy.action_log_prob(
            num_action_samples=self.actor_action_samples,
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            hidden_local_vars=batch.hidden_local_vars,
            hidden_global_vars=batch.hidden_global_vars,
            agent_mask=batch.agent_mask,
            **({} if batch.scenario_ids is None else {"scenario_ids": batch.scenario_ids}),
            previous_actions=batch.previous_actions,
            deterministic=False,
        )
        actor_action_dist_losses, actor_action_dist_metrics = self._compute_actor_action_dist_extra_losses(
            agent_mask=batch.agent_mask,
        )
        reduced_actor_action_dist_losses = self._reduce_actor_action_dist_extra_losses(
            batch=batch,
            extra_losses=actor_action_dist_losses,
        )
        log_prob_pi_mean = self._mean_agent_log_probs(log_prob_pi, batch.agent_mask)
        ent_coef, ent_coef_loss = self._update_entropy_coefficient(
            log_prob_mean=self._mean_action_samples(log_prob_pi_mean, self.actor_action_samples),
            batch=batch,
            update=update_actor,
        )
        target_entropy = self._target_entropy(
            batch=batch,
            dtype=log_prob_pi_mean.dtype,
            device=log_prob_pi_mean.device,
        )

        with torch.no_grad():
            (
                bootstrap_next_local_obs,
                bootstrap_next_global_obs,
                bootstrap_next_hidden_local_vars,
                bootstrap_next_hidden_global_vars,
                bootstrap_next_agent_mask,
            ) = self._tensor_operations.mask_terminal_observations(
                batch.terminal_mask,
                batch.next_local_obs,
                batch.next_global_obs,
                batch.next_hidden_local_vars,
                batch.next_hidden_global_vars,
                batch.next_agent_mask,
            )
            self._reset_train_gsde_noise(bootstrap_next_local_obs)
            target_q = self._target_forward_phase(
                batch,
                bootstrap_next_local_obs,
                bootstrap_next_global_obs,
                bootstrap_next_hidden_local_vars,
                bootstrap_next_hidden_global_vars,
                bootstrap_next_agent_mask,
                ent_coef,
            )

        current_q1, current_q2, critic_nop_latents, critic_loss = self._critic_forward_phase(
            batch,
            target_q,
        )
        if skip_multi_step_nop_loss:
            critic_nop_loss, critic_nop_metrics = None, {"nop_loss_skipped": 1.0}
        elif reuse_critic_nop_latents and critic_nop_latents is not None:
            critic_nop_loss, critic_nop_metrics = self.policy.compute_critic_nop_loss(
                nop_loss_batch,
                source_latents=critic_nop_latents,
            )
        else:
            critic_nop_loss, critic_nop_metrics = self.policy.compute_critic_nop_loss(nop_loss_batch)
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
            critic_parameters = self.policy.critic_parameters()
            self._set_requires_grad(critic_parameters, False)
            try:
                q1_pi, q2_pi, actor_loss = self._actor_forward_phase(
                    batch,
                    actions_pi,
                    log_prob_pi_mean,
                    ent_coef,
                )
            finally:
                self._set_requires_grad(critic_parameters, True)

            if skip_multi_step_nop_loss:
                actor_nop_loss, actor_nop_metrics = None, {}
            else:
                actor_nop_loss, actor_nop_metrics = self.policy.compute_actor_nop_loss(nop_loss_batch)
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

    def _target_forward_phase(
        self,
        batch: OffPolicyReplayBatch,
        next_local_obs: torch.Tensor,
        next_global_obs: torch.Tensor,
        next_hidden_local_vars: torch.Tensor,
        next_hidden_global_vars: torch.Tensor,
        next_agent_mask: torch.Tensor | None,
        ent_coef: torch.Tensor,
    ) -> torch.Tensor:
        next_actions, next_log_probs = self.policy.action_log_prob(
            num_action_samples=self.target_action_samples,
            local_obs=next_local_obs,
            global_obs=next_global_obs,
            hidden_local_vars=next_hidden_local_vars,
            hidden_global_vars=next_hidden_global_vars,
            agent_mask=next_agent_mask,
            **({} if batch.next_scenario_ids is None else {"scenario_ids": batch.next_scenario_ids}),
            previous_actions=batch.actions,
            deterministic=False,
            use_rsample=False,
        )
        next_log_prob_mean = self._mean_agent_log_probs(next_log_probs, next_agent_mask)
        next_actions = self._smooth_target_actions(next_actions, next_agent_mask)
        target_q1, target_q2 = self.policy.q_values_samples(
            num_action_samples=self.target_action_samples,
            target=True,
            local_obs=next_local_obs,
            global_obs=next_global_obs,
            hidden_local_vars=next_hidden_local_vars,
            hidden_global_vars=next_hidden_global_vars,
            agent_mask=next_agent_mask,
            **({} if batch.next_scenario_ids is None else {"scenario_ids": batch.next_scenario_ids}),
            actions=next_actions,
        )
        target_q = self._tensor_operations.bellman_target(
            batch.rewards,
            batch.terminal_mask,
            target_q1,
            target_q2,
            next_log_prob_mean,
            ent_coef,
            self.gamma,
        )
        return self._mean_action_samples(target_q, self.target_action_samples)

    def _critic_forward_phase(
        self,
        batch: OffPolicyReplayBatch,
        target_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        current_q1, current_q2, critic_nop_latents = self.policy.q_values_with_nop_latents(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            hidden_local_vars=batch.hidden_local_vars,
            hidden_global_vars=batch.hidden_global_vars,
            agent_mask=batch.agent_mask,
            **({} if batch.scenario_ids is None else {"scenario_ids": batch.scenario_ids}),
            actions=batch.actions,
        )
        critic_loss = self._tensor_operations.critic_loss(current_q1, current_q2, target_q)
        return current_q1, current_q2, critic_nop_latents, critic_loss

    def _actor_forward_phase(
        self,
        batch: OffPolicyReplayBatch,
        actions: torch.Tensor,
        log_prob_mean: torch.Tensor,
        ent_coef: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1, q2 = self.policy.q_values_samples(
            num_action_samples=self.actor_action_samples,
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            hidden_local_vars=batch.hidden_local_vars,
            hidden_global_vars=batch.hidden_global_vars,
            agent_mask=batch.agent_mask,
            **({} if batch.scenario_ids is None else {"scenario_ids": batch.scenario_ids}),
            actions=actions,
        )
        actor_loss = self._tensor_operations.actor_loss(q1, q2, log_prob_mean, ent_coef)
        return q1, q2, actor_loss

    @staticmethod
    def _mean_action_samples(value: torch.Tensor, count: int) -> torch.Tensor:
        return value.mean(dim=0) if count > 1 and value.ndim > 0 else value

    def _setup_entropy_coefficient(self) -> None:
        if isinstance(self.ent_coef, str):
            init_value = self._parse_auto_value(
                self.ent_coef,
                parameter_name="ent_coef",
                suffix_name="initial_value",
                default=1.0,
            )
            if init_value <= 0:
                raise ValueError(f"Initial entropy coefficient must be > 0, got {init_value}")
            self.log_ent_coef = torch.log(
                torch.ones((), device=self.train_device, dtype=torch.float32) * init_value
            ).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam(
                [self.log_ent_coef],
                lr=self._resolved_ent_coef_learning_rate(),
            )
            return

        ent_coef = float(self.ent_coef)
        if ent_coef < 0:
            raise ValueError(f"ent_coef must be >= 0, got {ent_coef}")
        self.ent_coef_tensor = torch.tensor(ent_coef, device=self.train_device, dtype=torch.float32)

    def _update_entropy_coefficient(
        self,
        *,
        log_prob_mean: torch.Tensor,
        batch: OffPolicyReplayBatch,
        update: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.log_ent_coef is None:
            assert self.ent_coef_tensor is not None
            return self.ent_coef_tensor, None

        if not update:
            return self.log_ent_coef.detach().exp(), None

        target_entropy = self._target_entropy(batch=batch, dtype=log_prob_mean.dtype, device=log_prob_mean.device)
        ent_coef_loss = self._tensor_operations.entropy_coefficient_loss(
            self.log_ent_coef,
            log_prob_mean,
            target_entropy,
        )
        assert self.ent_coef_optimizer is not None
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        assert self._ent_coef_optimizer_step is not None
        self._ent_coef_optimizer_step()
        return self.log_ent_coef.detach().exp(), ent_coef_loss.detach()

    def _target_entropy(
        self,
        *,
        batch: OffPolicyReplayBatch,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(self.target_entropy, str):
            auto_scale = self._target_entropy_auto_scale()
            target_entropy_per_agent = -auto_scale * float(self.agent_action_dim)
            return torch.full(batch.rewards.shape, target_entropy_per_agent, dtype=dtype, device=device)

        return torch.full(batch.rewards.shape, float(self.target_entropy), dtype=dtype, device=device)

    def _target_entropy_auto_scale(self) -> float:
        assert isinstance(self.target_entropy, str)
        scale = self._parse_auto_value(
            self.target_entropy,
            parameter_name="target_entropy",
            suffix_name="scale",
            default=1.0,
        )
        if scale <= 0.0:
            raise ValueError(f"target_entropy auto scale must be > 0, got {scale}")
        return scale

    @staticmethod
    def _parse_auto_value(
        value: str,
        *,
        parameter_name: str,
        suffix_name: str,
        default: float,
    ) -> float:
        normalized_value = value.lower()
        if normalized_value == "auto":
            return default
        for separator in ("_", "*"):
            prefix = f"auto{separator}"
            if normalized_value.startswith(prefix):
                return float(normalized_value.removeprefix(prefix))
        raise ValueError(f"{parameter_name} string must be 'auto', 'auto_{suffix_name}', or 'auto*{suffix_name}'")

    def _mean_agent_log_probs(
        self,
        log_probs: torch.Tensor,
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._tensor_operations.mean_agent_log_probs(log_probs, agent_mask)

    def _compute_actor_action_dist_extra_losses(
        self,
        *,
        agent_mask: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        action_dist = getattr(self.policy, "action_dist", None)
        if self.actor_action_samples > 1:
            agent_mask = expand_action_samples(agent_mask, self.actor_action_samples)
        compute_extra_losses_without_metrics = getattr(action_dist, "compute_extra_losses_without_metrics", None)
        if callable(compute_extra_losses_without_metrics):
            losses, metrics = compute_extra_losses_without_metrics(agent_mask=agent_mask), {}
        else:
            compute_extra_losses = getattr(action_dist, "compute_extra_losses", None)
            if not callable(compute_extra_losses):
                return {}, {}
            losses, metrics = compute_extra_losses(agent_mask=agent_mask)
        return {
            name: self._mean_action_samples(value, self.actor_action_samples) for name, value in losses.items()
        }, metrics

    def _reduce_actor_action_dist_extra_losses(
        self,
        *,
        batch: OffPolicyReplayBatch,
        extra_losses: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            name: self._reduce_actor_action_dist_extra_loss(batch=batch, value=value)
            for name, value in extra_losses.items()
        }

    def _reduce_actor_action_dist_extra_loss(
        self,
        *,
        batch: OffPolicyReplayBatch,
        value: torch.Tensor,
    ) -> torch.Tensor:
        if value.ndim == 0:
            return value
        if value.shape == batch.rewards.shape:
            return value.mean()

        expected_agent_shape = tuple(batch.local_obs.shape[:2])
        if tuple(value.shape) != expected_agent_shape:
            raise ValueError(
                f"Expected SAC actor action-dist extra loss shape {expected_agent_shape} or "
                f"{tuple(batch.rewards.shape)}, got {tuple(value.shape)}"
            )
        if batch.agent_mask is None:
            return value.sum(dim=1).mean()
        return (value * batch.agent_mask.to(dtype=value.dtype)).sum(dim=1).mean()

    def _reset_train_gsde_noise(self, local_obs: torch.Tensor) -> None:
        if not self.policy.gsde_enabled:
            return
        self.policy.action_dist.reset_temporal_correlations_on_step(batch_shape=tuple(local_obs.shape[:-1]))

    def _move_entropy_tensors_to_train_device(self) -> None:
        optimizer_replaced = False
        if self.log_ent_coef is not None and self.log_ent_coef.device != self.train_device:
            self.log_ent_coef = self.log_ent_coef.detach().to(self.train_device).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam(
                [self.log_ent_coef],
                lr=self._resolved_ent_coef_learning_rate(),
            )
            optimizer_replaced = True
        if self.ent_coef_tensor is not None and self.ent_coef_tensor.device != self.train_device:
            self.ent_coef_tensor = self.ent_coef_tensor.to(self.train_device)
        if optimizer_replaced:
            self._rebuild_optimizer_steps()

    def _get_optimizer_state_dict(self) -> dict[str, Any]:
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "entropy_agent_reduction": ENTROPY_AGENT_REDUCTION,
            "ent_coef_optimizer": None if self.ent_coef_optimizer is None else self.ent_coef_optimizer.state_dict(),
            "log_ent_coef": None if self.log_ent_coef is None else self.log_ent_coef.detach(),
            "ent_coef_tensor": None if self.ent_coef_tensor is None else self.ent_coef_tensor.detach(),
        }

    def _apply_optimizer_state_dict(
        self,
        state_dict: dict[str, Any],
        missing_keys: list[str],
        unexpected_keys: list[str],
    ) -> None:
        if missing_keys:
            raise NotImplementedError("Cannot safely load SAC optimizer state when policy keys are missing.")
        if unexpected_keys:
            logger.warning(f"Loading SAC optimizer state with unexpected policy keys: {unexpected_keys}")
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        critic_optimizer_state = state_dict["critic_optimizer"]
        saved_critic_parameter_count = sum(
            len(param_group["params"]) for param_group in critic_optimizer_state["param_groups"]
        )
        current_critic_parameter_count = sum(
            len(param_group["params"]) for param_group in self.critic_optimizer.param_groups
        )
        if saved_critic_parameter_count == current_critic_parameter_count:
            self.critic_optimizer.load_state_dict(critic_optimizer_state)
        else:
            logger.warning(
                "Critic optimizer parameter count changed "
                f"({saved_critic_parameter_count} -> {current_critic_parameter_count}); "
                "reinitializing critic optimizer state."
            )
            self.critic_optimizer = torch.optim.Adam(
                self.policy.critic_parameters(),
                lr=self._actor_critic_learning_rate_for_update(self.n_total_updates),
            )
        saved_entropy_agent_reduction = state_dict.get("entropy_agent_reduction")
        self.log_ent_coef = None
        self.ent_coef_optimizer = None
        self.ent_coef_tensor = None
        if saved_entropy_agent_reduction != ENTROPY_AGENT_REDUCTION:
            logger.warning(
                "Loaded a SAC checkpoint without compatible mean-agent entropy semantics; "
                "resetting the entropy coefficient from the current configuration."
            )
            self._setup_entropy_coefficient()
        else:
            log_ent_coef = state_dict.get("log_ent_coef", None)
            ent_coef_tensor = state_dict.get("ent_coef_tensor", None)
            if log_ent_coef is not None and ent_coef_tensor is not None:
                raise ValueError("Invalid SAC optimizer state: both log_ent_coef and ent_coef_tensor are set.")
            if log_ent_coef is not None:
                self.log_ent_coef = log_ent_coef.to(self.train_device).detach().requires_grad_(True)
                self.ent_coef_optimizer = torch.optim.Adam(
                    [self.log_ent_coef],
                    lr=self._resolved_ent_coef_learning_rate(),
                )
                ent_state = state_dict.get("ent_coef_optimizer", None)
                if ent_state is not None:
                    self.ent_coef_optimizer.load_state_dict(ent_state)
                self._apply_entropy_coefficient_learning_rate()
            if ent_coef_tensor is not None:
                self.ent_coef_tensor = ent_coef_tensor.to(self.train_device).detach()
        self._move_optimizer_state_to_device(self.actor_optimizer, self.train_device)
        self._move_optimizer_state_to_device(self.critic_optimizer, self.train_device)
        if self.ent_coef_optimizer is not None:
            self._move_optimizer_state_to_device(self.ent_coef_optimizer, self.train_device)
        self._rebuild_optimizer_steps()

    def _apply_learning_rate(self, lr: LearningRate) -> None:
        assert isinstance(lr, float)
        actor_critic_lr = self._actor_critic_learning_rate_for_update(self.n_total_updates)
        for optimizer in (self.actor_optimizer, self.critic_optimizer):
            for param_group in optimizer.param_groups:
                param_group["lr"] = actor_critic_lr
        self._apply_entropy_coefficient_learning_rate()

    def _apply_entropy_coefficient_learning_rate(self) -> None:
        if self.ent_coef_optimizer is None:
            return
        learning_rate = self._resolved_ent_coef_learning_rate()
        for param_group in self.ent_coef_optimizer.param_groups:
            param_group["lr"] = learning_rate

    def _resolved_ent_coef_learning_rate(self) -> float:
        if self.ent_coef_learning_rate is None:
            return self.learning_rate
        return self.ent_coef_learning_rate

    def _validate_hyper_parameters(self) -> None:
        super()._validate_hyper_parameters()
        validate_action_sampling(
            actor_action_samples=self.actor_action_samples,
            target_action_samples=self.target_action_samples,
        )
        if max(self.actor_action_samples, self.target_action_samples) > 1:
            if self.policy.gsde_enabled:
                raise ValueError("Multi-sample SAC does not support temporally correlated gSDE noise.")
        if self.ent_coef_learning_rate is not None and self.ent_coef_learning_rate <= 0.0:
            raise ValueError(f"ent_coef_learning_rate must be > 0 when set, got {self.ent_coef_learning_rate}")

    def _setup_extra_optimizers(self) -> None:
        self.log_ent_coef: torch.Tensor | None = None
        self.ent_coef_tensor: torch.Tensor | None = None
        self.ent_coef_optimizer: torch.optim.Optimizer | None = None
        self._setup_entropy_coefficient()

    def _rebuild_optimizer_steps(self) -> None:
        super()._rebuild_optimizer_steps()
        self._ent_coef_optimizer_step = (
            None
            if self.ent_coef_optimizer is None
            else build_optimizer_step(
                self.ent_coef_optimizer,
                compile_step=self.sac_compile_optimizer_steps,
                compile_mode=self.sac_compile_mode,
            )
        )

    def _move_extra_training_tensors_to_device(self) -> None:
        self._move_entropy_tensors_to_train_device()

    def _execute_command(self, cmd: str, params: str, extra_run_metadata: dict[str, Any] | None) -> bool:
        if cmd in {"set_ent_coef", "ent_coef"}:
            ent_coef = float(params)
            if ent_coef < 0:
                raise ValueError(f"ent_coef must be >= 0, got {ent_coef}")
            self.ent_coef = ent_coef
            self.log_ent_coef = None
            self.ent_coef_optimizer = None
            self._ent_coef_optimizer_step = None
            self.ent_coef_tensor = torch.tensor(ent_coef, device=self.train_device, dtype=torch.float32)
            logger.warning(f"Setting fixed ent_coef to {ent_coef}")
            return True
        return super()._execute_command(cmd, params, extra_run_metadata)
