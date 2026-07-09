from typing import Any

import torch
from loguru import logger
from torch.nn import functional as F

from swarmbots.learn.algos.base_algorithm import BaseAlgorithm, LearningRate
from swarmbots.learn.algos.off_policy import (
    OffPolicyReplayBuffer,
    OffPolicyRolloutState,
    collect_off_policy_steps,
    warmup_off_policy_steps,
)
from swarmbots.learn.algos.off_policy.replay_buffer import (
    NoEpisodeSegmentCandidatesError,
    OffPolicyReplayBatch,
    OffPolicyReplayEpisodeSegmentBatch,
)
from swarmbots.learn.algos.sac.base_sac_policy import BaseSACPolicy
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage
from swarmbots.learn.gsde_reset import GSDEResetMode, GSDEIntervalResetMode, GSDEProbabilityResetMode
from swarmbots.learn.metrics_logger import SUPPRESS_MISSING_CONSOLE_KEY_WARNINGS
from swarmbots.learn.metrics_list import MetricsLists
from swarmbots.learn.performance_timer import PerformanceTimer
from swarmbots.learn.summary_statistics import compute_summary_statistics
from swarmbots.learn.torch_device import as_device


class SAC(BaseAlgorithm):
    policy: BaseSACPolicy
    learning_rate: float

    def __init__(
            self,
            policy: BaseSACPolicy,
            env: BaseLearnEnvWrapper,
            learning_rate: float = 3e-4,
            buffer_capacity_per_env: int = 1_000_000,
            learning_starts: int = 10_000,
            batch_size: int = 256,
            rollout_steps_per_iteration: int | None = None,
            rollout_warmup_steps_per_env: int = 0,
            gradient_steps: int = 1,
            gamma: float = 0.99,
            tau: float = 0.005,
            ent_coef: float | str = "auto",
            target_entropy: float | str = "auto",
            target_update_interval: int = 1,
            max_grad_norm: float | None = 2.0,
            nop_steps: int = 4,
            nop_batch_size: int | None = None,
            gsde_reset_mode: GSDEResetMode | None = None,
            train_device: str | torch.device = "auto",
            rollout_device: str | torch.device = "cpu",
            record_device: str | torch.device | None = None,
            replay_storage_device: str | torch.device = "cuda",
            replay_storage_pin_memory: bool = False,
    ) -> None:
        if not isinstance(learning_rate, float):
            raise TypeError(f"learning_rate must be a float, got {type(learning_rate).__name__}")
        super().__init__(policy, env, learning_rate)

        self.learning_rate = learning_rate
        self.buffer_capacity_per_env = int(buffer_capacity_per_env)
        self.learning_starts = int(learning_starts)
        self.batch_size = int(batch_size)
        self.rollout_steps_per_iteration = (
            env.action_space.n_envs if rollout_steps_per_iteration is None else int(rollout_steps_per_iteration)
        )
        self.rollout_warmup_steps_per_env = int(rollout_warmup_steps_per_env)
        self._rollout_warmup_done = self.rollout_warmup_steps_per_env == 0
        self.gradient_steps = int(gradient_steps)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.ent_coef = ent_coef
        self.target_entropy = target_entropy
        self.target_update_interval = int(target_update_interval)
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.nop_steps = int(nop_steps)
        self.nop_batch_size = self.batch_size if nop_batch_size is None else int(nop_batch_size)
        self.gsde_reset_mode = gsde_reset_mode
        self.train_device = as_device(train_device)
        self.rollout_device = as_device(rollout_device)
        self.record_device = self.rollout_device if record_device is None else as_device(record_device)
        self.replay_storage_device = as_device(replay_storage_device)
        self.replay_storage_pin_memory = bool(replay_storage_pin_memory)
        self.agent_action_dim = int(env.action_space.total_agent_action_dim)
        self._rollout_state: OffPolicyRolloutState | None = None

        self._validate_hyper_parameters()
        self.replay_buffer = OffPolicyReplayBuffer(
            capacity_per_env=self.buffer_capacity_per_env,
            observation_space=env.observation_space,
            action_space=env.action_space,
            store_previous_actions=policy.requires_previous_actions(),
            storage_device=self.replay_storage_device,
            storage_dtype=torch.float32,
            storage_pin_memory=self.replay_storage_pin_memory,
            train_device=self.train_device,
            train_dtype=torch.float32,
        )

        self.policy.to(self.train_device)
        self.actor_optimizer = torch.optim.Adam(self.policy.actor_parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic_parameters(), lr=self.learning_rate)
        self.log_ent_coef: torch.Tensor | None = None
        self.ent_coef_tensor: torch.Tensor | None = None
        self.ent_coef_optimizer: torch.optim.Optimizer | None = None
        self._setup_entropy_coefficient()
        self._policy_num_params = self.policy.num_parameters(learnable_only=False)
        self._policy_num_trainable_params = self.policy.num_parameters()

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "buffer_capacity_per_env": self.buffer_capacity_per_env,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "rollout_steps_per_iteration": self.rollout_steps_per_iteration,
            "rollout_warmup_steps_per_env": self.rollout_warmup_steps_per_env,
            "gradient_steps": self.gradient_steps,
            "gamma": self.gamma,
            "tau": self.tau,
            "ent_coef": self.ent_coef,
            "target_entropy": self.target_entropy,
            "target_update_interval": self.target_update_interval,
            "max_grad_norm": self.max_grad_norm,
            "nop_steps": self.nop_steps,
            "nop_batch_size": self.nop_batch_size,
            "gsde_reset_mode": self._serialize_gsde_reset_mode(self.gsde_reset_mode),
            "train_device": str(self.train_device),
            "rollout_device": str(self.rollout_device),
            "record_device": str(self.record_device),
            "replay_storage_device": str(self.replay_storage_device),
            "replay_storage_pin_memory": self.replay_storage_pin_memory,
            "policy_num_params": self._policy_num_params,
            "policy_num_trainable_params": self._policy_num_trainable_params,
        }

    def _before_learn_loop(self) -> None:
        if self._rollout_warmup_done:
            return
        if (
                self.n_total_timesteps > 0
                or self.n_total_iterations > 0
                or self.n_total_updates > 0
                or self._rollout_state is not None
                or len(self.replay_buffer) > 0
        ):
            self._rollout_warmup_done = True
            return

        warmup_transitions = self.rollout_warmup_steps_per_env * self.replay_buffer.n_envs
        logger.info(
            f"Running SAC rollout warmup for {self.rollout_warmup_steps_per_env} vector steps "
            f"({warmup_transitions} transitions) without training or replay writes."
        )
        random_actions = self.n_total_timesteps < self.learning_starts
        with PerformanceTimer() as warmup_timer:
            self._rollout_state = warmup_off_policy_steps(
                env=self.env,
                replay_buffer=self.replay_buffer,
                n_steps=warmup_transitions,
                policy=None if random_actions else self.policy,
                rollout_state=None,
                random_actions=random_actions,
                deterministic=False,
                gsde_reset_mode=self.gsde_reset_mode,
                rollout_device=self.rollout_device,
            )
        self._rollout_warmup_done = True
        logger.info(f"Finished SAC rollout warmup in {warmup_timer.get_duration():.2f}s.")

    def perform_iteration(
            self,
            episode_return_ema: ExponentialMovingAverage,
            episode_success_rate_ema: ExponentialMovingAverage,
            update_ema: bool,
    ) -> tuple[dict[str, Any], int]:
        random_actions = self.n_total_timesteps < self.learning_starts
        if self.policy.gsde_enabled and self._rollout_state is not None:
            self._rollout_state.gsde_noise_initialized = False
        with PerformanceTimer() as rollout_timer:
            episode_infos, rollout_metrics, self._rollout_state = collect_off_policy_steps(
                env=self.env,
                replay_buffer=self.replay_buffer,
                n_steps=self.rollout_steps_per_iteration,
                policy=None if random_actions else self.policy,
                rollout_state=self._rollout_state,
                random_actions=random_actions,
                deterministic=False,
                gsde_reset_mode=self.gsde_reset_mode,
                rollout_device=self.rollout_device,
            )

        rollout_steps = int(rollout_metrics.get("transitions_collected", self.rollout_steps_per_iteration))
        self.n_total_timesteps += rollout_steps
        self.n_total_iterations += 1

        if update_ema:
            for ep_info in episode_infos:
                if "r" in ep_info:
                    episode_return_ema.update(ep_info["r"])
                if "success" in ep_info:
                    episode_success_rate_ema.update(float(ep_info["success"]))

        train_metrics: dict[str, Any]
        if self._should_train():
            train_metrics = self.train(gradient_steps=self._resolved_gradient_steps(rollout_steps))
        else:
            train_metrics = {
                "updates": 0,
                "total_updates": self.n_total_updates,
                "replay_size": len(self.replay_buffer),
                "random_actions": random_actions,
                "training_skipped": True,
                SUPPRESS_MISSING_CONSOLE_KEY_WARNINGS: True,
            }

        metrics = {
            **train_metrics,
            **rollout_metrics,
            **self._episode_metrics(episode_infos),
            "rollout_time": rollout_timer.get_duration(),
            "random_actions": random_actions,
            "replay_size": len(self.replay_buffer),
        }
        return metrics, rollout_steps

    def train(self, *, gradient_steps: int) -> dict[str, Any]:
        if gradient_steps <= 0:
            return {"updates": 0, "total_updates": self.n_total_updates}

        with PerformanceTimer() as to_train_device_timer:
            self.policy.train()
            self.policy.to(self.train_device)
            self._move_entropy_tensors_to_train_device()

        loss_metrics = MetricsLists[float]()
        actor_grad_norms: list[float] = []
        critic_grad_norms: list[float] = []
        update_timings: list[float] = []
        sample_timings: list[float] = []
        sample_timer = PerformanceTimer()
        update_timer = PerformanceTimer()
        train_timer = PerformanceTimer().start()

        n_updates = 0
        for _ in range(gradient_steps):
            with sample_timer:
                batch = self.replay_buffer.sample(self.batch_size)
                nop_batch = self._sample_nop_batch()
            sample_timings.append(sample_timer.get_duration())

            global_update_idx = self.n_total_updates + n_updates
            with update_timer:
                step_metrics, actor_grad_norm, critic_grad_norm = self._train_step(
                    batch,
                    nop_batch=nop_batch,
                    global_update_idx=global_update_idx,
                )
            update_timings.append(update_timer.get_duration())
            loss_metrics.add(step_metrics)
            actor_grad_norms.append(actor_grad_norm)
            critic_grad_norms.append(critic_grad_norm)
            n_updates += 1

        train_timer.stop()
        self.n_total_updates += n_updates

        with PerformanceTimer() as metrics_timer:
            metrics: dict[str, Any] = {
                **loss_metrics.compute_summary_statistics(),
                "updates": n_updates,
                "total_updates": self.n_total_updates,
                "replay_size": len(self.replay_buffer),
                "actor_grad_norm": compute_summary_statistics(actor_grad_norms, find_max=True, find_min=True),
                "critic_grad_norm": compute_summary_statistics(critic_grad_norms, find_max=True, find_min=True),
            }
            sampled_actions = batch.actions
            if batch.agent_mask is not None:
                sampled_actions = sampled_actions[batch.agent_mask]
            metrics.update(self.policy.action_dist.get_metrics(sampled_actions))

        return {
            **metrics,
            "to_train_device_time": to_train_device_timer.get_duration(),
            "sampling_time": compute_summary_statistics(sample_timings),
            "total_sampling_time": sum(sample_timings),
            "update_time": compute_summary_statistics(update_timings),
            "total_update_time": sum(update_timings),
            "metrics_time": metrics_timer.get_duration(),
            "train_time": train_timer.get_duration(),
        }

    def _train_step(
            self,
            batch: OffPolicyReplayBatch,
            *,
            nop_batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch | None = None,
            global_update_idx: int,
    ) -> tuple[dict[str, float], float, float]:
        nop_loss_batch = batch if nop_batch is None else nop_batch
        skip_multi_step_nop_loss = self._uses_multi_step_nop() and nop_batch is None
        self._reset_train_gsde_noise(batch.local_obs)
        actions_pi, log_prob_pi = self.policy.action_log_prob(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            hidden_local_vars=batch.hidden_local_vars,
            hidden_global_vars=batch.hidden_global_vars,
            agent_mask=batch.agent_mask,
            deterministic=False,
        )
        log_prob_pi_sum = self._reduce_agent_log_probs(log_prob_pi, batch.agent_mask)
        ent_coef, ent_coef_loss = self._update_entropy_coefficient(
            log_prob_sum=log_prob_pi_sum,
            batch=batch,
        )
        target_entropy = self._target_entropy(
            batch=batch,
            dtype=log_prob_pi_sum.dtype,
            device=log_prob_pi_sum.device,
        )

        with torch.no_grad():
            self._reset_train_gsde_noise(batch.next_local_obs)
            next_actions, next_log_probs = self.policy.action_log_prob(
                local_obs=batch.next_local_obs,
                global_obs=batch.next_global_obs,
                hidden_local_vars=batch.next_hidden_local_vars,
                hidden_global_vars=batch.next_hidden_global_vars,
                agent_mask=batch.next_agent_mask,
                deterministic=False,
                use_rsample=False,
            )
            next_log_prob_sum = self._reduce_agent_log_probs(next_log_probs, batch.next_agent_mask)
            target_q1, target_q2 = self.policy.target_q_values(
                local_obs=batch.next_local_obs,
                global_obs=batch.next_global_obs,
                hidden_local_vars=batch.next_hidden_local_vars,
                hidden_global_vars=batch.next_hidden_global_vars,
                agent_mask=batch.next_agent_mask,
                actions=next_actions,
            )
            next_q = torch.minimum(target_q1, target_q2) - ent_coef * next_log_prob_sum
            target_q = batch.rewards + (1.0 - batch.terminal_mask.to(dtype=batch.rewards.dtype)) * self.gamma * next_q

        current_q1, current_q2 = self.policy.q_values(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            hidden_local_vars=batch.hidden_local_vars,
            hidden_global_vars=batch.hidden_global_vars,
            agent_mask=batch.agent_mask,
            actions=batch.actions,
        )
        critic_loss = 0.5 * (
            F.mse_loss(current_q1, target_q)
            + F.mse_loss(current_q2, target_q)
        )
        if skip_multi_step_nop_loss:
            critic_nop_loss, critic_nop_metrics = None, {"nop_loss_skipped": 1.0}
        else:
            critic_nop_loss, critic_nop_metrics = self.policy.compute_critic_nop_loss(nop_loss_batch)
        critic_total_loss = critic_loss if critic_nop_loss is None else critic_loss + critic_nop_loss

        self.critic_optimizer.zero_grad()
        critic_total_loss.backward()
        critic_grad_norm = self._clip_grad_norm(self.policy.critic_parameters())
        self.critic_optimizer.step()

        critic_parameters = self.policy.critic_parameters()
        self._set_requires_grad(critic_parameters, False)
        try:
            q1_pi, q2_pi = self.policy.q_values(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                hidden_local_vars=batch.hidden_local_vars,
                hidden_global_vars=batch.hidden_global_vars,
                agent_mask=batch.agent_mask,
                actions=actions_pi,
            )
            actor_loss = (ent_coef * log_prob_pi_sum - torch.minimum(q1_pi, q2_pi)).mean()
        finally:
            self._set_requires_grad(critic_parameters, True)

        if skip_multi_step_nop_loss:
            actor_nop_loss, actor_nop_metrics = None, {}
        else:
            actor_nop_loss, actor_nop_metrics = self.policy.compute_actor_nop_loss(nop_loss_batch)
        actor_total_loss = actor_loss if actor_nop_loss is None else actor_loss + actor_nop_loss

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
            "log_prob": log_prob_pi_sum.mean().item(),
            "entropy": (-log_prob_pi_sum).mean().item(),
            "target_entropy": target_entropy.mean().item(),
            "ent_coef": ent_coef.item(),
        }
        if ent_coef_loss is not None:
            metrics["ent_coef_loss"] = ent_coef_loss.item()
        metrics.update(actor_nop_metrics)
        metrics.update(critic_nop_metrics)
        return metrics, actor_grad_norm, critic_grad_norm

    def _setup_entropy_coefficient(self) -> None:
        if isinstance(self.ent_coef, str):
            ent_coef_value = self.ent_coef.lower()
            if not ent_coef_value.startswith("auto"):
                raise ValueError("ent_coef string must be 'auto' or 'auto_<initial_value>'")
            init_value = 1.0
            if "_" in ent_coef_value:
                init_value = float(ent_coef_value.split("_", 1)[1])
            if init_value <= 0:
                raise ValueError(f"Initial entropy coefficient must be > 0, got {init_value}")
            self.log_ent_coef = torch.log(
                torch.ones((), device=self.train_device, dtype=torch.float32) * init_value
            ).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.learning_rate)
            return

        ent_coef = float(self.ent_coef)
        if ent_coef < 0:
            raise ValueError(f"ent_coef must be >= 0, got {ent_coef}")
        self.ent_coef_tensor = torch.tensor(ent_coef, device=self.train_device, dtype=torch.float32)

    def _update_entropy_coefficient(
            self,
            *,
            log_prob_sum: torch.Tensor,
            batch: OffPolicyReplayBatch,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.log_ent_coef is None:
            assert self.ent_coef_tensor is not None
            return self.ent_coef_tensor, None

        target_entropy = self._target_entropy(batch=batch, dtype=log_prob_sum.dtype, device=log_prob_sum.device)
        ent_coef_loss = -(self.log_ent_coef * (log_prob_sum + target_entropy).detach()).mean()
        assert self.ent_coef_optimizer is not None
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()
        return self.log_ent_coef.detach().exp(), ent_coef_loss.detach()

    def _target_entropy(
            self,
            *,
            batch: OffPolicyReplayBatch,
            dtype: torch.dtype,
            device: torch.device,
    ) -> torch.Tensor:
        if isinstance(self.target_entropy, str):
            if self.target_entropy.lower() != "auto":
                raise ValueError("target_entropy string must be 'auto'")
            if batch.agent_mask is None:
                active_agents = torch.full(
                    batch.rewards.shape,
                    float(self.env.n_agents),
                    dtype=dtype,
                    device=device,
                )
            else:
                active_agents = batch.agent_mask.to(dtype=dtype).sum(dim=1)
            return -float(self.agent_action_dim) * active_agents

        return torch.full(batch.rewards.shape, float(self.target_entropy), dtype=dtype, device=device)

    @staticmethod
    def _reduce_agent_log_probs(log_probs: torch.Tensor, agent_mask: torch.Tensor | None) -> torch.Tensor:
        if agent_mask is None:
            return log_probs.sum(dim=1)
        return (log_probs * agent_mask.to(dtype=log_probs.dtype)).sum(dim=1)

    def _clip_grad_norm(self, parameters: list[torch.nn.Parameter]) -> float:
        if self.max_grad_norm is None:
            return self.policy._grad_norm_from_parameters(parameters)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        return float(grad_norm)

    def _should_train(self) -> bool:
        return (
            self.n_total_timesteps >= self.learning_starts
            and len(self.replay_buffer) >= self.batch_size
        )

    def _sample_nop_batch(self) -> OffPolicyReplayEpisodeSegmentBatch | None:
        if not self._uses_multi_step_nop():
            return None
        try:
            return self.replay_buffer.sample_episode_segments(
                self.nop_batch_size,
                segment_length=self.nop_steps,
                require_initial_temporal_state=False,
            )
        except NoEpisodeSegmentCandidatesError:
            return None

    def _uses_multi_step_nop(self) -> bool:
        return self.nop_steps > 1 and self.policy.has_nop_loss()

    def _resolved_gradient_steps(self, rollout_steps: int) -> int:
        if self.gradient_steps == -1:
            return rollout_steps
        return self.gradient_steps

    def _validate_hyper_parameters(self) -> None:
        if self.buffer_capacity_per_env <= 0:
            raise ValueError(f"buffer_capacity_per_env must be > 0, got {self.buffer_capacity_per_env}")
        if self.learning_starts < 0:
            raise ValueError(f"learning_starts must be >= 0, got {self.learning_starts}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.rollout_steps_per_iteration <= 0:
            raise ValueError(
                f"rollout_steps_per_iteration must be > 0, got {self.rollout_steps_per_iteration}"
            )
        if self.rollout_steps_per_iteration % self.env.action_space.n_envs != 0:
            raise ValueError(
                f"rollout_steps_per_iteration must be a multiple of n_envs ({self.env.action_space.n_envs}), "
                f"got {self.rollout_steps_per_iteration}"
            )
        if self.rollout_warmup_steps_per_env < 0:
            raise ValueError(
                f"rollout_warmup_steps_per_env must be >= 0, got {self.rollout_warmup_steps_per_env}"
            )
        if self.gradient_steps == 0 or self.gradient_steps < -1:
            raise ValueError(f"gradient_steps must be -1 or > 0, got {self.gradient_steps}")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
        if self.target_update_interval <= 0:
            raise ValueError(f"target_update_interval must be > 0, got {self.target_update_interval}")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be > 0 when set, got {self.max_grad_norm}")
        if self.nop_steps <= 0:
            raise ValueError(f"nop_steps must be > 0, got {self.nop_steps}")
        if self.nop_batch_size <= 0:
            raise ValueError(f"nop_batch_size must be > 0, got {self.nop_batch_size}")
        if self.policy.gsde_enabled:
            if self.gsde_reset_mode is None:
                raise ValueError("gsde_reset_mode is required when the SAC policy uses GSDEConfig.")
            if isinstance(self.gsde_reset_mode, GSDEIntervalResetMode):
                if self.gsde_reset_mode.interval <= 0:
                    raise ValueError(
                        f"GSDEIntervalResetMode.interval must be > 0, got {self.gsde_reset_mode.interval}"
                    )
            elif isinstance(self.gsde_reset_mode, GSDEProbabilityResetMode):
                if not (0.0 < self.gsde_reset_mode.probability < 1.0):
                    raise ValueError(
                        "GSDEProbabilityResetMode.probability must be in (0, 1), "
                        f"got {self.gsde_reset_mode.probability}"
                    )
            else:
                raise TypeError(f"Unknown gsde_reset_mode type: {type(self.gsde_reset_mode)}")

    @staticmethod
    def _set_requires_grad(parameters: list[torch.nn.Parameter], value: bool) -> None:
        for parameter in parameters:
            parameter.requires_grad_(value)

    def _reset_train_gsde_noise(self, local_obs: torch.Tensor) -> None:
        if not self.policy.gsde_enabled:
            return
        self.policy.action_dist.reset_temporal_correlations_on_step(batch_shape=tuple(local_obs.shape[:-1]))

    @staticmethod
    def _episode_metrics(episode_infos: list[dict[str, Any]]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        key_map = {
            "r": "ep_rew",
            "l": "ep_len",
            "t": "ep_time",
            "progress_reward": "ep_progress_rew",
            "guidance_reward": "ep_guidance_rew",
        }
        for info_key, metric_key in key_map.items():
            values = [ep_info[info_key] for ep_info in episode_infos if info_key in ep_info]
            metrics[metric_key] = compute_summary_statistics(
                values,
                find_min=info_key in {"r", "l", "progress_reward", "guidance_reward"},
                find_max=info_key in {"r", "l", "progress_reward", "guidance_reward"},
                make_histogram=30 if info_key in {"r", "l", "progress_reward", "guidance_reward"} else False,
            )
        success_values = [float(ep_info["success"]) for ep_info in episode_infos if "success" in ep_info]
        if success_values:
            metrics["ep_success_rate"] = 100.0 * (sum(success_values) / len(success_values))
        return metrics

    def _move_entropy_tensors_to_train_device(self) -> None:
        if self.log_ent_coef is not None and self.log_ent_coef.device != self.train_device:
            self.log_ent_coef = self.log_ent_coef.detach().to(self.train_device).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.learning_rate)
        if self.ent_coef_tensor is not None and self.ent_coef_tensor.device != self.train_device:
            self.ent_coef_tensor = self.ent_coef_tensor.to(self.train_device)

    @staticmethod
    def _serialize_gsde_reset_mode(mode: GSDEResetMode | None) -> dict[str, Any] | None:
        if mode is None:
            return None
        if isinstance(mode, GSDEIntervalResetMode):
            return {"type": "interval", "interval": mode.interval}
        if isinstance(mode, GSDEProbabilityResetMode):
            return {"type": "probability", "probability": mode.probability}
        return {"type": type(mode).__name__}

    def _get_optimizer_state_dict(self) -> dict[str, Any]:
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
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
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.log_ent_coef = None
        self.ent_coef_optimizer = None
        self.ent_coef_tensor = None
        log_ent_coef = state_dict.get("log_ent_coef", None)
        ent_coef_tensor = state_dict.get("ent_coef_tensor", None)
        if log_ent_coef is not None and ent_coef_tensor is not None:
            raise ValueError("Invalid SAC optimizer state: both log_ent_coef and ent_coef_tensor are set.")
        if log_ent_coef is not None:
            self.log_ent_coef = log_ent_coef.to(self.train_device).detach().requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.learning_rate)
            ent_state = state_dict.get("ent_coef_optimizer", None)
            if ent_state is not None:
                self.ent_coef_optimizer.load_state_dict(ent_state)
        if ent_coef_tensor is not None:
            self.ent_coef_tensor = ent_coef_tensor.to(self.train_device).detach()
        self._move_optimizer_state_to_device(self.actor_optimizer, self.train_device)
        self._move_optimizer_state_to_device(self.critic_optimizer, self.train_device)
        if self.ent_coef_optimizer is not None:
            self._move_optimizer_state_to_device(self.ent_coef_optimizer, self.train_device)

    def _apply_learning_rate(self, lr: LearningRate) -> None:
        assert isinstance(lr, float)
        for optimizer in (self.actor_optimizer, self.critic_optimizer, self.ent_coef_optimizer):
            if optimizer is None:
                continue
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    @staticmethod
    def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and value.device != device:
                    state[key] = value.to(device)

    def _execute_command(
            self,
            cmd: str,
            params: str,
            extra_run_metadata: dict[str, Any] | None,
    ) -> bool:
        if cmd in {"set_batch_size", "batch_size"}:
            self.batch_size = int(params)
            if self.batch_size <= 0:
                raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
            logger.warning(f"Setting batch_size to {self.batch_size}")
            return True
        if cmd == "set_gamma":
            self.gamma = float(params)
            if not (0.0 <= self.gamma <= 1.0):
                raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
            logger.warning(f"Setting gamma to {self.gamma}")
            return True
        if cmd == "set_tau":
            self.tau = float(params)
            if not (0.0 < self.tau <= 1.0):
                raise ValueError(f"tau must be in (0, 1], got {self.tau}")
            logger.warning(f"Setting tau to {self.tau}")
            return True
        if cmd in {"set_ent_coef", "ent_coef"}:
            ent_coef = float(params)
            if ent_coef < 0:
                raise ValueError(f"ent_coef must be >= 0, got {ent_coef}")
            self.ent_coef = ent_coef
            self.log_ent_coef = None
            self.ent_coef_optimizer = None
            self.ent_coef_tensor = torch.tensor(ent_coef, device=self.train_device, dtype=torch.float32)
            logger.warning(f"Setting fixed ent_coef to {ent_coef}")
            return True
        if cmd in {"set_nop_loss_coef", "set_wm_loss_coef", "wm_loss_coef"}:
            coef = float(params)
            logger.warning(f"Setting NOP loss coefficient to {coef}")
            self.policy.update_loss_weights(nop_loss_coef=coef)
            return True
        return super()._execute_command(cmd, params, extra_run_metadata)
