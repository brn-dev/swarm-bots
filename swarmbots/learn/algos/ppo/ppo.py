import inspect
from dataclasses import dataclass
from typing import Optional, Any, Literal, TypeVar, Generic, Callable

import torch
import torch.nn as nn
from loguru import logger
from prompt_toolkit.key_binding.bindings.named_commands import self_insert

from swarmbots.learn.algos.base_algorithm import BaseAlgorithm, LearningRate
from swarmbots.learn.algos.ppo.ppo_policy import BasePPOPolicy, PPOPolicy
from swarmbots.learn.algos.ppo.ppo_rollout import collect_whole_episodes
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisode, PPORolloutBuffer, PPOSampler, PPOSamples
from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage
from swarmbots.learn.gsde_reset import GSDEResetMode, GSDEIntervalResetMode, GSDEProbabilityResetMode
from swarmbots.learn.metrics_list import MetricsLists
from swarmbots.learn.performance_timer import PerformanceTimer
from swarmbots.learn.summary_statistics import compute_summary_statistics
from swarmbots.learn.torch_device import as_device

TARGET_KL_MARGIN = 1.5

AGENTS_DIM = 1

try:
    logger.level("SAVE")
except ValueError:
    logger.level("SAVE", no=21, color="<magenta>")


@dataclass(slots=True)
class AutomaticLearningRate:
    initial_lr: float

    max_lr: float = 1e-2

    max_kl: float = 0.1
    max_kl_hit_decay_factor: float | Callable[[float], float] = 0.7

    min_epochs: float = 2
    min_epochs_hit_decay_factor: float | Callable[[int], float] = 0.9

    increase_after_n_iters: int = 2
    increase_factor: float = 1.4

PPOLearningRate = float | AutomaticLearningRate


PPOSamplesType = TypeVar('PPOSamplesType', bound=PPOSamples)
PPOSamplerType = TypeVar('PPOSamplerType', bound=PPOSampler)

# based on https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
class PPO(BaseAlgorithm, Generic[PPOSamplesType, PPOSamplerType]):

    policy: BasePPOPolicy
    learning_rate: float

    def __init__(
            self,
            policy: BasePPOPolicy,
            env: BaseLearnEnvWrapper,
            learning_rate: PPOLearningRate = 3e-4,
            n_episodes_per_rollout: int = 64,
            max_episode_length: int = 1000,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: float = 0.2,
            clip_range_vf: Optional[float] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            value_loss_fn: nn.Module | None = None,
            max_grad_norm: float = 0.5,
            target_kl: Optional[float] = None,
            gsde_reset_mode: GSDEResetMode | None = None,
            agent_logprob_reduction: Optional[Literal["sum", "mean"]] = None,
            train_device: str | torch.device = "auto",
            rollout_device: str | torch.device = "cpu",
    ):
        self.automatic_lr: AutomaticLearningRate | None = None
        self._auto_lr_enabled = False
        self._auto_lr__iters_without_kl_early_stop = 0

        initial_lr: float
        if isinstance(learning_rate, AutomaticLearningRate):
            if target_kl is None:
                raise ValueError("AutomaticLearningRate requires target_kl (otherwise KL early stopping cannot happen).")
            self.automatic_lr = learning_rate
            self._auto_lr_enabled = True
            initial_lr = learning_rate.initial_lr
        else:
            if not isinstance(learning_rate, float):
                raise TypeError(f"{learning_rate=} must be a float (or AutomaticLearningRate for auto LR).")
            initial_lr = learning_rate

        super().__init__(policy, env, initial_lr)

        self.learning_rate = initial_lr
        self.n_episodes_per_rollout = n_episodes_per_rollout
        self.max_episode_length = max_episode_length
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.value_loss_fn = value_loss_fn if value_loss_fn is not None else nn.MSELoss()
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.gsde_reset_mode = gsde_reset_mode
        assert not policy.gsde_enabled or self.gsde_reset_mode is not None
        if agent_logprob_reduction not in (None, "sum", "mean"):
            raise ValueError(f"{agent_logprob_reduction=} must be 'sum', 'mean', or None")
        if agent_logprob_reduction is not None and not isinstance(policy, PPOPolicy):
            logger.warning('agent_logprob_reduction is only intended for single agent PPO')
        self.agent_logprob_reduction: Optional[Literal["sum", "mean"]] = agent_logprob_reduction

        self.train_device = as_device(train_device)
        self.rollout_device = as_device(rollout_device)

        self.rollout_buffer = PPORolloutBuffer(
            n_episodes=n_episodes_per_rollout,
            max_episode_length=max_episode_length,
            observation_space=env.observation_space,
            action_space=env.action_space,
            gamma=gamma,
            gae_lambda=gae_lambda,
            rollout_device=self.rollout_device,
            rollout_dtype=torch.float32,
            train_device=self.train_device,
            train_dtype=torch.float32,
        )

        self._policy_num_params = sum(p.numel() for p in self.policy.parameters())
        self._policy_num_trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def get_hyper_parameters(self) -> dict[str, Any]:
        auto_lr: dict[str, Any] | None
        if self.automatic_lr is None:
            auto_lr = None
        else:
            if callable(self.automatic_lr.max_kl_hit_decay_factor):
                max_kl_hit_decay_factor = self._serialize_fn(self.automatic_lr.max_kl_hit_decay_factor)
            else:
                max_kl_hit_decay_factor = self.automatic_lr.max_kl_hit_decay_factor

            if callable(self.automatic_lr.min_epochs_hit_decay_factor):
                min_epochs_hit_decay_factor = self._serialize_fn(self.automatic_lr.min_epochs_hit_decay_factor)
            else:
                min_epochs_hit_decay_factor = self.automatic_lr.min_epochs_hit_decay_factor

            auto_lr = {
                "enabled": self._auto_lr_enabled,
                "initial_lr": self.automatic_lr.initial_lr,
                "max_lr": self.automatic_lr.max_lr,
                "max_kl": self.automatic_lr.max_kl,
                "max_kl_hit_decay_factor": max_kl_hit_decay_factor,
                "min_epochs": self.automatic_lr.min_epochs,
                "min_epochs_hit_decay_factor": min_epochs_hit_decay_factor,
                "increase_after_n_iters": self.automatic_lr.increase_after_n_iters,
                "increase_factor": self.automatic_lr.increase_factor,
                "iters_without_kl_early_stop": self._auto_lr__iters_without_kl_early_stop,
            }

        return {
            'learning_rate': self.learning_rate,
            'automatic_learning_rate': auto_lr,
            'n_episodes_per_rollout': self.n_episodes_per_rollout,
            'max_episode_length': self.max_episode_length,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'clip_range_vf': self.clip_range_vf,
            'normalize_advantage': self.normalize_advantage,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef,
            'value_loss_fn': str(self.value_loss_fn),
            'max_grad_norm': self.max_grad_norm,
            'target_kl': self.target_kl,
            'train_device': str(self.train_device),
            'rollout_device': str(self.rollout_device),
            'gsde_reset_mode': self._serialize_gsde_reset_mode(self.gsde_reset_mode),
            'agent_logprob_reduction': self.agent_logprob_reduction,
            'policy_num_params': self._policy_num_params,
            'policy_num_trainable_params': self._policy_num_trainable_params,
        }

    def compute_ppo_loss(
            self,
            batch: PPOSamples,
            entropies: torch.Tensor,
            log_probs: torch.Tensor,
            values: torch.Tensor,
    ) -> tuple[torch.Tensor, float, dict[str, Any]]:
        entropy, log_prob, old_log_prob = self.reduce_agents(batch, entropies, log_probs)

        advantages = batch.advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(log_prob - old_log_prob)

        if ratio.ndim == 2 and advantages.ndim == 1:
            advantages = advantages.unsqueeze(-1)

        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        if self.clip_range_vf is None:
            values_pred = values
        else:
            values_pred = batch.values + torch.clamp(
                values - batch.values, -self.clip_range_vf, self.clip_range_vf
            )

        value_loss: torch.Tensor = self.value_loss_fn(values_pred, batch.returns)

        if entropy is None:
            entropy_loss = -torch.mean(-log_prob)
        else:
            entropy_loss = -torch.mean(entropy)

        value_loss_scaled = self.vf_coef * value_loss
        entropy_loss_scaled = self.ent_coef * entropy_loss

        loss = policy_loss + entropy_loss_scaled + value_loss_scaled

        with torch.no_grad():
            log_ratio = log_prob - old_log_prob
            approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()

        clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
        metrics = {
            'act_loss': policy_loss.item(),
            'ent_loss': entropy_loss.item(),
            'ent_loss_scaled': entropy_loss_scaled.item(),
            'val_loss': value_loss.item(),
            'val_loss_scaled': value_loss_scaled.item(),
            'approx_kl': approx_kl_div,
            'clip_frac': clip_fraction,
            'ratio': compute_summary_statistics(ratio, find_min=True, find_max=True),
        }

        return loss, approx_kl_div, metrics

    def compute_loss(
            self,
            batch: PPOSamplesType,
    ) -> tuple[torch.Tensor, float, dict[str, Any]]:
        log_probs, entropies, values = self.policy.evaluate_actions(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            actions=batch.actions
        )

        return self.compute_ppo_loss(
            batch=batch,
            entropies=entropies,
            log_probs=log_probs,
            values=values,
        )

    def perform_iteration(
            self,
            episode_return_ema: ExponentialMovingAverage
    ) -> tuple[dict[str, Any], int]:
        with PerformanceTimer() as rollout_timer:
            episodes, episode_infos, rollout_metrics = collect_whole_episodes(
                env=self.env,
                policy=self.policy,
                buffer=self.rollout_buffer,
                gsde_reset_mode=self.gsde_reset_mode,
            )

            total_steps_in_rollout = sum(len(ep.rewards) for ep in episodes)
        self.n_total_timesteps += total_steps_in_rollout
        self.n_total_iterations += 1

        ep_rew = compute_summary_statistics([ep['r'] for ep in episode_infos], find_min=True, find_max=True)
        ep_len = compute_summary_statistics([ep['l'] for ep in episode_infos], find_min=True, find_max=True)
        ep_time = compute_summary_statistics([ep['t'] for ep in episode_infos])

        episode_return_ema.update(ep_rew.mean)

        update_metrics = self.train(episodes)
        metrics = {
            **update_metrics,
            **rollout_metrics,
            'rollout_time': rollout_timer.get_duration(),
            'ep_rew': ep_rew,
            'ep_len': ep_len,
            'ep_time': ep_time,
        }
        return metrics, total_steps_in_rollout

    def _make_sampler(self, episodes: list[PPOEpisode]) -> PPOSamplerType:
        return PPOSampler[PPOSamplesType](episodes)

    def train(self, episodes: list[PPOEpisode]) -> dict[str, Any]:
        with PerformanceTimer() as to_train_device_timer:
            self.policy.train()
            self.policy.to(self.train_device)
            self.value_loss_fn.to(self.train_device)

        with PerformanceTimer() as sampler_init_timer:
            sampler = self._make_sampler(episodes)

        y_pred = sampler.values.flatten()
        y_true = sampler.returns.flatten()
        var_y = torch.var(y_true)
        if not torch.isnan(var_y) and var_y > 1e-8:
            explained_var = (1 - torch.var(y_true - y_pred) / var_y).item()
        else:
            explained_var = 0.0

        loss_metrics = MetricsLists[float]()

        continue_training = True
        early_stopped_on_kl = False
        early_stop_epoch: int | None = None
        early_stop_kl_div: float | None = None
        n_updates = 0
        grad_norms: list[float] = []
        n_grad_clipped = 0

        sampling_timings: list[float] = []
        sample_timer = PerformanceTimer()

        update_timings: list[float] = []
        update_timer = PerformanceTimer()

        train_timer = PerformanceTimer().start()
        for epoch in range(self.n_epochs):
            sample_timer.start()
            for i, batch in enumerate(sampler.sample(self.batch_size)):
                sampling_timings.append(sample_timer.stop().get_duration())

                update_timer.start()

                loss, approx_kl_div, metrics = self.compute_loss(batch)
                loss_metrics.add(metrics)

                if self.target_kl is not None and approx_kl_div > TARGET_KL_MARGIN * self.target_kl:
                    continue_training = False
                    early_stopped_on_kl = True
                    early_stop_epoch = epoch
                    early_stop_kl_div = approx_kl_div
                    msg = f"Early stopping at epoch {epoch}, batch {i} due to reaching max kl: {approx_kl_div:.3f}"
                    if epoch == 0:
                        logger.warning(msg)
                    else:
                        logger.debug(msg)
                    break

                self.optimizer.zero_grad()
                loss.backward()
                total_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                total_grad_norm_f = float(total_grad_norm)
                grad_norms.append(total_grad_norm_f)
                if total_grad_norm_f > self.max_grad_norm:
                    n_grad_clipped += 1
                self.optimizer.step()
                self._after_optimizer_step()

                n_updates += 1

                update_timings.append(update_timer.stop().get_duration())
                sample_timer.start()

            if not continue_training:
                break

        train_timer.stop()

        auto_lr_metrics = self._maybe_update_automatic_lr(early_stopped_on_kl, early_stop_epoch, early_stop_kl_div)

        self.n_total_updates += n_updates

        metrics_timer = PerformanceTimer().start()
        with torch.no_grad():
            metrics: dict[str, Any] = {
                **{k: compute_summary_statistics(v, find_max=True) for k, v in loss_metrics.get().items()},
                'updates': n_updates,
                'total_updates': self.n_total_updates,
                'expl_var': explained_var,
                'grad_norm': compute_summary_statistics(grad_norms, find_max=True) if grad_norms else 0.0,
                'grad_clip_frac': (n_grad_clipped / len(grad_norms)) if grad_norms else 0.0,
                **auto_lr_metrics,
            }

            act_dim_sum = 0
            action_dims = self.policy.action_dist.action_dims
            for i, dist in enumerate(self.policy.action_dist.distributions):
                act_dim = action_dims[i]
                actions = sampler.actions[..., act_dim_sum:act_dim_sum + act_dim]
                act_dim_sum += act_dim
                metrics[f'act{i}'] = compute_summary_statistics(actions)
                if hasattr(dist, "log_stds"):
                    metrics[f'std{i}'] = compute_summary_statistics(torch.exp(dist.log_stds), find_min=True, find_max=True)
        metrics_timer.stop()

        return {
            **metrics,
            'to_train_device_time': to_train_device_timer.get_duration(),
            'sampler_init_time': sampler_init_timer.get_duration(),
            'sampling_time': compute_summary_statistics(sampling_timings),
            'total_sampling_time': sum(sampling_timings),
            'update_time': compute_summary_statistics(update_timings),
            'total_update_time': sum(update_timings),
            'metrics_time': metrics_timer.get_duration(),
            'train_time': train_timer.get_duration(),
        }

    def _maybe_update_automatic_lr(
            self,
            early_stopped_on_kl: bool,
            early_stop_epoch: int | None,
            early_stop_kl_div: float | None,
    ) -> dict[str, Any]:
        if self.automatic_lr is None:
            return {}

        if not self._auto_lr_enabled:
            return {"auto_lr_event": "disabled", "auto_lr": self.learning_rate}

        if early_stopped_on_kl:
            if early_stop_kl_div > self.automatic_lr.max_kl:
                self._auto_lr__iters_without_kl_early_stop = 0
                if callable(self.automatic_lr.max_kl_hit_decay_factor):
                    decay_factor = self.automatic_lr.max_kl_hit_decay_factor(early_stop_kl_div)
                else:
                    decay_factor = self.automatic_lr.max_kl_hit_decay_factor
                new_lr = self.learning_rate * decay_factor
                ratio = new_lr / self.learning_rate
                logger.warning(
                    f"Decaying LR from {self.learning_rate:.2e} to {new_lr:.2e} due to "
                    f"KL early stopping at epoch {early_stop_epoch} with kl div {early_stop_kl_div:.3f} "
                    f"({ratio=:.2f})"
                )
                self.set_learning_rate(new_lr)
                return {"auto_lr_event": "max_kl_hit_decay", "auto_lr": new_lr}
            if early_stop_epoch < self.automatic_lr.min_epochs:
                self._auto_lr__iters_without_kl_early_stop = 0
                if callable(self.automatic_lr.min_epochs_hit_decay_factor):
                    decay_factor = self.automatic_lr.min_epochs_hit_decay_factor(early_stop_epoch)
                else:
                    decay_factor = self.automatic_lr.min_epochs_hit_decay_factor
                new_lr = self.learning_rate * decay_factor
                ratio = new_lr / self.learning_rate
                logger.warning(
                    f"Decaying LR from {self.learning_rate:.2e} to {new_lr:.2e} due to "
                    f"KL early stopping at epoch {early_stop_epoch} with kl div {early_stop_kl_div:.3f} "
                    f"({ratio=:.2f})"
                )
                self.set_learning_rate(new_lr)
                return {"auto_lr_event": "min_epochs_hit_decay", "auto_lr": new_lr}

        self._auto_lr__iters_without_kl_early_stop += 1
        if self._auto_lr__iters_without_kl_early_stop < self.automatic_lr.increase_after_n_iters:
            return {"auto_lr_event": None, "auto_lr": self.learning_rate}

        unclamped_lr = self.learning_rate * self.automatic_lr.increase_factor
        new_lr = min(unclamped_lr, self.automatic_lr.max_lr)
        ratio = new_lr / self.learning_rate
        if new_lr < unclamped_lr:
            logger.warning(
                f"Auto LR capped at {new_lr:.2e} (requested {unclamped_lr:.2e}, max_lr={self.automatic_lr.max_lr:.2e}, "
                f"{ratio=:.2f})"
            )
        else:
            logger.warning(
                f"Increasing LR from {self.learning_rate:.2e} to {new_lr:.2e} due to no "
                f"critical KL early stopping for {self._auto_lr__iters_without_kl_early_stop} epochs "
                f"({ratio=:.2f})"
            )
        self._auto_lr__iters_without_kl_early_stop = 0
        self.set_learning_rate(new_lr)
        event = "no_kl_early_stop_increase" if new_lr == unclamped_lr else "no_kl_early_stop_increase_capped"
        return {"auto_lr_event": event, "auto_lr": new_lr}

    def _after_optimizer_step(self) -> None:
        pass

    def reduce_agents(
            self,
            batch: PPOSamples,
            entropies: torch.Tensor | None,
            log_probs: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        if self.agent_logprob_reduction is None:
            log_prob = log_probs
            old_log_prob = batch.log_probs
            entropy = entropies
        elif self.agent_logprob_reduction == "sum":
            log_prob = log_probs.sum(dim=AGENTS_DIM)
            old_log_prob = batch.log_probs.sum(dim=AGENTS_DIM)
            entropy = entropies.sum(dim=AGENTS_DIM) if entropies is not None else None
        elif self.agent_logprob_reduction == "mean":
            log_prob = log_probs.mean(dim=AGENTS_DIM)
            old_log_prob = batch.log_probs.mean(dim=AGENTS_DIM)
            entropy = entropies.mean(dim=AGENTS_DIM) if entropies is not None else None
        else:
            raise ValueError(f"Unhandled {self.agent_logprob_reduction=}")
        return entropy, log_prob, old_log_prob

    def _execute_command(
            self,
            cmd: str,
            params: str,
            extra_run_metadata: dict[str, Any] | None
    ) -> bool:
        """
        :return: True if the command was executed successfully and updated the hyper parameters, False otherwise
        """
        if cmd == 'set_std':
            std = float(params)
            logger.warning(f"Setting action std to {std}")
            self.policy.action_dist.set_std(std)
            return True
        elif cmd == 'scale_std':
            multiplier = float(params)
            logger.warning(f"Scaling action std by {multiplier}")
            self.policy.action_dist.scale_std(multiplier)
            return True
        elif cmd == 'set_n_epochs' or cmd == 'set_num_epochs':
            n_epochs = int(params)
            logger.warning(f'Setting num epochs to {n_epochs}')
            self.n_epochs = n_epochs
            return True
        elif cmd == 'set_target_kl':
            target_kl = float(params)
            logger.warning(f'Setting target KL to {target_kl}')
            self.target_kl = target_kl
            return True
        elif cmd == "set_clip_range":
            clip_range = float(params)
            logger.warning(f"Setting clip_range to {clip_range}")
            self.clip_range = clip_range
            return True
        elif cmd == "set_clip_range_vf":
            clip_range_vf = float(params)
            logger.warning(f"Setting clip_range_vf to {clip_range_vf}")
            self.clip_range_vf = clip_range_vf
            return True
        elif cmd == "set_ent_coef":
            ent_coef = float(params)
            logger.warning(f"Setting ent_coef to {ent_coef}")
            self.ent_coef = ent_coef
            return True
        elif cmd == "set_vf_coef":
            vf_coef = float(params)
            logger.warning(f"Setting vf_coef to {vf_coef}")
            self.vf_coef = vf_coef
            return True
        elif cmd == "set_gamma":
            gamma = float(params)
            logger.warning(f"Setting gamma to {gamma}")
            self.gamma = gamma
            self.rollout_buffer.gamma = gamma
            return True
        elif cmd == "set_gae_lambda":
            gae_lambda = float(params)
            logger.warning(f"Setting gae_lambda to {gae_lambda}")
            self.gae_lambda = gae_lambda
            self.rollout_buffer.gae_lambda = gae_lambda
            return True
        elif cmd in {"disable_auto_lr", "auto_lr_off", "disable_automatic_lr", "disable_auto_learning_rate"}:
            _ = params
            if self.automatic_lr is None:
                logger.warning("Automatic LR is not configured; nothing to disable.")
                return False
            if not self._auto_lr_enabled:
                logger.warning("Automatic LR is already disabled.")
                return False
            self._auto_lr_enabled = False
            logger.warning("Disabled automatic LR.")
            return True
        elif cmd in {"enable_auto_lr", "auto_lr_on", "enable_automatic_lr", "enable_auto_learning_rate"}:
            _ = params
            if self.automatic_lr is None:
                logger.warning("Automatic LR is not configured; cannot enable it.")
                return False
            if self.target_kl is None:
                raise ValueError("Cannot enable automatic LR when target_kl is None.")
            if self._auto_lr_enabled:
                logger.warning("Automatic LR is already enabled.")
                return False
            self._auto_lr_enabled = True
            self._auto_lr__iters_without_kl_early_stop = 0
            logger.warning("Enabled automatic LR (resetting no-early-stop counter).")
            return True
        else:
            return super()._execute_command(cmd, params, extra_run_metadata)

    def _get_optimizer_state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def _apply_optimizer_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)
        self._move_optimizer_state_to_device(self.train_device)
        try:
            lr = float(self.optimizer.param_groups[0]["lr"])
        except (KeyError, IndexError, TypeError, ValueError):
            return
        self.learning_rate = lr

    def _move_optimizer_state_to_device(self, device: torch.device) -> None:
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v) and v.device != device:
                    state[k] = v.to(device)

    def _apply_learning_rate(self, lr: LearningRate) -> None:
        assert isinstance(lr, float)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def _serialize_gsde_reset_mode(mode: GSDEResetMode | None) -> dict[str, Any] | None:
        if mode is None:
            return None
        if isinstance(mode, GSDEIntervalResetMode):
            return {"mode": "interval", "interval": mode.interval}
        if isinstance(mode, GSDEProbabilityResetMode):
            return {"mode": "probability", "probability": mode.probability}
        return {"mode": type(mode).__name__}

    @staticmethod
    def _serialize_fn(fn: Callable) -> dict[str, str]:
        fn_dict = {
            'repr': str(fn)
        }
        try:
            fn_dict['source'] = inspect.getsource(fn)
        except OSError as err:
            fn_dict['source'] = str(err)
        return fn_dict


