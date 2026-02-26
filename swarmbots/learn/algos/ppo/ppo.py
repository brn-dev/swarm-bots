import abc
import inspect
from dataclasses import dataclass
from typing import Optional, Any, Literal, TypeVar, Generic, Callable, Protocol, NotRequired, TypedDict

import torch
import torch.nn as nn
from loguru import logger

from swarmbots.learn.action_dists.action_dist import ActionDist
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliActionDist
from swarmbots.learn.algos.base_algorithm import BaseAlgorithm, LearningRate, _parse_bool
from swarmbots.learn.algos.ppo.ppo_policy import BasePPOPolicy, PPOPolicy
from swarmbots.learn.algos.ppo.ppo_rollout import PPORolloutState, collect_steps, collect_whole_episodes
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisode, PPORolloutBuffer, PPOSampler, PPOSamples
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage
from swarmbots.learn.gsde_reset import GSDEResetMode, GSDEIntervalResetMode, GSDEProbabilityResetMode
from swarmbots.learn.masking import masked_mean
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

class AutomaticLearningRateUpdateResult(TypedDict):
    new_lr: Optional[float]
    msg: NotRequired[str]
    event: NotRequired[str]

class AutomaticLearningRateUpdater(Protocol):

    def __call__(
            self,
            old_lr: float,
            state: dict[str, Any],
            n_iterations: int,
            n_model_updates: int,
            n_timesteps: int,
            early_stop_kl_div: Optional[float],
            early_stop_epoch: Optional[int],
            metrics: dict[str, Any]
    ) -> AutomaticLearningRateUpdateResult:
        ...


@dataclass(slots=True)
class AutomaticLearningRate:
    initial_lr: float
    updater: AutomaticLearningRateUpdater

    max_lr: float = 1e-2

PPOLearningRate = float | AutomaticLearningRate

class PPORolloutMode(abc.ABC):
    pass

@dataclass(slots=True, frozen=True)
class WholeEpisodesRolloutMode(PPORolloutMode):
    n_episodes_per_rollout: int


@dataclass(slots=True, frozen=True)
class StepsRolloutMode(PPORolloutMode):
    n_steps_per_rollout: int


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
            rollout_mode: PPORolloutMode = WholeEpisodesRolloutMode(6),
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
            max_grad_norm: float = 2.0,
            target_kl: Optional[float] = None,
            gsde_reset_mode: GSDEResetMode | None = None,
            agent_logprob_reduction: Optional[Literal["sum", "mean"]] = None,
            train_device: str | torch.device = "auto",
            rollout_device: str | torch.device = "cpu",
            metrics_action_splitters: list[Callable[[torch.Tensor], dict[str, torch.Tensor]] | None] | None = None,
            use_popart: bool = False,
    ):
        self.automatic_lr: AutomaticLearningRate | None = None
        self._auto_lr_enabled = False
        self._auto_lr_state: dict[str, Any] = {}

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
        self.rollout_mode = rollout_mode
        self._rollout_state: PPORolloutState | None = None
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
        self.use_popart = bool(use_popart)
        if self.use_popart and not self.policy.has_popart:
            raise ValueError(
                "use_popart=True, but policy.has_popart=False. "
                "Enable PopArt in the policy/critic first."
            )

        self.gsde_reset_mode = gsde_reset_mode
        assert not policy.gsde_enabled or self.gsde_reset_mode is not None
        if agent_logprob_reduction not in (None, "sum", "mean"):
            raise ValueError(f"{agent_logprob_reduction=} must be 'sum', 'mean', or None")
        if agent_logprob_reduction is not None and not isinstance(policy, PPOPolicy):
            logger.warning('agent_logprob_reduction is only intended for single agent PPO')
        self.agent_logprob_reduction: Optional[Literal["sum", "mean"]] = agent_logprob_reduction

        self.metrics_action_splitters: list[Callable[[torch.Tensor], dict[str, torch.Tensor]] | None]
        if metrics_action_splitters is not None:
            self.metrics_action_splitters = metrics_action_splitters
        else:
            self.metrics_action_splitters = [None] * len(self.policy.action_dist.distributions)

        self.train_device = as_device(train_device)
        self.rollout_device = as_device(rollout_device)

        self.rollout_buffer = PPORolloutBuffer(
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
            auto_lr = {
                "enabled": self._auto_lr_enabled,
                "initial_lr": self.automatic_lr.initial_lr,
                "max_lr": self.automatic_lr.max_lr,
                "updater": self._serialize_fn(self.automatic_lr.updater),
            }

        return {
            'learning_rate': self.learning_rate,
            'automatic_learning_rate': auto_lr,
            'rollout_mode': self._serialize_rollout_mode(self.rollout_mode),
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
            'use_popart': self.use_popart,
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
        valid_mask = self._build_agent_valid_mask(batch.agent_mask, log_prob)

        advantages = batch.advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(log_prob - old_log_prob)

        if ratio.ndim == 2 and advantages.ndim == 1:
            advantages = advantages.unsqueeze(-1)

        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -masked_mean(torch.min(policy_loss_1, policy_loss_2), valid_mask)

        if self.clip_range_vf is None:
            values_pred = values
        else:
            values_pred = batch.values + torch.clamp(
                values - batch.values, -self.clip_range_vf, self.clip_range_vf
            )
        value_targets = batch.returns
        if self.use_popart:
            values_pred = self.policy.normalize_values(values_pred)
            value_targets = self.policy.normalize_values(value_targets)

        value_loss: torch.Tensor = self.value_loss_fn(values_pred, value_targets)

        if entropy is None:
            entropy_loss = masked_mean(log_prob, valid_mask)
        else:
            entropy_loss = -masked_mean(entropy, valid_mask)

        value_loss_scaled = self.vf_coef * value_loss
        entropy_loss_scaled = self.ent_coef * entropy_loss

        loss = policy_loss + entropy_loss_scaled + value_loss_scaled

        with torch.no_grad():
            log_ratio = log_prob - old_log_prob
            approx_kl_div = masked_mean((torch.exp(log_ratio) - 1) - log_ratio, valid_mask).item()

        clip_fraction = masked_mean((torch.abs(ratio - 1) > self.clip_range).float(), valid_mask).item()
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
            actions=batch.actions,
            hidden_vars=batch.hidden_vars,
            agent_mask=batch.agent_mask,
        )

        return self.compute_ppo_loss(
            batch=batch,
            entropies=entropies,
            log_probs=log_probs,
            values=values,
        )

    def perform_iteration(
            self,
            episode_return_ema: ExponentialMovingAverage,
            update_ema: bool,
    ) -> tuple[dict[str, Any], int]:
        with PerformanceTimer() as rollout_timer:
            if isinstance(self.rollout_mode, WholeEpisodesRolloutMode):
                episodes, episode_infos, rollout_metrics = collect_whole_episodes(
                    env=self.env,
                    policy=self.policy,
                    buffer=self.rollout_buffer,
                    n_episodes=self.rollout_mode.n_episodes_per_rollout,
                    gsde_reset_mode=self.gsde_reset_mode,
                )
                self._rollout_state = None
            elif isinstance(self.rollout_mode, StepsRolloutMode):
                episodes, episode_infos, rollout_metrics, self._rollout_state = collect_steps(
                    env=self.env,
                    policy=self.policy,
                    buffer=self.rollout_buffer,
                    n_steps=self.rollout_mode.n_steps_per_rollout,
                    rollout_state=self._rollout_state,
                    gsde_reset_mode=self.gsde_reset_mode,
                )
            else:
                raise TypeError(f"Unknown rollout_mode type: {type(self.rollout_mode)}")

            total_steps_in_rollout = sum(len(ep.rewards) for ep in episodes)
        self.n_total_timesteps += total_steps_in_rollout
        self.n_total_iterations += 1

        ep_rew = compute_summary_statistics(
            [ep['r'] for ep in episode_infos],
            find_min=True, find_max=True,
            compute_skewness=True, compute_kurtosis=True,
            make_histogram=30
        )
        ep_len = compute_summary_statistics(
            [ep['l'] for ep in episode_infos],
            find_min=True, find_max=True,
            compute_skewness=True, compute_kurtosis=True,
            make_histogram=30
        )
        ep_time = compute_summary_statistics([ep['t'] for ep in episode_infos])
        ep_progress_rew = compute_summary_statistics(
            [ep['progress_reward'] for ep in episode_infos],
            find_min=True, find_max=True,
            compute_skewness=True, compute_kurtosis=True,
            make_histogram=30
        )
        ep_guidance_rew = compute_summary_statistics(
            [ep['guidance_reward'] for ep in episode_infos],
            find_min=True, find_max=True,
            compute_skewness=True, compute_kurtosis=True,
            make_histogram=30
        )

        if update_ema:
            for ep_info in episode_infos:
                episode_return_ema.update(ep_info['r'])

        update_metrics = self.train(episodes)
        metrics = {
            **update_metrics,
            **rollout_metrics,
            'rollout_time': rollout_timer.get_duration(),
            'ep_rew': ep_rew,
            'ep_len': ep_len,
            'ep_time': ep_time,
            'ep_progress_rew': ep_progress_rew,
            'ep_guidance_rew': ep_guidance_rew,
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

        if self.use_popart:
            self.policy.update_value_normalizer(sampler.returns)

        y_pred = sampler.values.flatten()
        y_true = sampler.returns.flatten()
        var_y = torch.var(y_true)
        if not torch.isnan(var_y) and var_y > 1e-8:
            explained_var = (1 - torch.var(y_true - y_pred) / var_y).item()
        else:
            explained_var = 0.0

        loss_metrics = MetricsLists[float]()

        continue_training = True
        early_stop_epoch: int | None = None
        early_stop_kl_div: float | None = None
        n_updates = 0
        grad_norms: list[float] = []
        n_grad_clipped = 0

        sampling_timings: list[float] = []
        sample_timer = PerformanceTimer()

        update_timings: list[float] = []
        update_timer = PerformanceTimer()

        sub_grad_norms = MetricsLists[float]()
        compute_grad_norms_timings: list[float] = []
        compute_grad_norms_timer = PerformanceTimer()

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

                with compute_grad_norms_timer:
                    sub_grad_norms.add(self.policy.get_grad_norms())
                compute_grad_norms_timings.append(compute_grad_norms_timer.get_duration())

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

        self.n_total_updates += n_updates

        metrics_timer = PerformanceTimer().start()
        with torch.no_grad():
            metrics: dict[str, Any] = {
                **loss_metrics.compute_summary_statistics(
                    find_min=True, find_max=True, compute_skewness=True, compute_kurtosis=True
                ),
                'updates': n_updates,
                'total_updates': self.n_total_updates,
                'expl_var': explained_var,
                'grad_norm': compute_summary_statistics(
                    grad_norms, find_max=True, find_min=True, compute_skewness=True, compute_kurtosis=True,
                ) if grad_norms else 0.0,
                'grad_clip_frac': (n_grad_clipped / len(grad_norms)) if grad_norms else 0.0,
                **sub_grad_norms.compute_summary_statistics(find_min=True, find_max=True, prefix='grad_norm_'),
                'total_compute_grad_norms_time': sum(compute_grad_norms_timings),
                'compute_grad_norms_time': compute_summary_statistics(
                    compute_grad_norms_timings, find_min=True, find_max=True
                )
            }

            auto_lr_metrics = self._maybe_update_automatic_lr(
                early_stop_kl_div=early_stop_kl_div,
                early_stop_epoch=early_stop_epoch,
                metrics=metrics,
            )
            metrics.update(auto_lr_metrics)
            metrics.update(self.policy.get_value_normalizer_metrics())

            act_dim_sum = 0
            action_dims = self.policy.action_dist.action_dims
            for i, (dist, act_splitter) in enumerate(zip(
                    self.policy.action_dist.distributions,
                    self.metrics_action_splitters
            )):
                dist: ActionDist
                act_splitter: Callable[[torch.Tensor], dict[str, torch.Tensor]] | None

                act_dim = action_dims[i]
                actions = sampler.actions[..., act_dim_sum:act_dim_sum + act_dim]
                act_dim_sum += act_dim

                hist_bins = 2 if isinstance(dist, BernoulliActionDist) else 20

                metrics[f'act{i}'] = compute_summary_statistics(actions, make_histogram=hist_bins)

                if act_splitter is not None:
                    split_actions = act_splitter(actions)
                    for key, sub_actions in split_actions.items():
                        metrics[f'act{i}_{key}'] = compute_summary_statistics(sub_actions, make_histogram=hist_bins)

                if hasattr(dist, "log_stds"):
                    std_values = torch.exp(dist.log_stds)
                    metrics[f'std{i}'] = compute_summary_statistics(
                        std_values, find_min=True, find_max=True,
                        compute_skewness=True, compute_kurtosis=True,
                        make_histogram=hist_bins
                    )
                    can_split_stds = not (std_values.ndim >= 1 and std_values.shape[-1] == 1 and act_dim > 1)
                    if act_splitter is not None and can_split_stds:
                        split_stds = act_splitter(std_values)
                        for key, sub_stds in split_stds.items():
                            metrics[f'std{i}_{key}'] = compute_summary_statistics(
                                sub_stds,
                                find_min=True,
                                find_max=True,
                                compute_skewness=True,
                                compute_kurtosis=True,
                                make_histogram=hist_bins,
                            )

        metrics_timer.stop()

        return {
            **metrics,
            'to_train_device_time': to_train_device_timer.get_duration(),
            'sampler_init_time': sampler_init_timer.get_duration(),
            'sampling_time': compute_summary_statistics(
                sampling_timings, find_min=True, find_max=True, compute_kurtosis=True, compute_skewness=True),
            'total_sampling_time': sum(sampling_timings),
            'update_time': compute_summary_statistics(
                update_timings, find_min=True, find_max=True, compute_kurtosis=True, compute_skewness=True),
            'total_update_time': sum(update_timings),
            'metrics_time': metrics_timer.get_duration(),
            'train_time': train_timer.get_duration(),
        }

    def _maybe_update_automatic_lr(
            self,
            early_stop_kl_div: Optional[float],
            early_stop_epoch: Optional[int],
            metrics: dict[str, Any]
    ) -> dict[str, Any]:
        if self.automatic_lr is None:
            return {}

        if not self._auto_lr_enabled:
            return {"auto_lr_event": "disabled", "auto_lr": self.learning_rate}

        update_result = self.automatic_lr.updater(
            old_lr=self.learning_rate,
            state=self._auto_lr_state,
            n_iterations=self.n_total_iterations,
            n_model_updates=self.n_total_updates,
            n_timesteps=self.n_total_timesteps,
            early_stop_kl_div=early_stop_kl_div,
            early_stop_epoch=early_stop_epoch,
            metrics=metrics,
        )
        requested_lr = update_result.get('new_lr', None)
        update_msg = update_result.get('msg', None)
        update_event = update_result.get('event', None)

        if requested_lr is None:
            return {"auto_lr_event": update_event, "auto_lr": self.learning_rate}

        new_lr = min(requested_lr, self.automatic_lr.max_lr)
        ratio = new_lr / self.learning_rate
        if new_lr > self.learning_rate:
            if new_lr < requested_lr:
                msg = f', {update_msg}' if update_msg else ''
                logger.warning(
                    f"Auto LR capped at {new_lr:.2e} (requested {requested_lr:.2e}{msg}, {ratio=:.2f})"
                )
            else:
                msg = f': {update_msg}' if update_msg else ''
                logger.warning(
                    f"Increasing LR to {new_lr:.2e}{msg} ({ratio=:.2f})"
                )
            self.set_learning_rate(new_lr)
            event = update_event or ("lr_increase" if new_lr == requested_lr else "lr_increase_capped")
            return {"auto_lr_event": event, "auto_lr": new_lr}

        if new_lr < self.learning_rate:
            msg = f': {update_msg}' if update_msg else ''
            logger.warning(
                f"Decaying LR to {new_lr:.2e}{msg} ({ratio=:.2f})"
            )
            self.set_learning_rate(new_lr)
            return {"auto_lr_event": update_event or "lr_decay", "auto_lr": new_lr}

        return {"auto_lr_event": update_event, "auto_lr": self.learning_rate}

    def _after_optimizer_step(self) -> None:
        pass

    def reduce_agents(
            self,
            batch: PPOSamples,
            entropies: torch.Tensor | None,
            log_probs: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        agent_mask = batch.agent_mask
        if agent_mask is not None:
            if agent_mask.dtype != torch.bool:
                raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
            if agent_mask.shape != log_probs.shape:
                raise ValueError(
                    f"Expected agent_mask shape {tuple(log_probs.shape)}, got {tuple(agent_mask.shape)}"
                )

        if self.agent_logprob_reduction is None:
            log_prob = log_probs
            old_log_prob = batch.log_probs
            entropy = entropies
        elif self.agent_logprob_reduction == "sum":
            if agent_mask is None:
                log_prob = log_probs.sum(dim=AGENTS_DIM)
                old_log_prob = batch.log_probs.sum(dim=AGENTS_DIM)
                entropy = entropies.sum(dim=AGENTS_DIM) if entropies is not None else None
            else:
                mask_f = agent_mask.to(dtype=log_probs.dtype)
                log_prob = (log_probs * mask_f).sum(dim=AGENTS_DIM)
                old_log_prob = (batch.log_probs * mask_f).sum(dim=AGENTS_DIM)
                entropy = (entropies * mask_f).sum(dim=AGENTS_DIM) if entropies is not None else None
        elif self.agent_logprob_reduction == "mean":
            if agent_mask is None:
                log_prob = log_probs.mean(dim=AGENTS_DIM)
                old_log_prob = batch.log_probs.mean(dim=AGENTS_DIM)
                entropy = entropies.mean(dim=AGENTS_DIM) if entropies is not None else None
            else:
                mask_f = agent_mask.to(dtype=log_probs.dtype)
                denom = mask_f.sum(dim=AGENTS_DIM).clamp_min(1.0)
                log_prob = (log_probs * mask_f).sum(dim=AGENTS_DIM) / denom
                old_log_prob = (batch.log_probs * mask_f).sum(dim=AGENTS_DIM) / denom
                entropy = (entropies * mask_f).sum(dim=AGENTS_DIM) / denom if entropies is not None else None
        else:
            raise ValueError(f"Unhandled {self.agent_logprob_reduction=}")
        return entropy, log_prob, old_log_prob

    @staticmethod
    def _build_agent_valid_mask(
            agent_mask: torch.Tensor | None,
            target: torch.Tensor,
    ) -> torch.Tensor | None:
        if agent_mask is None:
            return None
        if target.ndim == 2:
            if agent_mask.shape != target.shape:
                raise ValueError(
                    f"Expected agent_mask shape {tuple(target.shape)}, got {tuple(agent_mask.shape)}"
                )
            return agent_mask
        if target.ndim == 1:
            if agent_mask.ndim != 2 or agent_mask.shape[0] != target.shape[0]:
                raise ValueError(
                    f"Expected agent_mask shape (B, N) with B={target.shape[0]}, got {tuple(agent_mask.shape)}"
                )
            return agent_mask.any(dim=AGENTS_DIM)
        raise ValueError(f"Unsupported target ndim for agent mask: {target.ndim}")

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
        elif cmd in {"set_batch_size", "batch_size"}:
            batch_size = int(params)
            if batch_size <= 0:
                raise ValueError(f"batch_size must be > 0, got {batch_size}")
            logger.warning(f"Setting batch_size to {batch_size}")
            self.batch_size = batch_size
            return True
        elif cmd in {"set_normalize_advantage", "normalize_advantage"}:
            normalize_advantage = _parse_bool(params)
            logger.warning(f"Setting normalize_advantage to {normalize_advantage}")
            self.normalize_advantage = normalize_advantage
            return True
        elif cmd in {"set_gsde_prob", "set_gsde_probability", "gsde_prob"}:
            probability = float(params)
            if not (0.0 < probability < 1.0):
                raise ValueError(f"gsde probability must be in (0, 1), got {probability}")
            logger.warning(f"Setting gsde_reset_mode to probability={probability}")
            self.gsde_reset_mode = GSDEProbabilityResetMode(probability=probability)
            return True
        elif cmd in {"set_gsde_interval", "gsde_interval"}:
            interval = int(params)
            if interval <= 0:
                raise ValueError(f"gsde interval must be > 0, got {interval}")
            logger.warning(f"Setting gsde_reset_mode to interval={interval}")
            self.gsde_reset_mode = GSDEIntervalResetMode(interval=interval)
            return True
        elif cmd in {"disable_gsde_reset", "gsde_reset_off"}:
            logger.warning("Disabling gsde_reset_mode")
            self.gsde_reset_mode = None
            return True
        elif cmd in {"set_max_grad_norm", "max_grad_norm"}:
            max_grad_norm = float(params)
            if max_grad_norm <= 0:
                raise ValueError(f"max_grad_norm must be > 0, got {max_grad_norm}")
            logger.warning(f"Setting max_grad_norm to {max_grad_norm}")
            self.max_grad_norm = max_grad_norm
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
            self._auto_lr_state = {}
            logger.warning("Enabled automatic LR (resetting no-early-stop counter).")
            return True
        else:
            return super()._execute_command(cmd, params, extra_run_metadata)

    def _get_optimizer_state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def _apply_optimizer_state_dict(
            self,
            state_dict: dict[str, Any],
            missing_keys: list[str],
            unexpected_keys: list[str],
    ) -> None:
        if missing_keys:
            raise NotImplementedError()
        if unexpected_keys:
            state_dict = state_dict.copy()
            for k in unexpected_keys:
                state_dict.pop(k, None)

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
    def _serialize_rollout_mode(rollout_mode: PPORolloutMode) -> dict[str, Any]:
        if isinstance(rollout_mode, WholeEpisodesRolloutMode):
            return {"mode": "whole_episodes", "n_episodes_per_rollout": rollout_mode.n_episodes_per_rollout}
        if isinstance(rollout_mode, StepsRolloutMode):
            return {"mode": "steps", "n_steps_per_rollout": rollout_mode.n_steps_per_rollout}
        return {"mode": type(rollout_mode).__name__}

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
