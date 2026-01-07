from typing import Optional, Any, Literal

import torch
import torch.nn.functional as F
from loguru import logger

from swarmbots.learn.algos.base_algorithm import BaseAlgorithm, LearningRate
from swarmbots.learn.algos.ppo.ppo_policy import BasePPOPolicy, PPOPolicy
from swarmbots.learn.algos.ppo.ppo_rollout import collect_whole_episodes
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisode, PPORolloutBuffer, PPOSampler, PPOSamples
from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage
from swarmbots.learn.metrics_list import MetricsLists
from swarmbots.learn.performance_timer import PerformanceTimer
from swarmbots.learn.summary_statistics import compute_summary_statistics
from swarmbots.learn.torch_device import as_device

AGENTS_DIM = 1

try:
    logger.level("SAVE")
except ValueError:
    logger.level("SAVE", no=21, color="<magenta>")


# based on https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
class PPO(BaseAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    """

    policy: BasePPOPolicy
    learning_rate: LearningRate

    def __init__(
            self,
            policy: BasePPOPolicy,
            env: BaseLearnEnvWrapper,
            learning_rate: LearningRate = 3e-4,
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
            max_grad_norm: float = 0.5,
            target_kl: Optional[float] = None,
            gsde_sample_freq: int = -1,
            agent_logprob_reduction: Optional[Literal["sum", "mean"]] = None,
            train_device: str | torch.device = "auto",
            rollout_device: str | torch.device = "cpu",
    ):
        super().__init__(policy, env, learning_rate)

        self.learning_rate = learning_rate
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
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.gsde_sample_freq = gsde_sample_freq
        assert not policy.gsde_enabled or gsde_sample_freq > 0
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

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

    def get_hyper_parameters(self):
        return {
            'learning_rate': self.learning_rate,
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
            'max_grad_norm': self.max_grad_norm,
            'target_kl': self.target_kl,
            'train_device': str(self.train_device),
            'rollout_device': str(self.rollout_device),
            'gsde_sample_freq': self.gsde_sample_freq,
            'agent_logprob_reduction': self.agent_logprob_reduction,
        }

    def _get_optimizer_state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def _apply_optimizer_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)

    def _apply_learning_rate(self, lr: LearningRate) -> None:
        assert isinstance(lr, float)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _perform_iteration(
            self,
            episode_return_ema: ExponentialMovingAverage
    ) -> tuple[dict[str, Any], int]:
        with PerformanceTimer() as rollout_timer:
            episodes, episode_infos, rollout_metrics = collect_whole_episodes(
                env=self.env,
                policy=self.policy,
                buffer=self.rollout_buffer,
                gsde_sample_freq=self.gsde_sample_freq,
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

    def compute_loss(
            self,
            batch: PPOSamples,
            entropy: torch.Tensor | None,
            log_prob: torch.Tensor,
            old_log_prob: torch.Tensor,
            values: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
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

        value_loss = F.mse_loss(batch.returns, values_pred)

        if entropy is None:
            entropy_loss = -torch.mean(-log_prob)
        else:
            entropy_loss = -torch.mean(entropy)

        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float())
        metrics = {
            'ent_loss': entropy_loss.item(),
            'act_loss': policy_loss.item(),
            'val_loss': value_loss.item(),
            'clip_frac': clip_fraction.item(),
            'ratio': compute_summary_statistics(ratio, find_min=True, find_max=True),
        }

        return loss, metrics

    def train(self, episodes: list[PPOEpisode]) -> dict[str, Any]:
        with PerformanceTimer() as to_train_device_timer:
            self.policy.train()
            self.policy.to(self.train_device)

        with PerformanceTimer() as sampler_init_timer:
            sampler = PPOSampler(episodes, history_embeddings=None)

        y_pred = sampler.values.flatten()
        y_true = sampler.returns.flatten()
        var_y = torch.var(y_true)
        if not torch.isnan(var_y) and var_y > 1e-8:
            explained_var = (1 - torch.var(y_true - y_pred) / var_y).item()
        else:
            explained_var = 0.0

        loss_metrics = MetricsLists[float]()

        approx_kl_divs = []

        continue_training = True
        n_updates = 0

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

                log_probs, entropies, values = self.policy.evaluate_actions(
                    local_obs=batch.local_obs,
                    global_obs=batch.global_obs,
                    actions=batch.actions
                )

                entropy, log_prob, old_log_prob = self.reduce_agents(batch, entropies, log_probs)

                loss, metrics = self.compute_loss(batch, entropy, log_prob, old_log_prob, values)
                loss_metrics.add(metrics)

                with torch.no_grad():
                    log_ratio = log_prob - old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    msg = f"Early stopping at epoch {epoch}, batch {i} due to reaching max kl: {approx_kl_div:.3f}"
                    if epoch == 0:
                        logger.warning(msg)
                    else:
                        logger.debug(msg)
                    break

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

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
                **{k: compute_summary_statistics(v) for k, v in loss_metrics.get().items()},
                'approx_kl': compute_summary_statistics(approx_kl_divs, find_max=True),
                'upd': n_updates,
                'tot_upd': self.n_total_updates,
                'expl_var': explained_var,
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

    def _execute_command(self, cmd: str, params: str):
        if cmd == 'set_std':
            std = float(params)
            logger.warning(f"Setting action std to {std}")
            self.policy.action_dist.set_std(std)
        elif cmd == 'scale_std':
            multiplier = float(params)
            logger.warning(f"Scaling action std by {multiplier}")
            self.policy.action_dist.scale_std(multiplier)
        elif cmd == 'set_target_kl':
            target_kl = float(params)
            logger.warning(f'Setting target KL to {target_kl}')
            self.target_kl = target_kl
        elif cmd == "set_clip_range":
            clip_range = float(params)
            logger.warning(f"Setting clip_range to {clip_range}")
            self.clip_range = clip_range
        elif cmd == "set_clip_range_vf":
            clip_range_vf = float(params)
            logger.warning(f"Setting clip_range_vf to {clip_range_vf}")
            self.clip_range_vf = clip_range_vf
        elif cmd == "set_ent_coef":
            ent_coef = float(params)
            logger.warning(f"Setting ent_coef to {ent_coef}")
            self.ent_coef = ent_coef
        elif cmd == "set_vf_coef":
            vf_coef = float(params)
            logger.warning(f"Setting vf_coef to {vf_coef}")
            self.vf_coef = vf_coef
        elif cmd == "set_gamma":
            gamma = float(params)
            logger.warning(f"Setting gamma to {gamma}")
            self.gamma = gamma
            self.rollout_buffer.gamma = gamma
        elif cmd == "set_gae_lambda":
            gae_lambda = float(params)
            logger.warning(f"Setting gae_lambda to {gae_lambda}")
            self.gae_lambda = gae_lambda
            self.rollout_buffer.gae_lambda = gae_lambda
        else:
            super()._execute_command(cmd, params)

