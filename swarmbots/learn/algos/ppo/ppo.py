import json
import pathlib
from collections.abc import Collection
from datetime import datetime
from typing import Optional, Any, Literal

import torch
import torch.nn.functional as F
from loguru import logger

from swarmbots.learn.algos.base_algorithm import BaseAlgorithm
from swarmbots.learn.algos.ppo.ppo_policy import BasePPOPolicy, PPOPolicy
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisode, PPORolloutBuffer, PPOSampler, PPOSamples
from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage
from swarmbots.learn.metrics_list import MetricsLists
from swarmbots.learn.metrics_logger import MetricsLogger
from swarmbots.learn.performance_timer import PerformanceTimer
from swarmbots.learn.summary_statistics import compute_summary_statistics
from swarmbots.learn.torch_device import as_device

AGENTS_DIM = 1

MIN_ITERATIONS_FOR_BEST = 10

try:
    logger.level("SAVE")
except ValueError:
    logger.level("SAVE", no=21, color="<magenta>")


@torch.no_grad()
def collect_whole_episodes(
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy,
        buffer: PPORolloutBuffer,
        gsde_sample_freq: int = -1,
) -> tuple[list[PPOEpisode], list[dict], dict[str, Any]]:
    buffer.reset()
    obs, info = env.reset()
    is_final = torch.zeros((buffer.n_envs,), dtype=torch.bool, device=buffer.rollout_device)
    was_terminated = torch.zeros((buffer.n_envs,), dtype=torch.bool, device=buffer.rollout_device)

    with PerformanceTimer() as to_rollout_device_timer:
        policy.to(buffer.rollout_device)
        policy.eval()
    
    episode_infos = []
    rollout_step_idx = 0

    reset_noise_timings: list[float] = []
    policy_forward_timings: list[float] = []
    env_step_timings: list[float] = []
    buffer_add_timings: list[float] = []

    reset_noise_timer = PerformanceTimer()
    policy_forward_timer = PerformanceTimer()
    env_step_timer = PerformanceTimer()
    buffer_add_timer = PerformanceTimer()

    while not buffer.is_ready():
        local_obs = obs['local_obs']
        global_obs = obs['global_obs']

        if policy.gsde_enabled and gsde_sample_freq > 0 and (rollout_step_idx % gsde_sample_freq) == 0:
            with reset_noise_timer:
                policy.action_dist.reset_noise(batch_shape=tuple(local_obs.shape[:-1]))
            reset_noise_timings.append(reset_noise_timer.get_duration())

        with policy_forward_timer:
            actions, log_probs, values = policy(local_obs, global_obs)
        policy_forward_timings.append(policy_forward_timer.get_duration())
        
        values = values.masked_fill(was_terminated, 0.0)

        with env_step_timer:
            new_obs, rewards, terminations, truncations, infos = env.step(actions)
        env_step_timings.append(env_step_timer.get_duration())
        dones = torch.logical_or(terminations, truncations)

        if "episode" in infos:
            for i, has_ep_info in enumerate(infos["_episode"]):
                if has_ep_info:
                    episode_infos.append({
                        'r': infos['episode']['r'][i],
                        'l': infos['episode']['l'][i],
                        't': infos['episode']['t'][i],
                    })

        with buffer_add_timer:
            buffer.add(
                local_obs=local_obs,
                global_obs=global_obs,
                actions=actions,
                rewards=rewards,
                log_probs=log_probs,
                values=values,
                is_final=is_final,
            )
        buffer_add_timings.append(buffer_add_timer.get_duration())

        obs = new_obs
        is_final = dones
        was_terminated = terminations
        rollout_step_idx += 1

    with PerformanceTimer() as buffer_get_whole_episodes_timer:
        episodes = buffer.get_whole_episodes()
        
    metrics = {
        'to_rollout_device_time': to_rollout_device_timer.get_duration(),
        'reset_noise_time': compute_summary_statistics(reset_noise_timings),
        'total_reset_noise_time': sum(reset_noise_timings),
        'policy_forward_time': compute_summary_statistics(policy_forward_timings),
        'total_policy_forward_time': sum(policy_forward_timings),
        'env_step_time': compute_summary_statistics(env_step_timings),
        'total_env_step_time': sum(env_step_timings),
        'buffer_add_time': compute_summary_statistics(buffer_add_timings),
        'total_buffer_add_time': sum(buffer_add_timings),
        'buffer_get_whole_episodes_time': buffer_get_whole_episodes_timer.get_duration(),
    }
    return episodes, episode_infos, metrics


# based on https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
class PPO(BaseAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    """

    policy: BasePPOPolicy

    def __init__(
            self,
            policy: BasePPOPolicy,
            env: BaseLearnEnvWrapper,
            learning_rate: float = 3e-4,
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
        super().__init__(policy, env)

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

    def _apply_optimizer_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)

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

    def train(self, episodes: list[PPOEpisode], compute_agent_metrics: bool = True) -> dict[str, Any]:
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

                loss, metrics = self.compute_loss(batch, entropy, log_prob, old_log_prob, values)
                if compute_agent_metrics:
                    with torch.no_grad():
                        ratio_agent = torch.exp(log_probs - batch.log_probs)
                        metrics["ratio_agent"] = compute_summary_statistics(ratio_agent, find_min=True, find_max=True)
                        metrics["clip_frac_agent"] = torch.mean(
                            (torch.abs(ratio_agent - 1) > self.clip_range).float()
                        ).item()

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

    def learn(
            self,
            max_total_timesteps: int | None = None,
            additional_timesteps: int | None = None,
            run_dir: Optional[str | pathlib.Path] = None,
            log_interval: int = 1,
            save_interval: Optional[int] = None,
            save_optimizer: bool = True,
            extra_run_metadata: dict[str, Any] | None = None,
            episode_return_ema_alpha: float = 0.05,
            best_rotation_n: int = 1,
            wandb_project: str | None = None,
            wandb_entity: str | None = None,
            wandb_run_name: str | None = None,
            wandb_group: str | None = None,
            wandb_tags: list[str] | None = None,
            wandb_mode: str | None = None,
            wandb_kwargs: dict[str, Any] | None = None,
            logging_ignore_keys_for_persistence: list[str] | None = None,
            logging_console_keys: Collection[str] | Collection[tuple[str, str | None]] | None = None,
            compute_agent_metrics: bool = False,
    ):
        assert (
                (max_total_timesteps is not None and max_total_timesteps > 0 and additional_timesteps is None)
                or
                (additional_timesteps is not None and additional_timesteps > 0 and max_total_timesteps is None)
        )
        assert best_rotation_n >= 1

        if self.agent_logprob_reduction is None and compute_agent_metrics:
            logger.warning("compute_agent_metrics is unnecessary")

        if max_total_timesteps is None:
            max_total_timesteps = self.n_total_timesteps + additional_timesteps
        
        if run_dir is not None:
            run_dir = pathlib.Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_run_metadata(run_dir, extra_run_metadata)

        best_models_dir: pathlib.Path | None = None
        learn_started_at = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if run_dir is not None:
            best_models_dir = run_dir / "models" / "best" / learn_started_at
            
        wandb_config: dict[str, Any] | None = None
        if wandb_project is not None:
            wandb_config = {"hyper_parameters": self.get_hyper_parameters()}
            if extra_run_metadata:
                wandb_config["extra_run_metadata"] = json.loads(json.dumps(extra_run_metadata, default=str))
            if run_dir is not None:
                wandb_config["run_dir"] = str(run_dir)

            if wandb_run_name is None and run_dir is not None:
                wandb_run_name = run_dir.name

        metric_logger = MetricsLogger(
            log_dir=run_dir,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_run_name=wandb_run_name,
            wandb_group=wandb_group,
            wandb_tags=wandb_tags,
            wandb_config=wandb_config,
            wandb_mode=wandb_mode,
            wandb_kwargs=wandb_kwargs,
            wandb_step_key="timesteps",
            ignore_keys_for_persistence=logging_ignore_keys_for_persistence,
            console_keys=logging_console_keys,
        )
        episode_return_ema = ExponentialMovingAverage(alpha=episode_return_ema_alpha)
        best_episode_return_ema: float | None = None
        best_save_counter = 0

        while self.n_total_timesteps < max_total_timesteps:
            iter_timer = PerformanceTimer().start()
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

            current_episode_return_ema = episode_return_ema.update(ep_rew.mean)
            best_episode_return_ema, best_save_counter = self._maybe_save_best_ema_model(
                best_models_dir=best_models_dir,
                best_rotation_n=best_rotation_n,
                save_optimizer=save_optimizer,
                current_episode_return_ema=current_episode_return_ema,
                best_episode_return_ema=best_episode_return_ema,
                best_save_counter=best_save_counter,
            )

            update_metrics = self.train(episodes, compute_agent_metrics=compute_agent_metrics)
            metrics = {
                **update_metrics,
                **rollout_metrics,
                'rollout_time': rollout_timer.get_duration(),
                'ep_rew': ep_rew,
                'ep_len': ep_len,
                'ep_time': ep_time,
            }
            iter_duration = iter_timer.stop().get_duration()

            if log_interval is not None and self.n_total_iterations % log_interval == 0:
                fps = int(total_steps_in_rollout / iter_duration)

                metric_logger.log({
                    'learn_start': learn_started_at,
                    'iteration': self.n_total_iterations,
                    'timesteps': self.n_total_timesteps,
                    'lr': self.learning_rate,
                    **metrics,
                    'ep_rew_ema': current_episode_return_ema,
                    'best_ep_rew_ema': best_episode_return_ema,
                    'fps': fps,
                })

            if save_interval is not None and run_dir is not None and self.n_total_iterations % save_interval == 0:
                save_path = run_dir / f"models/model_{self.n_total_timesteps}_steps.pt"
                self.save(
                    save_path,
                    optimizer_state_dict=self.optimizer.state_dict() if save_optimizer else None,
                    return_ema=current_episode_return_ema
                )
                logger.log("SAVE", f"Saved model to {save_path.as_posix()}")

        if run_dir is not None:
            save_path = run_dir / f"models/model_{self.n_total_timesteps}_steps_final.pt"
            self.save(
                save_path,
                optimizer_state_dict=self.optimizer.state_dict() if save_optimizer else None,
                return_ema=episode_return_ema.get()
            )
            logger.log("SAVE", f"Saved final model to {save_path.as_posix()}")

        metric_logger.close()

        return self

    def _maybe_save_best_ema_model(
            self,
            best_models_dir: pathlib.Path | None,
            best_rotation_n: int,
            save_optimizer: bool,
            current_episode_return_ema: float,
            best_episode_return_ema: float | None,
            best_save_counter: int,
    ) -> tuple[float | None, int]:
        if ((best_episode_return_ema is not None and current_episode_return_ema <= best_episode_return_ema)
                or self.n_total_iterations < MIN_ITERATIONS_FOR_BEST):
            return best_episode_return_ema, best_save_counter

        best_episode_return_ema = current_episode_return_ema
        if best_models_dir is None:
            return best_episode_return_ema, best_save_counter

        if best_rotation_n == 1:
            best_save_path = best_models_dir / "model_best.pt"
        else:
            best_save_idx = best_save_counter % best_rotation_n
            best_save_path = best_models_dir / f"model_best_{best_save_idx}.pt"
            best_save_counter += 1

        self.save(
            best_save_path,
            optimizer_state_dict=self.optimizer.state_dict() if save_optimizer else None,
            return_ema=current_episode_return_ema
        )
        logger.log(
            "SAVE",
            f"Saved best-EMA model to {best_save_path.as_posix()} (ep_rew_ema={best_episode_return_ema:.4f})",
        )
        return best_episode_return_ema, best_save_counter

