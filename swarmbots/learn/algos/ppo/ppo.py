import json
import pathlib
import time
from datetime import datetime
from collections.abc import Collection
from typing import Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from swarmbots.learn.summary_statistics import compute_summary_statistics
from swarmbots.learn.torch_device import as_device
from swarmbots.learn.checkpointing import (
    apply_env_state,
    extract_env_state,
    extract_optimizer_state_dict,
    extract_policy_state_dict,
    load_checkpoint,
)
from swarmbots.learn.algos.ppo.ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisode, PPORolloutBuffer, PPOSampler
from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage
from swarmbots.learn.metrics_logger import MetricsLogger

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
) -> tuple[list[PPOEpisode], list[dict]]:
    buffer.reset()
    obs, info = env.reset()
    is_final = torch.zeros((buffer.n_envs,), dtype=torch.bool, device=buffer.rollout_device)
    was_terminated = torch.zeros((buffer.n_envs,), dtype=torch.bool, device=buffer.rollout_device)

    policy.to(buffer.rollout_device)
    policy.eval()
    
    episode_infos = []
    rollout_step_idx = 0

    while not buffer.is_ready():
        local_obs = obs['local_obs']
        global_obs = obs['global_obs']

        if policy.gsde_enabled and gsde_sample_freq > 0 and (rollout_step_idx % gsde_sample_freq) == 0:
            policy.action_dist.reset_noise(batch_shape=tuple(local_obs.shape[:-1]))

        actions, log_probs, values = policy(local_obs, global_obs)
        
        values = values.masked_fill(was_terminated, 0.0)

        new_obs, rewards, terminations, truncations, infos = env.step(actions)
        dones = torch.logical_or(terminations, truncations)

        if "episode" in infos:
            for i, has_ep_info in enumerate(infos["_episode"]):
                if has_ep_info:
                    episode_infos.append({
                        'r': infos['episode']['r'][i],
                        'l': infos['episode']['l'][i],
                        't': infos['episode']['t'][i],
                    })

        buffer.add(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            rewards=rewards,
            log_probs=log_probs,
            values=values,
            is_final=is_final,
        )

        obs = new_obs
        is_final = dones
        was_terminated = terminations
        rollout_step_idx += 1

    return buffer.get_whole_episodes(), episode_infos


# based on https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
class PPO:
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    """

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
            train_device: str | torch.device = "auto",
            rollout_device: str | torch.device = "cpu",
    ):
        self.policy = policy
        self.env = env
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
        self.n_total_updates = 0
        self.n_total_timesteps = 0

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
        }

    def train(self) -> dict[str, Any]:
        
        episodes, episode_infos = collect_whole_episodes(
            env=self.env,
            policy=self.policy,
            buffer=self.rollout_buffer,
            gsde_sample_freq=self.gsde_sample_freq,
        )

        self.policy.train()
        self.policy.to(self.train_device)
        
        sampler = PPOSampler(episodes, history_embeddings=None)

        y_pred = sampler.values.flatten()
        y_true = sampler.returns.flatten()
        var_y = torch.var(y_true)
        if not torch.isnan(var_y) and var_y > 1e-8:
            explained_var = (1 - torch.var(y_true - y_pred) / var_y).item()
        else:
            explained_var = 0.0

        entropy_losses = []
        pg_losses = []
        value_losses = []
        clip_fractions = []
        approx_kl_divs = []

        continue_training = True
        n_updates = 0

        for epoch in range(self.n_epochs):
            for i, batch in enumerate(sampler.sample(self.batch_size)):
                log_probs, entropies, values = self.policy.evaluate_actions(
                    local_obs=batch.local_obs,
                    global_obs=batch.global_obs,
                    actions=batch.actions
                )
                
                log_prob = log_probs.sum(dim=AGENTS_DIM)
                entropy = entropies.sum(dim=AGENTS_DIM) if entropies is not None else None
                old_log_prob = batch.log_probs.sum(dim=AGENTS_DIM)

                advantages = batch.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = batch.values + torch.clamp(
                        values - batch.values, -self.clip_range_vf, self.clip_range_vf
                    )
                
                value_loss = F.mse_loss(batch.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with torch.no_grad():
                    log_ratio = log_prob - old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    logger.debug(f'Early stopping at epoch {epoch}, batch {i} due to reaching max kl: {approx_kl_div:.3f}')
                    break

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                n_updates += 1

            if not continue_training:
                break

        self.n_total_updates += n_updates

        with torch.no_grad():
            metrics = {
                'ent_loss': compute_summary_statistics(entropy_losses),
                'act_loss': compute_summary_statistics(pg_losses),
                'val_loss': compute_summary_statistics(value_losses),
                'approx_kl': compute_summary_statistics(approx_kl_divs, find_max=True),
                'clip_frac': compute_summary_statistics(clip_fractions),
                'upd': n_updates,
                'tot_upd': self.n_total_updates,
                'expl_var': explained_var,
            }

            act_dim_sum = 0
            action_dims = self.policy.action_dist.action_dims
            for i, dist in enumerate(self.policy.action_dist.distributions):
                act_dim = action_dims[i]
                actions = sampler.actions[..., act_dim_sum:act_dim_sum+act_dim]
                act_dim_sum += act_dim
                metrics[f'act{i}'] = compute_summary_statistics(actions)
                if hasattr(dist, "log_stds"):
                    metrics[f'std{i}'] = compute_summary_statistics(torch.exp(dist.log_stds), find_min=True, find_max=True)

            metrics['ep_rew'] = compute_summary_statistics([ep['r'] for ep in episode_infos], find_min=True, find_max=True)
            metrics['ep_len'] = compute_summary_statistics([ep['l'] for ep in episode_infos], find_min=True, find_max=True)
            metrics['ep_time'] = compute_summary_statistics([ep['t'] for ep in episode_infos])

        return metrics

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
    ):
        assert (
                (max_total_timesteps is not None and max_total_timesteps > 0 and additional_timesteps is None)
                or
                (additional_timesteps is not None and additional_timesteps > 0 and max_total_timesteps is None)
        )
        assert best_rotation_n >= 1

        current_timesteps = self.n_total_timesteps
        iteration = 0

        if max_total_timesteps is None:
            max_total_timesteps = current_timesteps + additional_timesteps
        
        if run_dir is not None:
            run_dir = pathlib.Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_run_metadata(run_dir, extra_run_metadata)

        best_models_dir: pathlib.Path | None = None
        if run_dir is not None:
            learn_started_at = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

        while current_timesteps < max_total_timesteps:
            iter_start = time.time()
            metrics = self.train()
            iter_duration = time.time() - iter_start
            
            total_steps_in_rollout = sum(len(ep.rewards) for ep in self.rollout_buffer.episodes)
            current_timesteps += total_steps_in_rollout
            self.n_total_timesteps = current_timesteps
            iteration += 1

            current_episode_return_ema = episode_return_ema.update(metrics["ep_rew"].mean)
            best_episode_return_ema, best_save_counter = self._maybe_save_best_ema_model(
                iteration=iteration,
                best_models_dir=best_models_dir,
                best_rotation_n=best_rotation_n,
                save_optimizer=save_optimizer,
                current_episode_return_ema=current_episode_return_ema,
                best_episode_return_ema=best_episode_return_ema,
                best_save_counter=best_save_counter,
            )

            if log_interval is not None and iteration % log_interval == 0:
                fps = int(total_steps_in_rollout / iter_duration)

                metric_logger.log({
                    'iteration': iteration,
                    'timesteps': current_timesteps,
                    'lr': self.learning_rate,
                    **metrics,
                    'ep_rew_ema': current_episode_return_ema,
                    'best_ep_rew_ema': best_episode_return_ema,
                    'fps': fps,
                })

            if save_interval is not None and run_dir is not None and iteration % save_interval == 0:
                save_path = run_dir / f"models/model_{current_timesteps}_steps.pt"
                self.save(save_path, save_optimizer=save_optimizer, return_ema=current_episode_return_ema)
                logger.log("SAVE", f"Saved model to {save_path}")

        if run_dir is not None:
            save_path = run_dir / f"models/model_{current_timesteps}_steps_final.pt"
            self.save(save_path, save_optimizer=save_optimizer, return_ema=episode_return_ema.get())
            logger.log("SAVE", f"Saved final model to {save_path}")

        metric_logger.close()

        return self

    def _maybe_save_best_ema_model(
            self,
            iteration: int,
            best_models_dir: pathlib.Path | None,
            best_rotation_n: int,
            save_optimizer: bool,
            current_episode_return_ema: float,
            best_episode_return_ema: float | None,
            best_save_counter: int,
    ) -> tuple[float | None, int]:
        if ((best_episode_return_ema is not None and current_episode_return_ema <= best_episode_return_ema)
                or iteration < MIN_ITERATIONS_FOR_BEST):
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

        self.save(best_save_path, save_optimizer=save_optimizer, return_ema=current_episode_return_ema)
        logger.log(
            "SAVE",
            f"Saved best-EMA model to {best_save_path} (ep_rew_ema={best_episode_return_ema:.4f})",
        )
        return best_episode_return_ema, best_save_counter

    def _write_run_metadata(self, run_dir: pathlib.Path, extra_run_metadata: dict[str, Any] | None) -> None:
        metadata: dict[str, Any] = {
            "algorithm": "PPO",
            "hyper_parameters": self.get_hyper_parameters(),
            "policy_repr": str(self.policy),
            "env_repr": str(self.env),
        }
        if extra_run_metadata:
            metadata.update(extra_run_metadata)
        metadata["timesteps"] = int(self.n_total_timesteps)

        def _parse_step_from_metadata_filename(path: pathlib.Path) -> int | None:
            if path.name == "run_metadata.json":
                return None
            if not (path.name.startswith("run_metadata_") and path.name.endswith(".json")):
                return None

            suffix = path.name.removeprefix("run_metadata_").removesuffix(".json")
            step_str = suffix.split("_", 1)[0]
            if not step_str.isdigit():
                return None
            return int(step_str)

        def _read_metadata_json(path: pathlib.Path) -> dict[str, Any] | None:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None

        existing_metadata_files = list(run_dir.glob("run_metadata*.json"))
        if existing_metadata_files:
            def _sort_key(path: pathlib.Path) -> tuple[int, float, str]:
                step = _parse_step_from_metadata_filename(path)
                step_key = step if step is not None else -1
                return step_key, path.stat().st_mtime, path.name

            latest_path = max(existing_metadata_files, key=_sort_key)
            latest_metadata = _read_metadata_json(latest_path)
            if latest_metadata == metadata:
                return

        step = int(self.n_total_timesteps)
        base_path = run_dir / f"run_metadata_{step}.json"
        if base_path.exists():
            existing = _read_metadata_json(base_path)
            if existing == metadata:
                return

            suffix_idx = 1
            while (run_dir / f"run_metadata_{step}_{suffix_idx}.json").exists():
                suffix_idx += 1
            metadata_path = run_dir / f"run_metadata_{step}_{suffix_idx}.json"
        else:
            metadata_path = base_path

        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def save(self, path: str | pathlib.Path, save_optimizer: bool = True, return_ema: Optional[float] = None) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        env_state = []
        current_env = self.env
        while hasattr(current_env, 'env'):
            wrapper_state = {}
            if hasattr(current_env, 'local_obs_rms'):
                 wrapper_state['local_obs_rms'] = current_env.local_obs_rms
            if hasattr(current_env, 'global_obs_rms'):
                 wrapper_state['global_obs_rms'] = current_env.global_obs_rms
            if hasattr(current_env, 'return_rms'):
                 wrapper_state['return_rms'] = current_env.return_rms
            
            if wrapper_state:
                wrapper_state['wrapper_class'] = type(current_env).__name__
                env_state.append(wrapper_state)
            
            current_env = current_env.env
            
        save_dict = {
            'policy_state_dict': self.policy.state_dict(),
            'env_state': env_state,
            'n_total_updates': self.n_total_updates,
            'n_total_timesteps': self.n_total_timesteps,
            'return_ema': return_ema,
        }
        
        if save_optimizer:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
            
        torch.save(save_dict, path)

        metadata = {
            "n_total_updates": int(self.n_total_updates),
            "n_total_timesteps": int(self.n_total_timesteps),
            "return_ema": return_ema,
        }
        metadata_path = path.with_name(f"{path.name}.json")
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def load(self, path: str | pathlib.Path) -> None:
        checkpoint = load_checkpoint(path)
        self.policy.load_state_dict(extract_policy_state_dict(checkpoint))

        optimizer_state_dict = extract_optimizer_state_dict(checkpoint)
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)

        if isinstance(checkpoint, dict):
            self.n_total_updates = checkpoint.get("n_total_updates", 0)
            self.n_total_timesteps = checkpoint.get("n_total_timesteps", 0)

        apply_env_state(self.env, extract_env_state(checkpoint))

