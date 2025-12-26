import time
import pathlib
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.logger import MetricLogger
from swarmbots.learn.ppo.ppo_policy import PPOPolicy
from swarmbots.learn.ppo.ppo_rollout_buffer import PPOEpisode, PPORolloutBuffer, PPOSampler


@torch.no_grad()
def collect_whole_episodes(
        env: BaseLearnEnvWrapper,
        policy: PPOPolicy,
        buffer: PPORolloutBuffer,
        device: torch.device,
) -> tuple[list[PPOEpisode], list[dict]]:
    buffer.reset()
    obs, info = env.reset()
    is_final = torch.zeros((buffer.n_envs,), dtype=torch.bool, device=device)
    was_terminated = torch.zeros((buffer.n_envs,), dtype=torch.bool, device=device)
    
    episode_infos = []

    while not buffer.is_ready():
        local_obs = obs['local_obs']
        global_obs = obs['global_obs']

        actions, log_probs, values = policy(local_obs, global_obs)
        
        values = values.clone()
        values[was_terminated] = 0.0

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

    return buffer.get_whole_episodes(), episode_infos


# based on https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
class PPO:
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    """

    def __init__(
            self,
            policy: PPOPolicy,
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
            device: str | torch.device = "auto",
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

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.policy.to(self.device)

        self.rollout_buffer = PPORolloutBuffer(
            n_episodes=n_episodes_per_rollout,
            max_episode_length=max_episode_length,
            observation_space=env.observation_space,
            action_space=env.action_space,
            gamma=gamma,
            gae_lambda=gae_lambda,
            storage_device=self.device,
            sampling_device=self.device,
        )

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.n_total_updates = 0


    def train(self) -> dict[str, float]:
        """
        Update policy using the currently gathered rollout buffer.
        """
        self.policy.train()
        
        episodes, episode_infos = collect_whole_episodes(self.env, self.policy, self.rollout_buffer, self.device)
        
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
                
                # sum over agents
                log_prob = log_probs.sum(dim=1)
                entropy = entropies.sum(dim=1) if entropies is not None else None
                old_log_prob = batch.log_probs.sum(dim=1)

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
                    logger.info(f'Early stopping at epoch {epoch}, batch {i} due to reaching max kl: {approx_kl_div:.2f}')
                    break

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                n_updates += 1

            if not continue_training:
                break


        metrics = {}
        for i, dist in enumerate(self.policy.action_dist.distributions):
            if hasattr(dist, "log_stds"):
                metrics[f'std{i}'] = torch.exp(dist.log_stds).mean().item()

        self.n_total_updates += n_updates
        metrics.update({
            'ent_loss': np.mean(entropy_losses),
            'act_loss': np.mean(pg_losses),
            'val_loss': np.mean(value_losses),
            'approx_kl': np.mean(approx_kl_divs),
            'clip_frac': np.mean(clip_fractions),
            'upd': n_updates,
            'tot_upd': self.n_total_updates,
            'expl_var': explained_var,
        })
        
        if len(episode_infos) > 0:
            rewards = [ep['r'] for ep in episode_infos]
            lengths = [ep['l'] for ep in episode_infos]
            timings = [ep['t'] for ep in episode_infos]
            metrics['ep_rew'] = np.mean(rewards)
            metrics['ep_len'] = np.mean(lengths)
            metrics['ep_time'] = np.mean(timings)
        else:
            metrics['ep_rew'] = None
            metrics['ep_len'] = None
            metrics['ep_time'] = None

        return metrics

    def learn(
            self,
            total_timesteps: int,
            log_interval: int = 1,
            save_interval: Optional[int] = None,
            save_path_prefix: Optional[str | pathlib.Path] = None,
            save_optimizer: bool = True,
            csv_log_dir: Optional[str | pathlib.Path] = None,
    ):

        current_timesteps = 0
        iteration = 0
        
        metric_logger = MetricLogger(log_dir=csv_log_dir)

        start_time = time.time()

        while current_timesteps < total_timesteps:
            metrics = self.train()
            
            total_steps_in_rollout = sum(len(ep.rewards) for ep in self.rollout_buffer.episodes)
            current_timesteps += total_steps_in_rollout
            iteration += 1

            if log_interval is not None and iteration % log_interval == 0:
                fps = int(current_timesteps / (time.time() - start_time))

                metric_logger.log({
                    'iteration': iteration,
                    'timesteps': current_timesteps,
                    **metrics,
                    'fps': fps,
                })

            if save_interval is not None and save_path_prefix is not None and iteration % save_interval == 0:
                 save_path = f"{save_path_prefix}_{current_timesteps}_steps.pt"
                 self.save(save_path, save_optimizer=save_optimizer)
                 logger.info(f"Saved model to {save_path}")

        if save_path_prefix is not None:
             save_path = f"{save_path_prefix}_final.pt"
             self.save(save_path, save_optimizer=save_optimizer)
             logger.info(f"Saved final model to {save_path}")

        metric_logger.close()

        return self

    def save(self, path: str | pathlib.Path, save_optimizer: bool = True):
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
        }
        
        if save_optimizer:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
            
        torch.save(save_dict, path)

    def load(self, path: str | pathlib.Path):
        save_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(save_dict['policy_state_dict'])
        if 'optimizer_state_dict' in save_dict:
            self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])

        self.n_total_updates = save_dict.get('n_total_updates', 0)
        
        env_state = save_dict.get('env_state', [])
        
        current_env = self.env
        state_idx = 0
        while hasattr(current_env, 'env'):
            relevant = False
            if hasattr(current_env, 'local_obs_rms') or hasattr(current_env, 'global_obs_rms') or hasattr(current_env, 'return_rms'):
                relevant = True
            
            if relevant:
                 if state_idx < len(env_state):
                     saved_state = env_state[state_idx]
                     if saved_state['wrapper_class'] != type(current_env).__name__:
                         logger.warning(f"Wrapper type mismatch during load: {saved_state['wrapper_class']} vs {type(current_env).__name__}")
                     
                     if 'local_obs_rms' in saved_state and hasattr(current_env, 'local_obs_rms'):
                         self._copy_rms(saved_state['local_obs_rms'], current_env.local_obs_rms)
                     if 'global_obs_rms' in saved_state and hasattr(current_env, 'global_obs_rms'):
                         self._copy_rms(saved_state['global_obs_rms'], current_env.global_obs_rms)
                     if 'return_rms' in saved_state and hasattr(current_env, 'return_rms'):
                         self._copy_rms(saved_state['return_rms'], current_env.return_rms)
                     
                     state_idx += 1
            
            current_env = current_env.env
            
    def _copy_rms(self, src, dst):
        dst.mean = src.mean.copy()
        dst.var = src.var.copy()
        dst.count = src.count

