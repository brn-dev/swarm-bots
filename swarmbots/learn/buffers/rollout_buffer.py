from dataclasses import dataclass

import torch
from gymnasium import spaces


@dataclass
class Episode:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor

    returns: torch.Tensor = None
    advantages: torch.Tensor = None

    def compute_gae(self, gamma: float, gae_lambda: float, last_val: float | torch.Tensor = 0.0):
        self.advantages = torch.zeros_like(self.rewards)
        last_gae_lam = 0.0
        num_steps = len(self.rewards)

        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_val = last_val
            else:
                next_val = self.values[step + 1]

            delta = self.rewards[step] + gamma * next_val - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values


class EpisodeAccumulator:

    def __init__(
            self,
            n_envs: int,
            max_episode_length: int,
            n_agents: int,
            agent_obs_shape: tuple[int, ...],
            global_obs_shape: tuple[int, ...],
            n_agent_actions: int,
            storage_device: torch.device | str,
            storage_dtype: torch.dtype
    ):
        self.local_obs = torch.zeros(
            (n_envs, max_episode_length, n_agents, *agent_obs_shape),
            dtype=storage_dtype, device=storage_device
        )
        self.global_obs = torch.zeros(
            (n_envs, max_episode_length, *global_obs_shape),
            dtype=storage_dtype, device=storage_device
        )
        self.actions = torch.zeros(
            (n_envs, max_episode_length, n_agents, n_agent_actions),
            dtype=storage_dtype, device=storage_device
        )
        self.log_probs = torch.zeros(
            (n_envs, max_episode_length, n_agents),
            dtype=storage_dtype, device=storage_device
        )
        self.rewards = torch.zeros(
            (n_envs, max_episode_length),
            dtype=storage_dtype, device=storage_device
        )
        self.values = torch.zeros(
            (n_envs, max_episode_length),
            dtype=storage_dtype, device=storage_device
        )
        self.step = torch.zeros(n_envs, dtype=torch.long, device=storage_device)
        self.env_arange = torch.arange(n_envs, dtype=torch.long, device=storage_device)

    def add(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            log_probs: torch.Tensor,
            values: torch.Tensor,
            dones: torch.Tensor,
    ):
        self.local_obs[self.env_arange, self.step] = local_obs
        self.global_obs[self.env_arange, self.step] = global_obs
        self.actions[self.env_arange, self.step] = actions
        self.rewards[self.env_arange, self.step] = rewards
        self.log_probs[self.env_arange, self.step] = log_probs
        self.values[self.env_arange, self.step] = values

        self.step += 1

        done_envs = torch.where(dones)[0]
        for env in done_envs.tolist():
            step = int(self.step[env].item())
            yield Episode(
                local_obs=self.local_obs[env, :step].clone(),
                global_obs=self.global_obs[env, :step].clone(),
                actions=self.actions[env, :step].clone(),
                rewards=self.rewards[env, :step].clone(),
                log_probs=self.log_probs[env, :step].clone(),
                values=self.values[env, :step].clone(),
            )
            self.step[env] = 0


class RolloutBuffer:

    def __init__(
            self,
            n_episodes: int,
            max_episode_length: int,
            observation_space: spaces.Dict,
            action_space: spaces.Dict,
            storage_device: torch.device | str = 'cpu',
            storage_dtype: torch.dtype = torch.float32,
            sampling_device: torch.device | str = 'auto',
            sampling_dtype: torch.dtype = torch.float32,
            n_envs: int = 1,
    ):
        super().__init__()
        self.n_episodes = n_episodes
        self.max_episode_length = max_episode_length
        self.n_envs = n_envs

        self.observation_space = observation_space
        self.action_space = action_space

        self.local_obs_space = observation_space['local_obs']
        self.n_agents = self.local_obs_space.shape[0]
        self.agent_obs_shape = self.local_obs_space.shape[1:]

        self.global_obs_space = observation_space['global_obs']
        self.global_obs_shape = self.global_obs_space.shape

        self.n_agent_actions = 0
        for space in action_space.values():
            assert len(space.shape) == 2
            assert space.shape[0] == self.n_agents
            self.n_agent_actions += space.shape[1]

        self.storage_device = storage_device
        self.storage_dtype = storage_dtype
        self.sampling_device = sampling_device
        self.sampling_dtype = sampling_dtype

