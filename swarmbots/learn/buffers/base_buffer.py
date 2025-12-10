import abc
from dataclasses import dataclass

import torch
from gymnasium import spaces

@dataclass
class Episode:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor


class ShortTermEpisodeBuffer:
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
            (n_envs, max_episode_length + 1, n_agents, *agent_obs_shape),
            dtype=storage_dtype, device=storage_device
        )
        self.global_obs = torch.zeros(
            (n_envs, max_episode_length + 1, *global_obs_shape),
            dtype=storage_dtype, device=storage_device
        )
        self.actions = torch.zeros(
            (n_envs, max_episode_length, n_agents, n_agent_actions),
            dtype=storage_dtype, device=storage_device
        )
        self.rewards = torch.zeros(
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
            dones: torch.Tensor,
            next_local_obs: torch.Tensor,
            next_global_obs: torch.Tensor,
    ):
        self.local_obs[self.env_arange, self.step] = local_obs
        self.global_obs[self.env_arange, self.step] = global_obs
        self.actions[self.env_arange, self.step] = actions
        self.rewards[self.env_arange, self.step] = rewards

        self.step += 1

        done_envs = torch.where(dones)[0]
        for env in done_envs.tolist():
            step = int(self.step[env].item())
            self.local_obs[env, step] = next_local_obs[env]
            self.global_obs[env, step] = next_global_obs[env]
            yield Episode(
                local_obs=self.local_obs[env, :step+1].clone(),
                global_obs=self.global_obs[env, :step+1].clone(),
                actions=self.actions[env, :step].clone(),
                rewards=self.rewards[env, :step].clone(),
            )
            self.step[env] = 0



class BaseBuffer(abc.ABC):

    def __init__(
            self,
            n_episodes: int,
            max_episode_length: int,
            observation_space: spaces.Dict,
            action_space: spaces.Dict,
            storage_device: torch.device | str = "cpu",
            sampling_device: torch.device | str = "auto",
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
        for name, space in action_space.items():
            assert len(space.shape) == 2
            assert space.shape[0] == self.n_agents
            self.n_agent_actions += space.shape[1]

        self.storage_device = storage_device
        self.sampling_device = sampling_device
