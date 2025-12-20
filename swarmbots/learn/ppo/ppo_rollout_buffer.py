from dataclasses import dataclass
from typing import Optional

import torch
from gymnasium import spaces
from loguru import logger

from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace

MaybeTensor = Optional[torch.Tensor]


@dataclass
class PPOEpisode:
    local_obs: torch.Tensor  # (n_steps, n_agents, n_local_obs)
    global_obs: torch.Tensor  # (n_steps, n_global_obs)
    actions: torch.Tensor  # (n_steps, n_agents, n_actions_per_agent)
    rewards: MaybeTensor  # (n_steps)
    log_probs: torch.Tensor  # (n_steps, n_agents)
    values: MaybeTensor  # (n_steps)

    final_local_obs: MaybeTensor  # (n_agents, n_local_obs)
    final_global_obs: MaybeTensor  # (n_global_obs)
    final_value: MaybeTensor  # (1)

    returns: MaybeTensor = None  # (n_steps)
    advantages: MaybeTensor = None  # (n_steps)

    def compute_gae(self, gamma: float, gae_lambda: float):
        assert self.rewards is not None
        assert self.values is not None
        assert self.final_value is not None

        self.advantages = torch.zeros_like(self.rewards)
        last_gae_lam = 0.0
        num_steps = len(self.rewards)

        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_val = self.final_value
            else:
                next_val = self.values[step + 1]

            delta = self.rewards[step] + gamma * next_val - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values


class PPOEpisodeAccumulator:

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

    @property
    def total_steps(self) -> int:
        return self.step.sum().item()

    def add(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            log_probs: torch.Tensor,
            values: torch.Tensor,
            is_final: torch.Tensor,
    ):
        active_env_indices = torch.where(torch.logical_not(is_final))[0]
        if len(active_env_indices) > 0:
            step_indices = self.step[active_env_indices]
            self.local_obs[active_env_indices, step_indices] = local_obs[active_env_indices]
            self.global_obs[active_env_indices, step_indices] = global_obs[active_env_indices]
            self.actions[active_env_indices, step_indices] = actions[active_env_indices]
            self.rewards[active_env_indices, step_indices] = rewards[active_env_indices]
            self.log_probs[active_env_indices, step_indices] = log_probs[active_env_indices]
            self.values[active_env_indices, step_indices] = values[active_env_indices]

            self.step[active_env_indices] += 1

        final_env_indices = torch.where(is_final)[0]
        for final_env_idx in final_env_indices.tolist():
            yield self.construct_episode(
                final_env_idx,
                final_local_obs=local_obs[final_env_idx],
                final_global_obs=global_obs[final_env_idx],
                final_value=values[final_env_idx],
            )
            self.step[final_env_idx] = 0

    def construct_episode(
            self,
            env: int,
            final_local_obs: torch.Tensor,
            final_global_obs: torch.Tensor,
            final_value: torch.Tensor,
    ) -> PPOEpisode:
        step = int(self.step[env].item())
        return PPOEpisode(
            local_obs=self.local_obs[env, :step].clone(),
            global_obs=self.global_obs[env, :step].clone(),
            actions=self.actions[env, :step].clone(),
            rewards=self.rewards[env, :step].clone(),
            log_probs=self.log_probs[env, :step].clone(),
            values=self.values[env, :step].clone(),
            final_local_obs=final_local_obs.clone(),
            final_global_obs=final_global_obs.clone(),
            final_value=final_value.clone(),
        )

    def reset(self):
        self.step[:] = 0


class PPORolloutBuffer:

    def __init__(
            self,
            n_episodes: int,
            max_episode_length: int,
            observation_space: spaces.Dict,
            action_space: VectorHybridActionSpace,
            gamma: float,
            gae_lambda: float,
            storage_device: torch.device | str = 'cpu',
            storage_dtype: torch.dtype = torch.float32,
            sampling_device: torch.device | str = 'auto',
            sampling_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.n_episodes = n_episodes
        self.max_episode_length = max_episode_length

        self.observation_space = observation_space
        self.action_space = action_space

        self.local_obs_space = observation_space['local_obs']
        self.n_agents = self.local_obs_space.shape[1]
        self.agent_obs_shape = self.local_obs_space.shape[2:]

        self.global_obs_space = observation_space['global_obs']
        self.global_obs_shape = self.global_obs_space.shape[1:]

        self.n_envs = self.local_obs_space.shape[0]

        assert action_space.n_agents == self.n_agents
        self.n_agent_actions = action_space.total_agent_action_dim

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.storage_device = storage_device
        self.storage_dtype = storage_dtype
        if sampling_device == "auto":
            self.sampling_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.sampling_device = sampling_device
        self.sampling_dtype = sampling_dtype

        self.episodes = list()
        self.accumulator = PPOEpisodeAccumulator(
            n_envs=self.n_envs,
            max_episode_length=self.max_episode_length,
            n_agents=self.n_agents,
            agent_obs_shape=self.agent_obs_shape,
            global_obs_shape=self.global_obs_shape,
            n_agent_actions=self.n_agent_actions,
            storage_device=self.storage_device,
            storage_dtype=self.storage_dtype,
        )

    def is_ready(self):
        return len(self.episodes) >= self.n_episodes

    def add(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            log_probs: torch.Tensor,
            values: torch.Tensor,
            is_final: torch.Tensor,
    ):
        if self.is_ready():
            logger.warning('Adding into buffer despite being ready')

        new_episodes = self.accumulator.add(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            rewards=rewards,
            log_probs=log_probs,
            values=values,
            is_final=is_final,
        )
        for new_ep in new_episodes:
            new_ep.compute_gae(self.gamma, self.gae_lambda)
            self.episodes.append(new_ep)

    def reset(self):
        steps_remaining_in_acc = self.accumulator.total_steps
        if steps_remaining_in_acc > 0:
            logger.warning(f'Resetting buffer & accumulator with {steps_remaining_in_acc} steps in the accumulator')

        self.accumulator.reset()
        self.episodes = []

    def get_whole_episodes(self) -> list[PPOEpisode]:
        return [
            PPOEpisode(
                local_obs=ep.local_obs.to(device=self.sampling_device, dtype=self.sampling_dtype),
                global_obs=ep.global_obs.to(device=self.sampling_device, dtype=self.sampling_dtype),
                actions=ep.actions.to(device=self.sampling_device, dtype=self.sampling_dtype),
                rewards=ep.rewards.to(device=self.sampling_device, dtype=self.sampling_dtype),
                log_probs=ep.log_probs.to(device=self.sampling_device, dtype=self.sampling_dtype),
                values=ep.values.to(device=self.sampling_device, dtype=self.sampling_dtype),
                final_local_obs=ep.final_local_obs.to(device=self.sampling_device, dtype=self.sampling_dtype),
                final_global_obs=ep.final_global_obs.to(device=self.sampling_device, dtype=self.sampling_dtype),
                final_value=ep.final_value.to(device=self.sampling_device, dtype=self.sampling_dtype),
                returns=ep.returns.to(device=self.sampling_device, dtype=self.sampling_dtype),
                advantages=ep.advantages.to(device=self.sampling_device, dtype=self.sampling_dtype),
            )
            for ep in self.episodes
        ]

    def get_whole_episodes_minimal(self) -> list[PPOEpisode]:
        return [
            PPOEpisode(
                local_obs=ep.local_obs.to(device=self.sampling_device, dtype=self.sampling_dtype),
                global_obs=ep.global_obs.to(device=self.sampling_device, dtype=self.sampling_dtype),
                actions=ep.actions.to(device=self.sampling_device, dtype=self.sampling_dtype),
                rewards=None,
                log_probs=ep.log_probs.to(device=self.sampling_device, dtype=self.sampling_dtype),
                values=ep.values.to(device=self.sampling_device, dtype=self.sampling_dtype),
                final_local_obs=None,
                final_global_obs=None,
                final_value=None,
                returns=ep.returns.to(device=self.sampling_device, dtype=self.sampling_dtype),
                advantages=ep.advantages.to(device=self.sampling_device, dtype=self.sampling_dtype),
            )
            for ep in self.episodes
        ]


@dataclass
class PPOSamples:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    history_embeddings: Optional[torch.Tensor]


class PPOSampler:

    def __init__(
            self,
            episodes: list[PPOEpisode],
            history_embeddings: Optional[list[torch.Tensor]]
    ):
        self.local_obs = torch.concatenate(tuple(ep.local_obs for ep in episodes), dim=0)
        self.global_obs = torch.concatenate(tuple(ep.global_obs for ep in episodes), dim=0)
        self.actions = torch.concatenate(tuple(ep.actions for ep in episodes), dim=0)
        self.log_probs = torch.concatenate(tuple(ep.log_probs for ep in episodes), dim=0)
        self.values = torch.concatenate(tuple(ep.values for ep in episodes), dim=0)
        self.returns = torch.concatenate(tuple(ep.returns for ep in episodes), dim=0)
        self.advantages = torch.concatenate(tuple(ep.advantages for ep in episodes), dim=0)
        self.history_embeddings: Optional[torch.Tensor] = None
        if history_embeddings is not None:
            self.history_embeddings = torch.concatenate(tuple(history_embeddings), dim=0)

    def sample(self, batch_size: int, drop_last: bool = True):
        n_samples = self.local_obs.shape[0]
        indices = torch.randperm(n_samples)

        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]

            if drop_last and len(batch_indices) < batch_size:
                continue

            history_embeddings = None
            if self.history_embeddings is not None:
                history_embeddings = self.history_embeddings[batch_indices]

            yield PPOSamples(
                local_obs=self.local_obs[batch_indices],
                global_obs=self.global_obs[batch_indices],
                actions=self.actions[batch_indices],
                log_probs=self.log_probs[batch_indices],
                values=self.values[batch_indices],
                returns=self.returns[batch_indices],
                advantages=self.advantages[batch_indices],
                history_embeddings=history_embeddings
            )



