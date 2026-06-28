from dataclasses import dataclass
from typing import Optional

import torch
from gymnasium import spaces

from swarmbots.learn.base_sampler import BaseSampler, BaseSamplerConfig, BatchIndices
from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.torch_device import as_device

MaybeTensor = Optional[torch.Tensor]


@dataclass
class OffPolicyEpisodeSegment:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    hidden_local_vars: torch.Tensor
    hidden_global_vars: torch.Tensor
    agent_mask: MaybeTensor
    previous_actions: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    next_local_obs: torch.Tensor
    next_global_obs: torch.Tensor
    next_hidden_local_vars: torch.Tensor
    next_hidden_global_vars: torch.Tensor
    next_agent_mask: MaybeTensor
    next_previous_actions: torch.Tensor
    is_true_episode_start: bool = True
    rollout_env_idx: int | None = None
    rollout_start_step: int = 0

    @property
    def n_steps(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def dones(self) -> torch.Tensor:
        return torch.logical_or(self.terminations, self.truncations)


@dataclass
class OffPolicyTransitionBatch:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    hidden_local_vars: torch.Tensor
    hidden_global_vars: torch.Tensor
    agent_mask: MaybeTensor
    previous_actions: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    dones: torch.Tensor
    next_local_obs: torch.Tensor
    next_global_obs: torch.Tensor
    next_hidden_local_vars: torch.Tensor
    next_hidden_global_vars: torch.Tensor
    next_agent_mask: MaybeTensor
    next_previous_actions: torch.Tensor


@dataclass(frozen=True)
class OffPolicySamplerConfig(BaseSamplerConfig):
    replacement: bool = True


class OffPolicyEpisodeAccumulator:
    def __init__(
            self,
            n_envs: int,
            max_segment_length: int,
            n_agents: int,
            agent_obs_shape: tuple[int, ...],
            global_obs_shape: tuple[int, ...],
            hidden_local_vars_shape: tuple[int, ...],
            hidden_global_vars_shape: tuple[int, ...],
            n_agent_actions: int,
            has_agent_mask: bool,
            storage_device: torch.device | str,
            storage_dtype: torch.dtype,
    ):
        if max_segment_length <= 0:
            raise ValueError(f"max_segment_length must be > 0, got {max_segment_length}")

        self.max_segment_length = max_segment_length
        self.local_obs = torch.zeros(
            (n_envs, max_segment_length, n_agents, *agent_obs_shape),
            dtype=storage_dtype,
            device=storage_device,
        )
        self.global_obs = torch.zeros(
            (n_envs, max_segment_length, *global_obs_shape),
            dtype=storage_dtype,
            device=storage_device,
        )
        self.hidden_local_vars = torch.zeros(
            (n_envs, max_segment_length, n_agents, *hidden_local_vars_shape),
            dtype=storage_dtype,
            device=storage_device,
        )
        self.hidden_global_vars = torch.zeros(
            (n_envs, max_segment_length, *hidden_global_vars_shape),
            dtype=storage_dtype,
            device=storage_device,
        )
        self.agent_mask: MaybeTensor = torch.zeros(
            (n_envs, max_segment_length, n_agents),
            dtype=torch.bool,
            device=storage_device,
        ) if has_agent_mask else None
        self.previous_actions = torch.zeros(
            (n_envs, max_segment_length, n_agents, n_agent_actions),
            dtype=storage_dtype,
            device=storage_device,
        )
        self.actions = torch.zeros(
            (n_envs, max_segment_length, n_agents, n_agent_actions),
            dtype=storage_dtype,
            device=storage_device,
        )
        self.rewards = torch.zeros((n_envs, max_segment_length), dtype=storage_dtype, device=storage_device)
        self.terminations = torch.zeros((n_envs, max_segment_length), dtype=torch.bool, device=storage_device)
        self.truncations = torch.zeros((n_envs, max_segment_length), dtype=torch.bool, device=storage_device)
        self.next_local_obs = torch.zeros_like(self.local_obs)
        self.next_global_obs = torch.zeros_like(self.global_obs)
        self.next_hidden_local_vars = torch.zeros_like(self.hidden_local_vars)
        self.next_hidden_global_vars = torch.zeros_like(self.hidden_global_vars)
        self.next_agent_mask: MaybeTensor = torch.zeros_like(self.agent_mask) if self.agent_mask is not None else None
        self.next_previous_actions = torch.zeros_like(self.previous_actions)
        self.is_true_episode_start = torch.zeros(n_envs, dtype=torch.bool, device=storage_device)
        self.rollout_start_steps = torch.zeros(n_envs, dtype=torch.long, device=storage_device)
        self.step = torch.zeros(n_envs, dtype=torch.long, device=storage_device)

    def add(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor,
            hidden_global_vars: torch.Tensor,
            agent_mask: MaybeTensor,
            previous_actions: MaybeTensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            terminations: torch.Tensor,
            truncations: torch.Tensor,
            next_obs: dict[str, torch.Tensor],
            episode_start_mask: torch.Tensor,
            rollout_step_idx: int,
    ) -> list[OffPolicyEpisodeSegment]:
        env_indices = torch.arange(self.step.shape[0], device=self.step.device)
        step_indices = self.step
        new_segment_env_indices = torch.where(step_indices == 0)[0]
        if len(new_segment_env_indices) > 0:
            self.is_true_episode_start[new_segment_env_indices] = episode_start_mask[new_segment_env_indices]
            self.rollout_start_steps[new_segment_env_indices] = rollout_step_idx

        self.local_obs[env_indices, step_indices] = local_obs
        self.global_obs[env_indices, step_indices] = global_obs
        self.hidden_local_vars[env_indices, step_indices] = hidden_local_vars
        self.hidden_global_vars[env_indices, step_indices] = hidden_global_vars
        if agent_mask is not None:
            self.agent_mask[env_indices, step_indices] = agent_mask
        if previous_actions is not None:
            self.previous_actions[env_indices, step_indices] = previous_actions
        else:
            self.previous_actions[env_indices, step_indices].zero_()
        self.actions[env_indices, step_indices] = actions
        self.rewards[env_indices, step_indices] = rewards
        self.terminations[env_indices, step_indices] = terminations
        self.truncations[env_indices, step_indices] = truncations
        self.next_local_obs[env_indices, step_indices] = next_obs["local_obs"]
        self.next_global_obs[env_indices, step_indices] = next_obs["global_obs"]
        self.next_hidden_local_vars[env_indices, step_indices] = next_obs["hidden_local_vars"]
        self.next_hidden_global_vars[env_indices, step_indices] = next_obs["hidden_global_vars"]
        next_agent_mask = next_obs.get("agent_mask", None)
        if next_agent_mask is not None:
            self.next_agent_mask[env_indices, step_indices] = next_agent_mask
        self.next_previous_actions[env_indices, step_indices] = actions

        self.step += 1
        dones = torch.logical_or(terminations, truncations)
        segment_complete = torch.logical_or(dones, self.step >= self.max_segment_length)

        segments: list[OffPolicyEpisodeSegment] = []
        for env_idx in torch.nonzero(segment_complete, as_tuple=False).flatten().tolist():
            segments.append(self.construct_segment(env_idx))
            self.step[env_idx] = 0
            self.is_true_episode_start[env_idx] = False
            self.rollout_start_steps[env_idx] = rollout_step_idx + 1
        return segments

    def dump_partial_segments(self) -> list[OffPolicyEpisodeSegment]:
        segments = [
            self.construct_segment(env_idx)
            for env_idx in range(self.step.shape[0])
            if int(self.step[env_idx].item()) > 0
        ]
        self.step.zero_()
        self.is_true_episode_start.zero_()
        return segments

    def construct_segment(self, env: int) -> OffPolicyEpisodeSegment:
        step = int(self.step[env].item())
        if step <= 0:
            raise ValueError("Cannot construct an empty off-policy segment.")

        agent_mask = None if self.agent_mask is None else self.agent_mask[env, :step].clone()
        next_agent_mask = None if self.next_agent_mask is None else self.next_agent_mask[env, :step].clone()
        return OffPolicyEpisodeSegment(
            local_obs=self.local_obs[env, :step].clone(),
            global_obs=self.global_obs[env, :step].clone(),
            hidden_local_vars=self.hidden_local_vars[env, :step].clone(),
            hidden_global_vars=self.hidden_global_vars[env, :step].clone(),
            agent_mask=agent_mask,
            previous_actions=self.previous_actions[env, :step].clone(),
            actions=self.actions[env, :step].clone(),
            rewards=self.rewards[env, :step].clone(),
            terminations=self.terminations[env, :step].clone(),
            truncations=self.truncations[env, :step].clone(),
            next_local_obs=self.next_local_obs[env, :step].clone(),
            next_global_obs=self.next_global_obs[env, :step].clone(),
            next_hidden_local_vars=self.next_hidden_local_vars[env, :step].clone(),
            next_hidden_global_vars=self.next_hidden_global_vars[env, :step].clone(),
            next_agent_mask=next_agent_mask,
            next_previous_actions=self.next_previous_actions[env, :step].clone(),
            is_true_episode_start=bool(self.is_true_episode_start[env].item()),
            rollout_env_idx=env,
            rollout_start_step=int(self.rollout_start_steps[env].item()),
        )

    def reset(self) -> None:
        self.step.zero_()
        self.is_true_episode_start.zero_()
        self.rollout_start_steps.zero_()


class OffPolicyReplayBuffer:
    def __init__(
            self,
            capacity_transitions: int,
            max_segment_length: int,
            observation_space: spaces.Dict,
            action_space: VectorHybridActionSpace,
            rollout_device: torch.device | str = "cpu",
            rollout_dtype: torch.dtype = torch.float32,
            train_device: torch.device | str = "auto",
            train_dtype: torch.dtype = torch.float32,
    ):
        if capacity_transitions <= 0:
            raise ValueError(f"capacity_transitions must be > 0, got {capacity_transitions}")

        self.capacity_transitions = capacity_transitions
        self.max_segment_length = max_segment_length
        self.observation_space = observation_space
        self.action_space = action_space

        self.local_obs_space = observation_space["local_obs"]
        self.n_envs = self.local_obs_space.shape[0]
        self.n_agents = self.local_obs_space.shape[1]
        self.agent_obs_shape = self.local_obs_space.shape[2:]
        self.global_obs_shape = observation_space["global_obs"].shape[1:]
        self.hidden_local_vars_shape = observation_space["hidden_local_vars"].shape[2:]
        self.hidden_global_vars_shape = observation_space["hidden_global_vars"].shape[1:]
        self.has_agent_mask = "agent_mask" in observation_space.keys() and observation_space["agent_mask"] is not None

        if action_space.n_agents != self.n_agents:
            raise ValueError(f"Action space n_agents={action_space.n_agents} does not match obs n_agents={self.n_agents}")
        self.n_agent_actions = action_space.total_agent_action_dim

        self.rollout_device = as_device(rollout_device)
        self.rollout_dtype = rollout_dtype
        self.train_device = as_device(train_device)
        self.train_dtype = train_dtype
        self.segments: list[OffPolicyEpisodeSegment] = []
        self.n_transitions = 0
        self.accumulator = OffPolicyEpisodeAccumulator(
            n_envs=self.n_envs,
            max_segment_length=max_segment_length,
            n_agents=self.n_agents,
            agent_obs_shape=self.agent_obs_shape,
            global_obs_shape=self.global_obs_shape,
            hidden_local_vars_shape=self.hidden_local_vars_shape,
            hidden_global_vars_shape=self.hidden_global_vars_shape,
            n_agent_actions=self.n_agent_actions,
            has_agent_mask=self.has_agent_mask,
            storage_device=self.rollout_device,
            storage_dtype=self.rollout_dtype,
        )

    def add(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor,
            hidden_global_vars: torch.Tensor,
            agent_mask: MaybeTensor,
            previous_actions: MaybeTensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            terminations: torch.Tensor,
            truncations: torch.Tensor,
            next_obs: dict[str, torch.Tensor],
            episode_start_mask: torch.Tensor,
            rollout_step_idx: int,
    ) -> list[OffPolicyEpisodeSegment]:
        completed_segments = self.accumulator.add(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            next_obs=next_obs,
            episode_start_mask=episode_start_mask,
            rollout_step_idx=rollout_step_idx,
        )
        self.add_segments(completed_segments)
        return self._segments_to_train_dev(completed_segments)

    def add_segments(self, segments: list[OffPolicyEpisodeSegment]) -> None:
        for segment in segments:
            self.add_segment(segment)

    def add_segment(self, segment: OffPolicyEpisodeSegment) -> None:
        if segment.n_steps > self.capacity_transitions:
            raise ValueError(
                f"Cannot add segment with {segment.n_steps} transitions to replay capacity "
                f"{self.capacity_transitions}"
            )
        stored_segment = self._segment_to_train_dev(segment)
        self.segments.append(stored_segment)
        self.n_transitions += stored_segment.n_steps
        while self.n_transitions > self.capacity_transitions:
            removed_segment = self.segments.pop(0)
            self.n_transitions -= removed_segment.n_steps

    def flush_partial_segments(self) -> list[OffPolicyEpisodeSegment]:
        partial_segments = self.accumulator.dump_partial_segments()
        self.add_segments(partial_segments)
        return self._segments_to_train_dev(partial_segments)

    def make_sampler(self, config: OffPolicySamplerConfig) -> "OffPolicyTransitionSampler":
        if self.n_transitions <= 0:
            raise ValueError("Cannot sample from an empty off-policy replay buffer.")
        return OffPolicyTransitionSampler(self.segments, config=config)

    def sample_transition_batch(self, batch_size: int) -> OffPolicyTransitionBatch:
        return self.make_sampler(OffPolicySamplerConfig(batch_size=batch_size)).sample_batch()

    def reset(self) -> None:
        self.segments = []
        self.n_transitions = 0
        self.accumulator.reset()

    def _segments_to_train_dev(self, segments: list[OffPolicyEpisodeSegment]) -> list[OffPolicyEpisodeSegment]:
        return [self._segment_to_train_dev(segment) for segment in segments]

    def _segment_to_train_dev(self, segment: OffPolicyEpisodeSegment) -> OffPolicyEpisodeSegment:
        return OffPolicyEpisodeSegment(
            local_obs=segment.local_obs.to(device=self.train_device, dtype=self.train_dtype),
            global_obs=segment.global_obs.to(device=self.train_device, dtype=self.train_dtype),
            hidden_local_vars=segment.hidden_local_vars.to(device=self.train_device, dtype=self.train_dtype),
            hidden_global_vars=segment.hidden_global_vars.to(device=self.train_device, dtype=self.train_dtype),
            agent_mask=None if segment.agent_mask is None else segment.agent_mask.to(device=self.train_device),
            previous_actions=segment.previous_actions.to(device=self.train_device, dtype=self.train_dtype),
            actions=segment.actions.to(device=self.train_device, dtype=self.train_dtype),
            rewards=segment.rewards.to(device=self.train_device, dtype=self.train_dtype),
            terminations=segment.terminations.to(device=self.train_device),
            truncations=segment.truncations.to(device=self.train_device),
            next_local_obs=segment.next_local_obs.to(device=self.train_device, dtype=self.train_dtype),
            next_global_obs=segment.next_global_obs.to(device=self.train_device, dtype=self.train_dtype),
            next_hidden_local_vars=segment.next_hidden_local_vars.to(device=self.train_device, dtype=self.train_dtype),
            next_hidden_global_vars=segment.next_hidden_global_vars.to(
                device=self.train_device,
                dtype=self.train_dtype,
            ),
            next_agent_mask=None if segment.next_agent_mask is None else segment.next_agent_mask.to(
                device=self.train_device,
            ),
            next_previous_actions=segment.next_previous_actions.to(device=self.train_device, dtype=self.train_dtype),
            is_true_episode_start=segment.is_true_episode_start,
            rollout_env_idx=segment.rollout_env_idx,
            rollout_start_step=segment.rollout_start_step,
        )


class OffPolicyTransitionSampler(BaseSampler[OffPolicyTransitionBatch, OffPolicySamplerConfig]):
    def __init__(
            self,
            segments: list[OffPolicyEpisodeSegment],
            config: OffPolicySamplerConfig,
    ):
        if not segments:
            raise ValueError("segments must not be empty")

        self.local_obs = torch.concatenate(tuple(segment.local_obs for segment in segments), dim=0)
        self.global_obs = torch.concatenate(tuple(segment.global_obs for segment in segments), dim=0)
        self.hidden_local_vars = torch.concatenate(tuple(segment.hidden_local_vars for segment in segments), dim=0)
        self.hidden_global_vars = torch.concatenate(tuple(segment.hidden_global_vars for segment in segments), dim=0)
        has_agent_mask = any(segment.agent_mask is not None for segment in segments)
        has_missing_agent_mask = any(segment.agent_mask is None for segment in segments)
        if has_agent_mask and has_missing_agent_mask:
            raise ValueError("agent_mask must be provided for all segments or none")
        self.agent_mask = None if has_missing_agent_mask else torch.concatenate(
            tuple(segment.agent_mask for segment in segments),
            dim=0,
        )
        self.previous_actions = torch.concatenate(tuple(segment.previous_actions for segment in segments), dim=0)
        self.actions = torch.concatenate(tuple(segment.actions for segment in segments), dim=0)
        self.rewards = torch.concatenate(tuple(segment.rewards for segment in segments), dim=0)
        self.terminations = torch.concatenate(tuple(segment.terminations for segment in segments), dim=0)
        self.truncations = torch.concatenate(tuple(segment.truncations for segment in segments), dim=0)
        self.dones = torch.logical_or(self.terminations, self.truncations)
        self.next_local_obs = torch.concatenate(tuple(segment.next_local_obs for segment in segments), dim=0)
        self.next_global_obs = torch.concatenate(tuple(segment.next_global_obs for segment in segments), dim=0)
        self.next_hidden_local_vars = torch.concatenate(
            tuple(segment.next_hidden_local_vars for segment in segments),
            dim=0,
        )
        self.next_hidden_global_vars = torch.concatenate(
            tuple(segment.next_hidden_global_vars for segment in segments),
            dim=0,
        )
        has_next_agent_mask = any(segment.next_agent_mask is not None for segment in segments)
        has_missing_next_agent_mask = any(segment.next_agent_mask is None for segment in segments)
        if has_next_agent_mask and has_missing_next_agent_mask:
            raise ValueError("next_agent_mask must be provided for all segments or none")
        self.next_agent_mask = None if has_missing_next_agent_mask else torch.concatenate(
            tuple(segment.next_agent_mask for segment in segments),
            dim=0,
        )
        self.next_previous_actions = torch.concatenate(
            tuple(segment.next_previous_actions for segment in segments),
            dim=0,
        )

        super().__init__(
            config=config,
            n_samples=self.local_obs.shape[0],
            index_device=self.local_obs.device,
        )

    def sample_batch(self) -> OffPolicyTransitionBatch:
        if self.config.replacement:
            batch_indices = torch.randint(
                low=0,
                high=self.n_samples,
                size=(self.config.batch_size,),
                device=self.index_device,
            )
            return self._fetch_samples(batch_indices)
        if self.config.batch_size > self.n_samples:
            raise ValueError(
                f"Cannot sample batch_size={self.config.batch_size} without replacement from "
                f"{self.n_samples} transitions."
            )
        return next(self.sample(drop_last=True))

    def _fetch_samples(self, batch_indices: BatchIndices) -> OffPolicyTransitionBatch:
        return OffPolicyTransitionBatch(
            local_obs=self.local_obs[batch_indices],
            global_obs=self.global_obs[batch_indices],
            hidden_local_vars=self.hidden_local_vars[batch_indices],
            hidden_global_vars=self.hidden_global_vars[batch_indices],
            agent_mask=None if self.agent_mask is None else self.agent_mask[batch_indices],
            previous_actions=self.previous_actions[batch_indices],
            actions=self.actions[batch_indices],
            rewards=self.rewards[batch_indices],
            terminations=self.terminations[batch_indices],
            truncations=self.truncations[batch_indices],
            dones=self.dones[batch_indices],
            next_local_obs=self.next_local_obs[batch_indices],
            next_global_obs=self.next_global_obs[batch_indices],
            next_hidden_local_vars=self.next_hidden_local_vars[batch_indices],
            next_hidden_global_vars=self.next_hidden_global_vars[batch_indices],
            next_agent_mask=None if self.next_agent_mask is None else self.next_agent_mask[batch_indices],
            next_previous_actions=self.next_previous_actions[batch_indices],
        )
