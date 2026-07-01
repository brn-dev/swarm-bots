from dataclasses import dataclass
from typing import Any

import torch
from gymnasium import spaces

from swarmbots.learn.base_sampler import BaseSamplerConfig
from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.torch_device import as_device


@dataclass(slots=True)
class OffPolicyEpisodeSegment:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    hidden_local_vars: torch.Tensor
    hidden_global_vars: torch.Tensor
    agent_mask: torch.Tensor | None
    previous_actions: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    next_local_obs: torch.Tensor
    next_global_obs: torch.Tensor
    next_hidden_local_vars: torch.Tensor
    next_hidden_global_vars: torch.Tensor
    next_agent_mask: torch.Tensor | None
    next_previous_actions: torch.Tensor
    is_true_episode_start: bool
    rollout_env_idx: int
    rollout_start_step: int

    @property
    def n_steps(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def dones(self) -> torch.Tensor:
        return torch.logical_or(self.terminations, self.truncations)


@dataclass(slots=True)
class OffPolicyTransitionBatch:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    hidden_local_vars: torch.Tensor
    hidden_global_vars: torch.Tensor
    agent_mask: torch.Tensor | None
    previous_actions: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    next_local_obs: torch.Tensor
    next_global_obs: torch.Tensor
    next_hidden_local_vars: torch.Tensor
    next_hidden_global_vars: torch.Tensor
    next_agent_mask: torch.Tensor | None
    next_previous_actions: torch.Tensor

    @property
    def dones(self) -> torch.Tensor:
        return torch.logical_or(self.terminations, self.truncations)


@dataclass(slots=True)
class OffPolicyNStepTransitionBatch:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    hidden_local_vars: torch.Tensor
    hidden_global_vars: torch.Tensor
    agent_mask: torch.Tensor | None
    previous_actions: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    next_local_obs: torch.Tensor
    next_global_obs: torch.Tensor
    next_hidden_local_vars: torch.Tensor
    next_hidden_global_vars: torch.Tensor
    next_agent_mask: torch.Tensor | None
    next_previous_actions: torch.Tensor
    steps: torch.Tensor


@dataclass(slots=True)
class OffPolicySequenceBatch:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    hidden_local_vars: torch.Tensor
    hidden_global_vars: torch.Tensor
    agent_mask: torch.Tensor | None
    previous_actions: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    next_local_obs: torch.Tensor
    next_global_obs: torch.Tensor
    next_hidden_local_vars: torch.Tensor
    next_hidden_global_vars: torch.Tensor
    next_agent_mask: torch.Tensor | None
    next_previous_actions: torch.Tensor
    time_mask: torch.Tensor


@dataclass(frozen=True)
class OffPolicySamplerConfig(BaseSamplerConfig):
    replacement: bool = True


@dataclass(slots=True)
class _TransitionRefs:
    env_indices: torch.Tensor
    rows: torch.Tensor
    generations: torch.Tensor


@dataclass(slots=True)
class _SegmentState:
    rows: list[int]
    generations: list[int]
    is_true_episode_start: bool
    rollout_start_step: int


class OffPolicyTransitionSampler:
    def __init__(self, replay_buffer: "OffPolicyReplayBuffer", config: OffPolicySamplerConfig) -> None:
        if config.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {config.batch_size}")
        self.replay_buffer = replay_buffer
        self.config = config

    def sample_batch(self) -> OffPolicyTransitionBatch:
        return self.replay_buffer.sample_transition_batch(
            batch_size=self.config.batch_size,
            replacement=self.config.replacement,
        )


class OffPolicyReplayBuffer:
    def __init__(
            self,
            *,
            capacity_transitions: int,
            max_segment_length: int,
            observation_space: spaces.Dict,
            action_space: VectorHybridActionSpace,
            rollout_device: torch.device | str = "cpu",
            rollout_dtype: torch.dtype = torch.float32,
            train_device: torch.device | str = "auto",
            train_dtype: torch.dtype = torch.float32,
    ) -> None:
        if capacity_transitions <= 0:
            raise ValueError(f"capacity_transitions must be > 0, got {capacity_transitions}")
        if max_segment_length <= 0:
            raise ValueError(f"max_segment_length must be > 0, got {max_segment_length}")

        self.capacity_transitions = capacity_transitions
        self.max_segment_length = max_segment_length
        self.observation_space = observation_space
        self.action_space = action_space
        self.rollout_device = as_device(rollout_device)
        self.rollout_dtype = rollout_dtype
        self.train_device = as_device(train_device)
        self.train_dtype = train_dtype

        self.local_obs_space = observation_space["local_obs"]
        self.global_obs_space = observation_space["global_obs"]
        self.hidden_local_vars_space = observation_space["hidden_local_vars"]
        self.hidden_global_vars_space = observation_space["hidden_global_vars"]
        self.has_agent_mask = "agent_mask" in observation_space.keys() and observation_space["agent_mask"] is not None

        self.n_envs = int(self.local_obs_space.shape[0])
        self.n_agents = int(self.local_obs_space.shape[1])
        self.n_agent_actions = action_space.total_agent_action_dim
        if action_space.n_envs != self.n_envs:
            raise ValueError(f"Action space n_envs {action_space.n_envs} does not match obs n_envs {self.n_envs}")
        if action_space.n_agents != self.n_agents:
            raise ValueError(f"Action space n_agents {action_space.n_agents} does not match obs n_agents {self.n_agents}")

        self.capacity_per_env = (capacity_transitions + self.n_envs - 1) // self.n_envs
        self.obs_capacity_per_env = self.capacity_per_env + 1
        self._allocate_transition_storage()
        self._allocate_terminal_obs_storage()
        self._allocate_ref_storage()

        self._cursor = torch.zeros(self.n_envs, dtype=torch.long, device=self.rollout_device)
        self._obs_cursor = torch.zeros(self.n_envs, dtype=torch.long, device=self.rollout_device)
        self._step_in_episode = torch.zeros(self.n_envs, dtype=torch.long, device=self.rollout_device)
        self._episode_id = torch.zeros(self.n_envs, dtype=torch.long, device=self.rollout_device)
        self._segment_states = [
            _SegmentState(rows=[], generations=[], is_true_episode_start=True, rollout_start_step=0)
            for _ in range(self.n_envs)
        ]

    @property
    def n_transitions(self) -> int:
        return self._n_refs

    def _allocate_transition_storage(self) -> None:
        self.local_obs = self._zeros_from_space(
            self.local_obs_space,
            leading_shape=(self.n_envs, self.obs_capacity_per_env),
        )
        self.global_obs = self._zeros_from_space(
            self.global_obs_space,
            leading_shape=(self.n_envs, self.obs_capacity_per_env),
        )
        self.hidden_local_vars = self._zeros_from_space(
            self.hidden_local_vars_space,
            leading_shape=(self.n_envs, self.obs_capacity_per_env),
        )
        self.hidden_global_vars = self._zeros_from_space(
            self.hidden_global_vars_space,
            leading_shape=(self.n_envs, self.obs_capacity_per_env),
        )
        self.agent_mask = (
            torch.zeros(
                (self.n_envs, self.obs_capacity_per_env, self.n_agents),
                dtype=torch.bool,
                device=self.rollout_device,
            )
            if self.has_agent_mask
            else None
        )
        self.previous_actions = torch.zeros(
            (self.n_envs, self.capacity_per_env, self.n_agents, self.n_agent_actions),
            dtype=self.rollout_dtype,
            device=self.rollout_device,
        )
        self.actions = torch.zeros_like(self.previous_actions)
        self.next_previous_actions = torch.zeros_like(self.previous_actions)
        self.rewards = torch.zeros((self.n_envs, self.capacity_per_env), dtype=self.rollout_dtype, device=self.rollout_device)
        self.terminations = torch.zeros((self.n_envs, self.capacity_per_env), dtype=torch.bool, device=self.rollout_device)
        self.truncations = torch.zeros_like(self.terminations)
        self.episode_start_mask = torch.zeros_like(self.terminations)
        self.step_in_episode = torch.zeros((self.n_envs, self.capacity_per_env), dtype=torch.long, device=self.rollout_device)
        self.episode_ids = torch.zeros_like(self.step_in_episode)
        self.rollout_step_indices = torch.zeros_like(self.step_in_episode)
        self.generations = torch.zeros_like(self.step_in_episode)
        self.valid_transition_slots = torch.zeros_like(self.terminations)
        self.obs_slots = torch.zeros_like(self.step_in_episode)
        self.next_obs_slots = torch.full(
            (self.n_envs, self.capacity_per_env),
            -1,
            dtype=torch.long,
            device=self.rollout_device,
        )
        self.terminal_obs_refs = torch.full(
            (self.n_envs, self.capacity_per_env),
            -1,
            dtype=torch.long,
            device=self.rollout_device,
        )

    def _allocate_terminal_obs_storage(self) -> None:
        self.terminal_local_obs = self._zeros_from_space(
            self.local_obs_space,
            leading_shape=(self.capacity_transitions,),
        )
        self.terminal_global_obs = self._zeros_from_space(
            self.global_obs_space,
            leading_shape=(self.capacity_transitions,),
        )
        self.terminal_hidden_local_vars = self._zeros_from_space(
            self.hidden_local_vars_space,
            leading_shape=(self.capacity_transitions,),
        )
        self.terminal_hidden_global_vars = self._zeros_from_space(
            self.hidden_global_vars_space,
            leading_shape=(self.capacity_transitions,),
        )
        self.terminal_agent_mask = (
            torch.zeros((self.capacity_transitions, self.n_agents), dtype=torch.bool, device=self.rollout_device)
            if self.has_agent_mask
            else None
        )
        self._terminal_obs_cursor = 0

    def _allocate_ref_storage(self) -> None:
        self._ref_env_indices = torch.zeros(self.capacity_transitions, dtype=torch.long, device=self.rollout_device)
        self._ref_rows = torch.zeros_like(self._ref_env_indices)
        self._ref_generations = torch.zeros_like(self._ref_env_indices)
        self._ref_cursor = 0
        self._n_refs = 0

    def _zeros_from_space(self, space: spaces.Space, *, leading_shape: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(
            (*leading_shape, *space.shape[1:]),
            dtype=self.rollout_dtype,
            device=self.rollout_device,
        )

    def add(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor,
            hidden_global_vars: torch.Tensor,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            terminations: torch.Tensor,
            truncations: torch.Tensor,
            next_obs: dict[str, torch.Tensor],
            episode_start_mask: torch.Tensor,
            rollout_step_idx: int,
    ) -> list[OffPolicyEpisodeSegment]:
        local_obs = self._storage_tensor(local_obs)
        global_obs = self._storage_tensor(global_obs)
        hidden_local_vars = self._storage_tensor(hidden_local_vars)
        hidden_global_vars = self._storage_tensor(hidden_global_vars)
        previous_actions = self._zero_previous_actions() if previous_actions is None else self._storage_tensor(previous_actions)
        actions = self._storage_tensor(actions)
        rewards = self._storage_tensor(rewards).reshape(self.n_envs)
        terminations = terminations.to(device=self.rollout_device, dtype=torch.bool).reshape(self.n_envs)
        truncations = truncations.to(device=self.rollout_device, dtype=torch.bool).reshape(self.n_envs)
        episode_start_mask = episode_start_mask.to(device=self.rollout_device, dtype=torch.bool).reshape(self.n_envs)
        next_obs = {key: self._storage_tensor(value) for key, value in next_obs.items()}
        agent_mask = None if agent_mask is None else agent_mask.to(device=self.rollout_device, dtype=torch.bool)

        completed_segments: list[OffPolicyEpisodeSegment] = []
        for env_idx in range(self.n_envs):
            row = int(self._cursor[env_idx].item())
            obs_slot = int(self._obs_cursor[env_idx].item())
            generation = int(self.generations[env_idx, row].item()) + 1
            done = bool((terminations[env_idx] | truncations[env_idx]).item())

            self._store_transition_row(
                env_idx=env_idx,
                row=row,
                obs_slot=obs_slot,
                generation=generation,
                local_obs=local_obs[env_idx],
                global_obs=global_obs[env_idx],
                hidden_local_vars=hidden_local_vars[env_idx],
                hidden_global_vars=hidden_global_vars[env_idx],
                agent_mask=None if agent_mask is None else agent_mask[env_idx],
                previous_actions=previous_actions[env_idx],
                actions=actions[env_idx],
                rewards=rewards[env_idx],
                terminations=terminations[env_idx],
                truncations=truncations[env_idx],
                episode_start_mask=episode_start_mask[env_idx],
                rollout_step_idx=rollout_step_idx,
            )

            next_previous_actions = actions[env_idx].clone()
            if done:
                self.terminal_obs_refs[env_idx, row] = self._store_terminal_obs(env_idx=env_idx, obs=next_obs)
                self.next_obs_slots[env_idx, row] = -1
            else:
                next_obs_slot = (obs_slot + 1) % self.obs_capacity_per_env
                self._store_observation_row(env_idx=env_idx, obs_slot=next_obs_slot, obs=next_obs)
                self.next_obs_slots[env_idx, row] = next_obs_slot
                self.terminal_obs_refs[env_idx, row] = -1
            self.next_previous_actions[env_idx, row] = next_previous_actions

            self._append_ref(env_idx=env_idx, row=row, generation=generation)
            self._append_segment_ref(
                env_idx=env_idx,
                row=row,
                generation=generation,
                is_true_episode_start=bool(episode_start_mask[env_idx].item()),
                rollout_step_idx=rollout_step_idx,
            )

            segment_storage_limit = min(self.max_segment_length, self.capacity_per_env)
            if done or len(self._segment_states[env_idx].rows) >= segment_storage_limit:
                completed_segments.append(self._construct_segment(env_idx))

            self._cursor[env_idx] = (row + 1) % self.capacity_per_env
            self._obs_cursor[env_idx] = (obs_slot + 1) % self.obs_capacity_per_env
            if done:
                self._step_in_episode[env_idx] = 0
                self._episode_id[env_idx] += 1
            else:
                self._step_in_episode[env_idx] += 1

        return self._segments_to_train_device(completed_segments)

    def _storage_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to(device=self.rollout_device, dtype=self.rollout_dtype)

    def _zero_previous_actions(self) -> torch.Tensor:
        return torch.zeros(
            (self.n_envs, self.n_agents, self.n_agent_actions),
            dtype=self.rollout_dtype,
            device=self.rollout_device,
        )

    def _store_transition_row(
            self,
            *,
            env_idx: int,
            row: int,
            obs_slot: int,
            generation: int,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor,
            hidden_global_vars: torch.Tensor,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            terminations: torch.Tensor,
            truncations: torch.Tensor,
            episode_start_mask: torch.Tensor,
            rollout_step_idx: int,
    ) -> None:
        self.local_obs[env_idx, obs_slot] = local_obs
        self.global_obs[env_idx, obs_slot] = global_obs
        self.hidden_local_vars[env_idx, obs_slot] = hidden_local_vars
        self.hidden_global_vars[env_idx, obs_slot] = hidden_global_vars
        if self.agent_mask is not None and agent_mask is not None:
            self.agent_mask[env_idx, obs_slot] = agent_mask
        self.previous_actions[env_idx, row] = previous_actions
        self.actions[env_idx, row] = actions
        self.rewards[env_idx, row] = rewards
        self.terminations[env_idx, row] = terminations
        self.truncations[env_idx, row] = truncations
        self.episode_start_mask[env_idx, row] = episode_start_mask
        self.step_in_episode[env_idx, row] = self._step_in_episode[env_idx]
        self.episode_ids[env_idx, row] = self._episode_id[env_idx]
        self.rollout_step_indices[env_idx, row] = rollout_step_idx
        self.generations[env_idx, row] = generation
        self.valid_transition_slots[env_idx, row] = True
        self.obs_slots[env_idx, row] = obs_slot

    def _store_observation_row(
            self,
            *,
            env_idx: int,
            obs_slot: int,
            obs: dict[str, torch.Tensor],
    ) -> None:
        self.local_obs[env_idx, obs_slot] = obs["local_obs"][env_idx]
        self.global_obs[env_idx, obs_slot] = obs["global_obs"][env_idx]
        self.hidden_local_vars[env_idx, obs_slot] = obs["hidden_local_vars"][env_idx]
        self.hidden_global_vars[env_idx, obs_slot] = obs["hidden_global_vars"][env_idx]
        if self.agent_mask is not None and "agent_mask" in obs:
            self.agent_mask[env_idx, obs_slot] = obs["agent_mask"][env_idx]

    def _store_terminal_obs(self, *, env_idx: int, obs: dict[str, torch.Tensor]) -> int:
        terminal_idx = self._terminal_obs_cursor
        self.terminal_local_obs[terminal_idx] = obs["local_obs"][env_idx]
        self.terminal_global_obs[terminal_idx] = obs["global_obs"][env_idx]
        self.terminal_hidden_local_vars[terminal_idx] = obs["hidden_local_vars"][env_idx]
        self.terminal_hidden_global_vars[terminal_idx] = obs["hidden_global_vars"][env_idx]
        if self.terminal_agent_mask is not None and "agent_mask" in obs:
            self.terminal_agent_mask[terminal_idx] = obs["agent_mask"][env_idx]
        self._terminal_obs_cursor = (self._terminal_obs_cursor + 1) % self.capacity_transitions
        return terminal_idx

    def _append_ref(self, *, env_idx: int, row: int, generation: int) -> None:
        self._ref_env_indices[self._ref_cursor] = env_idx
        self._ref_rows[self._ref_cursor] = row
        self._ref_generations[self._ref_cursor] = generation
        self._ref_cursor = (self._ref_cursor + 1) % self.capacity_transitions
        self._n_refs = min(self._n_refs + 1, self.capacity_transitions)

    def _append_segment_ref(
            self,
            *,
            env_idx: int,
            row: int,
            generation: int,
            is_true_episode_start: bool,
            rollout_step_idx: int,
    ) -> None:
        state = self._segment_states[env_idx]
        if not state.rows:
            state.is_true_episode_start = is_true_episode_start
            state.rollout_start_step = rollout_step_idx
        state.rows.append(row)
        state.generations.append(generation)

    def flush_partial_segments(self) -> list[OffPolicyEpisodeSegment]:
        segments = [
            self._construct_segment(env_idx)
            for env_idx, state in enumerate(self._segment_states)
            if state.rows
        ]
        return self._segments_to_train_device(segments)

    def _construct_segment(self, env_idx: int) -> OffPolicyEpisodeSegment:
        state = self._segment_states[env_idx]
        if not state.rows:
            raise ValueError(f"No pending segment for env {env_idx}")

        rows = torch.tensor(state.rows, dtype=torch.long, device=self.rollout_device)
        generations = torch.tensor(state.generations, dtype=torch.long, device=self.rollout_device)
        env_indices = torch.full_like(rows, env_idx)
        refs = _TransitionRefs(env_indices=env_indices, rows=rows, generations=generations)
        self._validate_refs(refs)
        batch = self._fetch_transition_refs(refs)
        segment = OffPolicyEpisodeSegment(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            hidden_local_vars=batch.hidden_local_vars,
            hidden_global_vars=batch.hidden_global_vars,
            agent_mask=batch.agent_mask,
            previous_actions=batch.previous_actions,
            actions=batch.actions,
            rewards=batch.rewards,
            terminations=batch.terminations,
            truncations=batch.truncations,
            next_local_obs=batch.next_local_obs,
            next_global_obs=batch.next_global_obs,
            next_hidden_local_vars=batch.next_hidden_local_vars,
            next_hidden_global_vars=batch.next_hidden_global_vars,
            next_agent_mask=batch.next_agent_mask,
            next_previous_actions=batch.next_previous_actions,
            is_true_episode_start=state.is_true_episode_start,
            rollout_env_idx=env_idx,
            rollout_start_step=state.rollout_start_step,
        )
        state.rows.clear()
        state.generations.clear()
        state.is_true_episode_start = False
        state.rollout_start_step = 0
        return segment

    def make_sampler(self, config: OffPolicySamplerConfig) -> OffPolicyTransitionSampler:
        return OffPolicyTransitionSampler(self, config)

    def sample_transition_batch(self, *, batch_size: int, replacement: bool = True) -> OffPolicyTransitionBatch:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if self.n_transitions == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        if not replacement and batch_size > self.n_transitions:
            raise ValueError(
                f"Cannot sample batch_size={batch_size} without replacement from {self.n_transitions} transitions"
            )

        if replacement:
            logical_indices = torch.randint(self.n_transitions, (batch_size,), device=self.rollout_device)
        else:
            logical_indices = torch.randperm(self.n_transitions, device=self.rollout_device)[:batch_size]
        refs = self._refs_from_logical_indices(logical_indices)
        return self._transition_batch_to_train_device(self._fetch_transition_refs(refs))

    def sample_n_step_transition_batch(
            self,
            *,
            batch_size: int,
            n_steps: int,
            gamma: float,
            replacement: bool = True,
    ) -> OffPolicyNStepTransitionBatch:
        if n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {n_steps}")
        one_step_batch = self.sample_transition_batch(batch_size=batch_size, replacement=replacement)
        start_refs = self._last_sampled_refs
        rewards = torch.zeros(batch_size, dtype=self.train_dtype, device=self.train_device)
        discounts = torch.ones(batch_size, dtype=self.train_dtype, device=self.train_device)
        steps = torch.zeros(batch_size, dtype=torch.long, device=self.train_device)
        terminations = torch.zeros(batch_size, dtype=torch.bool, device=self.train_device)
        truncations = torch.zeros_like(terminations)

        next_refs = start_refs
        next_batch = one_step_batch
        exhausted = torch.zeros(batch_size, dtype=torch.bool, device=self.train_device)
        for step_offset in range(n_steps):
            fetched = self._transition_batch_to_train_device(self._fetch_transition_refs(next_refs))
            active = torch.logical_not(torch.logical_or(torch.logical_or(terminations, truncations), exhausted))
            rewards[active] += discounts[active] * fetched.rewards[active]
            steps[active] += 1
            terminations |= torch.logical_and(active, fetched.terminations)
            truncations |= torch.logical_and(active, fetched.truncations)
            next_batch = fetched
            discounts[active] *= gamma
            active_after_step = torch.logical_not(torch.logical_or(torch.logical_or(terminations, truncations), exhausted))
            if step_offset == n_steps - 1 or not bool(active_after_step.any().item()):
                break
            can_advance = self._continuation_available_mask(
                next_refs,
                active=active_after_step.to(device=self.rollout_device),
            )
            exhausted |= active_after_step & ~can_advance.to(device=self.train_device)
            if not bool(can_advance.any().item()):
                break
            next_refs = self._advance_refs(next_refs, can_advance)
        discounts[terminations] = 0.0

        return OffPolicyNStepTransitionBatch(
            local_obs=one_step_batch.local_obs,
            global_obs=one_step_batch.global_obs,
            hidden_local_vars=one_step_batch.hidden_local_vars,
            hidden_global_vars=one_step_batch.hidden_global_vars,
            agent_mask=one_step_batch.agent_mask,
            previous_actions=one_step_batch.previous_actions,
            actions=one_step_batch.actions,
            rewards=rewards,
            discounts=discounts,
            terminations=terminations,
            truncations=truncations,
            next_local_obs=next_batch.next_local_obs,
            next_global_obs=next_batch.next_global_obs,
            next_hidden_local_vars=next_batch.next_hidden_local_vars,
            next_hidden_global_vars=next_batch.next_hidden_global_vars,
            next_agent_mask=next_batch.next_agent_mask,
            next_previous_actions=next_batch.next_previous_actions,
            steps=steps,
        )

    def sample_sequence_batch(
            self,
            *,
            batch_size: int,
            horizon: int,
            require_full: bool = False,
    ) -> OffPolicySequenceBatch:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")

        start_logical_indices, lengths = self._sample_sequence_starts(
            batch_size=batch_size,
            horizon=horizon,
            require_full=require_full,
        )
        samples = [
            self._fetch_sequence(logical_start=int(start), length=int(length), horizon=horizon)
            for start, length in zip(start_logical_indices.tolist(), lengths.tolist(), strict=True)
        ]
        return self._stack_sequences(samples)

    def _sample_sequence_starts(
            self,
            *,
            batch_size: int,
            horizon: int,
            require_full: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_starts: list[int] = []
        lengths: list[int] = []
        for logical_idx in range(self.n_transitions):
            length = self._sequence_length_from(logical_idx=logical_idx, horizon=horizon)
            if length == 0:
                continue
            if require_full and length < horizon:
                continue
            valid_starts.append(logical_idx)
            lengths.append(length)

        if not valid_starts:
            mode = "full" if require_full else "non-empty"
            raise ValueError(f"No {mode} sequences with horizon={horizon} are available")

        start_tensor = torch.tensor(valid_starts, dtype=torch.long, device=self.rollout_device)
        length_tensor = torch.tensor(lengths, dtype=torch.long, device=self.rollout_device)
        chosen = torch.randint(len(valid_starts), (batch_size,), device=self.rollout_device)
        return start_tensor[chosen], length_tensor[chosen]

    def _sequence_length_from(self, *, logical_idx: int, horizon: int) -> int:
        first_ref = self._refs_from_logical_indices(torch.tensor([logical_idx], device=self.rollout_device))
        self._validate_refs(first_ref)
        env_idx = int(first_ref.env_indices[0].item())
        start_row = int(first_ref.rows[0].item())
        episode_id = int(self.episode_ids[env_idx, start_row].item())
        start_step = int(self.step_in_episode[env_idx, start_row].item())

        length = 0
        for offset in range(horizon):
            row = (start_row + offset) % self.capacity_per_env
            ref = _TransitionRefs(
                env_indices=torch.tensor([env_idx], dtype=torch.long, device=self.rollout_device),
                rows=torch.tensor([row], dtype=torch.long, device=self.rollout_device),
                generations=self.generations[env_idx, row].reshape(1),
            )
            try:
                self._validate_refs(ref)
            except ValueError:
                break
            if int(self.episode_ids[env_idx, row].item()) != episode_id:
                break
            if int(self.step_in_episode[env_idx, row].item()) != start_step + offset:
                break
            length += 1
            if bool((self.terminations[env_idx, row] | self.truncations[env_idx, row]).item()):
                break
        return length

    def _fetch_sequence(self, *, logical_start: int, length: int, horizon: int) -> OffPolicySequenceBatch:
        start_ref = self._refs_from_logical_indices(torch.tensor([logical_start], device=self.rollout_device))
        env_idx = int(start_ref.env_indices[0].item())
        start_row = int(start_ref.rows[0].item())
        rows = (torch.arange(start_row, start_row + length, device=self.rollout_device) % self.capacity_per_env).long()
        refs = _TransitionRefs(
            env_indices=torch.full((length,), env_idx, dtype=torch.long, device=self.rollout_device),
            rows=rows,
            generations=self.generations[env_idx, rows],
        )
        batch = self._transition_batch_to_train_device(self._fetch_transition_refs(refs))

        local_obs = self._pad_time(batch.local_obs, horizon)
        global_obs = self._pad_time(batch.global_obs, horizon)
        hidden_local_vars = self._pad_time(batch.hidden_local_vars, horizon)
        hidden_global_vars = self._pad_time(batch.hidden_global_vars, horizon)
        agent_mask = None if batch.agent_mask is None else self._pad_time(batch.agent_mask, horizon)
        previous_actions = self._pad_time(batch.previous_actions, horizon)
        actions = self._pad_time(batch.actions, horizon)
        rewards = self._pad_time(batch.rewards, horizon)
        terminations = self._pad_time(batch.terminations, horizon)
        truncations = self._pad_time(batch.truncations, horizon)
        next_local_obs = self._pad_time(batch.next_local_obs, horizon)
        next_global_obs = self._pad_time(batch.next_global_obs, horizon)
        next_hidden_local_vars = self._pad_time(batch.next_hidden_local_vars, horizon)
        next_hidden_global_vars = self._pad_time(batch.next_hidden_global_vars, horizon)
        next_agent_mask = None if batch.next_agent_mask is None else self._pad_time(batch.next_agent_mask, horizon)
        next_previous_actions = self._pad_time(batch.next_previous_actions, horizon)
        time_mask = torch.zeros(horizon, dtype=torch.bool, device=self.train_device)
        time_mask[:length] = True
        return OffPolicySequenceBatch(
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
            next_local_obs=next_local_obs,
            next_global_obs=next_global_obs,
            next_hidden_local_vars=next_hidden_local_vars,
            next_hidden_global_vars=next_hidden_global_vars,
            next_agent_mask=next_agent_mask,
            next_previous_actions=next_previous_actions,
            time_mask=time_mask,
        )

    def _stack_sequences(self, samples: list[OffPolicySequenceBatch]) -> OffPolicySequenceBatch:
        return OffPolicySequenceBatch(
            local_obs=torch.stack([sample.local_obs for sample in samples]),
            global_obs=torch.stack([sample.global_obs for sample in samples]),
            hidden_local_vars=torch.stack([sample.hidden_local_vars for sample in samples]),
            hidden_global_vars=torch.stack([sample.hidden_global_vars for sample in samples]),
            agent_mask=None if samples[0].agent_mask is None else torch.stack([sample.agent_mask for sample in samples]),
            previous_actions=torch.stack([sample.previous_actions for sample in samples]),
            actions=torch.stack([sample.actions for sample in samples]),
            rewards=torch.stack([sample.rewards for sample in samples]),
            terminations=torch.stack([sample.terminations for sample in samples]),
            truncations=torch.stack([sample.truncations for sample in samples]),
            next_local_obs=torch.stack([sample.next_local_obs for sample in samples]),
            next_global_obs=torch.stack([sample.next_global_obs for sample in samples]),
            next_hidden_local_vars=torch.stack([sample.next_hidden_local_vars for sample in samples]),
            next_hidden_global_vars=torch.stack([sample.next_hidden_global_vars for sample in samples]),
            next_agent_mask=(
                None if samples[0].next_agent_mask is None
                else torch.stack([sample.next_agent_mask for sample in samples])
            ),
            next_previous_actions=torch.stack([sample.next_previous_actions for sample in samples]),
            time_mask=torch.stack([sample.time_mask for sample in samples]),
        )

    def _pad_time(self, tensor: torch.Tensor, horizon: int) -> torch.Tensor:
        padded = torch.zeros((horizon, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        padded[:tensor.shape[0]] = tensor
        return padded

    def _refs_from_logical_indices(self, logical_indices: torch.Tensor) -> _TransitionRefs:
        physical_indices = (self._ref_cursor - self._n_refs + logical_indices.to(device=self.rollout_device)) % self.capacity_transitions
        refs = _TransitionRefs(
            env_indices=self._ref_env_indices[physical_indices],
            rows=self._ref_rows[physical_indices],
            generations=self._ref_generations[physical_indices],
        )
        self._last_sampled_refs = refs
        return refs

    def _validate_refs(self, refs: _TransitionRefs) -> None:
        stored_generations = self.generations[refs.env_indices, refs.rows]
        valid_slots = self.valid_transition_slots[refs.env_indices, refs.rows]
        if not bool(valid_slots.all().item()) or not torch.equal(stored_generations, refs.generations):
            raise ValueError("Replay sample points at overwritten transition storage")

    def _fetch_transition_refs(self, refs: _TransitionRefs) -> OffPolicyTransitionBatch:
        self._validate_refs(refs)
        env_indices = refs.env_indices
        rows = refs.rows
        obs_slots = self.obs_slots[env_indices, rows]
        next_obs = self._next_obs_for_refs(refs)
        return OffPolicyTransitionBatch(
            local_obs=self.local_obs[env_indices, obs_slots],
            global_obs=self.global_obs[env_indices, obs_slots],
            hidden_local_vars=self.hidden_local_vars[env_indices, obs_slots],
            hidden_global_vars=self.hidden_global_vars[env_indices, obs_slots],
            agent_mask=None if self.agent_mask is None else self.agent_mask[env_indices, obs_slots],
            previous_actions=self.previous_actions[env_indices, rows],
            actions=self.actions[env_indices, rows],
            rewards=self.rewards[env_indices, rows],
            terminations=self.terminations[env_indices, rows],
            truncations=self.truncations[env_indices, rows],
            next_local_obs=next_obs["local_obs"],
            next_global_obs=next_obs["global_obs"],
            next_hidden_local_vars=next_obs["hidden_local_vars"],
            next_hidden_global_vars=next_obs["hidden_global_vars"],
            next_agent_mask=next_obs.get("agent_mask", None),
            next_previous_actions=self.next_previous_actions[env_indices, rows],
        )

    def _next_obs_for_refs(self, refs: _TransitionRefs) -> dict[str, torch.Tensor]:
        env_indices = refs.env_indices
        rows = refs.rows
        terminal_refs = self.terminal_obs_refs[env_indices, rows]
        next_obs_slots = self.next_obs_slots[env_indices, rows]

        nonterminal_next_slots = torch.clamp(next_obs_slots, min=0)
        next_local_obs = self.local_obs[env_indices, nonterminal_next_slots].clone()
        next_global_obs = self.global_obs[env_indices, nonterminal_next_slots].clone()
        next_hidden_local_vars = self.hidden_local_vars[env_indices, nonterminal_next_slots].clone()
        next_hidden_global_vars = self.hidden_global_vars[env_indices, nonterminal_next_slots].clone()
        next_agent_mask = None if self.agent_mask is None else self.agent_mask[env_indices, nonterminal_next_slots].clone()

        terminal_mask = terminal_refs >= 0
        if bool(terminal_mask.any().item()):
            terminal_indices = terminal_refs[terminal_mask]
            next_local_obs[terminal_mask] = self.terminal_local_obs[terminal_indices]
            next_global_obs[terminal_mask] = self.terminal_global_obs[terminal_indices]
            next_hidden_local_vars[terminal_mask] = self.terminal_hidden_local_vars[terminal_indices]
            next_hidden_global_vars[terminal_mask] = self.terminal_hidden_global_vars[terminal_indices]
            if next_agent_mask is not None and self.terminal_agent_mask is not None:
                next_agent_mask[terminal_mask] = self.terminal_agent_mask[terminal_indices]

        obs = {
            "local_obs": next_local_obs,
            "global_obs": next_global_obs,
            "hidden_local_vars": next_hidden_local_vars,
            "hidden_global_vars": next_hidden_global_vars,
        }
        if next_agent_mask is not None:
            obs["agent_mask"] = next_agent_mask
        return obs

    def _advance_refs(self, refs: _TransitionRefs, active: torch.Tensor) -> _TransitionRefs:
        env_indices = refs.env_indices.clone()
        rows = refs.rows.clone()
        generations = refs.generations.clone()
        active = active.to(device=self.rollout_device, dtype=torch.bool)

        next_rows = (rows[active] + 1) % self.capacity_per_env
        active_env_indices = env_indices[active]
        current_rows = rows[active]
        expected_episode_ids = self.episode_ids[active_env_indices, current_rows]
        expected_step_indices = self.step_in_episode[active_env_indices, current_rows] + 1
        if not bool(torch.equal(self.episode_ids[active_env_indices, next_rows], expected_episode_ids)):
            raise ValueError("n-step replay sample crossed an episode boundary")
        if not bool(torch.equal(self.step_in_episode[active_env_indices, next_rows], expected_step_indices)):
            raise ValueError("n-step replay sample reached a transition that is not available yet")
        rows[active] = next_rows
        generations[active] = self.generations[active_env_indices, next_rows]
        next_refs = _TransitionRefs(env_indices=env_indices, rows=rows, generations=generations)
        self._validate_refs(next_refs)
        return next_refs

    def _continuation_available_mask(self, refs: _TransitionRefs, *, active: torch.Tensor) -> torch.Tensor:
        active = active.to(device=self.rollout_device, dtype=torch.bool)
        available = torch.zeros_like(active)
        if not bool(active.any().item()):
            return available

        env_indices = refs.env_indices[active]
        rows = refs.rows[active]
        next_rows = (rows + 1) % self.capacity_per_env
        expected_episode_ids = self.episode_ids[env_indices, rows]
        expected_step_indices = self.step_in_episode[env_indices, rows] + 1
        available[active] = (
            self.valid_transition_slots[env_indices, next_rows]
            & (self.episode_ids[env_indices, next_rows] == expected_episode_ids)
            & (self.step_in_episode[env_indices, next_rows] == expected_step_indices)
        )
        return available

    def _transition_batch_to_train_device(self, batch: OffPolicyTransitionBatch) -> OffPolicyTransitionBatch:
        return OffPolicyTransitionBatch(
            local_obs=batch.local_obs.to(device=self.train_device, dtype=self.train_dtype),
            global_obs=batch.global_obs.to(device=self.train_device, dtype=self.train_dtype),
            hidden_local_vars=batch.hidden_local_vars.to(device=self.train_device, dtype=self.train_dtype),
            hidden_global_vars=batch.hidden_global_vars.to(device=self.train_device, dtype=self.train_dtype),
            agent_mask=None if batch.agent_mask is None else batch.agent_mask.to(device=self.train_device),
            previous_actions=batch.previous_actions.to(device=self.train_device, dtype=self.train_dtype),
            actions=batch.actions.to(device=self.train_device, dtype=self.train_dtype),
            rewards=batch.rewards.to(device=self.train_device, dtype=self.train_dtype),
            terminations=batch.terminations.to(device=self.train_device),
            truncations=batch.truncations.to(device=self.train_device),
            next_local_obs=batch.next_local_obs.to(device=self.train_device, dtype=self.train_dtype),
            next_global_obs=batch.next_global_obs.to(device=self.train_device, dtype=self.train_dtype),
            next_hidden_local_vars=batch.next_hidden_local_vars.to(device=self.train_device, dtype=self.train_dtype),
            next_hidden_global_vars=batch.next_hidden_global_vars.to(device=self.train_device, dtype=self.train_dtype),
            next_agent_mask=None if batch.next_agent_mask is None else batch.next_agent_mask.to(device=self.train_device),
            next_previous_actions=batch.next_previous_actions.to(device=self.train_device, dtype=self.train_dtype),
        )

    def _segments_to_train_device(self, segments: list[OffPolicyEpisodeSegment]) -> list[OffPolicyEpisodeSegment]:
        return [
            OffPolicyEpisodeSegment(
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
                next_hidden_global_vars=segment.next_hidden_global_vars.to(device=self.train_device, dtype=self.train_dtype),
                next_agent_mask=None if segment.next_agent_mask is None else segment.next_agent_mask.to(device=self.train_device),
                next_previous_actions=segment.next_previous_actions.to(device=self.train_device, dtype=self.train_dtype),
                is_true_episode_start=segment.is_true_episode_start,
                rollout_env_idx=segment.rollout_env_idx,
                rollout_start_step=segment.rollout_start_step,
            )
            for segment in segments
        ]
