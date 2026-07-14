from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from gymnasium import spaces

from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.torch_device import as_device


MaybeTensor = torch.Tensor | None


class NoEpisodeSegmentCandidatesError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OffPolicyReplayBatch:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    hidden_local_vars: torch.Tensor
    hidden_global_vars: torch.Tensor
    agent_mask: MaybeTensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    previous_actions: MaybeTensor
    next_local_obs: torch.Tensor
    next_global_obs: torch.Tensor
    next_hidden_local_vars: torch.Tensor
    next_hidden_global_vars: torch.Tensor
    next_agent_mask: MaybeTensor
    episode_start_mask: torch.Tensor | None = None

    @property
    def episode_ends(self) -> torch.Tensor:
        return torch.logical_or(self.terminations, self.truncations)

    @property
    def terminal_mask(self) -> torch.Tensor:
        return self.terminations


@dataclass(frozen=True, slots=True)
class OffPolicyReplayEpisodeSegmentBatch:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    hidden_local_vars: torch.Tensor
    hidden_global_vars: torch.Tensor
    agent_mask: MaybeTensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    previous_actions: MaybeTensor
    next_local_obs: torch.Tensor
    next_global_obs: torch.Tensor
    next_hidden_local_vars: torch.Tensor
    next_hidden_global_vars: torch.Tensor
    next_agent_mask: MaybeTensor
    episode_start_mask: MaybeTensor
    train_mask: torch.Tensor
    initial_temporal_state: Any
    burn_in_steps: int

    @property
    def sequence_length(self) -> int:
        return int(self.actions.shape[1])

    @property
    def segment_length(self) -> int:
        return self.sequence_length - self.burn_in_steps

    @property
    def burn_in_mask(self) -> torch.Tensor:
        return torch.logical_not(self.train_mask)

    @property
    def episode_ends(self) -> torch.Tensor:
        return torch.logical_or(self.terminations, self.truncations)

    @property
    def terminal_mask(self) -> torch.Tensor:
        return self.terminations

    @property
    def origin_batch(self) -> OffPolicyReplayBatch:
        return OffPolicyReplayBatch(
            local_obs=self.local_obs[:, 0],
            global_obs=self.global_obs[:, 0],
            hidden_local_vars=self.hidden_local_vars[:, 0],
            hidden_global_vars=self.hidden_global_vars[:, 0],
            agent_mask=None if self.agent_mask is None else self.agent_mask[:, 0],
            actions=self.actions[:, 0],
            rewards=self.rewards[:, 0],
            terminations=self.terminations[:, 0],
            truncations=self.truncations[:, 0],
            previous_actions=None if self.previous_actions is None else self.previous_actions[:, 0],
            next_local_obs=self.next_local_obs[:, 0],
            next_global_obs=self.next_global_obs[:, 0],
            next_hidden_local_vars=self.next_hidden_local_vars[:, 0],
            next_hidden_global_vars=self.next_hidden_global_vars[:, 0],
            next_agent_mask=None if self.next_agent_mask is None else self.next_agent_mask[:, 0],
            episode_start_mask=(
                None if self.episode_start_mask is None else self.episode_start_mask[:, 0]
            ),
        )


class OffPolicyReplayBuffer:
    def __init__(
            self,
            capacity_per_env: int,
            observation_space: spaces.Dict,
            action_space: VectorHybridActionSpace,
            *,
            store_previous_actions: bool = False,
            temporal_state_store_interval: int | None = None,
            temporal_state_storage_dtype: torch.dtype | None = None,
            storage_device: torch.device | str = "cpu",
            storage_dtype: torch.dtype = torch.float32,
            storage_pin_memory: bool = False,
            train_device: torch.device | str = "auto",
            train_dtype: torch.dtype = torch.float32,
            non_blocking_train_transfer: bool | None = None,
    ) -> None:
        if capacity_per_env <= 0:
            raise ValueError(f"capacity_per_env must be > 0, got {capacity_per_env}")
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError(f"observation_space must be a gymnasium.spaces.Dict, got {observation_space}")
        if temporal_state_store_interval is not None and temporal_state_store_interval <= 0:
            raise ValueError(
                f"temporal_state_store_interval must be > 0 when set, got {temporal_state_store_interval}"
            )

        self.capacity_per_env = capacity_per_env
        self.observation_capacity_per_env = capacity_per_env + 1
        self.observation_space = observation_space
        self.action_space = action_space
        self.store_previous_actions = store_previous_actions
        self.temporal_state_store_interval = temporal_state_store_interval
        self.temporal_state_storage_dtype = (
            storage_dtype
            if temporal_state_storage_dtype is None
            else temporal_state_storage_dtype
        )
        self.storage_device = as_device(storage_device)
        self.storage_dtype = storage_dtype
        self.storage_pin_memory = storage_pin_memory
        self.train_device = as_device(train_device)
        self.train_dtype = train_dtype
        if self.storage_pin_memory and self.storage_device.type != "cpu":
            raise ValueError("storage_pin_memory=True requires storage_device='cpu'.")
        self.non_blocking_train_transfer = (
            self.storage_pin_memory and self.train_device.type == "cuda"
            if non_blocking_train_transfer is None
            else non_blocking_train_transfer
        )

        self.local_obs_space = observation_space["local_obs"]
        self.n_envs = int(self.local_obs_space.shape[0])
        self.n_agents = int(self.local_obs_space.shape[1])
        self.agent_obs_shape = tuple(self.local_obs_space.shape[2:])
        self.global_obs_shape = tuple(observation_space["global_obs"].shape[1:])
        self.hidden_local_vars_shape = tuple(observation_space["hidden_local_vars"].shape[2:])
        self.hidden_global_vars_shape = tuple(observation_space["hidden_global_vars"].shape[1:])
        self.has_agent_mask = "agent_mask" in observation_space.keys() and observation_space["agent_mask"] is not None

        if action_space.n_envs != self.n_envs:
            raise ValueError(f"Expected action_space.n_envs={self.n_envs}, got {action_space.n_envs}")
        if action_space.n_agents != self.n_agents:
            raise ValueError(f"Expected action_space.n_agents={self.n_agents}, got {action_space.n_agents}")
        self.n_agent_actions = action_space.total_agent_action_dim

        obs_shape = (self.n_envs, self.observation_capacity_per_env)
        transition_shape = (self.n_envs, self.capacity_per_env)
        self.local_obs = self._new_storage_tensor(
            (*obs_shape, self.n_agents, *self.agent_obs_shape),
            dtype=self.storage_dtype,
        )
        self.global_obs = self._new_storage_tensor(
            (*obs_shape, *self.global_obs_shape),
            dtype=self.storage_dtype,
        )
        self.hidden_local_vars = self._new_storage_tensor(
            (*obs_shape, self.n_agents, *self.hidden_local_vars_shape),
            dtype=self.storage_dtype,
        )
        self.hidden_global_vars = self._new_storage_tensor(
            (*obs_shape, *self.hidden_global_vars_shape),
            dtype=self.storage_dtype,
        )
        self.agent_mask: MaybeTensor = None
        if self.has_agent_mask:
            self.agent_mask = self._new_storage_tensor(
                (*obs_shape, self.n_agents),
                dtype=torch.bool,
            )

        self.actions = self._new_storage_tensor(
            (*transition_shape, self.n_agents, self.n_agent_actions),
            dtype=self.storage_dtype,
        )
        self.rewards = self._new_storage_tensor(transition_shape, dtype=self.storage_dtype)
        self.terminations = self._new_storage_tensor(transition_shape, dtype=torch.bool)
        self.truncations = self._new_storage_tensor(transition_shape, dtype=torch.bool)
        self.episode_starts: MaybeTensor = None
        if temporal_state_store_interval is not None:
            self.episode_starts = self._new_storage_tensor(transition_shape, dtype=torch.bool)
        self.previous_actions: MaybeTensor = None
        if store_previous_actions:
            self.previous_actions = self._new_storage_tensor(
                (*transition_shape, self.n_agents, self.n_agent_actions),
                dtype=self.storage_dtype,
            )
        self.temporal_states: Any = None
        self._temporal_state_available: torch.Tensor | None = None
        if temporal_state_store_interval is not None:
            self._temporal_state_available = self._new_storage_tensor(obs_shape, dtype=torch.bool)

        self._write_slot = 0
        self._current_obs_slots = torch.zeros(self.n_envs, dtype=torch.long, device=self.storage_device)
        self._obs_write_slots = torch.zeros(self.n_envs, dtype=torch.long, device=self.storage_device)
        self._transition_obs_slots = torch.zeros(
            transition_shape,
            dtype=torch.long,
            device=self.storage_device,
        )
        self._transition_next_obs_slots = torch.zeros(
            transition_shape,
            dtype=torch.long,
            device=self.storage_device,
        )
        self._terminal_obs_by_env_slot: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        self._terminal_envs_by_transition_slot: list[set[int]] = [set() for _ in range(self.capacity_per_env)]
        self._size_per_env = 0
        self._has_current_obs = False
        self._current_episode_start_mask: MaybeTensor = None
        if temporal_state_store_interval is not None:
            self._current_episode_start_mask = torch.ones(self.n_envs, dtype=torch.bool, device=self.storage_device)
        self._vector_steps_added = 0
        self.total_transitions_added = 0

    @property
    def size_per_env(self) -> int:
        return self._size_per_env

    @property
    def has_current_obs(self) -> bool:
        return self._has_current_obs

    def __len__(self) -> int:
        return self._size_per_env * self.n_envs

    @property
    def total_capacity(self) -> int:
        return self.capacity_per_env * self.n_envs

    def reset(self) -> None:
        self._write_slot = 0
        self._current_obs_slots.zero_()
        self._obs_write_slots.zero_()
        self._transition_obs_slots.zero_()
        self._transition_next_obs_slots.zero_()
        if self.episode_starts is not None:
            self.episode_starts.zero_()
        if self._temporal_state_available is not None:
            self._temporal_state_available.zero_()
        self.temporal_states = None
        self._terminal_obs_by_env_slot.clear()
        for terminal_envs in self._terminal_envs_by_transition_slot:
            terminal_envs.clear()
        self._size_per_env = 0
        self._has_current_obs = False
        if self._current_episode_start_mask is not None:
            self._current_episode_start_mask.fill_(True)
        self._vector_steps_added = 0
        self.total_transitions_added = 0

    def add(
            self,
            *,
            obs: dict[str, torch.Tensor],
            actions: torch.Tensor,
            rewards: torch.Tensor,
            terminations: torch.Tensor,
            truncations: torch.Tensor,
            next_obs: dict[str, torch.Tensor],
            terminal_obs: dict[str, torch.Tensor] | None = None,
            previous_actions: MaybeTensor = None,
            episode_start_mask: torch.Tensor | None = None,
            temporal_state: Any = None,
            next_temporal_state: Any = None,
            copy_current_obs: bool | None = None,
    ) -> None:
        """
        Append one full vector-env step from a continuous stream.

        After the stream is initialized, the current obs for this transition is the previous appended next_obs/reset
        obs. In that normal streaming case, this method ignores obs for storage and does not copy it into replay again.
        terminal_obs may either be full vector-env observations or rows packed in done-env order.
        """
        slot = self._write_slot
        dones = torch.logical_or(terminations, truncations)
        done_env_indices = torch.nonzero(dones, as_tuple=False).flatten()
        should_copy_current_obs = not self._has_current_obs if copy_current_obs is None else copy_current_obs
        storage_episode_start_mask: torch.Tensor | None = None
        if self.episode_starts is not None:
            if episode_start_mask is None:
                assert self._current_episode_start_mask is not None
                storage_episode_start_mask = self._current_episode_start_mask.clone()
            else:
                storage_episode_start_mask = episode_start_mask.to(device=self.storage_device, dtype=torch.bool)

        if len(done_env_indices) > 0 and terminal_obs is None:
            raise ValueError("terminal_obs is required for done transitions under SAME_STEP autoreset.")
        if not should_copy_current_obs and not self._has_current_obs:
            raise ValueError("copy_current_obs=False requires the current observation slot to be initialized.")
        if copy_current_obs is True and self._has_current_obs:
            raise ValueError(
                "copy_current_obs=True is only valid when starting a replay stream. Continue with "
                "copy_current_obs=False/None so next_obs/reset obs becomes the next transition's current obs."
            )

        self._clear_terminal_obs_for_slot_(slot)

        if should_copy_current_obs:
            self._current_obs_slots = self._obs_write_slots.clone()
            self._copy_obs_rows_at_slots_(obs=obs, obs_slots=self._current_obs_slots)
            self._obs_write_slots = self._advance_obs_slots(self._obs_write_slots)

        transition_obs_slots = self._current_obs_slots.clone()
        next_obs_slots = self._obs_write_slots.clone()
        self._copy_obs_rows_at_slots_(obs=next_obs, obs_slots=next_obs_slots)
        self._obs_write_slots = self._advance_obs_slots(self._obs_write_slots)

        if len(done_env_indices) > 0:
            assert terminal_obs is not None
            storage_done_env_indices = done_env_indices.to(device=self.storage_device)
            self._store_terminal_obs_at_slot_(
                slot=slot,
                terminal_obs=terminal_obs,
                done_env_indices=storage_done_env_indices,
            )

        self._current_obs_slots = next_obs_slots

        self._transition_obs_slots[:, slot] = transition_obs_slots
        self._transition_next_obs_slots[:, slot] = next_obs_slots
        if self.episode_starts is not None:
            assert storage_episode_start_mask is not None
            self.episode_starts[:, slot] = storage_episode_start_mask

        if self._should_store_temporal_state(self._vector_steps_added) and temporal_state is not None:
            self._copy_temporal_state_rows_at_slots_(state=temporal_state, obs_slots=transition_obs_slots)
        if self._should_store_temporal_state(self._vector_steps_added + 1) and next_temporal_state is not None:
            self._copy_temporal_state_rows_at_slots_(state=next_temporal_state, obs_slots=next_obs_slots)

        self.actions[:, slot] = actions.to(device=self.storage_device, dtype=self.storage_dtype)
        self.rewards[:, slot] = rewards.to(device=self.storage_device, dtype=self.storage_dtype)
        self.terminations[:, slot] = terminations.to(device=self.storage_device, dtype=torch.bool)
        self.truncations[:, slot] = truncations.to(device=self.storage_device, dtype=torch.bool)

        if self.previous_actions is not None:
            if previous_actions is None:
                self.previous_actions[:, slot].zero_()
            else:
                self.previous_actions[:, slot] = previous_actions.to(device=self.storage_device, dtype=self.storage_dtype)

        self._write_slot = (slot + 1) % self.capacity_per_env
        self._size_per_env = min(self._size_per_env + 1, self.capacity_per_env)
        self._has_current_obs = True
        if self._current_episode_start_mask is not None:
            self._current_episode_start_mask = dones.to(device=self.storage_device, dtype=torch.bool)
        self._vector_steps_added += 1
        self.total_transitions_added += self.n_envs

    def sample(
            self,
            batch_size: int,
            *,
            replacement: bool = True,
            generator: torch.Generator | None = None,
    ) -> OffPolicyReplayBatch:
        indices = self._sample_transition_indices(
            batch_size=batch_size,
            replacement=replacement,
            generator=generator,
        )
        return self._fetch_indices(indices)

    def sample_episode_windows(
            self,
            batch_size: int,
            *,
            num_next_steps: int,
            replacement: bool = True,
            generator: torch.Generator | None = None,
    ) -> OffPolicyReplayEpisodeSegmentBatch:
        if num_next_steps <= 0:
            raise ValueError(f"num_next_steps must be > 0, got {num_next_steps}")
        indices = self._sample_transition_indices(
            batch_size=batch_size,
            replacement=replacement,
            generator=generator,
        )
        return self._fetch_episode_windows(indices, num_next_steps=num_next_steps)

    def sample_episode_segments(
            self,
            batch_size: int,
            *,
            segment_length: int,
            burn_in_steps: int = 0,
            replacement: bool = True,
            require_initial_temporal_state: bool = True,
            generator: torch.Generator | None = None,
    ) -> OffPolicyReplayEpisodeSegmentBatch:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if segment_length <= 0:
            raise ValueError(f"segment_length must be > 0, got {segment_length}")
        if burn_in_steps < 0:
            raise ValueError(f"burn_in_steps must be >= 0, got {burn_in_steps}")
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        if require_initial_temporal_state and self._temporal_state_available is None:
            raise ValueError(
                "sample_episode_segments(require_initial_temporal_state=True) requires "
                "temporal_state_store_interval to be set on the replay buffer."
            )

        total_sequence_length = burn_in_steps + segment_length
        candidate_env_indices, candidate_logical_starts = self._episode_segment_candidates(
            total_sequence_length=total_sequence_length,
            require_initial_temporal_state=require_initial_temporal_state,
        )
        num_candidates = int(candidate_env_indices.numel())
        if num_candidates == 0:
            raise NoEpisodeSegmentCandidatesError(
                "Cannot sample episode segments: no contiguous replay windows satisfy the requested length, "
                "episode-boundary, and temporal-state checkpoint constraints."
            )
        if not replacement and batch_size > num_candidates:
            raise ValueError(
                f"Cannot sample batch_size={batch_size} episode segments without replacement from "
                f"{num_candidates} candidates."
            )

        candidate_indices = self._sample_indices(
            population_size=num_candidates,
            batch_size=batch_size,
            replacement=replacement,
            generator=generator,
        )
        env_indices = candidate_env_indices[candidate_indices]
        logical_starts = candidate_logical_starts[candidate_indices]
        sequence_offsets = torch.arange(total_sequence_length, dtype=torch.long, device=self.storage_device)
        logical_sequence_slots = logical_starts.unsqueeze(1) + sequence_offsets.unsqueeze(0)
        flat_indices = (env_indices.unsqueeze(1) * self._size_per_env + logical_sequence_slots).reshape(-1)
        flat_batch = self._fetch_indices(flat_indices)

        initial_temporal_state = None
        if require_initial_temporal_state:
            start_transition_slots = self._logical_to_transition_slots(logical_starts)
            start_obs_slots = self._transition_obs_slots[env_indices, start_transition_slots]
            initial_temporal_state = self._to_train_temporal_state(
                self._temporal_state_rows(env_indices=env_indices, obs_slots=start_obs_slots)
            )

        train_mask = torch.ones(
            (batch_size, total_sequence_length),
            dtype=torch.bool,
            device=self.storage_device,
        )
        if burn_in_steps > 0:
            train_mask[:, :burn_in_steps] = False
        return self._reshape_episode_segment_batch(
            flat_batch=flat_batch,
            batch_size=batch_size,
            total_sequence_length=total_sequence_length,
            burn_in_steps=burn_in_steps,
            train_mask=self._to_train(train_mask, dtype=torch.bool),
            initial_temporal_state=initial_temporal_state,
        )

    def _sample_transition_indices(
            self,
            *,
            batch_size: int,
            replacement: bool,
            generator: torch.Generator | None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        if not replacement and batch_size > len(self):
            raise ValueError(
                f"Cannot sample batch_size={batch_size} without replacement from {len(self)} transitions."
            )
        return self._sample_indices(
            population_size=len(self),
            batch_size=batch_size,
            replacement=replacement,
            generator=generator,
        )

    def _sample_indices(
            self,
            *,
            population_size: int,
            batch_size: int,
            replacement: bool,
            generator: torch.Generator | None,
    ) -> torch.Tensor:
        sampling_device = self.storage_device if generator is None else torch.device(generator.device)
        if replacement:
            indices = torch.randint(
                population_size,
                (batch_size,),
                generator=generator,
                device=sampling_device,
            )
        else:
            indices = torch.randperm(
                population_size,
                generator=generator,
                device=sampling_device,
            )[:batch_size]
        return indices.to(device=self.storage_device)

    def get_all(self) -> OffPolicyReplayBatch:
        if len(self) == 0:
            raise ValueError("Cannot fetch from an empty replay buffer.")
        return self._fetch_indices(torch.arange(len(self), device=self.storage_device))

    def _copy_obs_rows_at_slots_(
            self,
            *,
            obs: dict[str, torch.Tensor],
            obs_slots: torch.Tensor,
            target_env_indices: torch.Tensor | None = None,
            source_env_indices: torch.Tensor | None = None,
    ) -> None:
        if target_env_indices is None:
            target_env_indices = torch.arange(self.n_envs, dtype=torch.long, device=self.storage_device)
        if source_env_indices is None:
            source_env_indices = target_env_indices
        obs_slots = obs_slots.to(device=self.storage_device, dtype=torch.long)
        target_env_indices = target_env_indices.to(device=self.storage_device, dtype=torch.long)
        source_env_indices = source_env_indices.to(dtype=torch.long)

        local_obs = self._select_obs_rows_for_storage(obs["local_obs"], source_env_indices, dtype=self.storage_dtype)
        global_obs = self._select_obs_rows_for_storage(obs["global_obs"], source_env_indices, dtype=self.storage_dtype)
        hidden_local_vars = self._select_obs_rows_for_storage(
            obs["hidden_local_vars"],
            source_env_indices,
            dtype=self.storage_dtype,
        )
        hidden_global_vars = self._select_obs_rows_for_storage(
            obs["hidden_global_vars"],
            source_env_indices,
            dtype=self.storage_dtype,
        )

        self._clear_temporal_state_available_(obs_slots=obs_slots, target_env_indices=target_env_indices)
        self.local_obs[target_env_indices, obs_slots] = local_obs
        self.global_obs[target_env_indices, obs_slots] = global_obs
        self.hidden_local_vars[target_env_indices, obs_slots] = hidden_local_vars
        self.hidden_global_vars[target_env_indices, obs_slots] = hidden_global_vars
        if self.agent_mask is not None:
            agent_mask = obs.get("agent_mask", None)
            if agent_mask is None:
                raise ValueError("obs must include agent_mask because this replay buffer stores agent_mask.")
            self.agent_mask[target_env_indices, obs_slots] = self._select_obs_rows_for_storage(
                agent_mask,
                source_env_indices,
                dtype=torch.bool,
            )

    def _clear_terminal_obs_for_slot_(self, slot: int) -> None:
        terminal_envs = self._terminal_envs_by_transition_slot[slot]
        for env_idx in terminal_envs:
            self._terminal_obs_by_env_slot.pop((env_idx, slot), None)
        terminal_envs.clear()

    def _store_terminal_obs_at_slot_(
            self,
            *,
            slot: int,
            terminal_obs: dict[str, torch.Tensor],
            done_env_indices: torch.Tensor,
    ) -> None:
        source_env_indices = self._terminal_obs_source_indices(
            terminal_obs=terminal_obs,
            done_env_indices=done_env_indices,
        )
        terminal_rows = {
            "local_obs": self._select_obs_rows_for_storage(
                terminal_obs["local_obs"],
                source_env_indices,
                dtype=self.storage_dtype,
            ),
            "global_obs": self._select_obs_rows_for_storage(
                terminal_obs["global_obs"],
                source_env_indices,
                dtype=self.storage_dtype,
            ),
            "hidden_local_vars": self._select_obs_rows_for_storage(
                terminal_obs["hidden_local_vars"],
                source_env_indices,
                dtype=self.storage_dtype,
            ),
            "hidden_global_vars": self._select_obs_rows_for_storage(
                terminal_obs["hidden_global_vars"],
                source_env_indices,
                dtype=self.storage_dtype,
            ),
        }
        if self.agent_mask is not None:
            agent_mask = terminal_obs.get("agent_mask", None)
            if agent_mask is None:
                raise ValueError("terminal_obs must include agent_mask because this replay buffer stores agent_mask.")
            terminal_rows["agent_mask"] = self._select_obs_rows_for_storage(
                agent_mask,
                source_env_indices,
                dtype=torch.bool,
            )

        terminal_envs = self._terminal_envs_by_transition_slot[slot]
        for row_idx, env_idx in enumerate(done_env_indices.tolist()):
            self._terminal_obs_by_env_slot[(env_idx, slot)] = {
                key: self._clone_storage_row(value[row_idx])
                for key, value in terminal_rows.items()
            }
            terminal_envs.add(env_idx)

    def _terminal_obs_source_indices(
            self,
            *,
            terminal_obs: dict[str, torch.Tensor],
            done_env_indices: torch.Tensor,
    ) -> torch.Tensor:
        terminal_batch_size = int(terminal_obs["local_obs"].shape[0])
        done_count = int(done_env_indices.numel())
        if terminal_batch_size == self.n_envs:
            return done_env_indices
        if terminal_batch_size == done_count:
            return torch.arange(done_count, dtype=torch.long, device=self.storage_device)
        raise ValueError(
            "terminal_obs must be either full vector-env observations or rows packed in done-env order; "
            f"got leading dimension {terminal_batch_size} for {done_count} done envs across {self.n_envs} envs."
        )

    def _select_obs_rows_for_storage(
            self,
            tensor: torch.Tensor,
            source_env_indices: torch.Tensor,
            *,
            dtype: torch.dtype,
    ) -> torch.Tensor:
        source_env_indices = source_env_indices.to(device=tensor.device, dtype=torch.long)
        return tensor[source_env_indices].to(device=self.storage_device, dtype=dtype)

    def _fetch_indices(self, indices: torch.Tensor) -> OffPolicyReplayBatch:
        env_indices = torch.div(indices, self._size_per_env, rounding_mode="floor")
        logical_transition_slots = indices.remainder(self._size_per_env)
        transition_slots = self._logical_to_transition_slots(logical_transition_slots)
        obs_slots = self._transition_obs_slots[env_indices, transition_slots]
        next_obs_slots = self._transition_next_obs_slots[env_indices, transition_slots]
        terminations = self.terminations[env_indices, transition_slots]
        truncations = self.truncations[env_indices, transition_slots]
        dones = torch.logical_or(terminations, truncations)

        actions = self._to_train(self.actions[env_indices, transition_slots])
        next_local_obs = self.local_obs[env_indices, next_obs_slots]
        next_global_obs = self.global_obs[env_indices, next_obs_slots]
        next_hidden_local_vars = self.hidden_local_vars[env_indices, next_obs_slots]
        next_hidden_global_vars = self.hidden_global_vars[env_indices, next_obs_slots]
        next_agent_mask = None if self.agent_mask is None else self.agent_mask[env_indices, next_obs_slots]
        self._replace_done_next_obs_with_terminal_obs_(
            dones=dones,
            env_indices=env_indices,
            transition_slots=transition_slots,
            next_local_obs=next_local_obs,
            next_global_obs=next_global_obs,
            next_hidden_local_vars=next_hidden_local_vars,
            next_hidden_global_vars=next_hidden_global_vars,
            next_agent_mask=next_agent_mask,
        )

        return OffPolicyReplayBatch(
            local_obs=self._to_train(self.local_obs[env_indices, obs_slots]),
            global_obs=self._to_train(self.global_obs[env_indices, obs_slots]),
            hidden_local_vars=self._to_train(self.hidden_local_vars[env_indices, obs_slots]),
            hidden_global_vars=self._to_train(self.hidden_global_vars[env_indices, obs_slots]),
            agent_mask=None if self.agent_mask is None else self._to_train(
                self.agent_mask[env_indices, obs_slots],
                dtype=torch.bool,
            ),
            actions=actions,
            rewards=self._to_train(self.rewards[env_indices, transition_slots]),
            terminations=self._to_train(terminations, dtype=torch.bool),
            truncations=self._to_train(truncations, dtype=torch.bool),
            previous_actions=None if self.previous_actions is None else self._to_train(
                self.previous_actions[env_indices, transition_slots],
            ),
            next_local_obs=self._to_train(next_local_obs),
            next_global_obs=self._to_train(next_global_obs),
            next_hidden_local_vars=self._to_train(next_hidden_local_vars),
            next_hidden_global_vars=self._to_train(next_hidden_global_vars),
            next_agent_mask=None if next_agent_mask is None else self._to_train(next_agent_mask, dtype=torch.bool),
            episode_start_mask=None if self.episode_starts is None else self._to_train(
                self.episode_starts[env_indices, transition_slots],
                dtype=torch.bool,
            ),
        )

    def _fetch_episode_windows(
            self,
            indices: torch.Tensor,
            *,
            num_next_steps: int,
    ) -> OffPolicyReplayEpisodeSegmentBatch:
        batch_size = int(indices.numel())
        env_indices = torch.div(indices, self._size_per_env, rounding_mode="floor")
        logical_starts = indices.remainder(self._size_per_env)
        sequence_offsets = torch.arange(num_next_steps, dtype=torch.long, device=self.storage_device)
        logical_positions = logical_starts.unsqueeze(1) + sequence_offsets.unsqueeze(0)
        within_replay = logical_positions < self._size_per_env
        safe_logical_positions = logical_positions.clamp_max(self._size_per_env - 1)
        flat_indices = (
            env_indices.unsqueeze(1) * self._size_per_env + safe_logical_positions
        ).reshape(-1)
        flat_batch = self._fetch_indices(flat_indices)
        within_replay = self._to_train(within_replay, dtype=torch.bool)
        window_batch = self._reshape_episode_segment_batch(
            flat_batch=flat_batch,
            batch_size=batch_size,
            total_sequence_length=num_next_steps,
            burn_in_steps=0,
            train_mask=within_replay,
            initial_temporal_state=None,
        )
        previous_episode_end = torch.cat((
            torch.zeros((batch_size, 1), dtype=torch.bool, device=window_batch.actions.device),
            window_batch.episode_ends[:, :-1],
        ), dim=1).to(dtype=torch.long).cumsum(dim=1) > 0
        valid_steps = within_replay & ~previous_episode_end
        return self._mask_episode_window_batch(window_batch, valid_steps=valid_steps)

    @staticmethod
    def _mask_episode_window_batch(
            batch: OffPolicyReplayEpisodeSegmentBatch,
            *,
            valid_steps: torch.Tensor,
    ) -> OffPolicyReplayEpisodeSegmentBatch:
        def mask_tensor(tensor: torch.Tensor, value: bool | float = 0.0) -> torch.Tensor:
            invalid = ~valid_steps.reshape((*valid_steps.shape, *((1,) * (tensor.ndim - 2))))
            return tensor.masked_fill(invalid, value)

        def mask_optional_tensor(
                tensor: torch.Tensor | None,
                value: bool | float = 0.0,
        ) -> torch.Tensor | None:
            if tensor is None:
                return None
            return mask_tensor(tensor, value)

        return OffPolicyReplayEpisodeSegmentBatch(
            local_obs=mask_tensor(batch.local_obs),
            global_obs=mask_tensor(batch.global_obs),
            hidden_local_vars=mask_tensor(batch.hidden_local_vars),
            hidden_global_vars=mask_tensor(batch.hidden_global_vars),
            agent_mask=mask_optional_tensor(batch.agent_mask, True),
            actions=mask_tensor(batch.actions),
            rewards=mask_tensor(batch.rewards),
            terminations=mask_tensor(batch.terminations, False),
            truncations=mask_tensor(batch.truncations, False),
            previous_actions=mask_optional_tensor(batch.previous_actions),
            next_local_obs=mask_tensor(batch.next_local_obs),
            next_global_obs=mask_tensor(batch.next_global_obs),
            next_hidden_local_vars=mask_tensor(batch.next_hidden_local_vars),
            next_hidden_global_vars=mask_tensor(batch.next_hidden_global_vars),
            next_agent_mask=mask_optional_tensor(batch.next_agent_mask, True),
            episode_start_mask=mask_optional_tensor(batch.episode_start_mask, False),
            train_mask=valid_steps,
            initial_temporal_state=None,
            burn_in_steps=0,
        )

    def _episode_segment_candidates(
            self,
            *,
            total_sequence_length: int,
            require_initial_temporal_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._size_per_env < total_sequence_length:
            empty = torch.empty((0,), dtype=torch.long, device=self.storage_device)
            return empty, empty
        if require_initial_temporal_state and self.temporal_states is None:
            empty = torch.empty((0,), dtype=torch.long, device=self.storage_device)
            return empty, empty

        max_start_count = self._size_per_env - total_sequence_length + 1
        logical_positions = torch.arange(self._size_per_env, dtype=torch.long, device=self.storage_device)
        transition_slots_by_logical = self._logical_to_transition_slots(logical_positions)
        start_positions = torch.arange(max_start_count, dtype=torch.long, device=self.storage_device)
        episode_ends = torch.logical_or(
            self.terminations[:, transition_slots_by_logical],
            self.truncations[:, transition_slots_by_logical],
        )
        valid = torch.ones((self.n_envs, max_start_count), dtype=torch.bool, device=self.storage_device)
        if total_sequence_length > 1:
            end_prefix_sum = torch.cat((
                torch.zeros((self.n_envs, 1), dtype=torch.long, device=self.storage_device),
                episode_ends.to(dtype=torch.long).cumsum(dim=1),
            ), dim=1)
            ends_before_final_transition = (
                end_prefix_sum[:, start_positions + total_sequence_length - 1]
                - end_prefix_sum[:, start_positions]
            )
            valid &= ends_before_final_transition == 0
        if require_initial_temporal_state:
            assert self._temporal_state_available is not None
            start_transition_slots = transition_slots_by_logical[:max_start_count]
            start_obs_slots = self._transition_obs_slots[:, start_transition_slots]
            valid &= self._temporal_state_available.gather(1, start_obs_slots)

        return torch.nonzero(valid, as_tuple=True)

    def _reshape_episode_segment_batch(
            self,
            *,
            flat_batch: OffPolicyReplayBatch,
            batch_size: int,
            total_sequence_length: int,
            burn_in_steps: int,
            train_mask: torch.Tensor,
            initial_temporal_state: Any,
    ) -> OffPolicyReplayEpisodeSegmentBatch:
        def reshape_tensor(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch_size, total_sequence_length, *tensor.shape[1:])

        def reshape_optional_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return reshape_tensor(tensor)

        return OffPolicyReplayEpisodeSegmentBatch(
            local_obs=reshape_tensor(flat_batch.local_obs),
            global_obs=reshape_tensor(flat_batch.global_obs),
            hidden_local_vars=reshape_tensor(flat_batch.hidden_local_vars),
            hidden_global_vars=reshape_tensor(flat_batch.hidden_global_vars),
            agent_mask=reshape_optional_tensor(flat_batch.agent_mask),
            actions=reshape_tensor(flat_batch.actions),
            rewards=reshape_tensor(flat_batch.rewards),
            terminations=reshape_tensor(flat_batch.terminations),
            truncations=reshape_tensor(flat_batch.truncations),
            previous_actions=reshape_optional_tensor(flat_batch.previous_actions),
            next_local_obs=reshape_tensor(flat_batch.next_local_obs),
            next_global_obs=reshape_tensor(flat_batch.next_global_obs),
            next_hidden_local_vars=reshape_tensor(flat_batch.next_hidden_local_vars),
            next_hidden_global_vars=reshape_tensor(flat_batch.next_hidden_global_vars),
            next_agent_mask=reshape_optional_tensor(flat_batch.next_agent_mask),
            episode_start_mask=reshape_optional_tensor(flat_batch.episode_start_mask),
            train_mask=train_mask,
            initial_temporal_state=initial_temporal_state,
            burn_in_steps=burn_in_steps,
        )

    def _replace_done_next_obs_with_terminal_obs_(
            self,
            *,
            dones: torch.Tensor,
            env_indices: torch.Tensor,
            transition_slots: torch.Tensor,
            next_local_obs: torch.Tensor,
            next_global_obs: torch.Tensor,
            next_hidden_local_vars: torch.Tensor,
            next_hidden_global_vars: torch.Tensor,
            next_agent_mask: torch.Tensor | None,
    ) -> None:
        done_batch_indices = torch.nonzero(dones, as_tuple=False).flatten()
        if len(done_batch_indices) == 0:
            return

        terminal_local_obs: list[torch.Tensor] = []
        terminal_global_obs: list[torch.Tensor] = []
        terminal_hidden_local_vars: list[torch.Tensor] = []
        terminal_hidden_global_vars: list[torch.Tensor] = []
        terminal_agent_mask: list[torch.Tensor] = []

        done_entries = torch.stack((
            done_batch_indices,
            env_indices[done_batch_indices],
            transition_slots[done_batch_indices],
        ), dim=1).tolist()
        for batch_idx, env_idx, transition_slot in done_entries:
            terminal_obs = self._terminal_obs_by_env_slot.get((env_idx, transition_slot))
            if terminal_obs is None:
                raise ValueError(
                    f"Missing terminal_obs for done transition env_idx={env_idx}, transition_slot={transition_slot}."
                )
            terminal_local_obs.append(terminal_obs["local_obs"])
            terminal_global_obs.append(terminal_obs["global_obs"])
            terminal_hidden_local_vars.append(terminal_obs["hidden_local_vars"])
            terminal_hidden_global_vars.append(terminal_obs["hidden_global_vars"])
            if next_agent_mask is not None:
                terminal_agent_mask.append(terminal_obs["agent_mask"])

        next_local_obs[done_batch_indices] = torch.stack(terminal_local_obs, dim=0)
        next_global_obs[done_batch_indices] = torch.stack(terminal_global_obs, dim=0)
        next_hidden_local_vars[done_batch_indices] = torch.stack(terminal_hidden_local_vars, dim=0)
        next_hidden_global_vars[done_batch_indices] = torch.stack(terminal_hidden_global_vars, dim=0)
        if next_agent_mask is not None:
            next_agent_mask[done_batch_indices] = torch.stack(terminal_agent_mask, dim=0)

    def _new_storage_tensor(self, shape: tuple[int, ...], *, dtype: torch.dtype) -> torch.Tensor:
        if self.storage_pin_memory:
            return torch.empty(shape, dtype=dtype, device=self.storage_device, pin_memory=True).zero_()
        return torch.zeros(shape, dtype=dtype, device=self.storage_device)

    def _clone_storage_row(self, tensor: torch.Tensor) -> torch.Tensor:
        cloned = tensor.clone()
        if self.storage_pin_memory and cloned.device.type == "cpu" and not cloned.is_pinned():
            return cloned.pin_memory()
        return cloned

    def _to_train(self, tensor: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        target_dtype = dtype
        if target_dtype is None and tensor.is_floating_point():
            target_dtype = self.train_dtype
        if (
                self.storage_pin_memory
                and self.train_device.type == "cuda"
                and tensor.device.type == "cpu"
                and not tensor.is_pinned()
        ):
            tensor = tensor.pin_memory()
        return tensor.to(
            device=self.train_device,
            dtype=target_dtype if target_dtype is not None else tensor.dtype,
            non_blocking=self.non_blocking_train_transfer,
        )

    def _to_train_temporal_state(self, state: Any) -> Any:
        if state is None:
            return None
        if torch.is_tensor(state):
            return self._to_train(state)
        if isinstance(state, tuple):
            return tuple(self._to_train_temporal_state(item) for item in state)
        if isinstance(state, list):
            return [self._to_train_temporal_state(item) for item in state]
        if isinstance(state, Mapping):
            return type(state)((key, self._to_train_temporal_state(value)) for key, value in state.items())
        raise TypeError(f"Unsupported temporal state item: {type(state).__name__}")

    def _clear_temporal_state_available_(
            self,
            *,
            obs_slots: torch.Tensor,
            target_env_indices: torch.Tensor,
    ) -> None:
        if self._temporal_state_available is None:
            return
        self._temporal_state_available[target_env_indices, obs_slots] = False

    def _should_store_temporal_state(self, observation_step_idx: int) -> bool:
        return (
            self.temporal_state_store_interval is not None
            and observation_step_idx % self.temporal_state_store_interval == 0
        )

    def _copy_temporal_state_rows_at_slots_(
            self,
            *,
            state: Any,
            obs_slots: torch.Tensor,
            target_env_indices: torch.Tensor | None = None,
            source_env_indices: torch.Tensor | None = None,
    ) -> None:
        if self._temporal_state_available is None:
            return
        if target_env_indices is None:
            target_env_indices = torch.arange(self.n_envs, dtype=torch.long, device=self.storage_device)
        if source_env_indices is None:
            source_env_indices = target_env_indices
        obs_slots = obs_slots.to(device=self.storage_device, dtype=torch.long)
        target_env_indices = target_env_indices.to(device=self.storage_device, dtype=torch.long)
        source_env_indices = source_env_indices.to(dtype=torch.long)

        if self.temporal_states is None:
            self.temporal_states = self._new_temporal_state_storage(state)
        self._copy_temporal_state_tree_rows_(
            target=self.temporal_states,
            source=state,
            obs_slots=obs_slots,
            target_env_indices=target_env_indices,
            source_env_indices=source_env_indices,
        )
        self._temporal_state_available[target_env_indices, obs_slots] = True

    def _new_temporal_state_storage(self, state: Any) -> Any:
        if torch.is_tensor(state):
            if int(state.shape[0]) != self.n_envs:
                raise ValueError(f"Temporal state leading dimension must be {self.n_envs}, got {state.shape[0]}")
            dtype = self.temporal_state_storage_dtype if state.is_floating_point() else state.dtype
            return self._new_storage_tensor(
                (self.n_envs, self.observation_capacity_per_env, *state.shape[1:]),
                dtype=dtype,
            )
        if isinstance(state, tuple):
            return tuple(self._new_temporal_state_storage(item) for item in state)
        if isinstance(state, list):
            return [self._new_temporal_state_storage(item) for item in state]
        if isinstance(state, Mapping):
            return type(state)((key, self._new_temporal_state_storage(value)) for key, value in state.items())
        raise TypeError(f"Unsupported temporal state item: {type(state).__name__}")

    def _copy_temporal_state_tree_rows_(
            self,
            *,
            target: Any,
            source: Any,
            obs_slots: torch.Tensor,
            target_env_indices: torch.Tensor,
            source_env_indices: torch.Tensor,
    ) -> None:
        if torch.is_tensor(target) and torch.is_tensor(source):
            source_indices = source_env_indices.to(device=source.device, dtype=torch.long)
            source_rows = source[source_indices].to(device=self.storage_device, dtype=target.dtype)
            target[target_env_indices, obs_slots] = source_rows
            return
        if isinstance(target, tuple) and isinstance(source, tuple):
            for target_item, source_item in zip(target, source, strict=True):
                self._copy_temporal_state_tree_rows_(
                    target=target_item,
                    source=source_item,
                    obs_slots=obs_slots,
                    target_env_indices=target_env_indices,
                    source_env_indices=source_env_indices,
                )
            return
        if isinstance(target, list) and isinstance(source, list):
            for target_item, source_item in zip(target, source, strict=True):
                self._copy_temporal_state_tree_rows_(
                    target=target_item,
                    source=source_item,
                    obs_slots=obs_slots,
                    target_env_indices=target_env_indices,
                    source_env_indices=source_env_indices,
                )
            return
        if isinstance(target, Mapping) and isinstance(source, Mapping):
            if target.keys() != source.keys():
                raise ValueError("Temporal state mappings must have matching keys")
            for key in target:
                self._copy_temporal_state_tree_rows_(
                    target=target[key],
                    source=source[key],
                    obs_slots=obs_slots,
                    target_env_indices=target_env_indices,
                    source_env_indices=source_env_indices,
                )
            return
        raise TypeError(
            f"Temporal state structures do not match: {type(target).__name__} and {type(source).__name__}"
        )

    def _temporal_state_rows(
            self,
            *,
            env_indices: torch.Tensor,
            obs_slots: torch.Tensor,
    ) -> Any:
        return self._temporal_state_tree_rows(
            state=self.temporal_states,
            env_indices=env_indices,
            obs_slots=obs_slots,
        )

    def _temporal_state_tree_rows(
            self,
            *,
            state: Any,
            env_indices: torch.Tensor,
            obs_slots: torch.Tensor,
    ) -> Any:
        if torch.is_tensor(state):
            return state[env_indices, obs_slots]
        if isinstance(state, tuple):
            return tuple(
                self._temporal_state_tree_rows(state=item, env_indices=env_indices, obs_slots=obs_slots)
                for item in state
            )
        if isinstance(state, list):
            return [
                self._temporal_state_tree_rows(state=item, env_indices=env_indices, obs_slots=obs_slots)
                for item in state
            ]
        if isinstance(state, Mapping):
            return type(state)(
                (
                    key,
                    self._temporal_state_tree_rows(state=value, env_indices=env_indices, obs_slots=obs_slots),
                )
                for key, value in state.items()
            )
        raise TypeError(f"Unsupported temporal state item: {type(state).__name__}")

    def _advance_obs_slots(self, obs_slots: torch.Tensor) -> torch.Tensor:
        return (obs_slots + 1) % self.observation_capacity_per_env

    def _logical_to_transition_slots(self, logical_transition_slots: torch.Tensor) -> torch.Tensor:
        if self._size_per_env < self.capacity_per_env:
            return logical_transition_slots
        return (logical_transition_slots + self._write_slot) % self.capacity_per_env
