from collections.abc import Callable
from dataclasses import dataclass

import torch


GatheredReplayStorage = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]


def _advance_obs_slots(obs_slots: torch.Tensor, observation_capacity_per_env: int) -> torch.Tensor:
    return (obs_slots + 1).remainder(observation_capacity_per_env)


def _logical_to_transition_slots(
        logical_transition_slots: torch.Tensor,
        logical_transition_slot_offset: torch.Tensor,
        capacity_per_env: int,
) -> torch.Tensor:
    return (
        logical_transition_slots + logical_transition_slot_offset
    ).remainder(capacity_per_env)


def _gather_replay_storage(
        indices: torch.Tensor,
        size_per_env: torch.Tensor,
        logical_transition_slot_offset: torch.Tensor,
        capacity_per_env: int,
        transition_obs_slots: torch.Tensor,
        transition_next_obs_slots: torch.Tensor,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        hidden_local_vars: torch.Tensor,
        hidden_global_vars: torch.Tensor,
        scenario_ids: torch.Tensor | None,
        agent_mask: torch.Tensor | None,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        previous_actions: torch.Tensor | None,
        episode_starts: torch.Tensor | None,
) -> GatheredReplayStorage:
    env_indices = torch.div(indices, size_per_env, rounding_mode="floor")
    logical_transition_slots = indices.remainder(size_per_env)
    transition_slots = _logical_to_transition_slots(
        logical_transition_slots,
        logical_transition_slot_offset,
        capacity_per_env,
    )
    obs_slots = transition_obs_slots[env_indices, transition_slots]
    next_obs_slots = transition_next_obs_slots[env_indices, transition_slots]

    return (
        env_indices,
        transition_slots,
        local_obs[env_indices, obs_slots],
        global_obs[env_indices, obs_slots],
        hidden_local_vars[env_indices, obs_slots],
        hidden_global_vars[env_indices, obs_slots],
        None if agent_mask is None else agent_mask[env_indices, obs_slots],
        actions[env_indices, transition_slots],
        rewards[env_indices, transition_slots],
        terminations[env_indices, transition_slots],
        truncations[env_indices, transition_slots],
        None if previous_actions is None else previous_actions[env_indices, transition_slots],
        local_obs[env_indices, next_obs_slots],
        global_obs[env_indices, next_obs_slots],
        hidden_local_vars[env_indices, next_obs_slots],
        hidden_global_vars[env_indices, next_obs_slots],
        None if agent_mask is None else agent_mask[env_indices, next_obs_slots],
        None if episode_starts is None else episode_starts[env_indices, transition_slots],
        None if scenario_ids is None else scenario_ids[env_indices, obs_slots],
        None if scenario_ids is None else scenario_ids[env_indices, next_obs_slots],
    )


def _episode_window_indices(
        indices: torch.Tensor,
        size_per_env: torch.Tensor,
        sequence_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    env_indices = torch.div(indices, size_per_env, rounding_mode="floor")
    logical_starts = indices.remainder(size_per_env)
    logical_positions = logical_starts.unsqueeze(1) + sequence_offsets.unsqueeze(0)
    within_replay = logical_positions < size_per_env
    safe_logical_positions = logical_positions.clamp_max(size_per_env - 1)
    flat_indices = (
        env_indices.unsqueeze(1) * size_per_env + safe_logical_positions
    ).reshape(-1)
    return flat_indices, within_replay


def _episode_segment_indices(
        env_indices: torch.Tensor,
        logical_starts: torch.Tensor,
        size_per_env: torch.Tensor,
        sequence_offsets: torch.Tensor,
) -> torch.Tensor:
    logical_sequence_slots = logical_starts.unsqueeze(1) + sequence_offsets.unsqueeze(0)
    return (
        env_indices.unsqueeze(1) * size_per_env + logical_sequence_slots
    ).reshape(-1)


def _episode_segment_candidate_mask(
        logical_positions: torch.Tensor,
        logical_transition_slot_offset: torch.Tensor,
        capacity_per_env: int,
        total_sequence_length: int,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        transition_obs_slots: torch.Tensor,
        temporal_state_available: torch.Tensor | None,
        allow_episode_boundaries: bool,
) -> torch.Tensor:
    transition_slots_by_logical = _logical_to_transition_slots(
        logical_positions,
        logical_transition_slot_offset,
        capacity_per_env,
    )
    max_start_count = logical_positions.shape[0] - total_sequence_length + 1
    valid = torch.ones(
        (terminations.shape[0], max_start_count),
        dtype=torch.bool,
        device=terminations.device,
    )
    if not allow_episode_boundaries and total_sequence_length > 1:
        start_positions = torch.arange(max_start_count, dtype=torch.long, device=terminations.device)
        episode_ends = torch.logical_or(
            terminations[:, transition_slots_by_logical],
            truncations[:, transition_slots_by_logical],
        )
        end_prefix_sum = torch.cat((
            torch.zeros((terminations.shape[0], 1), dtype=torch.long, device=terminations.device),
            episode_ends.to(dtype=torch.long).cumsum(dim=1),
        ), dim=1)
        ends_before_final_transition = (
            end_prefix_sum[:, start_positions + total_sequence_length - 1]
            - end_prefix_sum[:, start_positions]
        )
        valid &= ends_before_final_transition == 0
    if temporal_state_available is not None:
        start_transition_slots = transition_slots_by_logical[:max_start_count]
        start_obs_slots = transition_obs_slots[:, start_transition_slots]
        valid &= temporal_state_available.gather(1, start_obs_slots)
    return valid


def _release_temporal_state_slots(
        temporal_state_available: torch.Tensor,
        temporal_state_indices: torch.Tensor,
        temporal_state_slots_in_use: torch.Tensor,
        target_env_indices: torch.Tensor,
        obs_slots: torch.Tensor,
) -> None:
    checkpoint_slots = temporal_state_indices[target_env_indices, obs_slots]
    has_checkpoint = checkpoint_slots >= 0
    safe_checkpoint_slots = checkpoint_slots.clamp_min(0)
    temporal_state_slots_in_use[target_env_indices, safe_checkpoint_slots] &= ~has_checkpoint
    temporal_state_indices[target_env_indices, obs_slots] = -1
    temporal_state_available[target_env_indices, obs_slots] = False


def _allocate_temporal_state_slots(
        temporal_state_indices: torch.Tensor,
        temporal_state_slots_in_use: torch.Tensor,
        target_env_indices: torch.Tensor,
        obs_slots: torch.Tensor,
) -> torch.Tensor:
    checkpoint_slots = temporal_state_indices[target_env_indices, obs_slots]
    needs_checkpoint_slot = checkpoint_slots < 0
    free_slots = ~temporal_state_slots_in_use[target_env_indices]
    allocated_slots = free_slots.to(dtype=torch.long).argmax(dim=1)
    checkpoint_slots = torch.where(needs_checkpoint_slot, allocated_slots, checkpoint_slots)
    temporal_state_slots_in_use[target_env_indices, checkpoint_slots] = True
    temporal_state_indices[target_env_indices, obs_slots] = checkpoint_slots
    return checkpoint_slots


@dataclass(frozen=True, slots=True)
class ReplayBufferTensorOperations:
    advance_obs_slots: Callable[[torch.Tensor, int], torch.Tensor]
    logical_to_transition_slots: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
    gather_replay_storage: Callable[..., GatheredReplayStorage]
    episode_window_indices: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    episode_segment_indices: Callable[..., torch.Tensor]
    episode_segment_candidate_mask: Callable[..., torch.Tensor]
    release_temporal_state_slots: Callable[..., None]
    allocate_temporal_state_slots: Callable[..., torch.Tensor]


def build_replay_buffer_tensor_operations(*, compile_operations: bool) -> ReplayBufferTensorOperations:
    operations = ReplayBufferTensorOperations(
        advance_obs_slots=_advance_obs_slots,
        logical_to_transition_slots=_logical_to_transition_slots,
        gather_replay_storage=_gather_replay_storage,
        episode_window_indices=_episode_window_indices,
        episode_segment_indices=_episode_segment_indices,
        episode_segment_candidate_mask=_episode_segment_candidate_mask,
        release_temporal_state_slots=_release_temporal_state_slots,
        allocate_temporal_state_slots=_allocate_temporal_state_slots,
    )
    if not compile_operations:
        return operations
    if not hasattr(torch, "compile") or not callable(torch.compile):
        raise RuntimeError("Compiling replay-buffer tensor operations requires torch.compile support.")

    return ReplayBufferTensorOperations(
        # These are single remainder kernels. Compiling them adds Dynamo dispatch and a large
        # first-write compilation stall without creating a useful fusion opportunity.
        advance_obs_slots=operations.advance_obs_slots,
        logical_to_transition_slots=operations.logical_to_transition_slots,
        # Sampling sizes can vary across public APIs and while replay fills, so let Dynamo
        # generalize them after observing variation. Temporal checkpoint bookkeeping always
        # processes a fixed number of env rows.
        gather_replay_storage=torch.compile(operations.gather_replay_storage, fullgraph=True),
        episode_window_indices=torch.compile(operations.episode_window_indices, fullgraph=True),
        episode_segment_indices=torch.compile(operations.episode_segment_indices, fullgraph=True),
        episode_segment_candidate_mask=torch.compile(
            operations.episode_segment_candidate_mask,
            fullgraph=True,
        ),
        release_temporal_state_slots=torch.compile(
            operations.release_temporal_state_slots,
            fullgraph=True,
            dynamic=False,
        ),
        allocate_temporal_state_slots=torch.compile(
            operations.allocate_temporal_state_slots,
            fullgraph=True,
            dynamic=False,
        ),
    )
