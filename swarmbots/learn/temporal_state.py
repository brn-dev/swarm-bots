from collections.abc import Mapping, Sequence
from typing import Any, Callable

import torch

TemporalState = Any


def clone_temporal_state(state: TemporalState) -> TemporalState:
    return _map_temporal_state(state, lambda tensor: tensor.clone())


def detach_temporal_state(state: TemporalState) -> TemporalState:
    return _map_temporal_state(state, lambda tensor: tensor.detach())


def clone_detach_temporal_state(state: TemporalState) -> TemporalState:
    return _map_temporal_state(state, lambda tensor: tensor.detach().clone())


def move_temporal_state(
        state: TemporalState,
        *,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
) -> TemporalState:
    def move_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if dtype is not None and tensor.is_floating_point():
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)

    return _map_temporal_state(state, move_tensor)


def index_temporal_state(
        state: TemporalState,
        indices: torch.Tensor | slice,
) -> TemporalState:
    return _map_temporal_state(state, lambda tensor: tensor[indices])


def index_temporal_state_batch_time(
        state: TemporalState,
        indices: torch.Tensor,
) -> TemporalState:
    return _map_temporal_state(
        state,
        lambda tensor: tensor[indices[:, 0], indices[:, 1]],
    )


def copy_temporal_state_rows_(
        target: TemporalState,
        source: TemporalState,
        indices: torch.Tensor,
) -> None:
    if target is None and source is None:
        return
    if torch.is_tensor(target) and torch.is_tensor(source):
        target[indices] = source[indices]
        return
    if isinstance(target, tuple) and isinstance(source, tuple):
        for target_item, source_item in zip(target, source, strict=True):
            copy_temporal_state_rows_(target_item, source_item, indices)
        return
    if isinstance(target, list) and isinstance(source, list):
        for target_item, source_item in zip(target, source, strict=True):
            copy_temporal_state_rows_(target_item, source_item, indices)
        return
    if isinstance(target, Mapping) and isinstance(source, Mapping):
        if target.keys() != source.keys():
            raise ValueError("Temporal state mappings must have matching keys")
        for key in target:
            copy_temporal_state_rows_(target[key], source[key], indices)
        return
    raise TypeError(
        f"Temporal state structures do not match: {type(target).__name__} and {type(source).__name__}"
    )


def flatten_temporal_state_batch_agents(state: TemporalState) -> TemporalState:
    return _flatten_temporal_state_first_two_dimensions(state)


def unflatten_temporal_state_batch_agents(
        state: TemporalState,
        *,
        batch_size: int,
        n_agents: int,
) -> TemporalState:
    return _map_temporal_state(
        state,
        lambda tensor: tensor.reshape(batch_size, n_agents, *tensor.shape[1:]),
    )


def stack_temporal_states(states: Sequence[TemporalState], *, dim: int) -> TemporalState:
    return _combine_temporal_states(states, lambda tensors: torch.stack(tensors, dim=dim))


def concatenate_temporal_states(states: Sequence[TemporalState]) -> TemporalState:
    return _combine_temporal_states(states, lambda tensors: torch.cat(tensors, dim=0))


def _combine_temporal_states(
        states: Sequence[TemporalState],
        tensor_fn: Callable[[Sequence[torch.Tensor]], torch.Tensor],
) -> TemporalState:
    if not states:
        raise ValueError("Expected at least one temporal state")

    first = states[0]
    if first is None:
        if any(state is not None for state in states):
            raise ValueError("Temporal states must either all be None or all contain state")
        return None
    if torch.is_tensor(first):
        return tensor_fn(states)
    if isinstance(first, tuple):
        return tuple(
            _combine_temporal_states([state[idx] for state in states], tensor_fn)
            for idx in range(len(first))
        )
    if isinstance(first, list):
        return [
            _combine_temporal_states([state[idx] for state in states], tensor_fn)
            for idx in range(len(first))
        ]
    if isinstance(first, Mapping):
        return type(first)(
            (key, _combine_temporal_states([state[key] for state in states], tensor_fn))
            for key in first
        )
    raise TypeError(f"Unsupported temporal state item: {type(first).__name__}")


def _map_temporal_state(
        state: TemporalState,
        tensor_fn: Callable[[torch.Tensor], torch.Tensor],
) -> TemporalState:
    if state is None:
        return None
    if torch.is_tensor(state):
        return tensor_fn(state)
    if isinstance(state, tuple):
        return tuple(_map_temporal_state(item, tensor_fn) for item in state)
    if isinstance(state, list):
        return [_map_temporal_state(item, tensor_fn) for item in state]
    if isinstance(state, Mapping):
        return type(state)((key, _map_temporal_state(value, tensor_fn)) for key, value in state.items())
    raise TypeError(f"Unsupported temporal state item: {type(state).__name__}")


def _flatten_temporal_state_first_two_dimensions(state: TemporalState) -> TemporalState:
    return _map_temporal_state(state, lambda tensor: tensor.flatten(0, 1))
