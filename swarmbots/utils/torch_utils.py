from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from copy import copy
from threading import RLock
from typing import Any

import numpy as np
import torch
from torch.utils import dlpack


_torch_rng_lock = RLock()


def resolve_torch_device(device: torch.device | str) -> torch.device:
    if device == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)

    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    if resolved.type == "cpu":
        return torch.device("cpu")
    return resolved


def _cuda_rng_devices(devices: Iterable[torch.device | str | None]) -> set[torch.device]:
    resolved = {resolve_torch_device(device) for device in devices if device is not None}
    return {device for device in resolved if device.type == "cuda"}


def seed_torch_rng(seed: int, device: torch.device | str, *other_devices: torch.device | str | None) -> None:
    """Seed the CPU generator and each selected CUDA generator."""
    cuda_devices = _cuda_rng_devices((device, *other_devices))
    torch.random.default_generator.manual_seed(seed)
    for cuda_device in cuda_devices:
        with torch.cuda.device(cuda_device):
            torch.cuda.manual_seed(seed)


@contextmanager
def _fork_lazy_cuda_rng() -> Iterator[None]:
    # CUDA has no public API to fork a pending manual_seed without initializing the runtime.
    # Keep the caller's queued operations intact if a CPU-only factory never touches a GPU.
    previous_calls = torch.cuda._queued_calls
    previous_seeds = torch.cuda._lazy_seed_tracker
    torch.cuda._queued_calls = previous_calls.copy()
    torch.cuda._lazy_seed_tracker = copy(previous_seeds)
    states: list[torch.Tensor] = []

    def capture_on_init() -> None:
        # Torch runs ordinary initialization callbacks before deferred manual_seed callbacks.
        # Capture the state the caller would have seen on its first draw.
        for pending in previous_seeds.get_calls():
            if pending is not None:
                pending[0]()
        states.extend(torch.cuda.get_rng_state_all())

    try:
        torch.cuda._lazy_call(capture_on_init)
        yield
    finally:
        torch.cuda._queued_calls = previous_calls
        torch.cuda._lazy_seed_tracker = previous_seeds
        if states:
            torch.cuda.set_rng_state_all(states)


@contextmanager
def fork_torch_rng() -> Iterator[None]:
    # Even evaluations on different devices save and restore the same CPU generator.
    with _torch_rng_lock:
        if not torch.cuda.is_initialized():
            with torch.random.fork_rng(devices=[]), _fork_lazy_cuda_rng():
                yield
        else:
            with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
                yield


@contextmanager
def seeded_torch_rng(seed: int, device: torch.device | str, *other_devices: torch.device | str | None) -> Iterator[None]:
    """Run with a reproducible Torch stream, restoring the caller's state on exit."""
    # Factories and policies can reseed every CUDA generator with one torch.manual_seed call.
    with fork_torch_rng():
        seed_torch_rng(seed, device, *other_devices)
        yield


def to_numpy_array(value: Any, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False) if dtype is not None else value
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return array.astype(dtype, copy=False) if dtype is not None else array
    if hasattr(value, "__dlpack__"):
        try:
            array = dlpack.from_dlpack(value).detach().cpu().numpy()
            return array.astype(dtype, copy=False) if dtype is not None else array
        except Exception:
            pass
    return np.asarray(value, dtype=dtype)
