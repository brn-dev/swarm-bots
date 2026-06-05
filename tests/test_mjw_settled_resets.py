from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import pytest
import torch

from swarmbots.mjw_env.mjw_swarm_bots_vector_env import MJWSwarmBotsVectorEnv, _SettledResetSnapshotBuffer


class _FakeRuntime:
    def __init__(self) -> None:
        self.raw_reset_calls: list[tuple[list[int], Any]] = []
        self.settled_reset_calls: list[tuple[list[int], list[Any]]] = []
        self.sampled_batches: list[list[str]] = []
        self.settled_specs: list[list[str]] = []

    def sample_reset_batch(self, *, n_reset: int, rng: torch.Generator) -> list[str]:
        _ = rng
        batch = [f"sample-{len(self.sampled_batches)}-{idx}" for idx in range(n_reset)]
        self.sampled_batches.append(batch)
        return batch

    def select_reset_batch(self, *, reset_batch: list[str], mask: torch.Tensor) -> list[str]:
        return [value for value, selected in zip(reset_batch, mask.tolist(), strict=True) if selected]

    def apply_reset_batch(self, *, world_idx: torch.Tensor, reset_batch: Any) -> None:
        self.raw_reset_calls.append((world_idx.detach().cpu().tolist(), reset_batch))

    def build_cpu_reset_specs(self, *, reset_batch: list[str]) -> list[str]:
        return [f"spec-{value}" for value in reset_batch]

    def settle_cpu_reset_specs(self, *, specs: list[str]) -> list[str]:
        self.settled_specs.append(specs)
        return [f"settled-{spec}" for spec in specs]

    def apply_settled_reset_batch(self, *, world_idx: torch.Tensor, snapshots: list[Any]) -> None:
        self.settled_reset_calls.append((world_idx.detach().cpu().tolist(), snapshots))


class _FakeLiveEpisodeRecorder:
    def is_active(self) -> bool:
        return False

    def close(self) -> None:
        return None


class _ManualExecutor:
    def __init__(self) -> None:
        self.submissions: list[tuple[Callable[..., list[Any]], dict[str, Any], Future[list[Any]]]] = []

    def submit(self, fn: Callable[..., list[Any]], **kwargs: Any) -> Future[list[Any]]:
        future: Future[list[Any]] = Future()
        self.submissions.append((fn, kwargs, future))
        return future

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        _ = wait, cancel_futures


class _FakeSettledResetBuffer:
    def __init__(self) -> None:
        self.clear_calls: list[bool] = []
        self.drain_ready_calls = 0
        self.fill_async_calls = 0

    def clear(self, *, wait: bool) -> None:
        self.clear_calls.append(wait)

    def drain_ready(self) -> None:
        self.drain_ready_calls += 1

    def fill_async(self) -> None:
        self.fill_async_calls += 1


def _make_uninitialized_env(*, use_settled_resets: bool = True) -> tuple[MJWSwarmBotsVectorEnv, _FakeRuntime]:
    env = object.__new__(MJWSwarmBotsVectorEnv)
    runtime = _FakeRuntime()
    env.num_envs = 4
    env.device = torch.device("cpu")
    env._scenario_runtime = runtime
    env._rng = torch.Generator(device="cpu")
    env._rng.manual_seed(123)
    env._use_settled_resets = use_settled_resets
    env._settled_reset_buffer = None
    env._settle_executor = None
    env._live_episode_recorder = _FakeLiveEpisodeRecorder()
    return env, runtime


def _make_snapshot_buffer(
    *,
    runtime: _FakeRuntime,
    rng: torch.Generator,
    executor: Any,
    capacity: int = 4,
    batch_size: int = 2,
) -> _SettledResetSnapshotBuffer:
    return _SettledResetSnapshotBuffer(
        scenario_runtime=runtime,
        rng=rng,
        executor=executor,
        capacity=capacity,
        batch_size=batch_size,
    )


def test_mjw_done_without_settled_resets_uses_raw_reset() -> None:
    env, runtime = _make_uninitialized_env(use_settled_resets=False)

    env._reset_done_worlds(torch.tensor([False, True, False, True], dtype=torch.bool))

    assert runtime.raw_reset_calls == [([1, 3], ["sample-0-0", "sample-0-1"])]
    assert runtime.settled_specs == []
    assert runtime.settled_reset_calls == []


def test_mjw_done_termination_without_buffer_uses_settled_reset() -> None:
    env, runtime = _make_uninitialized_env(use_settled_resets=True)

    env._reset_done_worlds(torch.tensor([False, True, False, True], dtype=torch.bool))

    assert runtime.raw_reset_calls == []
    assert runtime.settled_specs == [["spec-sample-0-0", "spec-sample-0-1"]]
    assert runtime.settled_reset_calls == [
        ([1, 3], ["settled-spec-sample-0-0", "settled-spec-sample-0-1"])
    ]


def test_mjw_done_uses_ready_settled_snapshot_buffer() -> None:
    env, runtime = _make_uninitialized_env(use_settled_resets=True)
    executor = _ManualExecutor()
    env._settled_reset_buffer = _make_snapshot_buffer(
        runtime=runtime,
        rng=env._rng,
        executor=executor,
    )
    env._settled_reset_buffer._snapshots.extend(["settled-buffer-0", "settled-buffer-1"])

    env._reset_done_worlds(torch.tensor([True, False, True, False], dtype=torch.bool))

    assert runtime.raw_reset_calls == []
    assert runtime.settled_reset_calls == [([0, 2], ["settled-buffer-0", "settled-buffer-1"])]
    assert len(executor.submissions) == 1


@pytest.mark.parametrize(
    ("capacity", "batch_size", "error_match"),
    [
        (0, 1, "capacity > 0"),
        (1, 0, "batch_size > 0"),
    ],
)
def test_mjw_settled_snapshot_buffer_rejects_non_positive_limits(
    *,
    capacity: int,
    batch_size: int,
    error_match: str,
) -> None:
    env, runtime = _make_uninitialized_env(use_settled_resets=True)
    executor = _ManualExecutor()

    with pytest.raises(ValueError, match=error_match):
        _make_snapshot_buffer(
            runtime=runtime,
            rng=env._rng,
            executor=executor,
            capacity=capacity,
            batch_size=batch_size,
        )


def test_mjw_settled_snapshot_buffer_fill_async_caps_batch_to_available_capacity() -> None:
    env, runtime = _make_uninitialized_env(use_settled_resets=True)
    executor = _ManualExecutor()
    buffer = _make_snapshot_buffer(runtime=runtime, rng=env._rng, executor=executor, capacity=4, batch_size=3)
    buffer._snapshots.extend(["ready-0", "ready-1"])

    buffer.fill_async()
    buffer.fill_async()

    assert runtime.sampled_batches == [["sample-0-0", "sample-0-1"]]
    assert len(executor.submissions) == 1
    assert executor.submissions[0][1] == {"specs": ["spec-sample-0-0", "spec-sample-0-1"]}


def test_mjw_settled_snapshot_buffer_drains_finished_future_without_refill() -> None:
    env, runtime = _make_uninitialized_env(use_settled_resets=True)
    executor = _ManualExecutor()
    buffer = _make_snapshot_buffer(runtime=runtime, rng=env._rng, executor=executor, capacity=4, batch_size=2)
    buffer.fill_async()
    executor.submissions[0][2].set_result(["settled-0", "settled-1"])

    assert buffer.ready_count() == 2

    assert len(executor.submissions) == 1
    assert buffer.take(2) == ["settled-0", "settled-1"]
    assert len(executor.submissions) == 2


def test_mjw_settled_snapshot_buffer_uses_finished_future_before_sync_fallback() -> None:
    env, runtime = _make_uninitialized_env(use_settled_resets=True)
    executor = _ManualExecutor()
    buffer = _make_snapshot_buffer(runtime=runtime, rng=env._rng, executor=executor, capacity=4, batch_size=2)
    buffer._snapshots.append("ready-0")
    buffer.fill_async()
    executor.submissions[0][2].set_result(["future-0", "future-1"])

    assert buffer.take(3) == ["ready-0", "future-0", "future-1"]

    assert runtime.settled_specs == []
    assert len(executor.submissions) == 2


def test_mjw_settled_snapshot_buffer_synchronously_settles_underflow() -> None:
    env, runtime = _make_uninitialized_env(use_settled_resets=True)
    executor = _ManualExecutor()
    buffer = _make_snapshot_buffer(runtime=runtime, rng=env._rng, executor=executor, capacity=3, batch_size=2)
    buffer._snapshots.append("ready-0")

    assert buffer.take(3) == ["ready-0", "settled-spec-sample-0-0", "settled-spec-sample-0-1"]

    assert runtime.settled_specs == [["spec-sample-0-0", "spec-sample-0-1"]]
    assert len(executor.submissions) == 1
    assert executor.submissions[0][1] == {"specs": ["spec-sample-1-0", "spec-sample-1-1"]}


def test_mjw_reset_with_seed_clears_settled_buffer_before_reseeding() -> None:
    env, runtime = _make_uninitialized_env(use_settled_resets=True)
    buffer = _FakeSettledResetBuffer()
    env._settled_reset_buffer = buffer
    env._initial_settled_reset_done = False
    env.settle_initial_reset = False
    env._build_obs = lambda: {"obs": torch.empty((env.num_envs, 0))}

    obs, info = env.reset(seed=999)

    assert obs["obs"].shape == (4, 0)
    assert info == {}
    assert buffer.clear_calls == [True]
    assert buffer.drain_ready_calls == 1
    assert buffer.fill_async_calls == 1
    assert runtime.raw_reset_calls == [([0, 1, 2, 3], ["sample-0-0", "sample-0-1", "sample-0-2", "sample-0-3"])]


def test_mjw_close_clears_settled_buffer_without_waiting_then_shuts_down_executor() -> None:
    env, _runtime = _make_uninitialized_env(use_settled_resets=True)
    buffer = _FakeSettledResetBuffer()
    env._settled_reset_buffer = buffer
    env._settle_executor = _ManualExecutor()
    env._max_nefc_since_overflow_check = torch.zeros(())
    env._maybe_notify_nefc_overflow = lambda: None

    env.close()

    assert buffer.clear_calls == [False]
    assert env._settled_reset_buffer is None
    assert env._settle_executor is None
