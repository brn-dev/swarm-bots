from __future__ import annotations

from typing import Any

import torch

from swarmbots.mjw_env.mjw_swarm_bots_vector_env import MJWSwarmBotsVectorEnv, _PendingSettledReset


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


class _PendingFuture:
    def __init__(self, result: list[str]) -> None:
        self._result = result
        self.result_called = False

    def done(self) -> bool:
        return False

    def result(self) -> list[str]:
        self.result_called = True
        return self._result


def _make_uninitialized_env(*, use_settled_resets: bool = True) -> tuple[MJWSwarmBotsVectorEnv, _FakeRuntime]:
    env = object.__new__(MJWSwarmBotsVectorEnv)
    runtime = _FakeRuntime()
    env.num_envs = 4
    env.device = torch.device("cpu")
    env._scenario_runtime = runtime
    env._rng = torch.Generator(device="cpu")
    env._rng.manual_seed(123)
    env._use_settled_resets = use_settled_resets
    env._pending_settled_reset = None
    return env, runtime


def test_mjw_done_termination_without_prefetch_uses_settled_reset() -> None:
    env, runtime = _make_uninitialized_env(use_settled_resets=True)

    env._reset_done_worlds(torch.tensor([False, True, False, True], dtype=torch.bool))

    assert runtime.raw_reset_calls == []
    assert runtime.settled_specs == [["spec-sample-0-0", "spec-sample-0-1"]]
    assert runtime.settled_reset_calls == [
        ([1, 3], ["settled-spec-sample-0-0", "settled-spec-sample-0-1"])
    ]


def test_mjw_done_prefetch_waits_for_settled_snapshot_instead_of_raw_reset() -> None:
    env, runtime = _make_uninitialized_env(use_settled_resets=True)
    future = _PendingFuture(["settled-prefetch-0", "settled-prefetch-1"])
    env._pending_settled_reset = _PendingSettledReset(
        world_idx=torch.tensor([0, 2], dtype=torch.long),
        sampled_batch=["prefetch-0", "prefetch-1"],
        future=future,
    )

    env._reset_done_worlds(torch.tensor([True, False, True, False], dtype=torch.bool))

    assert future.result_called
    assert runtime.raw_reset_calls == []
    assert runtime.settled_reset_calls == [([0, 2], ["settled-prefetch-0", "settled-prefetch-1"])]
    assert env._pending_settled_reset is None
