from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import pytest
import torch

from swarmbots.mjw_env.mjw_env_tensor_ops import (
    MJWActionLayout,
    MJWObservationLayout,
    build_mjw_env_tensor_operations,
)
from swarmbots.mjw_env.mjw_swarm_bots_vector_env import (
    MJWSwarmBotsVectorEnv,
    _resolve_torch_device,
    _SettledResetSnapshotBuffer,
)
from swarmbots.mjw_env.scenarios.base_mjw_scenario import MJWStepResult


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


class _FakeScenario:
    pass


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


def _make_uninitialized_env(
        *,
        use_settled_resets: bool = True,
        num_envs: int = 4,
) -> tuple[MJWSwarmBotsVectorEnv, _FakeRuntime]:
    env = object.__new__(MJWSwarmBotsVectorEnv)
    runtime = _FakeRuntime()
    env.num_envs = num_envs
    env.device = torch.device("cpu")
    env._scenario_runtime = runtime
    env._rng = torch.Generator(device="cpu")
    env._rng.manual_seed(123)
    env._use_settled_resets = use_settled_resets
    env._settled_reset_buffer = None
    env._settle_executor = None
    env._live_episode_recorder = _FakeLiveEpisodeRecorder()
    env._continuous_connector_actions = False
    env._pending_step = None
    env._step_status_host = torch.empty(num_envs + 1, dtype=torch.bool)
    env._step_status_ready = None
    env._nefc_overflow_host = torch.empty((), dtype=torch.float32)
    env._tensor_operations = build_mjw_env_tensor_operations(
        observation_layout=MJWObservationLayout(
            num_envs=num_envs,
            num_agents=1,
            num_connectors=1,
            free_joint_position=slice(0, 3),
            free_joint_rotation=slice(3, 7),
            hinge=slice(7, 9),
            qvel=slice(9, 17),
            connector=slice(17, 22),
            connector_position=slice(22, 25),
            use_rot6d=False,
            include_connector_positions=False,
        ),
        action_layout=MJWActionLayout(
            num_envs=num_envs,
            continuous_connectors=False,
        ),
        compile_operations=False,
        compile_mode="default",
    )
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


def test_mjw_indexless_cuda_device_uses_torch_current_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    assert _resolve_torch_device("cuda") == torch.device("cuda:3")
    assert _resolve_torch_device("cuda:1") == torch.device("cuda:1")
    assert _resolve_torch_device("cpu") == torch.device("cpu")


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


def test_mjw_deferred_step_preserves_terminal_obs_and_returns_same_step_reset_obs() -> None:
    env, runtime = _make_uninitialized_env(use_settled_resets=False, num_envs=2)
    env._n_agents = 1
    env._n_actuators = 1
    env._n_connectors = 1
    env._qpos = torch.zeros((2, 1), dtype=torch.float32)
    env._qvel = torch.zeros((2, 1), dtype=torch.float32)
    env.current_step = torch.zeros((2,), dtype=torch.long)
    env.is_first_episode = torch.ones((2,), dtype=torch.bool)
    env._episode_length_limit = torch.tensor([1, 2], dtype=torch.long)
    env._first_episode_length_limit = None
    env.simulation_unstable_reward = -1.0
    env._steps_since_nefc_overflow_check = 0
    env._nefc_overflow_check_interval_steps = 100
    reset_done_calls: list[torch.Tensor] = []

    def compute_step_rewards(*, stable_mask: torch.Tensor) -> MJWStepResult:
        assert torch.equal(stable_mask, torch.tensor([True, True]))
        return MJWStepResult(reward=torch.ones((2,), dtype=torch.float32), info={})

    runtime.compute_step_rewards = compute_step_rewards
    env._apply_actions = lambda *, actuators, connectors: None
    env._run_physics = lambda: None
    env._update_max_nefc_since_overflow_check = lambda: None
    env._maybe_notify_nefc_overflow = lambda: None
    env._build_obs = lambda: {"obs": env.current_step.clone().unsqueeze(-1)}
    env._apply_error_obs = lambda obs, unstable_mask: obs
    def reset_done_worlds(dones: torch.Tensor) -> None:
        reset_done_calls.append(dones.clone())
        env.current_step[dones] = 0

    env._reset_done_worlds = reset_done_worlds

    actions = {
        "actuators": torch.zeros((2, 1, 1), dtype=torch.float32),
        "connectors": torch.zeros((2, 1, 1), dtype=torch.bool),
    }
    env.begin_step(actions)

    assert env.has_pending_step
    assert reset_done_calls == []

    obs, rewards, terminations, truncations, infos = env.end_step()

    assert not env.has_pending_step
    assert torch.equal(rewards, torch.ones((2,), dtype=torch.float32))
    assert torch.equal(terminations, torch.tensor([False, False]))
    assert torch.equal(truncations, torch.tensor([True, False]))
    assert torch.equal(infos["_final_obs"], torch.tensor([True, False]))
    assert torch.equal(infos["final_obs"]["obs"], torch.tensor([[1], [1]]))
    assert torch.equal(obs["obs"], torch.tensor([[0], [1]]))
    assert len(reset_done_calls) == 1
    assert torch.equal(reset_done_calls[0], torch.tensor([True, False]))


def test_mjw_step_terminates_worlds_with_any_nonfinite_physics_state(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, runtime = _make_uninitialized_env(use_settled_resets=False)
    env.scenario = _FakeScenario()
    env._n_agents = 1
    env._n_actuators = 1
    env._n_connectors = 1
    env._qpos = torch.tensor([
        [float("nan"), 0.0],
        [float("inf"), 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    env._qvel = torch.tensor([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, float("-inf")],
        [0.0, 0.0],
    ])
    env.current_step = torch.zeros((4,), dtype=torch.long)
    env.is_first_episode = torch.ones((4,), dtype=torch.bool)
    env._episode_length_limit = torch.full((4,), 10, dtype=torch.long)
    env._first_episode_length_limit = None
    env.simulation_unstable_reward = -7.0
    env._steps_since_nefc_overflow_check = 0
    env._nefc_overflow_check_interval_steps = 100
    reset_done_calls: list[torch.Tensor] = []
    instability_notifications: list[dict[str, Any]] = []

    def compute_step_rewards(*, stable_mask: torch.Tensor) -> MJWStepResult:
        assert torch.equal(stable_mask, torch.tensor([False, False, False, True]))
        return MJWStepResult(reward=torch.ones((4,), dtype=torch.float32), info={})

    runtime.compute_step_rewards = compute_step_rewards
    env._apply_actions = lambda *, actuators, connectors: None
    env._run_physics = lambda: None
    env._update_max_nefc_since_overflow_check = lambda: None
    env._maybe_notify_nefc_overflow = lambda: None
    env._build_obs = lambda: {
        "local_obs": torch.ones((4, 1)),
        "global_obs": torch.ones((4, 1)),
        "hidden_local_vars": torch.ones((4, 1)),
        "hidden_global_vars": torch.ones((4, 1)),
    }
    env._reset_done_worlds = lambda dones: reset_done_calls.append(dones.clone())
    monkeypatch.setattr(
        "swarmbots.mjw_env.mjw_swarm_bots_vector_env.notify_mjw_simulation_instability_once",
        lambda **kwargs: instability_notifications.append(kwargs) or True,
    )

    _obs, rewards, terminations, truncations, infos = env.step(
        {
            "actuators": torch.zeros((4, 1, 1), dtype=torch.float32),
            "connectors": torch.zeros((4, 1, 1), dtype=torch.bool),
        }
    )

    expected_unstable = torch.tensor([True, True, True, False])
    torch.testing.assert_close(rewards, torch.tensor([-7.0, -7.0, -7.0, 1.0]))
    assert torch.equal(terminations, expected_unstable)
    assert not truncations.any()
    assert torch.equal(infos["_final_obs"], expected_unstable)
    for final_obs in infos["final_obs"].values():
        assert torch.equal(final_obs[:3], torch.zeros((3, 1)))
    assert len(reset_done_calls) == 1
    assert torch.equal(reset_done_calls[0], expected_unstable)
    assert instability_notifications == [{
        "scenario_name": "_FakeScenario",
        "num_envs": 4,
        "unstable_world_indices": [0, 1, 2],
    }]


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
