from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from gymnasium.vector import AutoresetMode

from swarmbots import ALL_BENCHMARK_IDS, make_env
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.mjw_env.mjw_swarm_bots_vector_env import MJWSwarmBotsVectorEnv
from swarmbots.mjw_env.scenarios.base_mjw_scenario import MJWStepResult
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def device(request: pytest.FixtureRequest) -> str:
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def _make_env(device: str, **kwargs: Any) -> MJWSwarmBotsVectorEnv:
    settle_time = kwargs.pop("settle_time", 0.0)
    return make_env(
        kwargs.pop("benchmark_id", "SwarmBots-MoveToGoal-v0"), num_envs=2, device=device,
        episode_length=kwargs.pop("episode_length", 3),
        compile_tensor_operations=kwargs.pop("compile_tensor_operations", False),
        scenario_kwargs={
            "unit_start_locations": MJWPreConnectedUnitLocationsConfig(
                num_units=1, pool_seeds=(42000, 42001), z_pos=0.5,
            ),
            "reset_settle_time": settle_time,
            "compile_reward_kernel": kwargs.pop("compile_reward_kernel", False),
        },
        **kwargs,
    )


def _actions(env: MJWSwarmBotsVectorEnv) -> dict[str, torch.Tensor]:
    return {key: torch.zeros(space.shape, device=env.device) for key, space in env.action_space.items()}


def _lane_observations(obs: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    return {key: value[index].clone() for key, value in obs.items()}


def _assert_error_observation(obs: dict[str, torch.Tensor], index: int, agent_mask: torch.Tensor) -> None:
    assert all(torch.isfinite(value[index]).all() for value in obs.values())
    for key in ("local_obs", "global_obs", "hidden_local_vars"):
        assert torch.count_nonzero(obs[key][index]) == 0
    assert torch.count_nonzero(obs["hidden_global_vars"][index, 1:]) == 0
    torch.testing.assert_close(obs["agent_mask"][index], agent_mask)
    assert obs["hidden_global_vars"][index, 0] == agent_mask.sum()


@pytest.mark.parametrize("mode", [AutoresetMode.DISABLED, "Disabled", "unknown"])
def test_unsupported_autoreset_modes_fail_before_physics_construction(mode: AutoresetMode | str) -> None:
    scenario = Mock()
    with pytest.raises(ValueError):
        MJWSwarmBotsVectorEnv(scenario, num_envs=2, device="cpu", autoreset_mode=mode)
    scenario.build_model.assert_not_called()


@pytest.mark.integration
def test_autoreset_metadata_and_default_are_independent_per_environment() -> None:
    with ExitStack() as cleanup:
        next_step = _make_env("cpu", autoreset_mode="NextStep", episode_length=1)
        cleanup.callback(next_step.close)
        same_step = _make_env("cpu", episode_length=1)
        cleanup.callback(same_step.close)
        assert next_step.autoreset_mode is AutoresetMode.NEXT_STEP
        assert next_step.metadata["autoreset_mode"] is AutoresetMode.NEXT_STEP
        assert next_step.get_settings()["autoreset_mode"] == "NextStep"
        assert same_step.metadata["autoreset_mode"] is AutoresetMode.SAME_STEP
        assert same_step.get_settings()["autoreset_mode"] == "SameStep"
        assert MJWSwarmBotsVectorEnv.metadata["autoreset_mode"] is AutoresetMode.SAME_STEP
        with pytest.raises(ValueError, match="requires VectorEnv autoreset_mode=SAME_STEP"):
            SwarmBotsLearnEnvWrapper(next_step)

        same_step.reset(seed=7)
        _, _, terminated, truncated, infos = same_step.step(_actions(same_step))
        assert not terminated.any()
        assert truncated.all()
        assert infos["_final_obs"].all()
        assert "final_obs" in infos
        assert same_step.current_step.tolist() == [0, 0]


@pytest.mark.integration
@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("settle_time", [0.0, 0.003])
def test_next_step_mixed_worlds_return_exact_initial_states_and_ignore_actions(
    device: str, deferred: bool, settle_time: float,
) -> None:
    with ExitStack() as cleanup:
        env = _make_env(
            device, autoreset_mode=AutoresetMode.NEXT_STEP, first_episode_lengths=[1, 3],
            settle_time=settle_time, settled_reset_buffer_size=4, settled_reset_batch_size=2,
        )
        cleanup.callback(env.close)
        reference = _make_env(
            device, autoreset_mode=AutoresetMode.NEXT_STEP, first_episode_lengths=[1, 3],
            settle_time=settle_time, settled_reset_buffer_size=4, settled_reset_batch_size=2,
        )
        cleanup.callback(reference.close)
        env.reset(seed=7)
        reference.reset(seed=7)
        obs, _, terminated, truncated, infos = env.step(_actions(env))
        reference.step(_actions(reference))
        assert not terminated.any()
        assert truncated.tolist() == [True, False]
        assert env.current_step.tolist() == [1, 1]
        assert "final_obs" not in infos and "_final_obs" not in infos
        terminal_qpos = env._qpos[0].clone()
        torch.testing.assert_close(_lane_observations(obs, 0), _lane_observations(env._build_obs(), 0))

        expected_obs, _ = reference.reset(options={"reset_mask": [True, False]})
        expected_reset = _lane_observations(expected_obs, 0)
        expected_qpos = reference._qpos[0].clone()
        expected_qvel = reference._qvel[0].clone()
        expected_time = reference._time[0].clone()
        assert torch.equal(terminal_qpos, env._qpos[0])

        actions = _actions(env)
        actions["actuators"][0] = float("nan")
        actions["connectors"][0] = 1
        if deferred:
            env.begin_step(actions)
            assert env.has_pending_step
            obs, rewards, terminated, truncated, infos = env.end_step()
        else:
            obs, rewards, terminated, truncated, infos = env.step(actions)
        reference_obs, reference_rewards, _, _, _ = reference.step(_actions(reference))

        assert torch.isnan(actions["actuators"][0]).all()
        assert env.current_step.tolist() == [0, 2]
        assert not terminated.any() and not truncated.any()
        assert rewards[0] == 0
        assert "final_obs" not in infos
        torch.testing.assert_close(_lane_observations(obs, 0), expected_reset, rtol=0, atol=0)
        torch.testing.assert_close(env._qpos[0], expected_qpos, rtol=0, atol=0)
        torch.testing.assert_close(env._qvel[0], expected_qvel, rtol=0, atol=0)
        torch.testing.assert_close(env._time[0], expected_time, rtol=0, atol=0)
        torch.testing.assert_close(_lane_observations(obs, 1), _lane_observations(reference_obs, 1), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(rewards[1], reference_rewards[1], rtol=1e-4, atol=1e-5)
        assert torch.equal(env._rng.get_state(), reference._rng.get_state())
        for value in infos.values():
            values = value.values() if isinstance(value, dict) else (value,)
            assert all(torch.count_nonzero(item[0]) == 0 for item in values)

        _, _, terminated, truncated, _ = env.step(_actions(env))
        assert not terminated.any()
        assert truncated.tolist() == [False, True]
        assert env.current_step.tolist() == [1, 3]


@pytest.mark.integration
def test_next_step_all_worlds_alternate_terminal_and_reset_calls(device: str) -> None:
    env = _make_env(device, autoreset_mode=AutoresetMode.NEXT_STEP, episode_length=1)
    try:
        env.reset(seed=7)
        for _ in range(3):
            _, _, terminated, truncated, infos = env.step(_actions(env))
            assert not terminated.any() and truncated.all()
            assert env.current_step.tolist() == [1, 1]
            assert "final_obs" not in infos
            obs, rewards, terminated, truncated, _ = env.step(_actions(env))
            assert torch.equal(rewards, torch.zeros_like(rewards))
            assert not terminated.any() and not truncated.any()
            assert env.current_step.tolist() == [0, 0]
            assert all(torch.isfinite(value).all() for value in obs.values())
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_next_step_with_compiled_tensor_and_reward_operations() -> None:
    env = _make_env(
        "cuda", autoreset_mode=AutoresetMode.NEXT_STEP, episode_length=2,
        first_episode_lengths=[1, 2], compile_tensor_operations=True, compile_reward_kernel=True,
    )
    try:
        env.reset(seed=7)
        for expected_steps, expected_dones in (
            ([1, 1], [True, False]), ([0, 2], [False, True]), ([1, 0], [False, False]),
            ([2, 1], [True, False]), ([0, 2], [False, True]),
        ):
            obs, rewards, terminated, truncated, _ = env.step(_actions(env))
            assert not terminated.any()
            assert truncated.tolist() == expected_dones
            assert env.current_step.tolist() == expected_steps
            assert torch.isfinite(rewards).all()
            assert all(torch.isfinite(value).all() for value in obs.values())
            for index, step in enumerate(expected_steps):
                if step == 0:
                    assert rewards[index] == 0
    finally:
        env.close()


@pytest.mark.integration
def test_next_step_explicit_partial_and_empty_resets_preserve_other_pending_worlds(device: str) -> None:
    env = _make_env(device, autoreset_mode=AutoresetMode.NEXT_STEP, first_episode_lengths=[1, 1])
    try:
        env.reset(seed=7)
        env.step(_actions(env))
        env.reset(options={"reset_mask": [False, False]})
        env.reset(options={"reset_mask": [True, False]})
        _, _, terminated, truncated, _ = env.step(_actions(env))
        assert not terminated.any() and not truncated.any()
        assert env.current_step.tolist() == [1, 0]

        env.reset()
        env.step(_actions(env))
        assert env.current_step.tolist() == [1, 1]
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize("unstable", [False, True])
def test_next_step_resets_scenario_terminations_and_unstable_worlds(
    device: str, unstable: bool,
) -> None:
    env = _make_env(device, autoreset_mode=AutoresetMode.NEXT_STEP, episode_length=4)
    try:
        env.reset(seed=7)
        if unstable:
            run_physics = env._run_physics
            first_step = True

            def run_with_one_failure() -> None:
                nonlocal first_step
                run_physics()
                if first_step:
                    env._qvel[0, 0] = float("nan")
                    first_step = False

            injected_step = patch.object(env, "_run_physics", run_with_one_failure)
        else:
            compute_rewards = env._scenario_runtime.compute_step_rewards

            def compute_with_termination(*, stable_mask: torch.Tensor) -> MJWStepResult:
                return replace(
                    compute_rewards(stable_mask=stable_mask),
                    terminations=torch.tensor([True, False], device=env.device),
                )

            injected_step = patch.object(env._scenario_runtime, "compute_step_rewards", compute_with_termination)

        with injected_step:
            obs, _, terminated, truncated, infos = env.step(_actions(env))
            assert terminated.tolist() == [True, False] and not truncated.any()
            assert all(torch.isfinite(value).all() for value in obs.values())
            assert "final_obs" not in infos
            obs, rewards, terminated, truncated, _ = env.step(_actions(env))
            assert not terminated.any() and not truncated.any()
            assert rewards[0] == 0
            assert env.current_step.tolist() == [0, 2]
            assert all(torch.isfinite(value).all() for value in obs.values())
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize("deferred, compile_operations", [(False, False), (True, False), (True, True)])
def test_next_step_preserves_error_observation_when_another_world_autoresets(
    device: str, deferred: bool, compile_operations: bool,
) -> None:
    if compile_operations and device == "cpu":
        pytest.skip("Compiled environment operations target CUDA")
    env = _make_env(
        device, autoreset_mode=AutoresetMode.NEXT_STEP, first_episode_lengths=[1, 3],
        compile_tensor_operations=compile_operations, compile_reward_kernel=compile_operations,
    )
    try:
        obs, _ = env.reset(seed=7)
        agent_mask = obs["agent_mask"][1].clone()
        env.step(_actions(env))
        run_physics = env._run_physics

        def run_with_failure() -> None:
            run_physics()
            env._qvel[1, 0] = float("nan")

        with patch.object(env, "_run_physics", run_with_failure):
            if deferred:
                env.begin_step(_actions(env))
                obs, rewards, terminated, truncated, _ = env.end_step()
            else:
                obs, rewards, terminated, truncated, _ = env.step(_actions(env))

        assert terminated.tolist() == [False, True] and not truncated.any()
        assert env.current_step.tolist() == [0, 1]
        assert rewards[0] == 0
        assert all(torch.isfinite(value).all() for value in obs.values())
        _assert_error_observation(obs, 1, agent_mask)

        obs, rewards, terminated, truncated, _ = env.step(_actions(env))
        assert env.current_step.tolist() == [1, 0]
        assert not terminated.any() and not truncated.any()
        assert rewards[1] == 0
        assert all(torch.isfinite(value).all() for value in obs.values())
        assert torch.count_nonzero(obs["local_obs"][1]) > 0
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize("reset_mask, compile_operations", [
    ([False, False], False), ([False, True], False), ([False, True], True),
])
def test_next_step_explicit_reset_preserves_error_observation_until_failed_world_resets(
    device: str, reset_mask: list[bool], compile_operations: bool,
) -> None:
    if compile_operations and device == "cpu":
        pytest.skip("Compiled environment operations target CUDA")
    env = _make_env(
        device, autoreset_mode=AutoresetMode.NEXT_STEP, episode_length=4,
        compile_tensor_operations=compile_operations, compile_reward_kernel=compile_operations,
    )
    try:
        obs, _ = env.reset(seed=7)
        agent_mask = obs["agent_mask"][0].clone()
        run_physics = env._run_physics

        def run_with_failure() -> None:
            run_physics()
            env._qvel[0, 0] = float("nan")

        with patch.object(env, "_run_physics", run_with_failure):
            obs, _, terminated, _, _ = env.step(_actions(env))
        assert terminated.tolist() == [True, False]
        _assert_error_observation(obs, 0, agent_mask)

        for _ in range(2):
            obs, _ = env.reset(options={"reset_mask": reset_mask})
            _assert_error_observation(obs, 0, agent_mask)
            assert torch.count_nonzero(obs["local_obs"][1]) > 0

        obs, _ = env.reset(options={"reset_mask": [True, False]})
        assert all(torch.isfinite(value).all() for value in obs.values())
        assert torch.count_nonzero(obs["local_obs"][0]) > 0
        obs, _, terminated, truncated, _ = env.step(_actions(env))
        assert not terminated.any() and not truncated.any()
        assert env.current_step[0] == 1
        assert torch.count_nonzero(obs["local_obs"][0]) > 0
    finally:
        env.close()


@pytest.mark.integration
def test_next_step_selective_and_full_resets_clear_error_observations_for_failed_worlds(
    device: str,
) -> None:
    env = _make_env(device, autoreset_mode=AutoresetMode.NEXT_STEP, episode_length=4)
    try:
        obs, _ = env.reset(seed=7)
        agent_mask = obs["agent_mask"].clone()
        run_physics = env._run_physics

        def run_with_failures() -> None:
            run_physics()
            env._qvel[:, 0] = float("nan")

        with patch.object(env, "_run_physics", run_with_failures):
            obs, _, terminated, _, _ = env.step(_actions(env))
        assert terminated.all()
        for index in range(env.num_envs):
            _assert_error_observation(obs, index, agent_mask[index])

        obs, _ = env.reset(options={"reset_mask": [True, False]})
        assert torch.count_nonzero(obs["local_obs"][0]) > 0
        _assert_error_observation(obs, 1, agent_mask[1])
        obs, _ = env.reset()
        assert all(torch.isfinite(value).all() for value in obs.values())
        assert all(torch.count_nonzero(value) > 0 for value in obs["local_obs"])
        obs, _, terminated, truncated, _ = env.step(_actions(env))
        assert not terminated.any() and not truncated.any()
        assert env.current_step.tolist() == [1, 1]
        assert all(torch.count_nonzero(value) > 0 for value in obs["local_obs"])
    finally:
        env.close()


@pytest.mark.integration
def test_next_step_recording_starts_new_episode_after_reset_without_extra_step() -> None:
    env = _make_env("cpu", autoreset_mode=AutoresetMode.NEXT_STEP, first_episode_lengths=[1, 3])
    try:
        env.reset(seed=7)
        env.step(_actions(env))
        recorder = Mock()
        recorder.is_active.return_value = True
        recorder.active_world_indices.return_value = np.array([1])
        env._live_episode_recorder = recorder
        env.step(_actions(env))

        kwargs = recorder.on_episode_starts.call_args.kwargs
        np.testing.assert_array_equal(kwargs["world_idx"], [0])
        assert kwargs["snapshots_by_world"][0].time == 0.0
        assert set(recorder.record_step.call_args.kwargs["snapshots_by_world"]) == {1}
        assert recorder.record_step.call_args.kwargs["rewards"][0] == 0
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize("benchmark_id", ALL_BENCHMARK_IDS)
def test_next_step_reset_restores_scenario_observations_and_reward_baselines(benchmark_id: str) -> None:
    with ExitStack() as cleanup:
        env = _make_env(
            "cpu", benchmark_id=benchmark_id, autoreset_mode=AutoresetMode.NEXT_STEP, episode_length=1,
        )
        cleanup.callback(env.close)
        reference = _make_env(
            "cpu", benchmark_id=benchmark_id, autoreset_mode=AutoresetMode.NEXT_STEP, episode_length=1,
        )
        cleanup.callback(reference.close)
        for current in (env, reference):
            current.reset(seed=7)
            current.step(_actions(current))

        expected, _ = reference.reset()
        expected = {key: value.clone() for key, value in expected.items()}
        obs, rewards, terminated, truncated, _ = env.step(_actions(env))
        torch.testing.assert_close(obs, expected, rtol=0, atol=0)
        assert not rewards.any() and not terminated.any() and not truncated.any()
        actual_step = env.step(_actions(env))
        expected_step = reference.step(_actions(reference))
        torch.testing.assert_close(actual_step, expected_step, rtol=1e-5, atol=1e-6)
