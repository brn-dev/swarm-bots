from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from swarmbots import ALL_BENCHMARK_IDS, make_env
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.multi_scenario_vector_env import MultiScenarioVectorEnv
from swarmbots.mjw_env.mjw_swarm_bots_vector_env import MJWSwarmBotsVectorEnv
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig
from tests.test_base_algorithm_lifecycle import _Algorithm


pytestmark = pytest.mark.integration


def _make_env(benchmark_id: str, *, device: str = "cpu", compiled: bool = False) -> MJWSwarmBotsVectorEnv:
    return make_env(
        benchmark_id, num_envs=1, device=device, seed=7, compile_tensor_operations=False,
        scenario_kwargs={
            "unit_start_locations": MJWPreConnectedUnitLocationsConfig(num_units=1, pool_seeds=(42000,), z_pos=0.5),
            "reset_settle_time": 0.0,
            "compile_reward_kernel": compiled,
        },
    )


@pytest.fixture
def env() -> Iterator[MJWSwarmBotsVectorEnv]:
    instance = _make_env("SwarmBots-MoveToGoal-v0")
    try:
        yield instance
    finally:
        instance.close()


def _make_algorithm(env: MJWSwarmBotsVectorEnv | MultiScenarioVectorEnv) -> _Algorithm:
    algorithm = _Algorithm()
    algorithm.env = SwarmBotsLearnEnvWrapper(env)
    return algorithm


@pytest.mark.parametrize("benchmark_id", ALL_BENCHMARK_IDS)
def test_live_weights_are_exposed_and_updated_for_every_benchmark(benchmark_id: str) -> None:
    env = _make_env(benchmark_id)
    try:
        original_weights = env.get_reward_weights()
        assert original_weights == {"progress_reward_weight": 1.0}

        updated_weights = env.update_reward_weights({"progress_reward_weight": 2.0})

        assert updated_weights == {"progress_reward_weight": 2.0}
        assert env.scenario.progress_reward_weight == 2.0
        assert env.get_settings()["scenario"]["reward_weights"] == updated_weights
        assert original_weights == {"progress_reward_weight": 1.0}
    finally:
        env.close()


@pytest.mark.parametrize("command", ("show_reward_weights", "show_rw"))
def test_show_reward_weights_reports_the_actual_environment_weights(
    env: MJWSwarmBotsVectorEnv, command: str,
) -> None:
    algorithm = _make_algorithm(env)
    with patch("swarmbots.learn.algos.base_algorithm.logger.info") as log:
        assert not algorithm.execute_command(command, "", None)
    log.assert_called_once_with({"reward_weights": {"progress_reward_weight": 1.0}})


@pytest.mark.parametrize("command", ("set_reward_weights", "set_rw"))
@pytest.mark.parametrize("nested", (False, True))
def test_reward_weight_commands_update_the_environment_and_metadata(
    env: MJWSwarmBotsVectorEnv, command: str, nested: bool,
) -> None:
    algorithm = _make_algorithm(env)
    metadata = {}
    params = '{"progress_reward_weight": 2.0}'
    if nested:
        params = '{"reward_weights": ' + params + '}'

    assert algorithm.execute_command(command, params, metadata)

    assert env.get_reward_weights() == {"progress_reward_weight": 2.0}
    assert metadata == {"updated_reward_weights": {"progress_reward_weight": 2.0}}


@pytest.mark.parametrize("params", (
    '{"progress_reward_weight": 2.0, "forward_reward_weight": 3.0}',
    '{"timestep": 0.01}',
    '{"progress_reward_weight": "nan"}',
    '{"progress_reward_weight": "inf"}',
    '{"progress_reward_weight": "invalid"}',
    '{"reward_weights": []}',
    '[]',
))
def test_invalid_reward_commands_preserve_weights_and_metadata(env: MJWSwarmBotsVectorEnv, params: str) -> None:
    algorithm = _make_algorithm(env)
    metadata = {}

    assert not algorithm.execute_command("set_rw", params, metadata)

    assert env.get_reward_weights() == {"progress_reward_weight": 1.0}
    assert metadata == {}


@pytest.mark.parametrize("weight", (0.0, -2.0, 3.0))
@pytest.mark.parametrize("device", ("cpu", pytest.param("cuda", marks=pytest.mark.cuda)))
def test_live_weights_scale_next_step_rewards_without_changing_physics(device: str, weight: float) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    env = _make_env("SwarmBots-MoveToGoal-v0", device=device, compiled=device == "cuda")
    try:
        actions = {key: torch.zeros(space.shape, device=env.device) for key, space in env.action_space.items()}

        def step_from_same_state() -> tuple[torch.Tensor, torch.Tensor]:
            env.reset(seed=7)
            direction = env._scenario_runtime.goal_position[0] - env._qpos[0, :2]
            env._qvel[0, :2] = direction / direction.norm()
            _, rewards, terminated, _, _ = env.step(actions)
            assert not terminated.any()
            return rewards.clone(), env._qpos.clone()

        original_rewards, original_qpos = step_from_same_state()
        assert original_rewards.abs().item() > 1e-4
        env.update_reward_weights({"progress_reward_weight": weight})
        updated_rewards, updated_qpos = step_from_same_state()

        torch.testing.assert_close(updated_rewards, weight * original_rewards)
        torch.testing.assert_close(updated_qpos, original_qpos)
    finally:
        env.close()


def test_multi_scenario_commands_report_each_scenario_and_update_all_children() -> None:
    with ExitStack() as cleanup:
        children = {name: _make_env(name) for name in ("SwarmBots-MoveToGoal-v0", "SwarmBots-PayloadPlane-v0")}
        for child in children.values():
            cleanup.callback(child.close)
        env = MultiScenarioVectorEnv(children)
        algorithm = _make_algorithm(env)
        metadata = {}

        assert algorithm.execute_command("set_rw", '{"progress_reward_weight": 2.0}', metadata)

        expected = {name: {"progress_reward_weight": 2.0} for name in children}
        assert env.get_reward_weights() == expected
        assert metadata == {"updated_reward_weights": expected}
        with patch("swarmbots.learn.algos.base_algorithm.logger.info") as log:
            assert not algorithm.execute_command("show_rw", "", None)
        log.assert_called_once_with({"reward_weights": expected})


def test_multi_scenario_update_rejects_pending_steps_before_changing_any_child() -> None:
    with ExitStack() as cleanup:
        children = {name: _make_env(name) for name in ("SwarmBots-MoveToGoal-v0", "SwarmBots-PayloadPlane-v0")}
        for child in children.values():
            cleanup.callback(child.close)
        env = MultiScenarioVectorEnv(children)
        env.reset(seed=7)
        second_child = env.envs[1]
        actions = {key: torch.zeros(space.shape) for key, space in second_child.action_space.items()}
        second_child.begin_step(actions)
        try:
            with pytest.raises(RuntimeError, match="step is pending"):
                env.update_reward_weights({"progress_reward_weight": 2.0})
            assert all(child.get_reward_weights() == {"progress_reward_weight": 1.0} for child in env.envs)
        finally:
            second_child.end_step()
