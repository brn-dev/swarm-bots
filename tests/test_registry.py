from __future__ import annotations

import pytest

from swarmbots import ALL_BENCHMARK_IDS, CORE_BENCHMARK_IDS, get_benchmark_spec, list_benchmarks, make_scenario
from swarmbots.mjw_env.parameter_distributions import UniformDistParams


def test_registry_ids_are_unique_and_versioned() -> None:
    specs = list_benchmarks()

    assert tuple(spec.id for spec in specs) == ALL_BENCHMARK_IDS
    assert len(set(ALL_BENCHMARK_IDS)) == len(ALL_BENCHMARK_IDS)
    assert all(benchmark_id.endswith("-v0") for benchmark_id in ALL_BENCHMARK_IDS)
    assert set(CORE_BENCHMARK_IDS) < set(ALL_BENCHMARK_IDS)


@pytest.mark.parametrize("benchmark_id", ALL_BENCHMARK_IDS)
def test_registered_scenarios_expose_multi_agent_spaces(benchmark_id: str) -> None:
    scenario = make_scenario(
        benchmark_id,
        seed=7,
        compile_reward_kernel=False,
        reset_settle_time=0.0,
    )

    observation_space = scenario.get_single_observation_space()
    action_space = scenario.get_single_action_space()

    assert observation_space["local_obs"].shape[0] == scenario.swarm.num_units
    assert observation_space["agent_mask"].shape == (scenario.swarm.num_units,)
    assert action_space["actuators"].shape[0] == scenario.swarm.num_units
    assert action_space["connectors"].shape[0] == scenario.swarm.num_units


def test_partial_observable_wall_randomizes_a_privileged_wall_position() -> None:
    scenario = make_scenario(
        "SwarmBots-POWallMedium-v0",
        compile_reward_kernel=False,
        reset_settle_time=0.0,
    )

    assert isinstance(scenario.wall_distance, UniformDistParams)
    assert scenario.get_single_observation_space()["global_obs"].shape == (0,)
    assert scenario.get_single_observation_space()["hidden_global_vars"].shape == (1,)


def test_unknown_benchmark_error_lists_available_ids() -> None:
    with pytest.raises(KeyError, match="SwarmBots-WallEasy-v0"):
        get_benchmark_spec("missing")
