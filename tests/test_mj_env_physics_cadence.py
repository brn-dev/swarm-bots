import pytest
import numpy as np

from swarmbots.mj_env.scenarios.scenario_presets import default_wall
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv


def test_wall_scenario_owns_timestep_and_action_repeat() -> None:
    scenario = default_wall(
        seed=123,
        unit_start_locations=[
            (0.0, 0.0, 0.5),
            (0.4, 0.0, 0.5),
        ],
        timestep=0.004,
        action_repeat=7,
    )

    assert scenario.timestep == 0.004
    assert scenario.action_repeat == 7
    assert np.isclose(float(scenario.dummy_model.opt.timestep), 0.004)
    assert scenario.get_settings()["timestep"] == 0.004
    assert scenario.get_settings()["action_repeat"] == 7


def test_swarm_bots_env_uses_scenario_action_repeat_by_default() -> None:
    scenario = default_wall(
        seed=123,
        unit_start_locations=[
            (0.0, 0.0, 0.5),
            (0.4, 0.0, 0.5),
        ],
        action_repeat=9,
    )

    env = SwarmBotsEnv(scenario=scenario)

    assert env.action_repeat == 9


def test_swarm_bots_env_no_longer_accepts_action_repeat_override() -> None:
    scenario = default_wall(
        seed=123,
        unit_start_locations=[
            (0.0, 0.0, 0.5),
            (0.4, 0.0, 0.5),
        ],
        action_repeat=9,
    )

    with pytest.raises(TypeError):
        SwarmBotsEnv(scenario=scenario, action_repeat=3)
