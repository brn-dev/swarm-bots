from __future__ import annotations

import numpy as np

from swarmbots.mj_env.float_or_dist_params import UniformDistParams
from swarmbots.mj_env.scenarios.scenario_presets import default_move_to as default_mj_move_to
from swarmbots.mjw_env.scenarios.mjw_scenario_presets import default_move_to as default_mjw_move_to
from swarmbots.scenario_presets.move_to_goal_config import AbsoluteGoalConfig, RelativePolarGoalConfig


def test_move_to_presets_sample_goal_positions_by_default() -> None:
    mj_scenario = default_mj_move_to(seed=123)
    mjw_scenario = default_mjw_move_to(seed=123)

    assert mj_scenario.goal == RelativePolarGoalConfig(
        distance=UniformDistParams(2.0, 4.0),
        angle=UniformDistParams(0.0, 2.0 * np.pi),
    )
    assert mjw_scenario.goal == mj_scenario.goal

    swarm_start = np.array([0.5, -0.25, 1.0], dtype=float)
    first_goal = mj_scenario._sample_goal_position(swarm_start_location=swarm_start)
    second_goal = mj_scenario._sample_goal_position(swarm_start_location=swarm_start)
    first_distance = np.linalg.norm(first_goal - swarm_start[:2])
    second_distance = np.linalg.norm(second_goal - swarm_start[:2])

    assert not np.allclose(first_goal, second_goal)
    assert 2.0 <= first_distance <= 4.0
    assert 2.0 <= second_distance <= 4.0


def test_move_to_absolute_goal_config_still_pins_goal() -> None:
    scenario = default_mj_move_to(goal=AbsoluteGoalConfig(x=1.25, y=-0.5))
    swarm_start = np.array([0.5, -0.25, 1.0], dtype=float)

    assert np.allclose(
        scenario._sample_goal_position(swarm_start_location=swarm_start),
        np.array([1.25, -0.5], dtype=float),
    )
