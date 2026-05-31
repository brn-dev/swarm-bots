from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from swarmbots.mj_env.scenarios.obstacle_street_scenario import ObstacleStreetScenario


def test_forward_reward_cap_is_applied_per_unit_in_mj_env() -> None:
    scenario = object.__new__(ObstacleStreetScenario)
    scenario.forward_reward_weight = 1.0
    scenario.forward_reward_max_y = 1.0
    scenario._qpos_indices = np.array([[0, 0], [1, 1]], dtype=int)
    scenario._compute_wall_pass_reward = lambda data, state: 0.0

    state = {
        "progress": 0.9,
        "units_active_mask": np.array([True, True], dtype=bool),
    }
    data = SimpleNamespace(qpos=np.array([1.6, 0.9], dtype=float))

    reward = scenario.compute_progress_reward(data, state)

    assert np.isclose(reward, 0.05)
    assert np.isclose(state["forward_reward"], 0.05)
    assert np.isclose(state["progress"], 0.95)


def test_wall_height_forward_reward_boost_applies_per_high_unit_in_mj_env() -> None:
    scenario = object.__new__(ObstacleStreetScenario)
    scenario.forward_reward_weight = 1.0
    scenario.forward_reward_max_y = None
    scenario.forward_reward_wall_boost_factor = 3.0
    scenario.forward_reward_wall_boost_distance = 0.5
    scenario.forward_reward_wall_boost_height_margin = 0.1
    scenario.wall_heights = [0.4]
    scenario.swarm = SimpleNamespace(body_radius=0.1)
    scenario._qpos_indices = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
    scenario._compute_wall_pass_reward = lambda data, state: 0.0
    scenario._compute_wall_climb_reward = lambda data, state: 0.0

    state = {
        "progress": 1.2,
        "forward_progress_unit_y": np.array([1.8, 0.6], dtype=float),
        "wall_y": np.array([1.0], dtype=float),
        "units_active_mask": np.array([True, True], dtype=bool),
    }
    data = SimpleNamespace(qpos=np.array([0.0, 0.7, 0.5, 0.0, 0.7, 0.49], dtype=float))

    reward = scenario.compute_progress_reward(data, state)

    assert np.isclose(reward, 0.2)
    assert np.isclose(state["forward_reward"], 0.2)


def test_wall_height_forward_reward_boost_penalizes_backtracking_symmetrically_in_mj_env() -> None:
    scenario = object.__new__(ObstacleStreetScenario)
    scenario.forward_reward_weight = 1.0
    scenario.forward_reward_max_y = None
    scenario.forward_reward_wall_boost_factor = 3.0
    scenario.forward_reward_wall_boost_distance = 0.5
    scenario.forward_reward_wall_boost_height_margin = 0.1
    scenario.wall_heights = [0.4]
    scenario.swarm = SimpleNamespace(body_radius=0.1)
    scenario.potential_reward_discount_factor = 1.0
    scenario._qpos_indices = np.array([[0, 1, 2]], dtype=int)
    scenario._compute_wall_pass_reward = lambda data, state: 0.0
    scenario._compute_wall_climb_reward = lambda data, state: 0.0

    state = {
        "progress": 2.1,
        "forward_progress_unit_y": np.array([2.1], dtype=float),
        "wall_y": np.array([1.0], dtype=float),
        "units_active_mask": np.array([True], dtype=bool),
    }
    backtrack = SimpleNamespace(qpos=np.array([0.0, 0.6, 0.5], dtype=float))
    retry = SimpleNamespace(qpos=np.array([0.0, 0.7, 0.5], dtype=float))

    backtrack_reward = scenario.compute_progress_reward(backtrack, state)
    retry_reward = scenario.compute_progress_reward(retry, state)

    assert np.isclose(backtrack_reward, -0.3)
    assert np.isclose(retry_reward, 0.3)
    assert np.isclose(backtrack_reward + retry_reward, 0.0)
