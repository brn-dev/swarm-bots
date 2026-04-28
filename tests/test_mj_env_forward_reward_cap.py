from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from swarmbots.mj_env.scenarios.obstacle_street_scenario import ObstacleStreetScenario


def test_forward_reward_cap_is_applied_per_unit_in_mj_env() -> None:
    scenario = object.__new__(ObstacleStreetScenario)
    scenario.payload_type = None
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
