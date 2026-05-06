from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

COMMON_SCENARIO_KWARGS: dict[str, object] = {
    "timestep": 0.003,
    "action_repeat": 10,
    "friction": [1.25, 7e-3, 1.25e-4],
    "actuator_strength": 15.0,
    "progress_reward_weight": 1.0,
    "guidance_reward_weight": 1.0,
    "units_without_connections_reward_weight": -1e-5,
    "reset_settle_time": 1.0,
    "reset_settle_timestep_scale": 3,
    "randomize_initial_swarm_z_rotation": False,
}

WALL_PASS_REWARD_KWARGS: dict[str, object] = {
    "forward_reward_weight": 1.0,
    "forward_reward_max_y": 1.5,
    "wall_pass_reward_weight": 10.0,
    "wall_pass_thresholds": [-0.1, 0.1, 0.3, 0.5],
}

BRIDGE_REWARD_KWARGS: dict[str, object] = {
    "fell_off_bridge_reward": -2.0,
}

PAYLOAD_PLANE_REWARD_KWARGS: dict[str, object] = {
    "forward_reward_weight": 1.0,
    "payload_centering_penalty_weight": 0.05,
    "payload_centering_penalty_power": 1.0,
    "payload_centering_tolerance": 0.25,
    "payload_shape": "box",
    "payload_radius": 0.2,
    "payload_mass": 1.0,
    "payload_offset_x": 0.0,
    "payload_offset_y": 0.75,
    "forward_reward_max_y": None,
}

MOVE_TO_REWARD_KWARGS: dict[str, object] = {
    "forward_reward_weight": 1.0,
    "goal_radius": 0.25,
}


def make_scenario_kwargs(*parts: Mapping[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for part in parts:
        merged.update(deepcopy(dict(part)))
    return merged
