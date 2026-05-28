from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Literal

from swarmbots.mj_env.float_or_dist_params import UniformDistParams
from swarmbots.scenario_presets.move_to_goal_config import RelativePolarGoalConfig

Difficulty = Literal['easy', 'medium', 'hard']

COMMON_SCENARIO_KWARGS: dict[str, object] = {
    "timestep": 0.003,
    "action_repeat": 10,
    "friction": [1.25, 7e-3, 1.25e-4],
    "actuator_strength": 15.0,
    "connection_dist_threshold": 0.1,
    "connection_angle_threshold": -0.5,
    "disconnect_potential_threshold": 5.0,
    "progress_reward_weight": 1.0,
    "guidance_reward_weight": 1.0,
    "units_without_connections_reward_weight": -1e-5,
    "include_connectors_xpos_in_obs": True,
    "include_connectors_xquat_in_obs": False,
    "quat_rot6d_representation": True,
    "reset_settle_time": 1.0,
    "reset_settle_timestep_scale": 3,
    "swarm_start_x": 0.0,
    "swarm_start_y": 0.0,
    "randomize_initial_swarm_z_rotation": False,
}

WALL_SCENARIO_KWARGS: dict[str, object] = {
    "num_walls": 1,
    "opening_width": 0.01,
    "swarm_start_x": 0.0,
    "swarm_start_y": UniformDistParams(0.25, 0.75),
    "first_wall_distance": 1.0,
    "inter_wall_distance": 4.0,
    "unusable_opening_offset": 2.0,
    "street_width": 10.0,
    "no_initial_ramp": True,
    "wall_height": 0.20,
    "forward_reward_weight": 1.0,
    "forward_reward_max_y": 1.5,
    "wall_pass_reward_weight": 10.0,
    "wall_pass_reward_skew": 0.0,
    "wall_pass_thresholds": [-0.1, 0.1, 0.3, 0.5],
}

WALL_DIFFICULTY_UPDATES: dict[Difficulty, dict[str, object]] = {
    "easy": {
        "wall_height": 0.2,
    },
    "medium": {
        "wall_height": 0.4,
    },
    "hard": {
        "wall_height": 0.8,
    },
}


def wall_scenario_difficulty_update(difficulty: Difficulty | None) -> dict[str, object]:
    if difficulty is None:
        return {}
    if difficulty in WALL_DIFFICULTY_UPDATES:
        return WALL_DIFFICULTY_UPDATES[difficulty].copy()
    raise ValueError(f'Unknown difficulty "{difficulty}"')


def wall_scenario_kwargs_with_difficulty(difficulty: Difficulty | None) -> dict[str, object]:
    kwargs = WALL_SCENARIO_KWARGS.copy()
    kwargs.update(wall_scenario_difficulty_update(difficulty))
    return kwargs


EASY_WALL_SCENARIO_KWARGS = wall_scenario_kwargs_with_difficulty("easy")
MEDIUM_WALL_SCENARIO_KWARGS = wall_scenario_kwargs_with_difficulty("medium")
HARD_WALL_SCENARIO_KWARGS = wall_scenario_kwargs_with_difficulty("hard")


BRIDGE_SCENARIO_KWARGS: dict[str, object] = {
    "street_width": 6.0,
    "bridge_width": 1.0,
    "bridge_length": 4.0,
    "bridge_x": 0.0,
    "platform_length": 4.0,
    "platform_height": 0.2,
    "fall_z_threshold": -1.0,
    "swarm_start_x": 0.0,
    "swarm_start_y": 0.0,
    "fell_off_bridge_reward": -2.0,
}

CLIMB_SCENARIO_KWARGS: dict[str, object] = {
    "plane_size": 100.0,
    "swarm_start_x": 0.0,
    "swarm_start_y": 0.0,
    "cuboid_size_x": 3.0,
    "cuboid_size_y": 3.0,
    "cuboid_size_z": 0.3,
    "cuboid_center_x": 0.0,
    "cuboid_center_y": 3.5,
    "horizontal_goal_radius": 0.3,
    "height_goal_radius": 0.1,
    "goal_height_offset": None,
    "horizontal_reward_weight": 10.0,
    "height_reward_weight": 10.0,
    "visualize_goal": True,
}

PAYLOAD_PLANE_SCENARIO_KWARGS: dict[str, object] = {
    "plane_size": 100.0,
    "swarm_start_x": 0.0,
    "swarm_start_y": 0.0,
    "forward_reward_weight": 10.0,
    "towards_payload_reward_weight": 1.0,
    "towards_payload_goal_radius": 0.5,
    "payload_centering_penalty_weight": 0.05,
    "payload_centering_penalty_power": 1.0,
    "payload_centering_tolerance": 0.5,
    "payload_shape": "box",
    "payload_radius": 0.3,
    "payload_mass": 3.0,
    "payload_offset_x": 0.0,
    "payload_offset_y": 1.0,
    "forward_reward_max_y": None,
}

DUAL_PAYLOAD_PLANE_SCENARIO_KWARGS: dict[str, object] = {
    **PAYLOAD_PLANE_SCENARIO_KWARGS,
    "payload_offset_x": (-0.35, 0.35),
    "payload_offset_y": (1.0, 1.0),
    "lagging_payload_weight": 0.75,
}

MOVE_TO_SCENARIO_KWARGS: dict[str, object] = {
    "plane_size": 100.0,
    "swarm_start_x": 0.0,
    "swarm_start_y": 0.0,
    "forward_reward_weight": 10.0,
    "goal_radius": 1.0,
    "goal": RelativePolarGoalConfig(
        distance=UniformDistParams(2.0, 4.0),
        angle=UniformDistParams(0.0, 2.0 * math.pi),
    ),
}


def make_scenario_kwargs(*parts: Mapping[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for part in parts:
        merged.update(deepcopy(dict(part)))
    return merged
