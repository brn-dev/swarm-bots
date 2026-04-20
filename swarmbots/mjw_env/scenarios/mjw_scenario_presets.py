from __future__ import annotations

import importlib.util
import shutil
import sys

import numpy as np
import torch

from swarmbots.mj_env.float_or_dist_params import UniformDistParams
from swarmbots.mj_env.swarm.unit_config import (
    UNIT_CONFIG_TETRAHEDRON_XY,
    UNIT_CONFIG_TETRAHEDRON_XYZ,
    UNIT_CONFIG_TETRAHEDRON_ZX,
)
from swarmbots.mjw_env.scenarios.mjw_obstacle_street_scenario import MJWObstacleStreetScenario
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWHomogeneousSwarm, MJWPreConnectedUnitLocationsConfig

DEFAULT_KWARGS = {
    "timestep": 0.003,
    "action_repeat": 10,
    "friction": [1.25, 7e-3, 1.25e-4],
    "actuator_strength": 15.0,
    "progress_reward_weight": 1.0,
    "guidance_reward_weight": 1.0,
    "units_without_connections_reward_weight": -1e-5,
    "reset_settle_time": 1.0,
    "reset_settle_timestep_scale": 3,
}
WALL_PASS_KWARGS = {
    "wall_pass_reward_weight": 5.0,
    "wall_pass_thresholds": [-0.1, 0.1, 0.3, 0.5],
}


def should_compile_reward_kernel_by_default() -> bool:
    if not hasattr(torch, "compile"):
        return False
    if importlib.util.find_spec("triton") is None:
        return False
    if sys.platform == "win32" and shutil.which("cl") is None:
        return False
    return True


def _resolve_swarm(
    *,
    swarm: MJWHomogeneousSwarm | None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None,
    quantize_connection_twist: int,
    joints: str,
) -> MJWHomogeneousSwarm:
    if swarm is not None:
        return swarm
    if unit_start_locations is None:
        unit_start_locations = MJWPreConnectedUnitLocationsConfig(
            num_units=6,
            num_unit_probs={2: 0.5, 3: 0.5, 4: 1.0, 5: 1.0, 6: 1.0},
            max_radius=1.5,
            unconnected_prob=0.03,
            z_pos=0.5,
        )

    joint_configs = {
        "zx": {
            "unit_config": UNIT_CONFIG_TETRAHEDRON_ZX,
            "hinge_range": (None, np.pi / 3),
            "hinge_armature": (0.025, 0.015),
            "hinge_damping": (0.3, 0.15),
            "hinge_frictionloss": (0.3, 0.15),
        },
        "xy": {
            "unit_config": UNIT_CONFIG_TETRAHEDRON_XY,
            "hinge_range": (np.pi / 3, np.pi / 3),
            "hinge_armature": (0.015, 0.015),
            "hinge_damping": (0.15, 0.15),
            "hinge_frictionloss": (0.15, 0.15),
        },
        "xyz": {
            "unit_config": UNIT_CONFIG_TETRAHEDRON_XYZ,
            "hinge_range": (np.pi / 3, np.pi / 3, None),
            "hinge_armature": (0.01, 0.01, 0.025),
            "hinge_damping": (0.1, 0.1, 0.5),
            "hinge_frictionloss": (0.1, 0.1, 0.5),
        },
    }

    return MJWHomogeneousSwarm(
        unit_start_locations=unit_start_locations,
        body_radius=0.1,
        leg_length=0.2,
        leg_radius=0.025,
        connection_torquescale=50.0,
        quantize_connection_twist=quantize_connection_twist,
        **joint_configs[joints],
    )


def default_wall(
    *,
    seed: int | None = None,
    swarm: MJWHomogeneousSwarm | None = None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None = None,
    quantize_connection_twist: int = 16,
    joints: str = "zx",
    **kwargs: object,
) -> MJWObstacleStreetScenario:
    scenario_kwargs = dict(DEFAULT_KWARGS)
    scenario_kwargs.update(WALL_PASS_KWARGS)
    scenario_kwargs.update(
        {
            "num_walls": 1,
            "opening_width": 0.01,
            "connection_dist_threshold": 0.1,
            "connection_angle_threshold": -0.5,
            "disconnect_potential_threshold": 5.0,
            "include_connectors_xpos_in_obs": True,
            "include_connectors_xquat_in_obs": False,
            "quat_rot6d_representation": True,
            "swarm_start_x": 0.0,
            "first_wall_distance": 1.0,
            "inter_wall_distance": 4.0,
            "unusable_opening_offset": 2.0,
            "street_width": 10.0,
            "no_initial_ramp": True,
            "wall_height": 0.20,
            "swarm_start_y": UniformDistParams(0.5, 0.75),
            "compile_reward_kernel": should_compile_reward_kernel_by_default(),
            "reward_kernel_compile_mode": "default",
        }
    )
    scenario_kwargs.update(kwargs)
    return MJWObstacleStreetScenario(
        swarm=_resolve_swarm(
            swarm=swarm,
            unit_start_locations=unit_start_locations,
            quantize_connection_twist=quantize_connection_twist,
            joints=joints,
        ),
        seed=seed,
        **scenario_kwargs,
    )
