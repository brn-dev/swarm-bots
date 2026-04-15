from typing import Any

import numpy as np

from swarmbots.mjx_env.mjx_float_or_dist_params import MjxUniformDistParams
from swarmbots.mjx_env.scenarios.mjx_base_scenario import MjxActuatorsActivationRewardType, MjxImplConfig
from swarmbots.mjx_env.scenarios.mjx_bridge_scenario import MjxBridgeScenario
from swarmbots.mjx_env.scenarios.mjx_obstacle_street_scenario import MjxObstacleStreetScenario
from swarmbots.mjx_env.swarm.mjx_base_swarm import MjxBaseSwarm
from swarmbots.mjx_env.swarm.mjx_homogeneous_swarm import MjxHomogeneousSwarm, MjxPreConnectedUnitLocationsConfig
from swarmbots.mjx_env.swarm.mjx_unit_config import (
    MJX_UNIT_CONFIG_TETRAHEDRON_XY,
    MJX_UNIT_CONFIG_TETRAHEDRON_XYZ,
    MJX_UNIT_CONFIG_TETRAHEDRON_ZX,
)


MJX_DEFAULT_KWARGS = {
    "friction": [2, 1e-2, 2e-4],
    "force_elliptic_cone": True,
    "actuator_strength": 15.0,
    "guidance_reward_weight": 1.00,
    "hinge_qvel_magnitude_reward_weight": -1e-6,
    "hinge_qvel_magnitude_reward_threshold": 8.0,
    "units_without_connections_reward_weight": -1e-5,
    "units_with_double_connection_reward_weight": -2e-4,
    "movement_reward_weight": 0e-1,
    "height_reward_weight": 0e-4,
    "connectors_stayed_active_reward_weight": 0e-5,
    "connectors_successfully_activated_reward_weight": 0e-3,
    "connectors_unsuccessfully_activated_reward_weight": -0e-5,
    "connectors_deactivated_reward_weight": 0e-3,
    "reset_settle_time": 0.0,
    "reset_settle_timestep_scale": 3,
}

MJX_WALL_PASS_KWARGS = {
    "wall_pass_reward_weight": 5.0,
    "wall_pass_thresholds": [-0.1, 0.1, 0.3, 0.5],
}


def _resolve_mjx_swarm(
    swarm: MjxBaseSwarm | None,
    unit_start_locations: MjxPreConnectedUnitLocationsConfig | None = None,
    joints: str = "zx",
) -> MjxBaseSwarm:
    if swarm is not None:
        return swarm
    if unit_start_locations is None:
        unit_start_locations = MjxPreConnectedUnitLocationsConfig(
            num_units=6,
            num_unit_probs={2: 0.5, 3: 0.5, 4: 1.0, 5: 1.0, 6: 1.0},
            max_radius=1.5,
            unconnected_prob=0.03,
            z_pos=0.5,
        )
    joint_configs = {
        "zx": {
            "unit_config": MJX_UNIT_CONFIG_TETRAHEDRON_ZX,
            "hinge_range": (None, np.pi / 3),
            "hinge_armature": (0.025, 0.015),
            "hinge_damping": (0.3, 0.15),
            "hinge_frictionloss": (0.3, 0.15),
        },
        "xy": {
            "unit_config": MJX_UNIT_CONFIG_TETRAHEDRON_XY,
            "hinge_range": (np.pi / 3, np.pi / 3),
            "hinge_armature": (0.015, 0.015),
            "hinge_damping": (0.15, 0.15),
            "hinge_frictionloss": (0.15, 0.15),
        },
        "xyz": {
            "unit_config": MJX_UNIT_CONFIG_TETRAHEDRON_XYZ,
            "hinge_range": (np.pi / 3, np.pi / 3, None),
            "hinge_armature": (0.01, 0.01, 0.025),
            "hinge_damping": (0.1, 0.1, 0.5),
            "hinge_frictionloss": (0.1, 0.1, 0.5),
        },
    }
    return MjxHomogeneousSwarm(
        **joint_configs[joints],
        unit_start_locations=unit_start_locations,
        body_radius=0.1,
        leg_length=0.2,
        leg_radius=0.025,
        connection_torquescale=50.0,
    )


def mjx_default_wall(
    seed: int | None = None,
    swarm: MjxBaseSwarm | None = None,
    unit_start_locations: MjxPreConnectedUnitLocationsConfig | None = None,
    reset_pool_size: int = 256,
    mjx_impl: MjxImplConfig = None,
    **kwargs: Any,
) -> MjxObstacleStreetScenario:
    scenario_kwargs = MJX_DEFAULT_KWARGS.copy()
    scenario_kwargs.update(MJX_WALL_PASS_KWARGS)
    scenario_kwargs.update(
        {
            "wall_height": 0.20,
            "swarm_start_y": MjxUniformDistParams(0.5, 0.75),
        }
    )
    scenario_kwargs.update(kwargs)
    return MjxObstacleStreetScenario(
        swarm=_resolve_mjx_swarm(swarm, unit_start_locations),
        payload_type=None,
        num_walls=1,
        opening_width=0.01,
        reset_pool_size=reset_pool_size,
        mjx_impl=mjx_impl,
        **scenario_kwargs,
        seed=seed,
    )


def mjx_default_bridge(
    seed: int | None = None,
    swarm: MjxBaseSwarm | None = None,
    unit_start_locations: MjxPreConnectedUnitLocationsConfig | None = None,
    reset_pool_size: int = 256,
    mjx_impl: MjxImplConfig = None,
    **kwargs: Any,
) -> MjxBridgeScenario:
    scenario_kwargs = MJX_DEFAULT_KWARGS.copy()
    scenario_kwargs.update({"fell_off_bridge_reward": -2.0})
    scenario_kwargs.update(kwargs)
    return MjxBridgeScenario(
        swarm=_resolve_mjx_swarm(swarm, unit_start_locations),
        payload_type=None,
        reset_pool_size=reset_pool_size,
        mjx_impl=mjx_impl,
        **scenario_kwargs,
        seed=seed,
    )
