from __future__ import annotations

from typing import Any

import numpy as np

from swarmbots.mj_env.float_or_dist_params import UniformDistParams
from swarmbots.mj_env.scenarios.bridge_scenario import BridgeScenario
from swarmbots.mj_env.scenarios.move_to_scenario import MoveToScenario
from swarmbots.mj_env.scenarios.obstacle_street_scenario import ObstacleStreetScenario
from swarmbots.mj_env.scenarios.payload_plane_scenario import PayloadPlaneScenario
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.homogeneous_swarm import HomogeneousSwarm, PreConnectedUnitLocationsConfig
from swarmbots.mj_env.swarm.unit_config import UNIT_CONFIG_TETRAHEDRON_XYZ, UNIT_CONFIG_TETRAHEDRON_ZX, \
    UNIT_CONFIG_TETRAHEDRON_XY
from swarmbots.move_to_goal_config import RelativePolarGoalConfig
from swarmbots.scenario_presets_kwargs import (
    COMMON_SCENARIO_KWARGS,
    MOVE_TO_REWARD_KWARGS as SHARED_MOVE_TO_REWARD_KWARGS,
    PAYLOAD_PLANE_REWARD_KWARGS as SHARED_PAYLOAD_PLANE_REWARD_KWARGS,
    WALL_PASS_REWARD_KWARGS as SHARED_WALL_PASS_REWARD_KWARGS,
    make_scenario_kwargs,
)

DEFAULT_KWARGS = make_scenario_kwargs(COMMON_SCENARIO_KWARGS, {"force_elliptic_cone": False})
WALL_PASS_REWARD_KWARGS = make_scenario_kwargs(SHARED_WALL_PASS_REWARD_KWARGS)
PAYLOAD_PLANE_REWARD_KWARGS = make_scenario_kwargs(SHARED_PAYLOAD_PLANE_REWARD_KWARGS)
MOVE_TO_REWARD_KWARGS = make_scenario_kwargs(SHARED_MOVE_TO_REWARD_KWARGS)

def _resolve_swarm(
        swarm: BaseSwarm | None,
        unit_start_locations: list[tuple[float, float, float]] | str | None = None,
        randomize_unit_orientations: bool = False,
        quantize_connection_twist: int | None = None,
        joints: str = 'zx'
) -> BaseSwarm:
    assert swarm is None or unit_start_locations is None

    if swarm is not None:
        if quantize_connection_twist is not None:
            raise ValueError("quantize_connection_twist can only be used when presets construct the swarm")
        return swarm

    if unit_start_locations is None:
        unit_start_locations = PreConnectedUnitLocationsConfig(
            num_units=5,
            num_unit_probs={
                4: 1.0,
                5: 1.0,
            },
            max_radius=1.5,
            unconnected_prob=0.02,
            z_pos=0.5,
            pool_seeds=tuple(range(42_000, 42_050)),
        )

    joint_configs = {
        'zx': {
            'unit_config': UNIT_CONFIG_TETRAHEDRON_ZX,
            'hinge_range': (None, np.pi / 3),
            'hinge_armature': (0.025, 0.015),
            'hinge_damping': (0.3, 0.15),
            'hinge_frictionloss': (0.3, 0.15),
        },
        'xy': {
            'unit_config': UNIT_CONFIG_TETRAHEDRON_XY,
            'hinge_range': (np.pi / 3, np.pi / 3),
            'hinge_armature': (0.015, 0.015),
            'hinge_damping': (0.15, 0.15),
            'hinge_frictionloss': (0.15, 0.15),
        },
        'xyz': {
            'unit_config': UNIT_CONFIG_TETRAHEDRON_XYZ,
            'hinge_range': (np.pi / 3, np.pi / 3, None),
            'hinge_armature': (0.01, 0.01, 0.025),
            'hinge_damping': (0.1, 0.1, 0.5),
            'hinge_frictionloss': (0.1, 0.1, 0.5),
        }
    }

    return HomogeneousSwarm(
        **joint_configs[joints],
        unit_start_locations=unit_start_locations,
        body_radius=0.1,
        leg_length=0.2,
        leg_radius=0.025,
        connection_torquescale=50.0,
        quantize_connection_twist=quantize_connection_twist,
        randomize_unit_orientations=randomize_unit_orientations,
    )


def default_wall(
        seed: int | None = None,
        swarm: BaseSwarm | None = None,
        unit_start_locations: list[tuple[float, float, float]] | str | Any | None = None,
        randomize_unit_orientations: bool = False,
        quantize_connection_twist: int | None = 8,
        joints: str = 'zx',
        **kwargs
) -> ObstacleStreetScenario:
    scenario_kwargs = make_scenario_kwargs(DEFAULT_KWARGS, WALL_PASS_REWARD_KWARGS)
    scenario_kwargs.update({
        'num_walls': 1,
        'opening_width': 0.01,
        'connection_dist_threshold': 0.1,
        'connection_angle_threshold': -0.5,
        'disconnect_potential_threshold': 5.0,
        'include_connectors_xpos_in_obs': True,
        'include_connectors_xquat_in_obs': False,
        'quat_rot6d_representation': True,
        'swarm_start_x': 0.0,
        'first_wall_distance': 1.0,
        'inter_wall_distance': 4.0,
        'unusable_opening_offset': 2.0,
        'street_width': 10.0,
        'no_initial_ramp': True,
        'wall_height': 0.20,
        'swarm_start_y': UniformDistParams(0.25, 0.75),
    })
    scenario_kwargs.update(kwargs)
    return ObstacleStreetScenario(
        swarm=_resolve_swarm(
            swarm,
            unit_start_locations,
            randomize_unit_orientations,
            quantize_connection_twist,
            joints=joints,
        ),
        **scenario_kwargs,
        seed=seed,
    )


def default_bridge(
        seed: int | None = None,
        swarm: BaseSwarm | None = None,
        unit_start_locations: list[tuple[float, float, float]] | str | None = None,
        randomize_unit_orientations: bool = False,
        quantize_connection_twist: int | None = None,
        **kwargs
) -> BridgeScenario:
    scenario_kwargs = make_scenario_kwargs(DEFAULT_KWARGS)
    scenario_kwargs.update({
        'fell_off_bridge_reward': -2.0,
    })
    scenario_kwargs.update(kwargs)
    return BridgeScenario(
        swarm=_resolve_swarm(
            swarm,
            unit_start_locations,
            randomize_unit_orientations,
            quantize_connection_twist,
        ),
        **scenario_kwargs,
        seed=seed,
    )


def default_payload_plane(
        seed: int | None = None,
        swarm: BaseSwarm | None = None,
        unit_start_locations: list[tuple[float, float, float]] | str | Any | None = None,
        randomize_unit_orientations: bool = False,
        quantize_connection_twist: int | None = 8,
        joints: str = 'zx',
        **kwargs
) -> PayloadPlaneScenario:
    scenario_kwargs = make_scenario_kwargs(DEFAULT_KWARGS, PAYLOAD_PLANE_REWARD_KWARGS)
    scenario_kwargs.update({
        'plane_size': 100.0,
        'connection_dist_threshold': 0.1,
        'connection_angle_threshold': -0.5,
        'disconnect_potential_threshold': 5.0,
        'include_connectors_xpos_in_obs': True,
        'include_connectors_xquat_in_obs': False,
        'quat_rot6d_representation': True,
        'swarm_start_x': 0.0,
        'swarm_start_y': 0.0,
    })
    scenario_kwargs.update(kwargs)
    return PayloadPlaneScenario(
        swarm=_resolve_swarm(
            swarm,
            unit_start_locations,
            randomize_unit_orientations,
            quantize_connection_twist,
            joints=joints,
        ),
        **scenario_kwargs,
        seed=seed,
    )


def default_move_to(
        seed: int | None = None,
        swarm: BaseSwarm | None = None,
        unit_start_locations: list[tuple[float, float, float]] | str | Any | None = None,
        randomize_unit_orientations: bool = False,
        quantize_connection_twist: int | None = 8,
        joints: str = 'zx',
        **kwargs
) -> MoveToScenario:
    scenario_kwargs = make_scenario_kwargs(DEFAULT_KWARGS, MOVE_TO_REWARD_KWARGS)
    scenario_kwargs.update({
        'plane_size': 100.0,
        'connection_dist_threshold': 0.1,
        'connection_angle_threshold': -0.5,
        'disconnect_potential_threshold': 5.0,
        'include_connectors_xpos_in_obs': True,
        'include_connectors_xquat_in_obs': False,
        'quat_rot6d_representation': True,
        'swarm_start_x': 0.0,
        'swarm_start_y': 0.0,
        'goal': RelativePolarGoalConfig(
            distance=UniformDistParams(2.0, 4.0),
            angle=UniformDistParams(0.0, 2.0 * np.pi),
        ),
    })
    scenario_kwargs.update(kwargs)
    return MoveToScenario(
        swarm=_resolve_swarm(
            swarm,
            unit_start_locations,
            randomize_unit_orientations,
            quantize_connection_twist,
            joints=joints,
        ),
        **scenario_kwargs,
        seed=seed,
    )
