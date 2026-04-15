from typing import Any

import numpy as np

from swarmbots.mj_env.float_or_dist_params import UniformDistParams
from swarmbots.mj_env.scenarios.bridge_scenario import BridgeScenario
from swarmbots.mj_env.scenarios.obstacle_street_scenario import ObstacleStreetScenario
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.homogeneous_swarm import HomogeneousSwarm, PreConnectedUnitLocationsConfig
from swarmbots.mj_env.swarm.unit_config import UNIT_CONFIG_TETRAHEDRON_XYZ, UNIT_CONFIG_TETRAHEDRON_ZX, \
    UNIT_CONFIG_TETRAHEDRON_XY

DEFAULT_KWARGS = {
    'friction': [2, 1e-2, 2e-4],
    'force_elliptic_cone': True,
    'actuator_strength': 15.0,
    'guidance_reward_weight': 1.00,
    'units_without_connections_reward_weight': -1e-5,
    'reset_settle_time': 1.0,
    'reset_settle_timestep_scale': 3,
}
WALL_PASS_KWARGS = {
    'wall_pass_reward_weight': 5.0,
    'wall_pass_thresholds': [-0.1, 0.1, 0.3, 0.5],
}

def _resolve_swarm(
        swarm: BaseSwarm | None,
        unit_start_locations: list[tuple[float, float, float]] | str | None = None,
        randomize_unit_orientations: bool = False,
        joints: str = 'zx'
) -> BaseSwarm:
    assert swarm is None or unit_start_locations is None

    if swarm is not None:
        return swarm

    if unit_start_locations is None:
        unit_start_locations = PreConnectedUnitLocationsConfig(
            num_units=6,
            num_unit_probs={
                2: 0.5,
                3: 0.5,
                4: 1.0,
                5: 1.0,
                6: 1.0,
            },
            max_radius=1.5,
            unconnected_prob=0.03,
            z_pos=0.5,
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
        randomize_unit_orientations=randomize_unit_orientations,
    )


def default_wall(
        seed: int | None = None,
        swarm: BaseSwarm | None = None,
        unit_start_locations: list[tuple[float, float, float]] | str | Any | None = None,
        randomize_unit_orientations: bool = False,
        **kwargs
) -> ObstacleStreetScenario:
    scenario_kwargs = DEFAULT_KWARGS.copy()
    scenario_kwargs.update(WALL_PASS_KWARGS)
    scenario_kwargs.update({
        'wall_height': 0.20,
        # 'swarm_start_y': UniformDistParams(0.0, 0.75),
        # 'swarm_start_y': 0.7,
        'swarm_start_y': UniformDistParams(0.5, 0.75),
    })
    scenario_kwargs.update(kwargs)
    return ObstacleStreetScenario(
        swarm=_resolve_swarm(swarm, unit_start_locations, randomize_unit_orientations),
        payload_type=None,
        num_walls=1,
        opening_width=0.01,
        **scenario_kwargs,
        seed=seed,
    )


def default_bridge(
        seed: int | None = None,
        swarm: BaseSwarm | None = None,
        unit_start_locations: list[tuple[float, float, float]] | str | None = None,
        randomize_unit_orientations: bool = False,
        **kwargs
) -> BridgeScenario:
    scenario_kwargs = DEFAULT_KWARGS.copy()
    scenario_kwargs.update({
        'fell_off_bridge_reward': -2.0,
    })
    scenario_kwargs.update(kwargs)
    return BridgeScenario(
        swarm=_resolve_swarm(swarm, unit_start_locations, randomize_unit_orientations),
        payload_type=None,
        **scenario_kwargs,
        seed=seed,
    )
