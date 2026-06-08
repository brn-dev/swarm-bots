from __future__ import annotations

import importlib.util
import shutil
import sys

import numpy as np
import torch

from swarmbots.mj_env.swarm.unit_config import (
    UNIT_CONFIG_TETRAHEDRON_XY,
    UNIT_CONFIG_TETRAHEDRON_XYZ,
    UNIT_CONFIG_TETRAHEDRON_ZX,
)
from swarmbots.mjw_env.scenarios.mjw_bridge_scenario import MJWBridgeScenario
from swarmbots.mjw_env.scenarios.mjw_climb_scenario import MJWClimbScenario
from swarmbots.mjw_env.scenarios.mjw_dual_payload_plane_scenario import MJWDualPayloadPlaneScenario
from swarmbots.mjw_env.scenarios.mjw_move_to_scenario import MJWMoveToScenario
from swarmbots.mjw_env.scenarios.mjw_obstacle_street_scenario import MJWObstacleStreetScenario
from swarmbots.mjw_env.scenarios.mjw_payload_plane_scenario import MJWPayloadPlaneScenario
from swarmbots.mjw_env.scenarios.mjw_payload_step_scenario import MJWPayloadStepScenario
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWHomogeneousSwarm, MJWPreConnectedUnitLocationsConfig
from swarmbots.scenario_presets.scenario_presets_kwargs import (
    BRIDGE_SCENARIO_KWARGS as SHARED_BRIDGE_SCENARIO_KWARGS,
    CLIMB_SCENARIO_KWARGS as SHARED_CLIMB_SCENARIO_KWARGS,
    COMMON_SCENARIO_KWARGS,
    Difficulty,
    DUAL_PAYLOAD_PLANE_SCENARIO_KWARGS as SHARED_DUAL_PAYLOAD_PLANE_SCENARIO_KWARGS,
    EASY_WALL_SCENARIO_KWARGS as SHARED_EASY_WALL_SCENARIO_KWARGS,
    HARD_WALL_SCENARIO_KWARGS as SHARED_HARD_WALL_SCENARIO_KWARGS,
    MEDIUM_WALL_SCENARIO_KWARGS as SHARED_MEDIUM_WALL_SCENARIO_KWARGS,
    MOVE_TO_SCENARIO_KWARGS as SHARED_MOVE_TO_SCENARIO_KWARGS,
    PAYLOAD_PLANE_SCENARIO_KWARGS as SHARED_PAYLOAD_PLANE_SCENARIO_KWARGS,
    PAYLOAD_STEP_SCENARIO_KWARGS as SHARED_PAYLOAD_STEP_SCENARIO_KWARGS,
    PO_WALL_MEDIUM_SCENARIO_KWARGS as SHARED_PO_WALL_MEDIUM_SCENARIO_KWARGS,
    WALL_SCENARIO_KWARGS as SHARED_WALL_SCENARIO_KWARGS,
    make_scenario_kwargs,
    wall_scenario_kwargs_with_difficulty,
)

DEFAULT_KWARGS = make_scenario_kwargs(COMMON_SCENARIO_KWARGS)
WALL_SCENARIO_KWARGS = make_scenario_kwargs(SHARED_WALL_SCENARIO_KWARGS)
EASY_WALL_SCENARIO_KWARGS = make_scenario_kwargs(SHARED_EASY_WALL_SCENARIO_KWARGS)
MEDIUM_WALL_SCENARIO_KWARGS = make_scenario_kwargs(SHARED_MEDIUM_WALL_SCENARIO_KWARGS)
HARD_WALL_SCENARIO_KWARGS = make_scenario_kwargs(SHARED_HARD_WALL_SCENARIO_KWARGS)
PO_WALL_MEDIUM_SCENARIO_KWARGS = make_scenario_kwargs(SHARED_PO_WALL_MEDIUM_SCENARIO_KWARGS)
BRIDGE_SCENARIO_KWARGS = make_scenario_kwargs(SHARED_BRIDGE_SCENARIO_KWARGS)
CLIMB_SCENARIO_KWARGS = make_scenario_kwargs(SHARED_CLIMB_SCENARIO_KWARGS)
PAYLOAD_PLANE_SCENARIO_KWARGS = make_scenario_kwargs(SHARED_PAYLOAD_PLANE_SCENARIO_KWARGS)
PAYLOAD_STEP_SCENARIO_KWARGS = make_scenario_kwargs(SHARED_PAYLOAD_STEP_SCENARIO_KWARGS)
DUAL_PAYLOAD_PLANE_SCENARIO_KWARGS = make_scenario_kwargs(SHARED_DUAL_PAYLOAD_PLANE_SCENARIO_KWARGS)
MOVE_TO_SCENARIO_KWARGS = make_scenario_kwargs(SHARED_MOVE_TO_SCENARIO_KWARGS)


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
    quantize_connection_twist: int = 8,
    joints: str = "zx",
    difficulty: Difficulty | None = None,
    **kwargs: object,
) -> MJWObstacleStreetScenario:
    scenario_kwargs = make_scenario_kwargs(DEFAULT_KWARGS, wall_scenario_kwargs_with_difficulty(difficulty))
    scenario_kwargs.update(
        {
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


def easy_wall(**kwargs: object) -> MJWObstacleStreetScenario:
    return default_wall(difficulty="easy", **kwargs)


def medium_wall(**kwargs: object) -> MJWObstacleStreetScenario:
    return default_wall(difficulty="medium", **kwargs)


def hard_wall(**kwargs: object) -> MJWObstacleStreetScenario:
    return default_wall(difficulty="hard", **kwargs)


def default_bridge(
    *,
    seed: int | None = None,
    swarm: MJWHomogeneousSwarm | None = None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None = None,
    quantize_connection_twist: int = 8,
    joints: str = "zx",
    **kwargs: object,
) -> MJWBridgeScenario:
    scenario_kwargs = make_scenario_kwargs(DEFAULT_KWARGS, BRIDGE_SCENARIO_KWARGS)
    scenario_kwargs.update(
        {
            "compile_reward_kernel": should_compile_reward_kernel_by_default(),
            "reward_kernel_compile_mode": "default",
        }
    )
    scenario_kwargs.update(kwargs)
    return MJWBridgeScenario(
        swarm=_resolve_swarm(
            swarm=swarm,
            unit_start_locations=unit_start_locations,
            quantize_connection_twist=quantize_connection_twist,
            joints=joints,
        ),
        seed=seed,
        **scenario_kwargs,
    )


def default_climb(
    *,
    seed: int | None = None,
    swarm: MJWHomogeneousSwarm | None = None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None = None,
    quantize_connection_twist: int = 8,
    joints: str = "zx",
    **kwargs: object,
) -> MJWClimbScenario:
    scenario_kwargs = make_scenario_kwargs(DEFAULT_KWARGS, CLIMB_SCENARIO_KWARGS)
    scenario_kwargs.update(
        {
            "compile_reward_kernel": should_compile_reward_kernel_by_default(),
            "reward_kernel_compile_mode": "default",
        }
    )
    scenario_kwargs.update(kwargs)
    return MJWClimbScenario(
        swarm=_resolve_swarm(
            swarm=swarm,
            unit_start_locations=unit_start_locations,
            quantize_connection_twist=quantize_connection_twist,
            joints=joints,
        ),
        seed=seed,
        **scenario_kwargs,
    )


def default_payload_plane(
    *,
    seed: int | None = None,
    swarm: MJWHomogeneousSwarm | None = None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None = None,
    quantize_connection_twist: int = 8,
    joints: str = "zx",
    **kwargs: object,
) -> MJWPayloadPlaneScenario:
    scenario_kwargs = make_scenario_kwargs(DEFAULT_KWARGS, PAYLOAD_PLANE_SCENARIO_KWARGS)
    scenario_kwargs.update(
        {
            "compile_reward_kernel": should_compile_reward_kernel_by_default(),
            "reward_kernel_compile_mode": "default",
        }
    )
    scenario_kwargs.update(kwargs)
    return MJWPayloadPlaneScenario(
        swarm=_resolve_swarm(
            swarm=swarm,
            unit_start_locations=unit_start_locations,
            quantize_connection_twist=quantize_connection_twist,
            joints=joints,
        ),
        seed=seed,
        **scenario_kwargs,
    )


def default_dual_payload_plane(
    *,
    seed: int | None = None,
    swarm: MJWHomogeneousSwarm | None = None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None = None,
    quantize_connection_twist: int = 8,
    joints: str = "zx",
    **kwargs: object,
) -> MJWDualPayloadPlaneScenario:
    scenario_kwargs = make_scenario_kwargs(DEFAULT_KWARGS, DUAL_PAYLOAD_PLANE_SCENARIO_KWARGS)
    scenario_kwargs.update(
        {
            "compile_reward_kernel": should_compile_reward_kernel_by_default(),
            "reward_kernel_compile_mode": "default",
        }
    )
    scenario_kwargs.update(kwargs)
    return MJWDualPayloadPlaneScenario(
        swarm=_resolve_swarm(
            swarm=swarm,
            unit_start_locations=unit_start_locations,
            quantize_connection_twist=quantize_connection_twist,
            joints=joints,
        ),
        seed=seed,
        **scenario_kwargs,
    )


def default_payload_step(
    *,
    seed: int | None = None,
    swarm: MJWHomogeneousSwarm | None = None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None = None,
    quantize_connection_twist: int = 8,
    joints: str = "zx",
    **kwargs: object,
) -> MJWPayloadStepScenario:
    scenario_kwargs = make_scenario_kwargs(DEFAULT_KWARGS, PAYLOAD_STEP_SCENARIO_KWARGS)
    scenario_kwargs.update(
        {
            "compile_reward_kernel": should_compile_reward_kernel_by_default(),
            "reward_kernel_compile_mode": "default",
        }
    )
    scenario_kwargs.update(kwargs)
    return MJWPayloadStepScenario(
        swarm=_resolve_swarm(
            swarm=swarm,
            unit_start_locations=unit_start_locations,
            quantize_connection_twist=quantize_connection_twist,
            joints=joints,
        ),
        seed=seed,
        **scenario_kwargs,
    )


def default_move_to(
    *,
    seed: int | None = None,
    swarm: MJWHomogeneousSwarm | None = None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None = None,
    quantize_connection_twist: int = 8,
    joints: str = "zx",
    **kwargs: object,
) -> MJWMoveToScenario:
    scenario_kwargs = make_scenario_kwargs(DEFAULT_KWARGS, MOVE_TO_SCENARIO_KWARGS)
    scenario_kwargs.update(
        {
            "compile_reward_kernel": should_compile_reward_kernel_by_default(),
            "reward_kernel_compile_mode": "default",
        }
    )
    scenario_kwargs.update(kwargs)
    return MJWMoveToScenario(
        swarm=_resolve_swarm(
            swarm=swarm,
            unit_start_locations=unit_start_locations,
            quantize_connection_twist=quantize_connection_twist,
            joints=joints,
        ),
        seed=seed,
        **scenario_kwargs,
    )
