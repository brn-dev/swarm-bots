from swarmbots.mj_env.scenarios.bridge_scenario import BridgeScenario
from swarmbots.mj_env.scenarios.obstacle_street_scenario import ObstacleStreetScenario
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.homogeneous_swarm import HomogeneousSwarm

DEFAULT_KWARGS = {
    'friction': [2, 1e-2, 2e-4],
    'force_elliptic_cone': True,
    'actuators_activation_reward_weight': -7e-3,
    'units_without_connections_reward_weight': -2e-3,
    'units_with_double_connection_reward_weight': -1e-3,
    'movement_reward_weight': 0e-1,
    'height_reward_weight': 0e-4,
    'connectors_stayed_active_reward_weight': 0e-5,
    'connectors_successfully_activated_reward_weight': 0e-3,
    'connectors_unsuccessfully_activated_reward_weight': -1e-4,
    'connectors_deactivated_reward_weight': 0e-3,
    'reset_settle_steps': 30,
}

def _resolve_swarm(
        swarm: BaseSwarm | None,
        unit_start_locations: list[tuple[float, float, float]] | str | None = None,
        randomize_unit_orientations: bool = False,
) -> BaseSwarm:
    assert swarm is None or unit_start_locations is None

    if swarm is not None:
        return swarm

    if unit_start_locations is None:
        unit_start_locations = '4:diamond'

    return HomogeneousSwarm(
        unit_start_locations=unit_start_locations,
        randomize_unit_orientations=randomize_unit_orientations
    )


def default_wall(
        seed: int | None = None,
        swarm: BaseSwarm | None = None,
        unit_start_locations: list[tuple[float, float, float]] | str | None = None,
        randomize_unit_orientations: bool = False,
        **kwargs
) -> ObstacleStreetScenario:
    scenario_kwargs = DEFAULT_KWARGS.copy()
    scenario_kwargs.update({
        'wall_height': 0.15,
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