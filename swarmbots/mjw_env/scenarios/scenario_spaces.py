from __future__ import annotations

import numpy as np
from gymnasium import spaces

from swarmbots.mjw_env.scenarios.base_mjw_scenario import BaseMJWScenario
from swarmbots.mjw_env.swarm.unit import get_unit_hinge_count
from swarmbots.utils.connector_actions import connector_action_space


def make_observation_space(
    scenario: BaseMJWScenario,
    *,
    global_obs_dim: int = 0,
    hidden_local_vars_dim: int = 0,
    hidden_global_vars_dim: int = 0,
) -> spaces.Dict:
    num_units = scenario.swarm.num_units
    limbs_per_unit = scenario.swarm.config.limbs_per_unit
    num_hinges = get_unit_hinge_count(scenario.swarm.config.unit_config)
    free_joint_rotation_dim = 6 if scenario.quat_rot6d_representation else 4
    qpos_obs_dim = 3 + free_joint_rotation_dim + 2 * num_hinges
    qvel_obs_dim = 6 + num_hinges
    connector_obs_dim = limbs_per_unit * 5
    connector_positions_dim = limbs_per_unit * 3 if scenario.include_connectors_xpos_in_obs else 0
    local_obs_dim = qpos_obs_dim + qvel_obs_dim + connector_obs_dim + connector_positions_dim
    return spaces.Dict({
        "local_obs": spaces.Box(-np.inf, np.inf, shape=(num_units, local_obs_dim), dtype=np.float32),
        "global_obs": spaces.Box(-np.inf, np.inf, shape=(global_obs_dim,), dtype=np.float32),
        "hidden_local_vars": spaces.Box(-np.inf, np.inf, shape=(num_units, hidden_local_vars_dim), dtype=np.float32),
        "hidden_global_vars": spaces.Box(-np.inf, np.inf, shape=(hidden_global_vars_dim,), dtype=np.float32),
        "agent_mask": spaces.MultiBinary((num_units,)),
    })


def make_action_space(scenario: BaseMJWScenario) -> spaces.Dict:
    num_units = scenario.swarm.num_units
    actuators_per_unit = get_unit_hinge_count(scenario.swarm.config.unit_config)
    return spaces.Dict({
        "actuators": spaces.Box(-1.0, 1.0, shape=(num_units, actuators_per_unit), dtype=np.float32),
        "connectors": connector_action_space(
            (num_units, scenario.swarm.config.limbs_per_unit),
            continuous=scenario.continuous_connector_actions,
        ),
    })
