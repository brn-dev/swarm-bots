from typing import NamedTuple, NotRequired, TypedDict

import jax
import jax.numpy as jnp
import mujoco.mjx as mjx


class MjxObsDict(TypedDict):
    local_obs: jax.Array
    global_obs: jax.Array
    hidden_local_vars: jax.Array
    hidden_global_vars: jax.Array
    agent_mask: NotRequired[jax.Array]


class MjxActDict(TypedDict):
    actuators: jax.Array
    connectors: jax.Array


class MjxSwarmConnections(NamedTuple):
    connections: jax.Array
    twist_bins: jax.Array
    twist_angles: jax.Array
    disconnect_potentials: jax.Array


class MjxEnvState(NamedTuple):
    data: mjx.Data
    connections: MjxSwarmConnections
    current_step: jax.Array
    progress: jax.Array
    unit_positions: jax.Array
    units_active_mask: jax.Array
    num_connectors_stayed_active: jax.Array
    num_connectors_successfully_activated: jax.Array
    num_connectors_unsuccessfully_activated: jax.Array
    num_connectors_deactivated: jax.Array
    weighted_progress_reward: jax.Array
    weighted_guidance_reward: jax.Array
    hidden_global_vars: jax.Array
    passed_thresholds_mask: jax.Array
    next_threshold_for_unit: jax.Array
    wall_pass_absolute_thresholds: jax.Array
    num_walls_passed: jax.Array
    walls_passed_reward: jax.Array
    fell_off_bridge: jax.Array


class MjxResetPool(NamedTuple):
    qpos: jax.Array
    qvel: jax.Array
    ctrl: jax.Array
    mocap_pos: jax.Array
    connections: MjxSwarmConnections
    units_active_mask: jax.Array
    hidden_global_vars: jax.Array
    passed_thresholds_mask: jax.Array
    next_threshold_for_unit: jax.Array
    wall_pass_absolute_thresholds: jax.Array


def empty_connections(num_units: int, limbs_per_unit: int) -> MjxSwarmConnections:
    return MjxSwarmConnections(
        connections=jnp.full((num_units, limbs_per_unit, 2), -1, dtype=jnp.int32),
        twist_bins=jnp.zeros((num_units, limbs_per_unit), dtype=jnp.int32),
        twist_angles=jnp.zeros((num_units, limbs_per_unit), dtype=jnp.float32),
        disconnect_potentials=jnp.zeros((num_units, limbs_per_unit), dtype=jnp.float32),
    )
