import abc
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
from gymnasium import spaces

from swarmbots.mjx_env.mjx_float_or_dist_params import MjxFloatOrDistParams, mjx_eval_fodp_2d
from swarmbots.mjx_env.mjx_quat import mjx_quat_to_rot6d
from swarmbots.mjx_env.swarm.mjx_base_swarm import MjxBaseSwarm, MjxSwarmReset
from swarmbots.mjx_env.swarm.mjx_unit import mjx_get_limb_segment_names
from swarmbots.mjx_env.types import MjxActDict, MjxEnvState, MjxObsDict, MjxResetPool, MjxSwarmConnections

MjxImplConfig = mjx.Impl | str | None


class MjxActuatorsActivationRewardType(Enum):
    MONOMIAL = 1
    LOG1M = 2


DEFAULT_GEOM_FRICTION: tuple[float, float, float] = (1.0, 0.005, 0.0001)
HIGH_FRICTION_SLIDING_THRESHOLD = 2.0
DEFAULT_INACTIVE_AREA_LOCATION = (50.0, 0.0, 0.1)
MJX_CONTACT_LIMIT_PER_LIMB = 3


@dataclass(frozen=True)
class MjxModelIndices:
    qpos_indices: jax.Array
    qvel_indices: jax.Array
    ctrl_indices: jax.Array
    connector_body_indices: jax.Array
    eq_indices: jax.Array
    pair_i: jax.Array
    pair_j: jax.Array
    flat_units: jax.Array
    flat_connectors: jax.Array


class _ResetSample(NamedTuple):
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    mocap_pos: np.ndarray
    swarm_reset: MjxSwarmReset
    hidden_global_vars: np.ndarray
    passed_thresholds_mask: np.ndarray
    next_threshold_for_unit: np.ndarray
    wall_pass_absolute_thresholds: np.ndarray


def _validate_geom_friction(
    friction: float | Iterable[float] | None,
) -> tuple[float, float, float] | None:
    if friction is None:
        return None
    if isinstance(friction, (int, float)):
        return (float(friction), DEFAULT_GEOM_FRICTION[1], DEFAULT_GEOM_FRICTION[2])
    friction_tuple = tuple(float(x) for x in friction)
    if len(friction_tuple) != 3:
        raise ValueError(f"Expected friction to be a float or 3-tuple, got {friction_tuple!r}")
    return friction_tuple


def _body_ids_for_prefix(model: mujoco.MjModel, prefix: str) -> list[int]:
    return [idx for idx in range(model.nbody) if model.body(idx).name.startswith(prefix)]


def _qpos_indices_for_body(model: mujoco.MjModel, body_name: str) -> list[int]:
    body_id = model.body(body_name).id
    jadr = model.body_jntadr[body_id]
    jnum = model.body_jntnum[body_id]
    if jadr < 0 or jnum == 0:
        return []
    qpos_width = {0: 7, 1: 4, 2: 1, 3: 1}
    out: set[int] = set()
    for jnt_id in range(jadr, jadr + jnum):
        qadr = int(model.jnt_qposadr[jnt_id])
        width = qpos_width[int(model.jnt_type[jnt_id])]
        out.update(range(qadr, qadr + width))
    return sorted(out)


def _qpos_indices_for_prefix(model: mujoco.MjModel, prefix: str) -> list[int]:
    out: set[int] = set()
    for body_id in _body_ids_for_prefix(model, prefix):
        out.update(_qpos_indices_for_body(model, model.body(body_id).name))
    return sorted(out)


def _dof_indices_for_body(model: mujoco.MjModel, body_name: str) -> list[int]:
    body_id = model.body(body_name).id
    dof_adr = model.body_dofadr[body_id]
    dof_num = model.body_dofnum[body_id]
    if dof_adr < 0 or dof_num == 0:
        return []
    return list(range(int(dof_adr), int(dof_adr + dof_num)))


def _dof_indices_for_prefix(model: mujoco.MjModel, prefix: str) -> list[int]:
    out: set[int] = set()
    for body_id in _body_ids_for_prefix(model, prefix):
        out.update(_dof_indices_for_body(model, model.body(body_id).name))
    return sorted(out)


def _ctrl_indices_for_prefix(model: mujoco.MjModel, prefix: str) -> list[int]:
    return [idx for idx in range(model.nu) if model.actuator(idx).name.startswith(prefix)]


def _resolve_mjx_impl(impl: MjxImplConfig) -> mjx.Impl | None:
    if impl is None or isinstance(impl, mjx.Impl):
        return impl
    try:
        return mjx.Impl(str(impl).lower())
    except ValueError as error:
        valid = ", ".join(value.value for value in mjx.Impl)
        raise ValueError(f"Unsupported MJX impl {impl!r}. Expected one of: {valid}") from error


def _put_mjx_model(model: mujoco.MjModel, impl: mjx.Impl | None) -> mjx.Model:
    try:
        return mjx.put_model(model, impl=impl)
    except RuntimeError as error:
        if impl == mjx.Impl.WARP:
            raise RuntimeError(
                "Failed to initialize MJX with impl='warp'. Warp requires a CUDA-capable JAX runtime "
                "and NVIDIA Warp installed. Install the CUDA JAX build and the repo's warp extra "
                "before using the MJX Warp run script."
            ) from error
        raise


def _prefixed_body_name(unit_prefix: str, body_name: str) -> str:
    return f"{unit_prefix}{body_name}"


def _set_numeric(spec: mujoco.MjSpec, name: str, value: float) -> None:
    numeric = spec.add_numeric()
    numeric.name = name
    numeric.size = 1
    numeric.data = [float(value)]


class MjxBaseScenario(abc.ABC):
    def __init__(
        self,
        swarm: MjxBaseSwarm,
        actuator_strength: float,
        progress_reward_weight: float,
        guidance_reward_weight: float,
        hinge_qvel_magnitude_reward_weight: float,
        hinge_qvel_magnitude_reward_threshold: float,
        units_without_connections_reward_weight: float,
        units_with_double_connection_reward_weight: float,
        movement_reward_weight: float,
        height_reward_weight: float,
        connectors_stayed_active_reward_weight: float,
        connectors_successfully_activated_reward_weight: float,
        connectors_unsuccessfully_activated_reward_weight: float,
        connectors_deactivated_reward_weight: float,
        average_connectors_reward: bool,
        include_connectors_xpos_in_obs: bool,
        include_connectors_xquat_in_obs: bool,
        quat_rot6d_representation: bool,
        connection_dist_threshold: float,
        connection_angle_threshold: float,
        disconnect_potential_threshold: float,
        friction: float | Iterable[float] | None,
        force_elliptic_cone: bool,
        reset_settle_time: float,
        reset_settle_timestep_scale: float,
        swarm_start_x: MjxFloatOrDistParams,
        swarm_start_y: MjxFloatOrDistParams,
        inactive_area_location: Iterable[float] | None,
        seed: int | None,
        reset_pool_size: int,
        mjx_impl: MjxImplConfig = None,
    ) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.swarm = swarm
        self.num_units = swarm.config.num_units
        self.limbs_per_unit = swarm.config.limbs_per_unit
        self.num_connectors = self.num_units * self.limbs_per_unit
        self.actuator_strength = float(actuator_strength)
        self.connection_dist_threshold = float(connection_dist_threshold)
        self.connection_angle_threshold = float(connection_angle_threshold)
        self.disconnect_potential_threshold = float(disconnect_potential_threshold)
        self.swarm_start_x = swarm_start_x
        self.swarm_start_y = swarm_start_y
        self.reset_settle_time = float(reset_settle_time)
        self.reset_settle_timestep_scale = float(reset_settle_timestep_scale)
        self.reset_pool_size = int(reset_pool_size)
        self.mjx_impl = _resolve_mjx_impl(mjx_impl)
        if self.reset_pool_size <= 0:
            raise ValueError(f"Expected reset_pool_size > 0, got {reset_pool_size}")
        if self.reset_settle_time < 0:
            raise ValueError(f"Expected reset_settle_time >= 0, got {reset_settle_time}")
        if self.reset_settle_timestep_scale <= 0:
            raise ValueError(f"Expected reset_settle_timestep_scale > 0, got {reset_settle_timestep_scale}")

        self.reward_weights = {
            "progress_reward_weight": float(progress_reward_weight),
            "guidance_reward_weight": float(guidance_reward_weight),
            "hinge_qvel_magnitude_reward_weight": float(hinge_qvel_magnitude_reward_weight),
            "hinge_qvel_magnitude_reward_threshold": float(hinge_qvel_magnitude_reward_threshold),
            "units_without_connections_reward_weight": float(units_without_connections_reward_weight),
            "units_with_double_connection_reward_weight": float(units_with_double_connection_reward_weight),
            "movement_reward_weight": float(movement_reward_weight),
            "height_reward_weight": float(height_reward_weight),
            "connectors_stayed_active_reward_weight": float(connectors_stayed_active_reward_weight),
            "connectors_successfully_activated_reward_weight": float(connectors_successfully_activated_reward_weight),
            "connectors_unsuccessfully_activated_reward_weight": float(
                connectors_unsuccessfully_activated_reward_weight
            ),
            "connectors_deactivated_reward_weight": float(connectors_deactivated_reward_weight),
        }
        self.average_connectors_reward = bool(average_connectors_reward)
        self.include_connectors_xpos_in_obs = bool(include_connectors_xpos_in_obs)
        self.include_connectors_xquat_in_obs = bool(include_connectors_xquat_in_obs)
        self.quat_rot6d_representation = bool(quat_rot6d_representation)
        self.friction = _validate_geom_friction(friction)
        self.force_elliptic_cone = bool(force_elliptic_cone)

        if inactive_area_location is None:
            inactive_area_location = DEFAULT_INACTIVE_AREA_LOCATION
        self.inactive_area_location = np.asarray(inactive_area_location, dtype=float)
        self.inactive_unit_positions = self._generate_inactive_unit_positions(swarm, self.inactive_area_location)

        self.spec = self.create_scenario_spec()
        self.model = self.spec.compile()
        self._configure_model(self.model)
        self.mjx_model = _put_mjx_model(self.model, self.mjx_impl)
        self.settle_mjx_model = self._make_settle_model()
        self.template_data = mjx.make_data(self.mjx_model)
        self.indices = self._build_indices()
        self.reset_settle_steps = self._compute_reset_settle_steps()
        self.reset_pool = self._build_reset_pool()

    @abc.abstractmethod
    def _create_scenario_spec(self) -> mujoco.MjSpec:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_progress(self, state: MjxEnvState) -> jax.Array:
        raise NotImplementedError

    def create_scenario_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        swarm_spec = self.swarm.create_swarm_spec(np.random.default_rng(self.seed if self.seed is not None else 42))
        swarm_site = spec.worldbody.add_site(pos=self.get_swarm_start_location(), name="swarm_site")
        spec.attach(swarm_spec, "", site=swarm_site)
        scenario_site = spec.worldbody.add_site(pos=[0, 0, 0], name="scenario_site")
        spec.attach(self._create_scenario_spec(), "", site=scenario_site)
        self._configure_spec(spec)
        return spec

    def get_settings(self) -> dict[str, Any]:
        return {
            "swarm": self.swarm.get_settings(),
            "actuator_strength": self.actuator_strength,
            "reward_weights": dict(self.reward_weights),
            "average_connectors_reward": self.average_connectors_reward,
            "include_connectors_xpos_in_obs": self.include_connectors_xpos_in_obs,
            "include_connectors_xquat_in_obs": self.include_connectors_xquat_in_obs,
            "quat_rot6d_representation": self.quat_rot6d_representation,
            "connection_dist_threshold": self.connection_dist_threshold,
            "connection_angle_threshold": self.connection_angle_threshold,
            "disconnect_potential_threshold": self.disconnect_potential_threshold,
            "swarm_start_x": self.swarm_start_x,
            "swarm_start_y": self.swarm_start_y,
            "friction": self.friction,
            "force_elliptic_cone": self.force_elliptic_cone,
            "seed": self.seed,
            "reset_settle_time": self.reset_settle_time,
            "reset_settle_timestep_scale": self.reset_settle_timestep_scale,
            "inactive_area_location": self.inactive_area_location.tolist(),
            "reset_pool_size": self.reset_pool_size,
            "mjx_impl": None if self.mjx_impl is None else self.mjx_impl.value,
        }

    def get_swarm_start_location(self) -> np.ndarray:
        start_xy = mjx_eval_fodp_2d((self.swarm_start_x, self.swarm_start_y), self.rng)
        return np.array([start_xy[0], start_xy[1], self.swarm.max_unit_extent * 1.1], dtype=float)

    def reset(self, key: jax.Array) -> MjxEnvState:
        idx = jax.random.randint(key, shape=(), minval=0, maxval=self.reset_pool_size)
        pool = self.reset_pool
        connections = MjxSwarmConnections(
            connections=pool.connections.connections[idx],
            twist_bins=pool.connections.twist_bins[idx],
            twist_angles=pool.connections.twist_angles[idx],
            disconnect_potentials=pool.connections.disconnect_potentials[idx],
        )
        data = self.template_data.replace(
            qpos=pool.qpos[idx],
            qvel=pool.qvel[idx],
            ctrl=pool.ctrl[idx],
            mocap_pos=pool.mocap_pos[idx],
            eq_active=self.build_eq_active(connections.connections, connections.twist_bins),
        )
        data = mjx.forward(self.mjx_model, data)
        if self.reset_settle_steps > 0:
            data = jax.lax.fori_loop(
                0,
                self.reset_settle_steps,
                lambda _, loop_data: mjx.step(self.settle_mjx_model, loop_data),
                data,
            )
            data = mjx.forward(self.mjx_model, data)

        state = MjxEnvState(
            data=data,
            connections=connections,
            current_step=jnp.array(0, dtype=jnp.int32),
            progress=jnp.array(0.0, dtype=jnp.float32),
            unit_positions=data.qpos[self.indices.qpos_indices[:, :3]],
            units_active_mask=pool.units_active_mask[idx],
            num_connectors_stayed_active=jnp.array(0, dtype=jnp.int32),
            num_connectors_successfully_activated=jnp.array(0, dtype=jnp.int32),
            num_connectors_unsuccessfully_activated=jnp.array(0, dtype=jnp.int32),
            num_connectors_deactivated=jnp.array(0, dtype=jnp.int32),
            weighted_progress_reward=jnp.array(0.0, dtype=jnp.float32),
            weighted_guidance_reward=jnp.array(0.0, dtype=jnp.float32),
            hidden_global_vars=pool.hidden_global_vars[idx],
            passed_thresholds_mask=pool.passed_thresholds_mask[idx],
            next_threshold_for_unit=pool.next_threshold_for_unit[idx],
            wall_pass_absolute_thresholds=pool.wall_pass_absolute_thresholds[idx],
            num_walls_passed=jnp.array(0, dtype=jnp.int32),
            walls_passed_reward=jnp.array(0.0, dtype=jnp.float32),
            fell_off_bridge=jnp.array(False),
        )
        return self.finalize_reset_state(state)

    def finalize_reset_state(self, state: MjxEnvState) -> MjxEnvState:
        state = self.enforce_inactive_units_state(state)
        return state._replace(
            progress=self.compute_progress(state),
            unit_positions=state.data.qpos[self.indices.qpos_indices[:, :3]],
        )

    def get_obs(self, state: MjxEnvState) -> MjxObsDict:
        qpos = state.data.qpos[self.indices.qpos_indices]
        qvel = state.data.qvel[self.indices.qvel_indices]
        free_joint_quats = qpos[:, 3:7]
        if self.quat_rot6d_representation:
            free_joint_quats = mjx_quat_to_rot6d(free_joint_quats, axis=-1)

        hinge_qpos = qpos[:, 7:]
        hinge_obs = jnp.stack([jnp.sin(hinge_qpos), jnp.cos(hinge_qpos)], axis=-1).reshape(self.num_units, -1)
        qpos_obs = jnp.concatenate([qpos[:, :3], free_joint_quats, hinge_obs], axis=1)

        is_active = state.connections.connections[:, :, 0] != -1
        connector_obs = jnp.stack(
            [
                (~is_active).astype(jnp.float32),
                is_active.astype(jnp.float32),
                jnp.where(is_active, jnp.sin(state.connections.twist_angles), 0.0),
                jnp.where(is_active, jnp.cos(state.connections.twist_angles), 0.0),
                jnp.where(is_active, state.connections.disconnect_potentials, 0.0),
            ],
            axis=-1,
        ).reshape(self.num_units, -1)

        obs_parts = [qpos_obs, qvel, connector_obs]
        if self.include_connectors_xpos_in_obs:
            obs_parts.append(state.data.xpos[self.indices.connector_body_indices].reshape(self.num_units, -1))
        if self.include_connectors_xquat_in_obs:
            conn_xquat = state.data.xquat[self.indices.connector_body_indices]
            if self.quat_rot6d_representation:
                conn_xquat = mjx_quat_to_rot6d(conn_xquat, axis=-1)
            obs_parts.append(conn_xquat.reshape(self.num_units, -1))

        obs: MjxObsDict = {
            "local_obs": jnp.concatenate(obs_parts, axis=1).astype(jnp.float32),
            "global_obs": jnp.empty((0,), dtype=jnp.float32),
            "hidden_local_vars": jnp.zeros((self.num_units, 0), dtype=jnp.float32),
            "hidden_global_vars": jnp.array([jnp.sum(state.units_active_mask)], dtype=jnp.float32),
        }
        if self.swarm.can_have_inactive_units:
            obs["agent_mask"] = state.units_active_mask
        return obs

    def apply_action(self, state: MjxEnvState, action: MjxActDict) -> MjxEnvState:
        active_mask = state.units_active_mask
        actuators = jnp.asarray(action["actuators"], dtype=jnp.float32)
        actuators = jnp.where(active_mask[:, None], actuators, 0.0)
        ctrl = jnp.zeros_like(state.data.ctrl).at[self.indices.ctrl_indices].set(actuators * self.actuator_strength)
        data = state.data.replace(ctrl=ctrl)

        connectors_action = jnp.asarray(action["connectors"]).astype(bool) & active_mask[:, None]
        currently_active = state.connections.connections[:, :, 0] != -1
        stayed_active = connectors_action & currently_active
        newly_activated = connectors_action & (~currently_active)
        newly_deactivated = (~connectors_action) & currently_active

        connections, successful_pairs = self.try_connect(data, state.connections, newly_activated)
        connections, num_deactivated = self.update_disconnects(connections, currently_active, newly_deactivated)
        data = data.replace(eq_active=self.build_eq_active(connections.connections, connections.twist_bins))

        num_successful = successful_pairs * 2
        num_newly_activated = jnp.sum(newly_activated.astype(jnp.int32))
        return state._replace(
            data=data,
            connections=connections,
            num_connectors_stayed_active=jnp.sum(stayed_active.astype(jnp.int32)),
            num_connectors_successfully_activated=num_successful,
            num_connectors_unsuccessfully_activated=num_newly_activated - num_successful,
            num_connectors_deactivated=num_deactivated,
        )

    def try_connect(
        self,
        data: mjx.Data,
        connections: MjxSwarmConnections,
        newly_activated_mask: jax.Array,
    ) -> tuple[MjxSwarmConnections, jax.Array]:
        flat_new = newly_activated_mask.reshape(-1)
        connector_body_ids = self.indices.connector_body_indices.reshape(-1)
        xpos = data.xpos[connector_body_ids]
        xmat = data.xmat[connector_body_ids].reshape(self.num_connectors, 3, 3)
        i = self.indices.pair_i
        j = self.indices.pair_j

        rel_pos = xpos[j] - xpos[i]
        mat_i = xmat[i]
        mat_j = xmat[j]
        z_i = mat_i[:, :, 2]
        z_j = mat_j[:, :, 2]
        distances = jnp.linalg.norm(rel_pos, axis=1)
        valid = (
            flat_new[i]
            & flat_new[j]
            & (self.indices.flat_units[i] != self.indices.flat_units[j])
            & (distances < self.connection_dist_threshold)
            & (jnp.sum(z_i * z_j, axis=1) <= self.connection_angle_threshold)
            & (jnp.sum(rel_pos * z_i, axis=1) >= 0.0)
        )
        sort_keys = jnp.where(valid, distances, jnp.inf)
        order = jnp.argsort(sort_keys)
        bin_width = 2.0 * jnp.pi / self.swarm.config.twist_bins

        def scan_pair(carry: tuple[jax.Array, MjxSwarmConnections, jax.Array], pair_order_idx: jax.Array):
            used, conn_state, successful_pairs = carry
            pair_idx = order[pair_order_idx]
            conn_i = i[pair_idx]
            conn_j = j[pair_idx]
            accept = jnp.isfinite(sort_keys[pair_idx]) & (~used[conn_i]) & (~used[conn_j])

            u1 = self.indices.flat_units[conn_i]
            c1 = self.indices.flat_connectors[conn_i]
            u2 = self.indices.flat_units[conn_j]
            c2 = self.indices.flat_connectors[conn_j]
            x1 = mat_i[pair_idx, :, 0]
            y1 = mat_i[pair_idx, :, 1]
            x2 = mat_j[pair_idx, :, 0]
            twist = jnp.arctan2(jnp.sum(x2 * y1), jnp.sum(x2 * x1))
            twist_bin = jnp.mod(jnp.floor(jnp.mod(twist, 2.0 * jnp.pi) / bin_width + 0.5), self.swarm.config.twist_bins)
            twist_bin = twist_bin.astype(jnp.int32)
            twist_angle = twist_bin.astype(jnp.float32) * bin_width

            old_1 = conn_state.connections[u1, c1]
            old_2 = conn_state.connections[u2, c2]
            new_connections = conn_state.connections.at[u1, c1].set(
                jnp.where(accept, jnp.array([u2, c2], dtype=jnp.int32), old_1)
            )
            new_connections = new_connections.at[u2, c2].set(
                jnp.where(accept, jnp.array([u1, c1], dtype=jnp.int32), old_2)
            )
            new_twist_bins = conn_state.twist_bins.at[u1, c1].set(jnp.where(accept, twist_bin, conn_state.twist_bins[u1, c1]))
            new_twist_bins = new_twist_bins.at[u2, c2].set(jnp.where(accept, twist_bin, conn_state.twist_bins[u2, c2]))
            new_twist_angles = conn_state.twist_angles.at[u1, c1].set(
                jnp.where(accept, twist_angle, conn_state.twist_angles[u1, c1])
            )
            new_twist_angles = new_twist_angles.at[u2, c2].set(
                jnp.where(accept, twist_angle, conn_state.twist_angles[u2, c2])
            )
            new_disconnect = conn_state.disconnect_potentials.at[u1, c1].set(
                jnp.where(accept, 0.0, conn_state.disconnect_potentials[u1, c1])
            )
            new_disconnect = new_disconnect.at[u2, c2].set(
                jnp.where(accept, 0.0, conn_state.disconnect_potentials[u2, c2])
            )
            used = used.at[conn_i].set(used[conn_i] | accept)
            used = used.at[conn_j].set(used[conn_j] | accept)
            return (
                used,
                MjxSwarmConnections(new_connections, new_twist_bins, new_twist_angles, new_disconnect),
                successful_pairs + accept.astype(jnp.int32),
            ), None

        initial = (jnp.zeros((self.num_connectors,), dtype=bool), connections, jnp.array(0, dtype=jnp.int32))
        (_, new_connections, successful_pairs), _ = jax.lax.scan(scan_pair, initial, jnp.arange(order.shape[0]))
        return new_connections, successful_pairs

    def update_disconnects(
        self,
        connections: MjxSwarmConnections,
        currently_active: jax.Array,
        newly_deactivated: jax.Array,
    ) -> tuple[MjxSwarmConnections, jax.Array]:
        partner = connections.connections
        partner_u = jnp.maximum(partner[:, :, 0], 0)
        partner_c = jnp.maximum(partner[:, :, 1], 0)
        partner_deactivated = currently_active & newly_deactivated[partner_u, partner_c]
        update = newly_deactivated.astype(jnp.float32) + partner_deactivated.astype(jnp.float32)
        stayed_active = currently_active & (update == 0.0)
        potentials = connections.disconnect_potentials + update
        potentials = jnp.where(stayed_active, jnp.maximum(0.0, potentials - 2.0), potentials)
        should_disconnect = currently_active & (
            (potentials >= self.disconnect_potential_threshold)
            | (potentials[partner_u, partner_c] >= self.disconnect_potential_threshold)
        )
        num_deactivated = jnp.sum(should_disconnect.astype(jnp.int32))
        cleared_connections = jnp.where(
            should_disconnect[:, :, None],
            -jnp.ones_like(connections.connections),
            connections.connections,
        )
        potentials = jnp.where(should_disconnect, 0.0, potentials)
        return (
            MjxSwarmConnections(
                connections=cleared_connections,
                twist_bins=connections.twist_bins,
                twist_angles=connections.twist_angles,
                disconnect_potentials=potentials,
            ),
            num_deactivated,
        )

    def build_eq_active(self, connections: jax.Array, twist_bins: jax.Array) -> jax.Array:
        flat_connections = connections.reshape(self.num_connectors, 2)
        flat_twist_bins = twist_bins.reshape(self.num_connectors)
        flat_u = self.indices.flat_units
        flat_c = self.indices.flat_connectors
        partner_u = jnp.maximum(flat_connections[:, 0], 0)
        partner_c = jnp.maximum(flat_connections[:, 1], 0)
        valid = flat_connections[:, 0] >= 0
        eq_idx = self.indices.eq_indices[flat_u, flat_c, partner_u, partner_c, flat_twist_bins]
        valid = valid & (eq_idx >= 0)
        safe_eq_idx = jnp.where(valid, eq_idx, 0)
        eq_active = jnp.zeros((self.model.neq,), dtype=jnp.uint8)
        return eq_active.at[safe_eq_idx].max(valid.astype(jnp.uint8))

    def enforce_inactive_units_state(self, state: MjxEnvState) -> MjxEnvState:
        if not self.swarm.can_have_inactive_units:
            return state

        inactive = ~state.units_active_mask

        def enforce(_: None) -> MjxEnvState:
            current_qpos_xyz = state.data.qpos[self.indices.qpos_indices[:, :3]]
            qpos = state.data.qpos.at[self.indices.qpos_indices[:, :3]].set(
                jnp.where(
                    inactive[:, None],
                    jnp.asarray(self.inactive_unit_positions, dtype=jnp.float32),
                    current_qpos_xyz,
                )
            )
            current_qvel = state.data.qvel[self.indices.qvel_indices]
            qvel = state.data.qvel.at[self.indices.qvel_indices].set(jnp.where(inactive[:, None], 0.0, current_qvel))
            data = mjx.forward(self.mjx_model, state.data.replace(qpos=qpos, qvel=qvel))
            return state._replace(data=data)

        return jax.lax.cond(jnp.any(inactive), enforce, lambda _: state, operand=None)

    def evaluate_step(self, state: MjxEnvState, action: MjxActDict) -> tuple[MjxEnvState, jax.Array, jax.Array]:
        state = self.enforce_inactive_units_state(state)
        old_progress = state.progress
        new_progress = self.compute_progress(state)
        progress_reward = new_progress - old_progress
        guidance_reward, state = self.compute_guidance_reward(state, action)
        weighted_progress_reward = progress_reward * self.reward_weights["progress_reward_weight"]
        weighted_guidance_reward = guidance_reward * self.reward_weights["guidance_reward_weight"]
        state = state._replace(
            progress=new_progress,
            weighted_progress_reward=weighted_progress_reward,
            weighted_guidance_reward=weighted_guidance_reward,
        )
        return state, weighted_progress_reward + weighted_guidance_reward, jnp.array(False)

    def compute_guidance_reward(self, state: MjxEnvState, action: MjxActDict) -> tuple[jax.Array, MjxEnvState]:
        rw = self.reward_weights
        active = state.units_active_mask
        active_f = active.astype(jnp.float32)
        active_count = jnp.maximum(jnp.sum(active_f), 1.0)
        reward = jnp.array(0.0, dtype=jnp.float32)

        qvel = state.data.qvel[self.indices.qvel_indices]
        if qvel.shape[1] > 6:
            hinge_qvel = jnp.abs(qvel[:, 6:])
            hinge_threshold = rw["hinge_qvel_magnitude_reward_threshold"]
            hinge_qvel = jnp.where(hinge_qvel <= hinge_threshold, 0.0, hinge_qvel)
            hinge_denom = jnp.maximum(active_count * hinge_qvel.shape[1], 1.0)
            hinge_qvel_magnitude = jnp.sum(jnp.where(active[:, None], hinge_qvel, 0.0)) / hinge_denom
        else:
            hinge_qvel_magnitude = jnp.array(0.0, dtype=jnp.float32)
        reward += hinge_qvel_magnitude * rw["hinge_qvel_magnitude_reward_weight"]

        connection_mask = state.connections.connections[:, :, 0] != -1
        units_without_connections = active & (~jnp.any(connection_mask, axis=1))
        reward += (
            jnp.sum(units_without_connections.astype(jnp.float32))
            / active_count
            * rw["units_without_connections_reward_weight"]
        )

        connection_targets = state.connections.connections[:, :, 0]
        sorted_targets = jnp.sort(connection_targets, axis=1)
        double_connected = jnp.any(
            (sorted_targets[:, 1:] == sorted_targets[:, :-1]) & (sorted_targets[:, 1:] != -1),
            axis=1,
        )
        reward += jnp.sum((double_connected & active).astype(jnp.float32)) * rw[
            "units_with_double_connection_reward_weight"
        ]

        prev_unit_positions = state.unit_positions
        unit_positions = state.data.qpos[self.indices.qpos_indices[:, :3]]
        movement = jnp.linalg.norm(unit_positions - prev_unit_positions, axis=1)
        reward += jnp.sum(jnp.where(active, movement, 0.0)) / active_count * rw["movement_reward_weight"]
        reward += jnp.sum(jnp.where(active, unit_positions[:, 2], 0.0)) / active_count * rw["height_reward_weight"]

        connectors_reward = (
            state.num_connectors_stayed_active * rw["connectors_stayed_active_reward_weight"]
            + state.num_connectors_successfully_activated * rw["connectors_successfully_activated_reward_weight"]
            + state.num_connectors_unsuccessfully_activated * rw["connectors_unsuccessfully_activated_reward_weight"]
            + state.num_connectors_deactivated * rw["connectors_deactivated_reward_weight"]
        )
        if self.average_connectors_reward:
            connectors_reward = connectors_reward / jnp.maximum(active_count * self.limbs_per_unit, 1.0)
        reward += connectors_reward
        return reward, state._replace(unit_positions=unit_positions)

    def get_obs_space(self) -> spaces.Dict:
        quat_dim = 6 if self.quat_rot6d_representation else 4
        hinge_qpos_dim = int(self.indices.qpos_indices.shape[1]) - 7
        local_obs_dim = 3 + quat_dim + 2 * hinge_qpos_dim
        local_obs_dim += int(self.indices.qvel_indices.shape[1])
        local_obs_dim += self.limbs_per_unit * 5
        if self.include_connectors_xpos_in_obs:
            local_obs_dim += self.limbs_per_unit * 3
        if self.include_connectors_xquat_in_obs:
            local_obs_dim += self.limbs_per_unit * quat_dim
        obs_space = spaces.Dict(
            {
                "local_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_units, local_obs_dim), dtype=np.float32),
                "global_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self._global_obs_size(),), dtype=np.float32),
                "hidden_local_vars": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=self._hidden_local_vars_shape(),
                    dtype=np.float32,
                ),
                "hidden_global_vars": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._hidden_global_vars_size(),),
                    dtype=np.float32,
                ),
            }
        )
        if self.swarm.can_have_inactive_units:
            obs_space["agent_mask"] = spaces.MultiBinary((self.num_units,))
        return obs_space

    def _global_obs_size(self) -> int:
        return 0

    def _hidden_local_vars_shape(self) -> tuple[int, int]:
        return self.num_units, 0

    def _hidden_global_vars_size(self) -> int:
        return 1

    def get_action_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "actuators": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=tuple(self.indices.ctrl_indices.shape),
                    dtype=np.float32,
                ),
                "connectors": spaces.MultiBinary((self.num_units, self.limbs_per_unit)),
            }
        )

    def _build_reset_pool(self) -> MjxResetPool:
        samples = [self._make_reset_sample() for _ in range(self.reset_pool_size)]
        return MjxResetPool(
            qpos=jnp.asarray(np.stack([sample.qpos for sample in samples]), dtype=jnp.float32),
            qvel=jnp.asarray(np.stack([sample.qvel for sample in samples]), dtype=jnp.float32),
            ctrl=jnp.asarray(np.stack([sample.ctrl for sample in samples]), dtype=jnp.float32),
            mocap_pos=jnp.asarray(np.stack([sample.mocap_pos for sample in samples]), dtype=jnp.float32),
            connections=MjxSwarmConnections(
                connections=jnp.asarray(np.stack([sample.swarm_reset.connections for sample in samples]), dtype=jnp.int32),
                twist_bins=jnp.asarray(np.stack([sample.swarm_reset.twist_bins for sample in samples]), dtype=jnp.int32),
                twist_angles=jnp.asarray(np.stack([sample.swarm_reset.twist_angles for sample in samples]), dtype=jnp.float32),
                disconnect_potentials=jnp.asarray(
                    np.stack([sample.swarm_reset.disconnect_potentials for sample in samples]),
                    dtype=jnp.float32,
                ),
            ),
            units_active_mask=jnp.asarray(
                np.stack([sample.swarm_reset.units_active_mask for sample in samples]),
                dtype=bool,
            ),
            hidden_global_vars=jnp.asarray(
                np.stack([sample.hidden_global_vars for sample in samples]),
                dtype=jnp.float32,
            ),
            passed_thresholds_mask=jnp.asarray(
                np.stack([sample.passed_thresholds_mask for sample in samples]),
                dtype=bool,
            ),
            next_threshold_for_unit=jnp.asarray(
                np.stack([sample.next_threshold_for_unit for sample in samples]),
                dtype=jnp.int32,
            ),
            wall_pass_absolute_thresholds=jnp.asarray(
                np.stack([sample.wall_pass_absolute_thresholds for sample in samples]),
                dtype=jnp.float32,
            ),
        )

    def _make_reset_sample(self) -> _ResetSample:
        qpos = np.array(self.model.qpos0, dtype=np.float32, copy=True)
        qvel = np.zeros((self.model.nv,), dtype=np.float32)
        ctrl = np.zeros((self.model.nu,), dtype=np.float32)
        mocap_pos = np.zeros((self.model.nmocap, 3), dtype=np.float32)
        swarm_start_location = self.get_swarm_start_location()
        swarm_reset = self.swarm.make_reset(self.rng, swarm_start_location, self.inactive_unit_positions)
        qpos_indices = np.asarray(self.indices.qpos_indices)
        for unit_idx in range(self.num_units):
            qpos[qpos_indices[unit_idx, :3]] = swarm_reset.positions[unit_idx]
            qpos[qpos_indices[unit_idx, 3:7]] = swarm_reset.quats[unit_idx]
        return self._augment_reset_sample(
            _ResetSample(
                qpos=qpos,
                qvel=qvel,
                ctrl=ctrl,
                mocap_pos=mocap_pos,
                swarm_reset=swarm_reset,
                hidden_global_vars=np.zeros((1,), dtype=np.float32),
                passed_thresholds_mask=np.zeros((self.num_units, 0), dtype=bool),
                next_threshold_for_unit=np.zeros((self.num_units,), dtype=np.int32),
                wall_pass_absolute_thresholds=np.zeros((0,), dtype=np.float32),
            ),
            swarm_start_location,
        )

    def _augment_reset_sample(self, sample: _ResetSample, swarm_start_location: np.ndarray) -> _ResetSample:
        return sample

    def _configure_model(self, model: mujoco.MjModel) -> None:
        if self.friction is None:
            return
        model.geom_friction[:] = np.asarray(self.friction, dtype=float)
        sliding = float(self.friction[0])
        if self.force_elliptic_cone or (
            sliding >= HIGH_FRICTION_SLIDING_THRESHOLD
            and int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
        ):
            model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

    def _make_settle_model(self) -> mjx.Model:
        effective_timestep = float(self.model.opt.timestep) * self.reset_settle_timestep_scale
        if effective_timestep == float(self.model.opt.timestep):
            return self.mjx_model
        return self.mjx_model.replace(opt=self.mjx_model.opt.replace(timestep=effective_timestep))

    def _configure_spec(self, spec: mujoco.MjSpec) -> None:
        self._add_same_unit_long_segment_excludes(spec)
        self._set_contact_limit_numerics(spec)

    def _add_same_unit_long_segment_excludes(self, spec: mujoco.MjSpec) -> None:
        unit_config = self.swarm.config.unit_config
        for unit_prefix in self.swarm.config.unit_prefixes:
            main_body_name = _prefixed_body_name(unit_prefix, "-main_body")
            long_segment_body_names: list[str] = []
            for limb_idx, limb_config in enumerate(unit_config):
                segment_names = mjx_get_limb_segment_names(limb_idx, limb_config)
                long_segment_body_names.extend(
                    _prefixed_body_name(unit_prefix, segment_name) for segment_name in segment_names[1:]
                )

            for long_segment_body_name in long_segment_body_names:
                exclude = spec.add_exclude()
                exclude.bodyname1 = main_body_name
                exclude.bodyname2 = long_segment_body_name

            for body_idx1, body_name1 in enumerate(long_segment_body_names[:-1]):
                for body_name2 in long_segment_body_names[body_idx1 + 1 :]:
                    exclude = spec.add_exclude()
                    exclude.bodyname1 = body_name1
                    exclude.bodyname2 = body_name2

    def _set_contact_limit_numerics(self, spec: mujoco.MjSpec) -> None:
        contact_limit = self.num_units * self.limbs_per_unit * MJX_CONTACT_LIMIT_PER_LIMB
        _set_numeric(spec, "max_geom_pairs", contact_limit)
        _set_numeric(spec, "max_contact_points", contact_limit)

    def _compute_reset_settle_steps(self) -> int:
        if self.reset_settle_time <= 0:
            return 0
        effective_timestep = float(self.model.opt.timestep) * self.reset_settle_timestep_scale
        return int(math.ceil(self.reset_settle_time / effective_timestep))

    def _build_indices(self) -> MjxModelIndices:
        unit_prefixes = self.swarm.config.unit_prefixes
        qpos_indices = np.array([_qpos_indices_for_prefix(self.model, prefix) for prefix in unit_prefixes], dtype=np.int32)
        qvel_indices = np.array([_dof_indices_for_prefix(self.model, prefix) for prefix in unit_prefixes], dtype=np.int32)
        ctrl_indices = np.array([_ctrl_indices_for_prefix(self.model, prefix) for prefix in unit_prefixes], dtype=np.int32)
        connector_body_indices = np.zeros((self.num_units, self.limbs_per_unit), dtype=np.int32)
        for unit in range(self.num_units):
            for connector in range(self.limbs_per_unit):
                connector_body_indices[unit, connector] = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.swarm.config.get_connector_name(unit, connector),
                )

        eq_indices = np.full(
            (self.num_units, self.limbs_per_unit, self.num_units, self.limbs_per_unit, self.swarm.config.twist_bins),
            -1,
            dtype=np.int32,
        )
        for unit1 in range(self.num_units - 1):
            for unit2 in range(unit1 + 1, self.num_units):
                for conn1 in range(self.limbs_per_unit):
                    for conn2 in range(self.limbs_per_unit):
                        for twist_bin in range(self.swarm.config.twist_bins):
                            eq_id = mujoco.mj_name2id(
                                self.model,
                                mujoco.mjtObj.mjOBJ_EQUALITY,
                                self.swarm.config.get_eq_name(unit1, conn1, unit2, conn2, twist_bin),
                            )
                            eq_indices[unit1, conn1, unit2, conn2, twist_bin] = eq_id
                            eq_indices[unit2, conn2, unit1, conn1, twist_bin] = eq_id

        flat_units = np.repeat(np.arange(self.num_units, dtype=np.int32), self.limbs_per_unit)
        flat_connectors = np.tile(np.arange(self.limbs_per_unit, dtype=np.int32), self.num_units)
        pair_i, pair_j = np.triu_indices(self.num_connectors, k=1)
        return MjxModelIndices(
            qpos_indices=jnp.asarray(qpos_indices),
            qvel_indices=jnp.asarray(qvel_indices),
            ctrl_indices=jnp.asarray(ctrl_indices),
            connector_body_indices=jnp.asarray(connector_body_indices),
            eq_indices=jnp.asarray(eq_indices),
            pair_i=jnp.asarray(pair_i, dtype=jnp.int32),
            pair_j=jnp.asarray(pair_j, dtype=jnp.int32),
            flat_units=jnp.asarray(flat_units),
            flat_connectors=jnp.asarray(flat_connectors),
        )

    @staticmethod
    def _generate_inactive_unit_positions(swarm: MjxBaseSwarm, inactive_area_location: np.ndarray) -> np.ndarray:
        num_units = swarm.config.num_units
        num_units_sqrt = int(math.ceil(math.sqrt(num_units)))
        unit_spacing = swarm.max_unit_extent * 3.0
        inactive_positions = np.zeros((num_units, 3), dtype=float)
        inactive_positions[:, 2] = swarm.max_unit_extent
        for idx in range(num_units):
            inactive_positions[idx, 0] = (idx // num_units_sqrt) * unit_spacing
            inactive_positions[idx, 1] = (idx % num_units_sqrt) * unit_spacing
        inactive_positions[:, :2] -= inactive_positions[:, :2].mean(axis=0)
        return inactive_area_location + inactive_positions
