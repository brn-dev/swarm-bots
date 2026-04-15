from typing import Iterable

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from swarmbots.mjx_env.mjx_float_or_dist_params import (
    MjxBoundedDistParams,
    MjxFloatOrBoundedDistParams,
    mjx_eval_fodp,
)
from swarmbots.mjx_env.scenarios.mjx_base_scenario import MjxActuatorsActivationRewardType, MjxImplConfig, _ResetSample
from swarmbots.mjx_env.scenarios.mjx_payload_scenario import MjxPayloadScenario
from swarmbots.mjx_env.swarm.mjx_base_swarm import MjxBaseSwarm
from swarmbots.mjx_env.types import MjxActDict, MjxEnvState, MjxObsDict


class MjxBridgeScenario(MjxPayloadScenario):
    def __init__(
        self,
        swarm: MjxBaseSwarm,
        payload_type: None | str,
        payload_size: Iterable[float] = (0.2, 0.2, 0.2),
        payload_mass: float = 5.0,
        payload_start_location_offset: Iterable[float] = (0, 1, 0),
        street_width: float = 6.0,
        bridge_width: float = 1.0,
        bridge_length: float = 4.0,
        bridge_x: MjxFloatOrBoundedDistParams = 0.0,
        platform_length: float = 4.0,
        platform_height: float = 0.2,
        fall_z_threshold: float = -1.0,
        fell_off_bridge_reward: float = -1.0,
        actuator_strength: float = 8.0,
        connection_dist_threshold: float = 0.1,
        connection_angle_threshold: float = -0.5,
        disconnect_potential_threshold: float = 5.0,
        friction: float | Iterable[float] | None = None,
        force_elliptic_cone: bool = False,
        progress_reward_weight: float = 1.0,
        guidance_reward_weight: float = 1.0,
        hinge_qvel_magnitude_reward_weight: float = 0.0,
        hinge_qvel_magnitude_reward_threshold: float = 0.0,
        units_without_connections_reward_weight: float = 0.0,
        units_with_double_connection_reward_weight: float = 0.0,
        movement_reward_weight: float = 0.0,
        height_reward_weight: float = 0.0,
        connectors_stayed_active_reward_weight: float = 0.0,
        connectors_successfully_activated_reward_weight: float = 0.0,
        connectors_unsuccessfully_activated_reward_weight: float = 0.0,
        connectors_deactivated_reward_weight: float = 0.0,
        average_connectors_reward: bool = True,
        include_connectors_xpos_in_obs: bool = True,
        include_connectors_xquat_in_obs: bool = False,
        quat_rot6d_representation: bool = True,
        reset_settle_time: float = 0,
        reset_settle_timestep_scale: float = 1.0,
        seed: int | None = None,
        reset_pool_size: int = 256,
        mjx_impl: MjxImplConfig = None,
    ) -> None:
        self.street_width = float(street_width)
        self.bridge_width = float(bridge_width)
        self.bridge_length = float(bridge_length)
        self.platform_length = float(platform_length)
        self.platform_height = float(platform_height)
        self.fall_z_threshold = float(fall_z_threshold)
        self.fell_off_bridge_reward = float(fell_off_bridge_reward)
        self.side_wall_x = self.street_width / 2.0
        self.bridge_x_param = bridge_x
        self.platform1_min_y = -50.0
        self.platform1_max_y = self.platform_length / 2.0
        self.platform1_center_y = (self.platform1_min_y + self.platform1_max_y) / 2.0
        self.platform1_half_length = (self.platform1_max_y - self.platform1_min_y) / 2.0
        self.platform2_center_y = self.platform_length + self.bridge_length
        self.bridge_center_y = (self.platform_length / 2.0) + (self.bridge_length / 2.0)
        self._validate_bridge_x_param()
        super().__init__(
            swarm=swarm,
            payload_type=payload_type,
            payload_size=payload_size,
            payload_mass=payload_mass,
            payload_start_location_offset=payload_start_location_offset,
            actuator_strength=actuator_strength,
            connection_dist_threshold=connection_dist_threshold,
            connection_angle_threshold=connection_angle_threshold,
            disconnect_potential_threshold=disconnect_potential_threshold,
            friction=friction,
            force_elliptic_cone=force_elliptic_cone,
            progress_reward_weight=progress_reward_weight,
            guidance_reward_weight=guidance_reward_weight,
            hinge_qvel_magnitude_reward_weight=hinge_qvel_magnitude_reward_weight,
            hinge_qvel_magnitude_reward_threshold=hinge_qvel_magnitude_reward_threshold,
            units_without_connections_reward_weight=units_without_connections_reward_weight,
            units_with_double_connection_reward_weight=units_with_double_connection_reward_weight,
            movement_reward_weight=movement_reward_weight,
            height_reward_weight=height_reward_weight,
            connectors_stayed_active_reward_weight=connectors_stayed_active_reward_weight,
            connectors_successfully_activated_reward_weight=connectors_successfully_activated_reward_weight,
            connectors_unsuccessfully_activated_reward_weight=connectors_unsuccessfully_activated_reward_weight,
            connectors_deactivated_reward_weight=connectors_deactivated_reward_weight,
            average_connectors_reward=average_connectors_reward,
            include_connectors_xpos_in_obs=include_connectors_xpos_in_obs,
            include_connectors_xquat_in_obs=include_connectors_xquat_in_obs,
            quat_rot6d_representation=quat_rot6d_representation,
            reset_settle_time=reset_settle_time,
            reset_settle_timestep_scale=reset_settle_timestep_scale,
            inactive_area_location=[-street_width * 1.5, 0, 0.1],
            seed=seed,
            reset_pool_size=reset_pool_size,
            mjx_impl=mjx_impl,
        )

    def _validate_bridge_x_param(self) -> None:
        min_x = -self.side_wall_x + self.bridge_width / 2.0
        max_x = self.side_wall_x - self.bridge_width / 2.0
        if isinstance(self.bridge_x_param, MjxBoundedDistParams):
            if self.bridge_x_param.low < min_x or self.bridge_x_param.high > max_x:
                raise ValueError(f"Expected bridge_x bounds within [{min_x}, {max_x}], got {self.bridge_x_param}")
            return
        value = float(self.bridge_x_param)
        if value < min_x or value > max_x:
            raise ValueError(f"Expected bridge_x within [{min_x}, {max_x}], got {value}")

    def _create_scenario_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody = spec.worldbody
        worldbody.add_light(pos=[0, 0, 100], dir=[0, 0, -1])
        worldbody.add_light(pos=[0, 100, 100], dir=[-1, -1, -1])
        for x in (self.side_wall_x, -self.side_wall_x):
            worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 100, 5], rgba=[0.3, 0.4, 0.5, 0.1], pos=[x, 0, 0])

        platform_top_z = 0.0
        platform_center_z = platform_top_z - self.platform_height / 2.0
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.street_width / 2.0, self.platform1_half_length, self.platform_height / 2.0],
            rgba=[0.45, 0.45, 0.5, 1],
            pos=[0, self.platform1_center_y, platform_center_z],
        )
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.street_width / 2.0, self.platform_length / 2.0, self.platform_height / 2.0],
            rgba=[0.45, 0.45, 0.5, 1],
            pos=[0, self.platform2_center_y, platform_center_z],
        )
        bridge = worldbody.add_body(name="Bridge", mocap=True, pos=[0, 0, 0])
        bridge.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.bridge_width / 2.0, self.bridge_length / 2.0, self.platform_height / 2.0],
            rgba=[0.7, 0.6, 0.3, 1],
        )
        self._maybe_add_payload_spec(spec)
        return spec

    def _augment_reset_sample(self, sample: _ResetSample, swarm_start_location: np.ndarray) -> _ResetSample:
        sample = super()._augment_reset_sample(sample, swarm_start_location)
        bridge_x = float(mjx_eval_fodp(self.bridge_x_param, self.rng))
        mocap_pos = sample.mocap_pos.copy()
        bridge_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Bridge")
        mocap_pos[self.model.body_mocapid[bridge_body_id]] = [
            bridge_x,
            self.bridge_center_y,
            -(self.platform_height / 2.0),
        ]
        return sample._replace(mocap_pos=mocap_pos, hidden_global_vars=np.asarray([bridge_x], dtype=np.float32))

    def get_obs(self, state: MjxEnvState) -> MjxObsDict:
        obs = super().get_obs(state)
        obs["hidden_global_vars"] = state.hidden_global_vars
        return obs

    def compute_progress(self, state: MjxEnvState) -> jax.Array:
        if self.payload_type is None:
            unit_y = state.data.qpos[self.indices.qpos_indices[:, 1]]
            active = state.units_active_mask
            return jnp.sum(jnp.where(active, unit_y, 0.0)) / jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)
        return state.data.xpos[self.payload_body_id, 1]

    def evaluate_step(self, state: MjxEnvState, action: MjxActDict) -> tuple[MjxEnvState, jax.Array, jax.Array]:
        state, reward, terminated = super().evaluate_step(state, action)
        unit_z = state.data.qpos[self.indices.qpos_indices[:, 2]]
        active_unit_z = jnp.where(state.units_active_mask, unit_z, jnp.inf)
        fell = jnp.any(active_unit_z < self.fall_z_threshold)
        if self.payload_type is not None:
            fell = fell | (state.data.xpos[self.payload_body_id, 2] < self.fall_z_threshold)
        reward = reward + jnp.where(fell, self.fell_off_bridge_reward, 0.0)
        return state._replace(fell_off_bridge=fell), reward, terminated | fell
