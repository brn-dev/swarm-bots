import abc
from typing import Iterable

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from swarmbots.mjx_env.mjx_float_or_dist_params import MjxFloatOrDistParams
from swarmbots.mjx_env.mjx_quat import mjx_quat_to_rot6d
from swarmbots.mjx_env.scenarios.mjx_base_scenario import (
    MjxImplConfig,
    MjxActuatorsActivationRewardType,
    MjxBaseScenario,
    _ResetSample,
)
from swarmbots.mjx_env.swarm.mjx_base_swarm import MjxBaseSwarm
from swarmbots.mjx_env.types import MjxEnvState, MjxObsDict


class MjxPayloadScenario(MjxBaseScenario, abc.ABC):
    def __init__(
        self,
        swarm: MjxBaseSwarm,
        payload_type: None | str,
        payload_size: Iterable[float] = (0.2, 0.2, 0.2),
        payload_mass: float = 5.0,
        payload_start_location_offset: Iterable[float] = (0, 1, 0),
        actuator_strength: float = 8.0,
        connection_dist_threshold: float = 0.1,
        connection_angle_threshold: float = -0.5,
        disconnect_potential_threshold: float = 5.0,
        friction: float | Iterable[float] | None = None,
        force_elliptic_cone: bool = False,
        progress_reward_weight: float = 1.0,
        guidance_reward_weight: float = 1.0,
        actuators_activation_reward_weight: float = 0.0,
        actuators_activation_reward_power: int = 8,
        actuators_activation_reward_threshold: float = 0.0,
        actuators_activation_reward_type: MjxActuatorsActivationRewardType = MjxActuatorsActivationRewardType.MONOMIAL,
        actuators_activation_reward_clip: float = 20.0,
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
        swarm_start_x: MjxFloatOrDistParams = 0.0,
        swarm_start_y: MjxFloatOrDistParams = 0.0,
        inactive_area_location: Iterable[float] | None = None,
        seed: int | None = None,
        reset_pool_size: int = 256,
        mjx_impl: MjxImplConfig = None,
    ) -> None:
        self.payload_type = payload_type
        self.payload_size = tuple(float(x) for x in payload_size)
        self.payload_mass = float(payload_mass)
        self.payload_start_location_offset = np.asarray(payload_start_location_offset, dtype=float)
        self.payload_body_id = -1
        self.payload_qpos_adr = -1
        self.payload_dof_adr = -1
        super().__init__(
            swarm=swarm,
            actuator_strength=actuator_strength,
            progress_reward_weight=progress_reward_weight,
            guidance_reward_weight=guidance_reward_weight,
            actuators_activation_reward_weight=actuators_activation_reward_weight,
            actuators_activation_reward_power=actuators_activation_reward_power,
            actuators_activation_reward_threshold=actuators_activation_reward_threshold,
            actuators_activation_reward_type=actuators_activation_reward_type,
            actuators_activation_reward_clip=actuators_activation_reward_clip,
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
            connection_dist_threshold=connection_dist_threshold,
            connection_angle_threshold=connection_angle_threshold,
            disconnect_potential_threshold=disconnect_potential_threshold,
            friction=friction,
            force_elliptic_cone=force_elliptic_cone,
            reset_settle_time=reset_settle_time,
            reset_settle_timestep_scale=reset_settle_timestep_scale,
            swarm_start_x=swarm_start_x,
            swarm_start_y=swarm_start_y,
            inactive_area_location=inactive_area_location,
            seed=seed,
            reset_pool_size=reset_pool_size,
            mjx_impl=mjx_impl,
        )
        if self.payload_type is not None:
            self.payload_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Payload")
            if self.payload_body_id == -1:
                raise RuntimeError("payload_type is set, but body 'Payload' was not found")
            payload_joint_adr = int(self.model.body_jntadr[self.payload_body_id])
            self.payload_qpos_adr = int(self.model.jnt_qposadr[payload_joint_adr])
            self.payload_dof_adr = int(self.model.jnt_dofadr[payload_joint_adr])

    def _maybe_add_payload_spec(self, spec: mujoco.MjSpec) -> None:
        if self.payload_type is None:
            return
        payload_body = spec.worldbody.add_body(name="Payload", pos=[0, 0, 0])
        payload_geom_type = {
            "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
            "box": mujoco.mjtGeom.mjGEOM_BOX,
            "cylinder": mujoco.mjtGeom.mjGEOM_CAPSULE,
            "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
            "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        }[self.payload_type]
        payload_body.add_geom(
            type=payload_geom_type,
            size=self.payload_size,
            mass=self.payload_mass,
            rgba=[0.8, 0.3, 0.3, 0.9],
        )
        payload_body.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)

    def _augment_reset_sample(self, sample: _ResetSample, swarm_start_location: np.ndarray) -> _ResetSample:
        if self.payload_type is None:
            return sample
        if self.payload_qpos_adr < 0:
            self._resolve_payload_indices()
        qpos = sample.qpos.copy()
        qvel = sample.qvel.copy()
        payload_pos = swarm_start_location + self.payload_start_location_offset
        qpos[self.payload_qpos_adr:self.payload_qpos_adr + 3] = payload_pos
        qpos[self.payload_qpos_adr + 3:self.payload_qpos_adr + 7] = [1, 0, 0, 0]
        qvel[self.payload_dof_adr:self.payload_dof_adr + 6] = 0.0
        return sample._replace(qpos=qpos, qvel=qvel)

    def _resolve_payload_indices(self) -> None:
        self.payload_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Payload")
        if self.payload_body_id == -1:
            raise RuntimeError("payload_type is set, but body 'Payload' was not found")
        payload_joint_adr = int(self.model.body_jntadr[self.payload_body_id])
        self.payload_qpos_adr = int(self.model.jnt_qposadr[payload_joint_adr])
        self.payload_dof_adr = int(self.model.jnt_dofadr[payload_joint_adr])

    def get_obs(self, state: MjxEnvState) -> MjxObsDict:
        obs = super().get_obs(state)
        if self.payload_type is not None:
            payload_quat = state.data.xquat[self.payload_body_id]
            if self.quat_rot6d_representation:
                payload_quat = mjx_quat_to_rot6d(payload_quat, axis=-1)
            obs["global_obs"] = jnp.concatenate([state.data.xpos[self.payload_body_id], payload_quat]).astype(jnp.float32)
        return obs

    def _global_obs_size(self) -> int:
        if self.payload_type is None:
            return 0
        return 3 + (6 if self.quat_rot6d_representation else 4)
