import abc
from typing import Any, Iterable

import mujoco
import numpy as np
from mujoco import MjsBody

from swarmbots.mj_env.quat_rot6d import quat_to_rot6d
from swarmbots.mj_env.scenarios.base_scenario import BaseScenario, SwarmObsDict
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections


class PayloadScenario(BaseScenario, abc.ABC):

    def __init__(
            self,
            swarm: BaseSwarm,
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
            reset_settle_time: int = 0,
            reset_settle_timestep_scale: float = 1.0,
            inactive_area_location: Iterable[float] | None = None,
            seed: int | None = None,
            _reset_in_init: bool = True,
    ) -> None:
        self.payload_type = payload_type
        self.payload_body_id: int = -1
        self.payload_size = payload_size
        self.payload_mass = payload_mass
        self.payload_start_location_offset = payload_start_location_offset

        super().__init__(
            swarm=swarm,
            actuator_strength=actuator_strength,
            progress_reward_weight=progress_reward_weight,
            guidance_reward_weight=guidance_reward_weight,
            actuators_activation_reward_weight=actuators_activation_reward_weight,
            actuators_activation_reward_power=actuators_activation_reward_power,
            units_without_connections_reward_weight=units_without_connections_reward_weight,
            units_with_double_connection_reward_weight=units_with_double_connection_reward_weight,
            movement_reward_weight=movement_reward_weight,
            height_reward_weight=height_reward_weight,
            connectors_stayed_active_reward_weight=connectors_stayed_active_reward_weight,
            connectors_successfully_activated_reward_weight=connectors_successfully_activated_reward_weight,
            connectors_unsuccessfully_activated_reward_weight=connectors_unsuccessfully_activated_reward_weight,
            connectors_deactivated_reward_weight=connectors_deactivated_reward_weight,
            average_connectors_reward=average_connectors_reward,
            seed=seed,
            include_connectors_xpos_in_obs=include_connectors_xpos_in_obs,
            include_connectors_xquat_in_obs=include_connectors_xquat_in_obs,
            quat_rot6d_representation=quat_rot6d_representation,
            friction=friction,
            connection_dist_threshold=connection_dist_threshold,
            connection_angle_threshold=connection_angle_threshold,
            disconnect_potential_threshold=disconnect_potential_threshold,
            force_elliptic_cone=force_elliptic_cone,
            reset_settle_time=reset_settle_time,
            reset_settle_timestep_scale=reset_settle_timestep_scale,
            inactive_area_location=inactive_area_location,
            _reset_in_init=_reset_in_init
        )

        self._dummy_state, self._dummy_connections = self.reset_scenario(self.dummy_model, self.dummy_data)

    def get_settings(self) -> dict[str, Any]:
        settings = super().get_settings()
        settings.update({
            'payload_type': self.payload_type,
            'payload_size': self.payload_size,
            'payload_mass': self.payload_mass,
            'payload_start_location_offset': self.payload_start_location_offset,
        })
        return settings


    def _maybe_add_payload_spec(self, spec: mujoco.MjSpec):
        if self.payload_type is not None:
            worldbody: mujoco.MjsBody = spec.worldbody

            payload_start_position = (
                    np.array(self.get_swarm_start_location()) + np.array(self.payload_start_location_offset)
            )
            payload_body: MjsBody = worldbody.add_body(name='Payload', pos=payload_start_position)

            payload_geom_type = {
                'sphere': mujoco.mjtGeom.mjGEOM_SPHERE,
                'box': mujoco.mjtGeom.mjGEOM_BOX,
                'cylinder': mujoco.mjtGeom.mjGEOM_CYLINDER,
                'capsule': mujoco.mjtGeom.mjGEOM_CAPSULE,
                'ellipsoid': mujoco.mjtGeom.mjGEOM_ELLIPSOID
            }[self.payload_type]

            payload_rgba = [0.8, 0.3, 0.3, 0.9]
            payload_body.add_geom(
                type=payload_geom_type,
                size=self.payload_size,
                mass=self.payload_mass,
                rgba=payload_rgba
            )
            payload_body.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)

    def reset_scenario(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[dict, SwarmConnections]:
        state, connections = super().reset_scenario(model, data)

        if self.payload_type is None:
            self.payload_body_id = -1
        else:
            self.payload_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Payload")
            if self.payload_body_id == -1:
                raise RuntimeError("payload_type is set, but body 'Payload' was not found in the model")

        return state, connections

    def get_obs(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
            connections: SwarmConnections
    ) -> SwarmObsDict:
        obs = super().get_obs(model, data, state, connections)

        if self.payload_type is not None:
            payload_quat = data.xquat[self.payload_body_id]
            if self.quat_rot6d_representation:
                payload_quat = quat_to_rot6d(payload_quat, axis=-1)
            obs['global_obs'] = np.concatenate([data.xpos[self.payload_body_id], payload_quat])

        return obs
