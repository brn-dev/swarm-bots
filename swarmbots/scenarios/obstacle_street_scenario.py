from typing import Literal, Any, Iterable

import mujoco
import numpy as np
from mujoco import MjsBody

from swarmbots.scenarios.base_scenario import BaseScenario, SwarmObsDict
from swarmbots.swarm.base_swarm import BaseSwarm
from swarmbots.swarm.swarm_connections import SwarmConnections

class ObstacleStreetScenario(BaseScenario):

    def __init__(
            self,
            swarm: BaseSwarm,
            payload_type: None | str,
            payload_size: Iterable[float] = (0.2, 0.2, 0.2),
            payload_start_location_offset: Iterable[float] = (0, 1, 0),
            actuators_activation_reward_weight: float = -1e-3,
            units_without_connections_reward_weight: float = -2e-3,
            connectors_stayed_active_reward_weight: float = 2e-4,
            connectors_successfully_activated_reward_weight: float = 1e-3,
            connectors_unsuccessfully_activated_reward_weight: float = -5e-5,
            connectors_deactivated_reward_weight: float = -1e-3,
            average_connectors_reward: bool = True,
            seed: int = None
    ):
        self.payload_type = payload_type
        self.payload_size = payload_size
        self.payload_start_location_offset = payload_start_location_offset

        self.side_wall_x = 10.0
        self.wall_fixed_width = 25.0
        self.wall_height = 1.0
        self.wall_distance = 4.0

        self.opening_width = 2.0
        self.ramp_length = 3.0
        self.ramp_range_x = 8.5
        self.ramp_angle = np.asin(self.wall_height / self.ramp_length)
        self.ramp_distance_to_wall = self.ramp_length * np.cos(self.ramp_angle)

        super().__init__(
            swarm=swarm,
            actuators_activation_reward_weight=actuators_activation_reward_weight,
            units_without_connections_reward_weight=units_without_connections_reward_weight,
            connectors_stayed_active_reward_weight=connectors_stayed_active_reward_weight,
            connectors_successfully_activated_reward_weight=connectors_successfully_activated_reward_weight,
            connectors_unsuccessfully_activated_reward_weight=connectors_unsuccessfully_activated_reward_weight,
            connectors_deactivated_reward_weight=connectors_deactivated_reward_weight,
            average_connectors_reward=average_connectors_reward,
            seed=seed,
            _reset_in_init=False
        )
        
        self.payload_body_id = mujoco.mj_name2id(self.dummy_model, mujoco.mjtObj.mjOBJ_BODY, 'Payload')

        self._dummy_state, self._dummy_connections = self.reset_scenario(self.dummy_model, self.dummy_data)

    def _create_scenario_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody: mujoco.MjsBody = spec.worldbody

        # base stuff
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[100, 100, 0.1],
            rgba=[0.2, 0.3, 0.4, 1],
            pos=[0, 0, 0]
        )

        worldbody.add_light(pos=[0, 0, 100], dir=[0, 0, -1])
        worldbody.add_light(pos=[0, 100, 100], dir=[-1, -1, -1])

        # side walls
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.1, 100, 5],
            rgba=[0.3, 0.4, 0.5, 0.1],
            pos=[self.side_wall_x, 0, 0]
        )
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.1, 100, 5],
            rgba=[0.3, 0.4, 0.5, 0.1],
            pos=[-self.side_wall_x, 0, 0]
        )

        for i in range(5):
            y = self.wall_distance + self.wall_distance * i

            body_left = worldbody.add_body(name=f'Wall_{i}_Left', mocap=True, pos=[0, y, 0])
            body_left.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[self.wall_fixed_width / 2, 0.1, self.wall_height],
                rgba=[0.5, 0.5, 0.6, 1],
            )

            body_right = worldbody.add_body(name=f'Wall_{i}_Right', mocap=True, pos=[0, y, 0])
            body_right.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[self.wall_fixed_width / 2, 0.1, self.wall_height],
                rgba=[0.5, 0.5, 0.6, 1],
            )

            body_ramp = worldbody.add_body(
                name=f'Ramp_{i}', mocap=True,
                pos=[0, y - self.ramp_distance_to_wall/2, self.wall_height/2],
                euler=[self.ramp_angle, 0, 0]
            )
            body_ramp.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[1, self.ramp_length * 1.2 / 2, 0.1],
                rgba=[0.5, 0.5, 0.6, 1],
            )

        if self.payload_type is not None:
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
                rgba=payload_rgba
            )
            payload_body.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)

        return spec

    def reset_scenario(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[dict, SwarmConnections]:
        state, connections = super().reset_scenario(model, data)

        rng = self.rng

        for i in range(5):
            y = self.wall_distance + self.wall_distance * i
            
            opening_x = (rng.random() - 0.5) * 2 * (self.side_wall_x + 1)  # small chance that there is no usable opening
                                                                      # -> must use ramp
            
            first_wall_end_x = opening_x - self.opening_width / 2
            wall_left_pos_x = first_wall_end_x - (self.wall_fixed_width / 2)
            
            wall_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'Wall_{i}_Left')
            if wall_left_id != -1:
                mocap_id = model.body_mocapid[wall_left_id]
                if mocap_id != -1:
                    data.mocap_pos[mocap_id] = [wall_left_pos_x, y, 0]

            second_wall_start_x = opening_x + self.opening_width / 2

            wall_right_pos_x = second_wall_start_x + (self.wall_fixed_width / 2)

            wall_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'Wall_{i}_Right')
            if wall_right_id != -1:
                mocap_id = model.body_mocapid[wall_right_id]
                if mocap_id != -1:
                    data.mocap_pos[mocap_id] = [wall_right_pos_x, y, 0]

            ramp_x = (rng.random() - 0.5) * 2 * self.ramp_range_x

            ramp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'Ramp_{i}')
            if ramp_id != -1:
                mocap_id = model.body_mocapid[ramp_id]
                if mocap_id != -1:
                    data.mocap_pos[mocap_id] = [ramp_x, y - self.ramp_distance_to_wall / 2, self.wall_height / 2]

        mujoco.mj_forward(model, data)

        state['progress'] = self._compute_progress(data)

        return state, connections

    def evaluate_step(
            self,
            action: dict[str, Any],
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
            connections: SwarmConnections
    ) -> tuple[float, bool]:
        """
        :return: (reward, done)
        """

        old_progress = state['progress']
        new_progress = self._compute_progress(data)
        state['progress'] = new_progress

        progress_reward = new_progress - old_progress
        state['progress_reward'] = progress_reward

        action_reward = self.compute_action_reward(action, state, connections)
        state['action_reward'] = action_reward

        return progress_reward + action_reward, False

    def get_obs(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
            connections: SwarmConnections
    ) -> SwarmObsDict:
        obs = super().get_obs(model, data, state, connections)

        if self.payload_type is not None:
            obs['global_obs'] = np.concatenate([
                data.xpos[self.payload_body_id],
                data.xquat[self.payload_body_id]
            ])

        return obs

    def _compute_progress(
            self,
            data: mujoco.MjData
    ):
        if self.payload_type is None:
            return data.qpos[self._qpos_indices[:, 1]].mean()  # avg y pos of the unit bodies

        return data.xpos[self.payload_body_id, 1]
