from typing import Union, Literal

import mujoco
import numpy as np
from mujoco import MjsBody

from swarmbots.mujoco_utils import quat_z2vec
from swarmbots.scenarios.base_scenario import BaseScenario
from swarmbots.swarm.base_swarm import BaseSwarm

PayloadType = Literal['ball', 'box'] | None


class ObstacleDungeonScenario(BaseScenario):

    def __init__(self, swarm: BaseSwarm, payload_type: PayloadType, payload_size: float = 0.3):

        self.payload_type = payload_type
        self.payload_size = payload_size

        super().__init__(swarm)

    def create_scenario_spec(self) -> mujoco.MjSpec:
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

        worldbody.add_camera(
            pos=[12, 0, 5], euler=[0, np.pi / 3, np.pi / 2],
            mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACK, targetbody='Unit1--main_body')

        # side walls
        side_wall_x = 10.0
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.1, 100, 5],
            rgba=[0.3, 0.4, 0.5, 0.1],
            pos=[side_wall_x, 0, 0]
        )
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.1, 100, 5],
            rgba=[0.3, 0.4, 0.5, 0.1],
            pos=[-side_wall_x, 0, 0]
        )

        # rng = np.random.default_rng() # moved to reset
        for i in range(5):
            y = 4 + 4 * i
            wall_height = 1.0
            wall_width = 15.0

            # Wall Left
            body_left = worldbody.add_body(name=f'Wall_{i}_Left', mocap=True, pos=[0, y, 0])
            body_left.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[wall_width / 2, 0.1, wall_height],
                rgba=[0.5, 0.5, 0.6, 1],
            )

            # Wall Right
            body_right = worldbody.add_body(name=f'Wall_{i}_Right', mocap=True, pos=[0, y, 0])
            body_right.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[wall_width / 2, 0.1, wall_height],
                rgba=[0.5, 0.5, 0.6, 1],
            )

            # Ramp
            ramp_length = 3
            ramp_angle = np.asin(wall_height / ramp_length)
            # ramp_distance_to_wall = ramp_length * np.cos(ramp_angle)

            body_ramp = worldbody.add_body(name=f'Ramp_{i}', mocap=True, pos=[0, y, wall_height/2], euler=[ramp_angle, 0, 0])
            body_ramp.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[1, ramp_length * 1.2 / 2, 0.1],
                rgba=[0.5, 0.5, 0.6, 1],
            )

        # payload
        # if self.payload_type is not None:
        #     payload_start_position = np.array(self.get_swarm_start_location()) + np.array([0, 2, 1])
        #     payload_body: MjsBody = worldbody.add_body(name='Payload', pos=payload_start_position)
        #
        #     payload_rgba = [0.8, 0.3, 0.3, 0.9]
        #     if self.payload_type == 'ball':
        #         payload_body.add_geom(
        #             type=mujoco.mjtGeom.mjGEOM_SPHERE,
        #             size=[self.payload_size, 0, 0],
        #             rgba=payload_rgba
        #         )
        #     elif self.payload_type == 'box':
        #         payload_body.add_geom(
        #             type=mujoco.mjtGeom.mjGEOM_BOX,
        #             size=[self.payload_size, self.payload_size, self.payload_size],
        #             rgba=payload_rgba
        #         )
        #     payload_body.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)

        return spec

    def reset_scenario(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict:
        rng = self.rng
        side_wall_x = 10.0
        wall_height = 1.0

        for i in range(5):
            y = 4 + 4 * i
            opening_x = (rng.random() - 0.5) * 2 * (side_wall_x + 1)

            # Left Wall
            # Center = opening_x - 0.5 - 7.5 = opening_x - 8.0
            wall_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'Wall_{i}_Left')
            if wall_left_id != -1:
                mocap_id = model.body_mocapid[wall_left_id]
                if mocap_id != -1:
                    data.mocap_pos[mocap_id] = [opening_x - 8.0, y, 0]

            # Right Wall
            # Center = opening_x + 0.5 + 7.5 = opening_x + 8.0
            wall_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'Wall_{i}_Right')
            if wall_right_id != -1:
                mocap_id = model.body_mocapid[wall_right_id]
                if mocap_id != -1:
                    data.mocap_pos[mocap_id] = [opening_x + 8.0, y, 0]

            # Ramp
            ramp_x = (rng.random() - 0.5) * 2 * 8.5
            ramp_length = 3
            ramp_angle = np.asin(wall_height / ramp_length)
            ramp_distance_to_wall = ramp_length * np.cos(ramp_angle)

            ramp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'Ramp_{i}')
            if ramp_id != -1:
                mocap_id = model.body_mocapid[ramp_id]
                if mocap_id != -1:
                    data.mocap_pos[mocap_id] = [ramp_x, y - ramp_distance_to_wall / 2, wall_height / 2]

        mujoco.mj_forward(model, data)

        return {
            'progress': self._compute_progress(model, data)
        }

    def evaluate_step(
            self,
            action: np.ndarray,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
    ) -> tuple[float, bool]:

        old_progress = state['progress']
        new_progress = self._compute_progress(model, data)

        reward = new_progress - old_progress

        state['progress'] = new_progress

        return reward, False

    def _compute_progress(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData
    ):
        return data.qpos[self._qpos_indices[:, 1]].mean()  # avg y pos of the unit bodies