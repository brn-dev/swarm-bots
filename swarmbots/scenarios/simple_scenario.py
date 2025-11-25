import mujoco
import numpy as np

from swarmbots.mujoco_utils import quat_z2vec
from swarmbots.scenarios.base_scenario import BaseScenario


class SimpleScenario(BaseScenario[dict]):
    def build_scenario_spec(self) -> mujoco.MjSpec:
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

        worldbody.add_light(pos=[0, 0, 10], dir=[0, 0, -1])
        worldbody.add_light(pos=[0, 10, 10], dir=[-1, -1, -1])

        worldbody.add_camera(
            pos=[9, 0, 5], quat=quat_z2vec([0, 0, 1]),
            mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody='Unit1--main_body')


        # major walls
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.1, 100, 5],
            rgba=[0.3, 0.4, 0.5, 0.1],
            pos=[10, 0, 0]
        )
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.1, 100, 5],
            rgba=[0.3, 0.4, 0.5, 0.1],
            pos=[-10, 0, 0]
        )

        rng = np.random.default_rng()
        for i in range(5):
            opening_x = (rng.random() - 0.5) * 2 * 8.5
            opening_width = 1.0
            wall_height = 2.0
            y = 1 + 6 * i

            first_wall_start_x = -10.0
            first_wall_end_x = opening_x - opening_width / 2
            first_wall_length_x = first_wall_end_x - first_wall_start_x
            first_wall_mid_x = first_wall_start_x + first_wall_length_x / 2.0

            worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[first_wall_length_x / 2, 0.1, wall_height],
                rgba=[0.5, 0.5, 0.6, 1],
                pos=[first_wall_mid_x, y, 0]
            )

            second_wall_start_x = opening_x + opening_width / 2
            second_wall_end_x = 10.0
            second_wall_length_x = second_wall_end_x - second_wall_start_x
            second_wall_mid_x = second_wall_start_x + second_wall_length_x / 2.0

            worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[second_wall_length_x / 2, 0.1, wall_height],
                rgba=[0.5, 0.5, 0.6, 1],
                pos=[second_wall_mid_x, y, 0]
            )

            ramp_x = (rng.random() - 0.5) * 2 * 8.5
            ramp_length = 5
            ramp_angle = np.asin(wall_height / ramp_length)
            ramp_distance_to_wall = ramp_length * np.cos(ramp_angle)

            worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[1, ramp_length * 1.2 / 2, 0.1],
                rgba=[0.5, 0.5, 0.6, 1],
                pos=[ramp_x, y - ramp_distance_to_wall/2, wall_height/2],
                euler=[ramp_angle, 0, 0]
            )

        return spec

    def reset_scenario(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict:
        return dict()

    def modify_obs(self, model: mujoco.MjModel, data: mujoco.MjData, state: dict, obs: np.ndarray) -> np.ndarray:
        return obs

    def scenario_step(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            old_state: dict | None
    ) -> tuple[dict, float, bool]:
        # avg y vel
        reward = np.mean(data.qvel[1::18])
        return old_state, reward, False