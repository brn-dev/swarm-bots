import mujoco
import numpy as np

from swarmbots.scenarios.base_scenario import BaseScenario


class SimpleScenario(BaseScenario[dict]):
    def build_scenario_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        worldbody: mujoco.MjsBody = spec.worldbody

        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[10, 10, 0.1],
            rgba=[0.2, 0.3, 0.4, 1],
            pos=[0, 0, 0]
        )

        worldbody.add_light(pos=[0, 0, 6], dir=[0, 0, -1])
        worldbody.add_light(pos=[2, 2, 6], dir=[-1, -1, -1])

        worldbody.add_camera(pos=[-3, 0, 3], mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM, targetbody='Unit1--main_body')
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
        return old_state, 0, False