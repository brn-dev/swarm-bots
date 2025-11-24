import mujoco
import numpy as np
from mujoco import MjsBody

from swarmbots.swarm.base_swarm import BaseSwarm
from swarmbots.unit import init_unit


class SimpleSwarm(BaseSwarm):

    def build_swarm_spec(self) -> mujoco.MjsBody:
        spec = mujoco.MjSpec()
        worldbody: MjsBody = spec.worldbody

        swarm_body = worldbody.add_body(name="Swarm")

        unit = init_unit(
            body_radius=0.1,
            leg_length=0.2,
            leg_radius=0.025,
            hinge_range=np.pi / 4
        )
        unit.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)

        swarm_body.add_frame(pos=[0, 0, 0]).attach_body(unit, 'Unit1--', '')  # , quat=quat_z2vec([1, 1, 0])

        unit = init_unit(
            body_radius=0.1,
            leg_length=0.2,
            leg_radius=0.025,
            hinge_range=np.pi / 4
        )
        unit.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)

        swarm_body.add_frame(pos=[0, 0.6, 0]).attach_body(unit, 'Unit2--', '')

        return swarm_body

    def reset_swarm(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        # mujoco reset should be enough
        pass

    def get_unit_prefixes(self) -> set[str]:
        return set(f'Unit{i}--' for i in range(1, 3))