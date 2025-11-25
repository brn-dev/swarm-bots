import mujoco
import numpy as np
from mujoco import MjsBody

from swarmbots.swarm.base_swarm import BaseSwarm
from swarmbots.unit import init_unit


class SimpleSwarm(BaseSwarm):

    def create_swarm_spec(self) -> mujoco.MjsBody:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody: MjsBody = spec.worldbody

        unit = init_unit(
            body_radius=0.1,
            leg_length=0.2,
            leg_radius=0.025,
            hinge_range=np.pi / 4
        )
        unit.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)

        worldbody.add_frame(pos=[0, 0, 0]).attach_body(unit, 'Unit1--', '')  # , quat=quat_z2vec([1, 1, 0])

        unit = init_unit(
            body_radius=0.1,
            leg_length=0.2,
            leg_radius=0.025,
            hinge_range=np.pi / 4
        )
        unit.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)

        worldbody.add_frame(pos=[0, 0.6, 0]).attach_body(unit, 'Unit2--', '')

        eq = spec.add_equality(
            name="site_weld",
            type=mujoco.mjtEq.mjEQ_WELD,  # or mjEQ_CONNECT
            objtype=mujoco.mjtObj.mjOBJ_BODY,  # tells MuJoCo the objs are sites
            name1="Unit1--limb_yp-tip",
            name2="Unit2--limb_yn-tip",

            #   data[0:7]  = relpose (3 pos + 4 quat)
            #   data[7:10] = anchor  (3)
            #   data[10]   = torquescale
            # data=[0,0,0,0,1,0,0,  0,0,0,  10.0]
        )
        eq.data[10] = 50

        return spec

    def reset_swarm(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        # mujoco reset should be enough
        pass

    def get_unit_prefixes(self) -> set[str]:
        return set(f'Unit{i}--' for i in range(1, 3))