import mujoco
import numpy as np
from mujoco import MjsBody

from swarmbots.swarm.base_swarm import BaseSwarm
from swarmbots.swarm.swarm_config import SwarmConfig
from swarmbots.swarm.swarm_connections import SwarmConnections
from swarmbots.unit import init_unit, CUBE_ZX_LIMB_CONFIGS


class SimpleSwarm(BaseSwarm):

    def __init__(self, seed: int):
        super().__init__(SwarmConfig(2, 6), seed)

    def create_swarm_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody: MjsBody = spec.worldbody

        unit_prefixes = self.get_unit_prefixes()

        unit = init_unit(
            body_radius=0.1,
            leg_length=0.2,
            leg_radius=0.025,
            hinge_range=np.pi / 4,
            limb_configs=CUBE_ZX_LIMB_CONFIGS,
        )
        unit.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)

        worldbody.add_frame(pos=[0, 0, 0]).attach_body(unit, unit_prefixes[0], '')  # , quat=quat_z2vec([1, 1, 0])

        unit = init_unit(
            body_radius=0.1,
            leg_length=0.2,
            leg_radius=0.025,
            hinge_range=np.pi / 4,
            limb_configs=CUBE_ZX_LIMB_CONFIGS,
        )
        unit.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)

        worldbody.add_frame(pos=[0, 0.6, 0]).attach_body(unit, unit_prefixes[1], '')

        eq = spec.add_equality(
            name="site_weld",
            type=mujoco.mjtEq.mjEQ_WELD,  # or mjEQ_CONNECT
            objtype=mujoco.mjtObj.mjOBJ_BODY,  # tells MuJoCo the objs are sites
            name1="Unit1--limb_yp-connector",
            name2="Unit2--limb_yn-connector",
        )

        theta = 0.5 * np.pi

        eq.data[:3] = [0, 0, 0]  # anchor
        eq.data[3:6] = [0, 0, 0]  # relpose pos
        eq.data[6:10] = [0, np.cos(theta / 2), np.sin(theta / 2), 0]  # relpose quat
        eq.data[10] = 50  # torquescale

        return spec

    def reset_swarm(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[dict, SwarmConnections]:



