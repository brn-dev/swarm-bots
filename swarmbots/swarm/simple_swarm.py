import mujoco
import numpy as np
from mujoco import MjsBody

from swarmbots.swarm.base_swarm import BaseSwarm
from swarmbots.swarm.swarm_config import SwarmConfig
from swarmbots.swarm.swarm_connections import SwarmConnections
from swarmbots.swarm.unit_config import UNIT_CONFIG_CUBE_ZX
from swarmbots.swarm.unit import init_unit


class SimpleSwarm(BaseSwarm):

    def __init__(self, connection_torquescale: float):
        super().__init__(SwarmConfig(
            num_units=2,
            unit_config=UNIT_CONFIG_CUBE_ZX,
            connection_torquescale=connection_torquescale,
        ))

    def _create_swarm_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody: MjsBody = spec.worldbody

        unit = init_unit(
            body_radius=0.1,
            leg_length=0.2,
            leg_radius=0.025,
            hinge_range=np.pi / 4,
            unit_config=UNIT_CONFIG_CUBE_ZX,
        )
        unit.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
        worldbody.add_frame(pos=[0, 0, 0]).attach_body(unit, self.config.unit_prefixes[0], '')

        unit = init_unit(
            body_radius=0.1,
            leg_length=0.2,
            leg_radius=0.025,
            hinge_range=np.pi / 4,
            unit_config=UNIT_CONFIG_CUBE_ZX,
        )
        unit.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
        worldbody.add_frame(pos=[0, 0.601, 0], euler=[0, np.pi * 0.2, 0]).attach_body(unit, self.config.unit_prefixes[1], '')
        # , euler=[0, np.pi * 1.9, 0]

        # unit = init_unit(
        #     body_radius=0.1,
        #     leg_length=0.2,
        #     leg_radius=0.025,
        #     hinge_range=np.pi / 4,
        #     unit_config=UNIT_CONFIG_CUBE_ZX,
        # )
        # unit.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
        # worldbody.add_frame(pos=[1, 1, 0]).attach_body(unit, self.config.unit_prefixes[2], '')
        #
        # unit = init_unit(
        #     body_radius=0.1,
        #     leg_length=0.2,
        #     leg_radius=0.025,
        #     hinge_range=np.pi / 4,
        #     unit_config=UNIT_CONFIG_CUBE_ZX,
        # )
        # unit.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
        # worldbody.add_frame(pos=[-1, 1, 0]).attach_body(unit, self.config.unit_prefixes[3], '')
        #
        # unit = init_unit(
        #     body_radius=0.1,
        #     leg_length=0.2,
        #     leg_radius=0.025,
        #     hinge_range=np.pi / 4,
        #     unit_config=UNIT_CONFIG_CUBE_ZX,
        # )
        # unit.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
        # worldbody.add_frame(pos=[0, -1, 0]).attach_body(unit, self.config.unit_prefixes[4], '')

        return spec

    def reset_swarm(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> SwarmConnections:
        connections = SwarmConnections(self.config)
        connections.connect(
            0,
            self.config.limb_name_to_idx['yp'],
            1,
            self.config.limb_name_to_idx['yn'],
            0
        )
        return connections

