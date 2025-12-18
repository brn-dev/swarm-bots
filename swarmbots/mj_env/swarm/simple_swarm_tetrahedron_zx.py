import mujoco
import numpy as np
from mujoco import MjsBody

from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.swarm_config import SwarmConfig
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections
from swarmbots.mj_env.swarm.unit_config import UNIT_CONFIG_TETRAHEDRON_ZX
from swarmbots.mj_env.swarm.unit import init_unit


class SimpleSwarmTetrahedronZX(BaseSwarm):

    def __init__(
            self,
            connection_torquescale: float = 1.0,
            connection_dist_threshold: float = 0.1,
            connection_angle_threshold: float = -0.5
    ):
        super().__init__(SwarmConfig(
            num_units=3,
            unit_config=UNIT_CONFIG_TETRAHEDRON_ZX,
            connection_torquescale=connection_torquescale,
            connection_dist_threshold=connection_dist_threshold,
            connection_angle_threshold=connection_angle_threshold,
        ))

        self.body_radius = 0.1
        self.leg_length = 0.2
        self.leg_radius = 0.025
        self.hinge_range = np.pi / 4


    def _create_swarm_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody: MjsBody = spec.worldbody

        unit = init_unit(
            body_radius=self.body_radius,
            leg_length=self.leg_length,
            leg_radius=self.leg_radius,
            hinge_range=self.hinge_range,
            unit_config=UNIT_CONFIG_TETRAHEDRON_ZX,
        )
        unit.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
        worldbody.add_frame(pos=[0, 0, 0]).attach_body(unit, self.config.unit_prefixes[0], '')

        unit = init_unit(
            body_radius=self.body_radius,
            leg_length=self.leg_length,
            leg_radius=self.leg_radius,
            hinge_range=self.hinge_range,
            unit_config=UNIT_CONFIG_TETRAHEDRON_ZX,
        )
        unit.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)

        worldbody.add_frame(pos=[1, 0, 0]).attach_body(unit, self.config.unit_prefixes[1], '')


        unit = init_unit(
            body_radius=self.body_radius,
            leg_length=self.leg_length,
            leg_radius=self.leg_radius,
            hinge_range=self.hinge_range,
            unit_config=UNIT_CONFIG_TETRAHEDRON_ZX,
        )
        unit.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
        worldbody.add_frame(pos=[0.5, 0.5, 0]).attach_body(unit, self.config.unit_prefixes[2], '')

        return spec

    def reset_swarm(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            rng: np.random.Generator,
            start_location: np.ndarray
    ) -> SwarmConnections:
        connections = SwarmConnections(self.config)

        unit0_main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                               self.config.unit_prefixes[0] + '-main_body')
        jnt_adr = model.body_jntadr[unit0_main_body_id]
        qpos_adr = model.jnt_qposadr[jnt_adr]
        data.qpos[qpos_adr:qpos_adr + 3] = start_location

        # Place unit 1 such that connector 3 is next to unit 0's connector 1
        v1 = np.array([1, -1, -1], dtype=float)
        v1 /= np.linalg.norm(v1)
        d = self.body_radius + self.leg_length
        pos = 2 * d * v1 + start_location
        # Rotate -90 deg around X axis
        quat = [np.cos(-np.pi/4), np.sin(-np.pi/4), 0, 0]

        unit1_main_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                               self.config.unit_prefixes[1] + '-main_body')
        jnt_adr = model.body_jntadr[unit1_main_body_id]
        qpos_adr = model.jnt_qposadr[jnt_adr]
        data.qpos[qpos_adr:qpos_adr+3] = pos
        data.qpos[qpos_adr+3:qpos_adr+7] = quat

        connections.connect(
            0, 1,
            1, 3,
            0
        )
        return connections

