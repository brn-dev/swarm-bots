import mujoco
import numpy as np
from mujoco import MjsBody

from swarmbots.mj_env.random_utils import random_quat_shoemake
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.swarm_config import SwarmConfig
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections
from swarmbots.mj_env.swarm.unit import init_unit
from swarmbots.mj_env.swarm.unit_config import UnitConfig, UNIT_CONFIG_TETRAHEDRON_YX

UNIT_START_LOCATION_PRESETS = {
    '4:diamond': [
        (0.0, 0.0, 0.0),
        (0.4, -0.4, 0),
        (-0.4, -0.4, 0),
        (0.0, -0.8, 0.0),
    ],
    '5:X': [
        (0.0, 0.0, 0.0),
        (0.4, 0.4, 0),
        (0.4, -0.4, 0),
        (-0.4, 0.4, 0),
        (-0.4, -0.4, 0),
    ],
    '5:T': [
        (0.0, 0.0, 0.0),
        (0.0, -0.4, 0.0),
        (0.0, 0.4, 0.0),
        (0.0, 0.4, -0.4),
        (0.0, 0.4, 0.4),
    ],
    '5:T-reverse': [
        (0.0, 0.0, 0.0),
        (0.0, 0.4, 0.0),
        (0.0, -0.4, 0.0),
        (0.0, -0.4, -0.4),
        (0.0, -0.4, 0.4),
    ],
    '5:3+2': [
        (0.0, 0.0, 0.0),
        (-0.6, 0.0, 0.0),
        (-0.3, -0.4, 0.0),
        (0.8, 0.0, 0.0),
        (1.3, 0.0, 0.0),
    ],
    '5:W': [
        (0.0, 0.0, 0.0),
        (0.8, 0.0, 0.0),
        (-0.8, 0.0, 0.0),
        (0.4, -0.4, 0.0),
        (-0.4, -0.4, 0.0),
    ],
    '5:M': [
        (0.0, -0.4, 0.0),
        (0.8, -0.4, 0.0),
        (-0.8, -0.4, 0.0),
        (0.4, 0.0, 0.0),
        (-0.4, 0.0, 0.0),
    ],
}


class HomogeneousSwarm(BaseSwarm):

    def __init__(
            self,
            unit_start_locations: list[tuple[float, float, float]] | str,
            unit_start_quats: list[tuple[float, float, float, float]] = None,
            unit_config: UnitConfig = UNIT_CONFIG_TETRAHEDRON_YX,
            body_radius: float = 0.1,
            leg_length: float = 0.2,
            leg_radius: float = 0.025,
            hinge_range: float = np.pi / 3,
            connection_torquescale: float = 10.0,
            randomize_unit_orientations: bool = False
    ):
        assert unit_start_quats is None or not randomize_unit_orientations

        if isinstance(unit_start_locations, str):
            if unit_start_locations not in UNIT_START_LOCATION_PRESETS:
                raise ValueError(f'Unknown unit start location preset "{unit_start_locations}", available presets: '
                                 + str(list(UNIT_START_LOCATION_PRESETS.keys())))
            unit_start_locations = UNIT_START_LOCATION_PRESETS.get(unit_start_locations)


        self.num_units = len(unit_start_locations)
        self.unit_start_locations = unit_start_locations
        self.unit_start_quats = unit_start_quats
        self.randomize_unit_orientations = randomize_unit_orientations

        assert unit_start_quats is None or len(unit_start_quats) == self.num_units

        super().__init__(SwarmConfig(
            num_units=self.num_units,
            unit_config=unit_config,
            connection_torquescale=connection_torquescale,
        ))

        self.body_radius = body_radius
        self.leg_length = leg_length
        self.leg_radius = leg_radius
        self.hinge_range = hinge_range

    def get_settings(self):
        settings = super().get_settings()
        settings.update({
            'unit_start_locations': self.unit_start_locations,
            'unit_start_quats': self.unit_start_quats,
            'body_radius': self.body_radius,
            'leg_length': self.leg_length,
            'leg_radius': self.leg_radius,
            'hinge_range': self.hinge_range,
            'randomize_unit_orientations': self.randomize_unit_orientations,
        })
        return settings

    def _create_swarm_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody: MjsBody = spec.worldbody

        for i, unit_start_location in enumerate(self.unit_start_locations):
            unit_start_location = np.array(unit_start_location)
            unit_start_quat = np.zeros(4, dtype=float)
            if self.unit_start_quats is not None:
                unit_start_quat = np.array(self.unit_start_quats[i])

            unit = init_unit(
                body_radius=self.body_radius,
                leg_length=self.leg_length,
                leg_radius=self.leg_radius,
                hinge_range=self.hinge_range,
                unit_config=self.config.unit_config,
            )
            unit.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
            (worldbody
             .add_frame(pos=unit_start_location, quat=unit_start_quat)
             .attach_body(unit, self.config.unit_prefixes[i], ''))

        return spec

    def reset_swarm(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            rng: np.random.Generator,
            start_location: np.ndarray
    ) -> SwarmConnections:
        connections = SwarmConnections(self.config)

        if self.randomize_unit_orientations:
            for i, unit_start_location in enumerate(self.unit_start_locations):
                random_quat = random_quat_shoemake()

                body_name = f"{self.config.unit_prefixes[i]}-main_body"
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                jnt_adr = model.body_jntadr[body_id]

                qpos_adr = model.jnt_qposadr[jnt_adr]
                data.qpos[qpos_adr:qpos_adr + 3] = start_location + unit_start_location
                data.qpos[qpos_adr + 3:qpos_adr + 7] = random_quat

                dof_adr = model.jnt_dofadr[jnt_adr]
                data.qvel[dof_adr:dof_adr + 6] = 0

        return connections

