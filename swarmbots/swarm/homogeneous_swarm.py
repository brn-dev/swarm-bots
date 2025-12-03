import mujoco
import numpy as np
from mujoco import MjsBody

from swarmbots.swarm.base_swarm import BaseSwarm
from swarmbots.swarm.swarm_config import SwarmConfig
from swarmbots.swarm.swarm_connections import SwarmConnections
from swarmbots.swarm.unit import init_unit
from swarmbots.swarm.unit_config import UnitConfig, UNIT_CONFIG_TETRAHEDRON_YX


class HomogeneousSwarm(BaseSwarm):

    def __init__(
            self,
            unit_start_locations: list[tuple[float, float, float]],
            unit_start_quats: list[tuple[float, float, float, float]] = None,
            unit_config: UnitConfig = UNIT_CONFIG_TETRAHEDRON_YX,
            body_radius: float = 0.1,
            leg_length: float = 0.2,
            leg_radius: float = 0.025,
            hinge_range: float = np.pi / 3,
            connection_torquescale: float = 1.0,
            connection_dist_threshold: float = 0.1,
            connection_angle_threshold: float = -0.5
    ):
        self.num_units = len(unit_start_locations)
        self.unit_start_locations = unit_start_locations
        self.unit_start_quats = unit_start_quats

        assert unit_start_quats is None or len(unit_start_quats) == self.num_units

        super().__init__(SwarmConfig(
            num_units=self.num_units,
            unit_config=unit_config,
            connection_torquescale=connection_torquescale,
            connection_dist_threshold=connection_dist_threshold,
            connection_angle_threshold=connection_angle_threshold,
        ))

        self.body_radius = body_radius
        self.leg_length = leg_length
        self.leg_radius = leg_radius
        self.hinge_range = hinge_range


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
        return connections

