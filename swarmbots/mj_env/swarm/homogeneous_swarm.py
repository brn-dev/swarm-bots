from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np
from mujoco import MjsBody

from swarmbots.mj_env.float_or_dist import FloatOrDistParams, eval_fod, DistParams
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

@dataclass
class RandomUnitLocationsConfig:
    num_units: int
    pairwise_distance: float

    max_distance: float = 1e8
    z_pos: float = 0.0
    center: bool = True

UnitStartLocations = list[tuple[float, float, float]] | RandomUnitLocationsConfig

class HomogeneousSwarm(BaseSwarm):

    unit_start_locations: UnitStartLocations

    def __init__(
            self,
            unit_start_locations: UnitStartLocations | str,
            unit_start_quats: list[tuple[float, float, float, float]] | None = None,
            unit_config: UnitConfig = UNIT_CONFIG_TETRAHEDRON_YX,
            body_radius: float = 0.1,
            leg_length: float = 0.2,
            leg_radius: float = 0.025,
            hinge_range: float = np.pi / 3,
            hinge_armature: float = 0.001,
            connection_torquescale: float = 10.0,
            randomize_unit_orientations: bool = False
    ):
        assert unit_start_quats is None or not randomize_unit_orientations

        self.random_unit_start_locations = False
        if isinstance(unit_start_locations, RandomUnitLocationsConfig):
            assert unit_start_locations.max_distance > unit_start_locations.pairwise_distance
            self.num_units = unit_start_locations.num_units
            self.unit_start_locations = unit_start_locations
            self.random_unit_start_locations = True

        elif isinstance(unit_start_locations, str):
            if unit_start_locations in UNIT_START_LOCATION_PRESETS:
                self.unit_start_locations = UNIT_START_LOCATION_PRESETS.get(unit_start_locations)
                self.num_units = len(self.unit_start_locations)
            else:
                raise ValueError(f'Unknown unit start location preset "{unit_start_locations}", available presets: '
                             + str(list(UNIT_START_LOCATION_PRESETS.keys())))
        else:
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
        self.hinge_armature = hinge_armature

    def get_settings(self):
        settings = super().get_settings()
        settings.update({
            'unit_start_locations': self.unit_start_locations,
            'unit_start_quats': self.unit_start_quats,
            'body_radius': self.body_radius,
            'leg_length': self.leg_length,
            'leg_radius': self.leg_radius,
            'hinge_range': self.hinge_range,
            'hinge_armature': self.hinge_armature,
            'randomize_unit_orientations': self.randomize_unit_orientations,
        })
        return settings

    def _create_swarm_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody: MjsBody = spec.worldbody

        if self.random_unit_start_locations:
            unit_start_locations = self._generate_random_start_locations(rng=np.random.default_rng())
        else:
            unit_start_locations = self.unit_start_locations

        for i, unit_start_location in enumerate(unit_start_locations):
            unit_start_location = np.array(unit_start_location)
            unit_start_quat = np.zeros(4, dtype=float)
            if self.unit_start_quats is not None:
                unit_start_quat = np.array(self.unit_start_quats[i])

            unit = init_unit(
                body_radius=self.body_radius,
                leg_length=self.leg_length,
                leg_radius=self.leg_radius,
                hinge_range=self.hinge_range,
                hinge_armature=self.hinge_armature,
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

        if self.random_unit_start_locations:
            unit_start_locations = self._generate_random_start_locations(rng=rng)
        else:
            unit_start_locations = self.unit_start_locations

        for i in range(self.num_units):
            qpos_adr, dof_adr = self._get_unit_main_body_addresses(model, i)

            data.qpos[qpos_adr:qpos_adr + 3] = start_location + np.array(unit_start_locations[i])

            if self.randomize_unit_orientations:
                data.qpos[qpos_adr + 3:qpos_adr + 7] = random_quat_shoemake()
            else:
                data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]

            data.qvel[dof_adr:dof_adr+6] = 0

        data.ctrl[:] = 0

        return connections

    def _get_unit_main_body_addresses(self, model: mujoco.MjModel, unit_idx: int):
        body_name = f"{self.config.unit_prefixes[unit_idx]}-main_body"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        jnt_adr = model.body_jntadr[body_id]
        qpos_adr = model.jnt_qposadr[jnt_adr]
        dof_adr = model.jnt_dofadr[jnt_adr]
        return qpos_adr, dof_adr

    def _generate_random_start_locations(
            self,
            rng: np.random.Generator,
            eps: float = 1e-6,
    ) -> list[tuple[float, float, float]]:
        random_config: RandomUnitLocationsConfig = self.unit_start_locations
        num_units = random_config.num_units
        pairwise_distance = random_config.pairwise_distance
        max_distance = random_config.max_distance

        points = np.zeros((num_units, 2))
        points_found = 1
        rejected_count = 0

        while points_found < num_units:
            source_point = points[rng.choice(points_found)]

            angle = rng.random() * 2 * np.pi
            proposal_x = np.cos(angle) * pairwise_distance
            proposal_y = np.sin(angle) * pairwise_distance

            target_point = source_point + np.array([proposal_x, proposal_y])

            if np.linalg.norm(target_point) > max_distance:
                rejected_count += 1
                continue

            distances = np.linalg.norm(points[:points_found] - target_point, axis=-1)

            if np.any(distances + eps < pairwise_distance):
                rejected_count += 1
            else:
                points[points_found] = target_point
                points_found += 1
            
            if rejected_count > 1000:
                raise RuntimeError(f"Failed to generate random start locations after 1000 attempts: {random_config=}")

        if random_config.center:
            mean = points.mean(axis=0, keepdims=True)
            points -= mean

            while np.any(np.linalg.norm(points, axis=-1) > max_distance):
                points += mean / 4


        return [(float(p[0]), float(p[1]), random_config.z_pos) for p in points]
