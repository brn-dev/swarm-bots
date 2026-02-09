from dataclasses import dataclass
from typing import Optional, TypeVar, TypeAlias

import mujoco
import numpy as np
from mujoco import MjsBody

from swarmbots.mj_env.float_or_dist_params import FloatOrDistParams, eval_fodp, DistParams, eval_fodp_3d
from swarmbots.mj_env.random_utils import random_quat_shoemake
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.swarm_config import SwarmConfig
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections
from swarmbots.mj_env.swarm.unit import init_unit
from swarmbots.mj_env.swarm.unit_config import UnitConfig, UNIT_CONFIG_TETRAHEDRON_YX

UNIT_START_LOCATION_PRESETS = {
    '4:diamond': [
        (0.0, 0.0, 0.0),
        (0.4, -0.4, 0.0),
        (-0.4, -0.4, 0.0),
        (0.0, -0.8, 0.0),
    ],
    '5:X': [
        (0.0, 0.0, 0.0),
        (0.4, 0.4, 0.0),
        (0.4, -0.4, 0.0),
        (-0.4, 0.4, 0.0),
        (-0.4, -0.4, 0.0),
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
    '8:hourglass': [
        (0.0, 0.0, 0.0),
        (0.8, 0.0, 0.0),
        (-0.8, 0.0, 0.0),
        (0.4, -0.4, 0.0),
        (-0.4, -0.4, 0.0),
        (0.0, -0.8, 0.0),
        (0.8, -0.8, 0.0),
        (-0.8, -0.8, 0.0),
    ],
}

T = TypeVar('T')
Tuple2: TypeAlias = tuple[T, T]
Tuple3: TypeAlias = tuple[T, T, T]
Tuple2or3: TypeAlias = Tuple2 | Tuple3

def _normalize_tuple2or3(tup: Tuple2or3[T], default_val: T) -> Tuple3[T]:
    l = len(tup)
    if l == 2:
        # noinspection PyTypeChecker
        return tup + (default_val,)
    if l == 3:
        return tup
    raise ValueError(tup)

def _normalize_tuples2or3(tuples: list[Tuple2or3[T]], default_val: T) -> list[Tuple3[T]]:
    return [_normalize_tuple2or3(tup, default_val) for tup in tuples]

@dataclass
class RandomLatticeUnitLocationsConfig:
    num_units: int
    pairwise_distance: float

    num_unit_probs: Optional[dict[int, float]] = None
    max_radius: float = 1e8
    z_pos: float = 0.0
    center: bool = True

@dataclass
class RandomWiggleUnitLocationsConfig:
    num_units: int
    unit_locations: list[Tuple3[float]]
    wiggle_params: list[Tuple3[FloatOrDistParams]]

    def __init__(
            self,
            unit_locations: list[Tuple2or3[float]] | str,
            wiggle_params: DistParams | Tuple2or3[DistParams] | list[Tuple2or3[FloatOrDistParams]],
    ):
        if isinstance(unit_locations, str):
            self.unit_locations = UNIT_START_LOCATION_PRESETS[unit_locations]
        else:
            self.unit_locations = _normalize_tuples2or3(unit_locations, 0.0)

        self.num_units = len(self.unit_locations)
        if isinstance(wiggle_params, DistParams):
            self.wiggle_params = [(wiggle_params,) * 3] * self.num_units
        elif isinstance(wiggle_params, tuple):
            self.wiggle_params = [_normalize_tuple2or3(wiggle_params, 0.0)] * self.num_units
        else:
            assert len(wiggle_params) == self.num_units
            self.wiggle_params = _normalize_tuples2or3(wiggle_params, 0.0)

UnitStartLocations = (
        list[Tuple2or3[FloatOrDistParams]]
        | RandomLatticeUnitLocationsConfig
        | RandomWiggleUnitLocationsConfig
)

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
            hinge_armature: float = 0.003,
            connection_torquescale: float = 10.0,
            randomize_unit_orientations: bool = False
    ):
        assert unit_start_quats is None or not randomize_unit_orientations

        self._can_have_inactive_units = False

        if isinstance(unit_start_locations, str):
            self.unit_start_locations = UNIT_START_LOCATION_PRESETS[unit_start_locations]
            self.num_units = len(self.unit_start_locations)
        elif isinstance(unit_start_locations, RandomLatticeUnitLocationsConfig):
            assert unit_start_locations.max_radius > unit_start_locations.pairwise_distance
            self.num_units = unit_start_locations.num_units
            self.unit_start_locations = unit_start_locations
            if unit_start_locations.num_unit_probs is not None:
                counts = np.array(list(unit_start_locations.num_unit_probs.keys()), dtype=int)
                probs = np.array(list(unit_start_locations.num_unit_probs.values()), dtype=float)
                if counts.max() != self.num_units:
                    raise ValueError("num_unit_probs must contain an entry for count == num_units")
                if (counts < 1).any():
                    raise ValueError("num_unit_probs must only contain counts >= 1")
                if (counts > unit_start_locations.num_units).any():
                    raise ValueError("num_unit_probs must not exceed num_units")
                if (probs < 0).any():
                    raise ValueError("num_unit_probs must not contain negative probabilities")
                probs_sum = probs.sum()
                if probs_sum <= 0:
                    raise ValueError("num_unit_probs must sum to a positive value")
                probs = probs / probs_sum
                unit_start_locations.num_unit_probs = dict(zip(counts.tolist(), probs.tolist()))
                self._can_have_inactive_units = True
        elif isinstance(unit_start_locations, RandomWiggleUnitLocationsConfig):
            self.num_units = unit_start_locations.num_units
            self.unit_start_locations = unit_start_locations
        elif isinstance(unit_start_locations, list):
            self.num_units = len(unit_start_locations)
            self.unit_start_locations = _normalize_tuples2or3(unit_start_locations, 0.0)
        else:
            raise ValueError(unit_start_locations)

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

    @property
    def can_have_inactive_units(self) -> bool:
        return self._can_have_inactive_units

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

    def _create_swarm_spec(self, rng: np.random.Generator | None = None) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody: MjsBody = spec.worldbody
        if rng is None:
            rng = np.random.default_rng()

        unit_start_locations: list[Tuple3[FloatOrDistParams]]
        if isinstance(self.unit_start_locations, RandomLatticeUnitLocationsConfig):
            unit_start_locations = self._generate_random_lattice_start_locations(
                rng=rng,
                force_full=True
            )
        elif isinstance(self.unit_start_locations, RandomWiggleUnitLocationsConfig):
            unit_start_locations = self.unit_start_locations.unit_locations
        else:
            unit_start_locations = self.unit_start_locations

        for i, unit_start_location in enumerate(unit_start_locations):
            unit_start_location = np.array([eval_fodp(coord, rng) for coord in unit_start_location])
            unit_start_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
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
            start_location: np.ndarray,
            parking_location: np.ndarray,
    ) -> tuple[SwarmConnections, Optional[np.ndarray]]:
        connections = SwarmConnections(self.config)

        start_locations: list[Tuple3[float]]
        if isinstance(self.unit_start_locations, RandomLatticeUnitLocationsConfig):
            start_locations = self._generate_random_lattice_start_locations(rng)
        elif isinstance(self.unit_start_locations, RandomWiggleUnitLocationsConfig):
            start_locations = []
            for unit_loc, unit_wiggle in zip(
                    self.unit_start_locations.unit_locations,
                    self.unit_start_locations.wiggle_params
            ):
                # noinspection PyTypeChecker
                start_locations.append(tuple(
                    coord + eval_fodp(wiggle, rng)
                    for coord, wiggle in zip(unit_loc, unit_wiggle)
                ))
        else:
            start_locations = [
                eval_fodp_3d(unit_loc, rng)
                for unit_loc in self.unit_start_locations
            ]

        for i, unit_start_location in enumerate(start_locations):
            qpos_adr, dof_adr = self._get_unit_main_body_addresses(model, i)

            data.qpos[qpos_adr:qpos_adr + 3] = start_location + np.array(unit_start_location)

            if self.randomize_unit_orientations:
                data.qpos[qpos_adr + 3:qpos_adr + 7] = random_quat_shoemake()
            elif self.unit_start_quats is not None:
                data.qpos[qpos_adr + 3:qpos_adr + 7] = self.unit_start_quats[i]
            else:
                data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]

            data.qvel[dof_adr:dof_adr+6] = 0

        for i in range(len(start_locations), self.num_units):
            qpos_adr, dof_adr = self._get_unit_main_body_addresses(model, i)

            data.qpos[qpos_adr:qpos_adr + 3] = parking_location + np.array([0.0, -1.0, 0.0]) * i  # todo: improve
            data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
            data.qvel[dof_adr:dof_adr + 6] = 0

        data.ctrl[:] = 0

        units_active_mask: Optional[np.ndarray] = None
        if self.can_have_inactive_units:
            units_active_mask = np.zeros(self.num_units, dtype=bool)
            units_active_mask[:len(start_locations)] = True

        return connections, units_active_mask

    def _get_unit_main_body_addresses(self, model: mujoco.MjModel, unit_idx: int):
        body_name = f"{self.config.unit_prefixes[unit_idx]}-main_body"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        jnt_adr = model.body_jntadr[body_id]
        qpos_adr = model.jnt_qposadr[jnt_adr]
        dof_adr = model.jnt_dofadr[jnt_adr]
        return qpos_adr, dof_adr

    def _generate_random_lattice_start_locations(
            self,
            rng: np.random.Generator,
            eps: float = 1e-6,
            force_full: bool = False
    ) -> list[tuple[float, float, float]]:
        random_config: RandomLatticeUnitLocationsConfig = self.unit_start_locations
        num_units = random_config.num_units
        pairwise_distance = random_config.pairwise_distance
        max_distance = random_config.max_radius

        if force_full or random_config.num_unit_probs is None:
            num_active_units = num_units
        else:
            unit_probs = random_config.num_unit_probs
            num_active_units = rng.choice(list(unit_probs.keys()), p=list(unit_probs.values()))
        if num_active_units < 1:
            raise ValueError("num_active_units must be >= 1")

        points = np.zeros((num_active_units, 2))
        points_found = 1
        rejected_count = 0

        while points_found < num_active_units:
            source_point = points[rng.choice(points_found)]

            angle = rng.random() * 2 * np.pi
            proposal_x = np.cos(angle) * pairwise_distance
            proposal_y = np.sin(angle) * pairwise_distance

            target_point = source_point + np.array([proposal_x, proposal_y])

            if float(np.linalg.norm(target_point)) > max_distance:
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

            counter = 0
            while np.any(np.linalg.norm(points, axis=-1) > max_distance) and counter < 4:
                points += mean / 4
                counter += 1


        return [(float(p[0]), float(p[1]), random_config.z_pos) for p in points]
