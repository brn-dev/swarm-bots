from dataclasses import dataclass
from typing import Collection, TypeAlias

import mujoco
import numpy as np

from swarmbots.mjx_env.mjx_quat import mjx_random_quat_shoemake
from swarmbots.mjx_env.swarm.mjx_base_swarm import MjxBaseSwarm, MjxSwarmReset, mjx_empty_swarm_reset
from swarmbots.mjx_env.swarm.mjx_swarm_config import MjxSwarmConfig
from swarmbots.mjx_env.swarm.mjx_unit import MjxHingeJointParam, mjx_init_unit
from swarmbots.mjx_env.swarm.mjx_unit_config import MJX_UNIT_CONFIG_TETRAHEDRON_XY, MjxUnitConfig

Tuple3: TypeAlias = tuple[float, float, float]


@dataclass
class MjxPreConnectedUnitLocationsConfig:
    num_units: int
    num_unit_probs: dict[int, float] | None = None
    unconnected_prob: float = 0.0
    max_radius: float = 1e8
    z_pos: float = 0.0
    center: bool = True
    pool_seeds: Collection[int] | None = None


class _GeneratedSwarm:
    def __init__(
        self,
        positions: np.ndarray,
        quats: np.ndarray,
        connections: np.ndarray,
        twist_angles: np.ndarray,
    ) -> None:
        self.positions = positions
        self.quats = quats
        self.connections = connections
        self.twist_angles = twist_angles


class MjxHomogeneousSwarm(MjxBaseSwarm):
    def __init__(
        self,
        unit_start_locations: MjxPreConnectedUnitLocationsConfig,
        unit_config: MjxUnitConfig = MJX_UNIT_CONFIG_TETRAHEDRON_XY,
        body_radius: float = 0.1,
        leg_length: float = 0.2,
        leg_radius: float = 0.025,
        hinge_range: tuple[float | None, ...] = (np.pi / 3, np.pi / 3),
        hinge_armature: MjxHingeJointParam = (0.0, 0.0),
        hinge_damping: MjxHingeJointParam = (0.0, 0.0),
        hinge_frictionloss: MjxHingeJointParam = (0.0, 0.0),
        connection_torquescale: float = 50.0,
    ) -> None:
        if not isinstance(unit_start_locations, MjxPreConnectedUnitLocationsConfig):
            raise TypeError("MJX HomogeneousSwarm currently supports only MjxPreConnectedUnitLocationsConfig")
        if not 0.0 <= unit_start_locations.unconnected_prob <= 1.0:
            raise ValueError("unconnected_prob must be in [0.0, 1.0]")
        if unit_start_locations.num_unit_probs is not None:
            counts = np.array(list(unit_start_locations.num_unit_probs.keys()), dtype=int)
            probs = np.array(list(unit_start_locations.num_unit_probs.values()), dtype=float)
            if counts.max() != unit_start_locations.num_units:
                raise ValueError("num_unit_probs must contain an entry for count == num_units")
            if (counts < 1).any() or (counts > unit_start_locations.num_units).any():
                raise ValueError("num_unit_probs entries must be in [1, num_units]")
            if (probs < 0).any() or probs.sum() <= 0:
                raise ValueError("num_unit_probs probabilities must be non-negative and sum positive")
            probs = probs / probs.sum()
            unit_start_locations.num_unit_probs = dict(zip(counts.tolist(), probs.tolist()))
        if unit_start_locations.pool_seeds is not None:
            pool_seeds = tuple(sorted(set(int(seed) for seed in unit_start_locations.pool_seeds)))
            if not pool_seeds:
                raise ValueError("pool_seeds must contain at least one seed")
            unit_start_locations.pool_seeds = pool_seeds

        self.unit_start_locations = unit_start_locations
        self.num_units = unit_start_locations.num_units
        self.body_radius = float(body_radius)
        self.leg_length = float(leg_length)
        self.leg_radius = float(leg_radius)
        self.hinge_range = hinge_range
        self.hinge_armature = hinge_armature
        self.hinge_damping = hinge_damping
        self.hinge_frictionloss = hinge_frictionloss
        self._can_have_inactive_units = unit_start_locations.num_unit_probs is not None

        super().__init__(
            config=MjxSwarmConfig(
                num_units=self.num_units,
                unit_config=unit_config,
                connection_torquescale=connection_torquescale,
            ),
            max_unit_extent=body_radius + leg_length,
        )

    @property
    def can_have_inactive_units(self) -> bool:
        return self._can_have_inactive_units

    def get_settings(self) -> dict:
        settings = super().get_settings()
        settings.update(
            {
                "unit_start_locations": self.unit_start_locations,
                "body_radius": self.body_radius,
                "leg_length": self.leg_length,
                "leg_radius": self.leg_radius,
                "hinge_range": self.hinge_range,
                "hinge_armature": self.hinge_armature,
                "hinge_damping": self.hinge_damping,
                "hinge_frictionloss": self.hinge_frictionloss,
            }
        )
        return settings

    def _create_swarm_spec(self, rng: np.random.Generator | None = None) -> mujoco.MjSpec:
        if rng is None:
            rng = np.random.default_rng(42)
        generated = self._generate_preconnected_swarm(rng, force_full=True)

        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        for unit_idx in range(self.num_units):
            shade = 1.0 - (unit_idx / (self.num_units - 1)) if self.num_units > 1 else 1.0
            unit = mjx_init_unit(
                body_radius=self.body_radius,
                leg_length=self.leg_length,
                leg_radius=self.leg_radius,
                hinge_range=self.hinge_range,
                hinge_armature=self.hinge_armature,
                hinge_damping=self.hinge_damping,
                hinge_frictionloss=self.hinge_frictionloss,
                unit_config=self.config.unit_config,
                body_rgba=(shade, shade, shade, 0.35),
            )
            unit.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
            spec.worldbody.add_frame(
                pos=generated.positions[unit_idx].tolist(),
                quat=generated.quats[unit_idx].tolist(),
            ).attach_body(unit, self.config.unit_prefixes[unit_idx], "")

        return spec

    def make_reset(
        self,
        rng: np.random.Generator,
        swarm_start_location: np.ndarray,
        inactive_unit_positions: np.ndarray,
    ) -> MjxSwarmReset:
        generated = self._generate_preconnected_swarm(rng, force_full=False)
        positions = np.array(inactive_unit_positions, dtype=float, copy=True)
        quats = np.zeros((self.num_units, 4), dtype=float)
        quats[:, 0] = 1.0

        num_active_units = generated.positions.shape[0]
        positions[:num_active_units] = swarm_start_location + generated.positions
        quats[:num_active_units] = generated.quats

        units_active_mask = np.ones(self.num_units, dtype=bool)
        if self.can_have_inactive_units:
            units_active_mask[:] = False
            units_active_mask[:num_active_units] = True

        reset = mjx_empty_swarm_reset(
            num_units=self.num_units,
            limbs_per_unit=self.config.limbs_per_unit,
            positions=positions,
            quats=quats,
            units_active_mask=units_active_mask,
        )
        reset.connections[:] = generated.connections
        reset.twist_angles[:] = generated.twist_angles
        reset.twist_bins[:] = self._twist_angles_to_bins(generated.twist_angles)
        reset.twist_angles[:] = reset.twist_bins * (2.0 * np.pi / self.config.twist_bins)
        return reset

    def _generate_preconnected_swarm(self, rng: np.random.Generator, force_full: bool) -> _GeneratedSwarm:
        config = self.unit_start_locations
        if config.pool_seeds is not None:
            rng = np.random.default_rng(int(rng.choice(tuple(config.pool_seeds))))

        num_active_units = config.num_units
        if not force_full and config.num_unit_probs is not None:
            num_active_units = int(rng.choice(list(config.num_unit_probs.keys()), p=list(config.num_unit_probs.values())))
        if num_active_units < 1:
            raise ValueError("num_active_units must be >= 1")

        unit_config = self.config.unit_config
        limb_rot_mats = np.zeros((len(unit_config), 3, 3), dtype=float)
        for limb_idx, limb in enumerate(unit_config):
            quat = np.empty(4, dtype=float)
            mujoco.mju_quatZ2Vec(quat, limb.vec)
            mat = np.empty(9, dtype=float)
            mujoco.mju_quat2Mat(mat, quat)
            limb_rot_mats[limb_idx] = mat.reshape(3, 3)

        positions = np.zeros((num_active_units, 3), dtype=float)
        quats = np.zeros((num_active_units, 4), dtype=float)
        rot_mats = np.zeros((num_active_units, 3, 3), dtype=float)
        connections = np.full((self.num_units, self.config.limbs_per_unit, 2), -1, dtype=np.int32)
        twist_angles = np.zeros((self.num_units, self.config.limbs_per_unit), dtype=np.float32)

        connector_distance = self.body_radius + self.leg_length
        min_center_distance = 2.0 * self.max_unit_extent
        available_connectors = [list(range(len(unit_config))) for _ in range(num_active_units)]

        quats[0] = mjx_random_quat_shoemake(rng)
        rot_mats[0] = self._quat_to_mat(quats[0])

        locations_found = 1
        rejected_count = 0
        while locations_found < num_active_units:
            choices = [
                (unit_idx, conn_idx)
                for unit_idx in range(locations_found)
                for conn_idx in available_connectors[unit_idx]
            ]
            if not choices:
                raise RuntimeError("No available connectors left to build a preconnected swarm")

            unit1, conn1 = choices[int(rng.integers(len(choices)))]
            conn2 = available_connectors[locations_found][int(rng.integers(len(available_connectors[locations_found])))]

            conn1_rot = rot_mats[unit1] @ limb_rot_mats[conn1]
            z1 = conn1_rot[:, 2]
            pos1 = positions[unit1] + z1 * connector_distance
            z2 = -z1
            pos2 = pos1 + z1 * connector_distance

            if float(np.linalg.norm(pos2)) > config.max_radius:
                rejected_count += 1
                if rejected_count > 1000:
                    raise RuntimeError(f"Failed to generate preconnected swarm: {config}")
                continue

            distances = np.linalg.norm(positions[:locations_found] - pos2, axis=1)
            if np.any(distances + 1e-6 < min_center_distance):
                rejected_count += 1
                if rejected_count > 1000:
                    raise RuntimeError(f"Failed to generate preconnected swarm: {config}")
                continue

            twist = float(rng.random() * 2.0 * np.pi)
            x1 = conn1_rot[:, 0]
            y1 = conn1_rot[:, 1]
            x2 = np.cos(twist) * x1 + np.sin(twist) * y1
            x2 /= np.linalg.norm(x2)
            y2 = np.cross(z2, x2)
            y2 /= np.linalg.norm(y2)
            conn2_rot = np.stack([x2, y2, z2], axis=1)
            unit2_rot = conn2_rot @ limb_rot_mats[conn2].T

            positions[locations_found] = pos2
            rot_mats[locations_found] = unit2_rot
            quats[locations_found] = self._mat_to_quat(unit2_rot)
            if config.unconnected_prob <= 0.0 or rng.random() >= config.unconnected_prob:
                self._connect(connections, twist_angles, unit1, conn1, locations_found, conn2, twist)

            available_connectors[unit1].remove(conn1)
            available_connectors[locations_found].remove(conn2)
            locations_found += 1
            rejected_count = 0

        if config.center:
            mean = positions.mean(axis=0, keepdims=True)
            positions -= mean
            counter = 0
            while np.any(np.linalg.norm(positions, axis=-1) > config.max_radius) and counter < 4:
                positions += mean / 4.0
                counter += 1

        positions[:, 2] += config.z_pos
        return _GeneratedSwarm(positions=positions, quats=quats, connections=connections, twist_angles=twist_angles)

    @staticmethod
    def _connect(
        connections: np.ndarray,
        twist_angles: np.ndarray,
        unit1: int,
        conn1: int,
        unit2: int,
        conn2: int,
        twist: float,
    ) -> None:
        connections[unit1, conn1] = [unit2, conn2]
        connections[unit2, conn2] = [unit1, conn1]
        twist_angles[unit1, conn1] = twist
        twist_angles[unit2, conn2] = twist

    def _twist_angles_to_bins(self, twist_angles: np.ndarray) -> np.ndarray:
        bin_width = 2.0 * np.pi / self.config.twist_bins
        return np.mod(np.floor(np.mod(twist_angles, 2.0 * np.pi) / bin_width + 0.5), self.config.twist_bins).astype(
            np.int32
        )

    @staticmethod
    def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
        mat = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(mat, quat)
        return mat.reshape(3, 3)

    @staticmethod
    def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
        quat = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(quat, mat.reshape(9))
        return quat
