import abc
import math
from dataclasses import dataclass
from typing import NamedTuple

import mujoco
import numpy as np

from swarmbots.mjx_env.swarm.mjx_swarm_config import MjxSwarmConfig


class MjxSwarmReset(NamedTuple):
    positions: np.ndarray
    quats: np.ndarray
    connections: np.ndarray
    twist_bins: np.ndarray
    twist_angles: np.ndarray
    disconnect_potentials: np.ndarray
    units_active_mask: np.ndarray


class MjxBaseSwarm(abc.ABC):
    def __init__(self, config: MjxSwarmConfig, max_unit_extent: float) -> None:
        self.config = config
        self.max_unit_extent = float(max_unit_extent)

    @property
    @abc.abstractmethod
    def can_have_inactive_units(self) -> bool:
        raise NotImplementedError

    def get_settings(self) -> dict:
        return {"config": self.config.get_settings()}

    @abc.abstractmethod
    def _create_swarm_spec(self, rng: np.random.Generator | None = None) -> mujoco.MjSpec:
        raise NotImplementedError

    @abc.abstractmethod
    def make_reset(
        self,
        rng: np.random.Generator,
        swarm_start_location: np.ndarray,
        inactive_unit_positions: np.ndarray,
    ) -> MjxSwarmReset:
        raise NotImplementedError

    def create_swarm_spec(self, rng: np.random.Generator | None = None) -> mujoco.MjSpec:
        if rng is None:
            rng = np.random.default_rng(42)
        spec = self._create_swarm_spec(rng)
        self.add_eq_constraints(spec)
        return spec

    def add_eq_constraints(self, spec: mujoco.MjSpec) -> None:
        twist_bin_width = 2.0 * math.pi / self.config.twist_bins
        for unit1 in range(self.config.num_units - 1):
            for unit2 in range(unit1 + 1, self.config.num_units):
                for conn1 in range(self.config.limbs_per_unit):
                    for conn2 in range(self.config.limbs_per_unit):
                        for twist_bin in range(self.config.twist_bins):
                            twist = twist_bin * twist_bin_width
                            eq = spec.add_equality(
                                name=self.config.get_eq_name(unit1, conn1, unit2, conn2, twist_bin),
                                type=mujoco.mjtEq.mjEQ_WELD,
                                objtype=mujoco.mjtObj.mjOBJ_BODY,
                                name1=self.config.get_connector_name(unit1, conn1),
                                name2=self.config.get_connector_name(unit2, conn2),
                                active=False,
                            )
                            eq.data[:3] = [0, 0, 0]
                            eq.data[3:6] = [0, 0, 0]
                            eq.data[6:10] = [0, math.cos(twist / 2.0), math.sin(twist / 2.0), 0]
                            eq.data[10] = self.config.connection_torquescale


def mjx_empty_swarm_reset(
    num_units: int,
    limbs_per_unit: int,
    positions: np.ndarray,
    quats: np.ndarray,
    units_active_mask: np.ndarray,
) -> MjxSwarmReset:
    return MjxSwarmReset(
        positions=positions,
        quats=quats,
        connections=np.full((num_units, limbs_per_unit, 2), -1, dtype=np.int32),
        twist_bins=np.zeros((num_units, limbs_per_unit), dtype=np.int32),
        twist_angles=np.zeros((num_units, limbs_per_unit), dtype=np.float32),
        disconnect_potentials=np.zeros((num_units, limbs_per_unit), dtype=np.float32),
        units_active_mask=units_active_mask.astype(bool, copy=False),
    )
