import abc
from dataclasses import dataclass

import mujoco
import numpy as np

from swarmbots.swarm.swarm_config import SwarmConfig
from swarmbots.swarm.swarm_connections import SwarmConnections


class BaseSwarm(abc.ABC):

    def __init__(self, config: SwarmConfig, seed: int):
        self.rng = np.random.default_rng(seed)
        self.config = config

    @abc.abstractmethod
    def create_swarm_spec(self) -> mujoco.MjSpec:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset_swarm(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[dict, SwarmConnections]:
        """
        initialize state and potentially randomize swarm
        """
        raise NotImplementedError()

    def get_unit_prefixes(self) -> list[str]:
        return [f'Unit{i}--' for i in range(0, self.config.num_units)]

    def add_eq_constraints(self, spec: mujoco.MjSpec):
        for i in range(self.config.num_units - 1):
            for j in range(i + 1, self.config.num_units):
                for conn1 in range(self.config.limbs_per_unit):
                    for conn2 in range(self.config.limbs_per_unit):
                        eq = spec.add_equality(
                            name="site_weld",
                            type=mujoco.mjtEq.mjEQ_WELD,  # or mjEQ_CONNECT
                            objtype=mujoco.mjtObj.mjOBJ_BODY,  # tells MuJoCo the objs are sites
                            name1=f"Unit{i}--connector",
                            name2=f"Unit{j}--connector",
                        )

                        eq.data[:3] = [0, 0, 0]  # anchor
                        eq.data[3:6] = [0, 0, 0]  # relpose pos
                        eq.data[6:10] = [0, 1, 0, 0]  # relpose quat
                        eq.data[10] = 50  # torquescale

