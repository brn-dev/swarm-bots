import abc

import mujoco
import numpy as np

from swarmbots.swarm.swarm_config import SwarmConfig
from swarmbots.swarm.swarm_connections import SwarmConnections


class BaseSwarm(abc.ABC):

    def __init__(self, config: SwarmConfig):
        self.config = config

    @abc.abstractmethod
    def _create_swarm_spec(self) -> mujoco.MjSpec:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset_swarm(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            rng: np.random.Generator,
            start_location: np.ndarray
    ) -> SwarmConnections:
        """
        initialize state and potentially randomize swarm
        """
        raise NotImplementedError()

    def create_swarm_spec(self) -> mujoco.MjSpec:
        spec = self._create_swarm_spec()
        self.add_eq_constraints(spec)
        return spec

    def add_eq_constraints(self, spec: mujoco.MjSpec):
        for unit1 in range(self.config.num_units - 1):
            for unit2 in range(unit1 + 1, self.config.num_units):
                for conn1 in range(self.config.limbs_per_unit):
                    for conn2 in range(self.config.limbs_per_unit):
                        eq = spec.add_equality(
                            name=self.config.get_eq_name(unit1, conn1, unit2, conn2),
                            type=mujoco.mjtEq.mjEQ_WELD,
                            objtype=mujoco.mjtObj.mjOBJ_BODY,
                            name1=self.config.get_connector_name(unit1, conn1),
                            name2=self.config.get_connector_name(unit2, conn2),
                            active=False,
                        )

                        eq.data[:3] = [0, 0, 0]  # anchor
                        eq.data[3:6] = [0, 0, 0]  # relpose pos
                        eq.data[6:10] = [0, 1, 0, 0]  # relpose quat
                        eq.data[10] = self.config.connection_torquescale  # torquescale

