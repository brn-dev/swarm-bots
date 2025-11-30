import numpy as np

from swarmbots.swarm.swarm_config import SwarmConfig


class SwarmConnections:

    def __init__(self, config: SwarmConfig):
        self.connections = np.full((config.num_units, config.limbs_per_unit, 2), -1, dtype=int)
        self.twist_angles = np.zeros((config.num_units, config.limbs_per_unit), dtype=np.float32)

    def reset(self):
        self.connections[:] = -1

    def is_connected(
            self,
            unit: int,
            unit_connector: int
    ):
        return self.connections[unit, unit_connector, 0] != -1

    def connect(
            self,
            unit1: int,
            unit1_connector: int,
            unit2: int,
            unit2_connector: int,
            twist_angle: float
    ):
        if unit1 == unit2:
            raise ValueError(f'Cannot connect {unit1} to itself')
        if self.is_connected(unit1, unit1_connector):
            raise ValueError(f'Unit {unit1} connector {unit1_connector} already connected')
        if self.is_connected(unit2, unit2_connector):
            raise ValueError(f'Unit {unit2} connector {unit2_connector} already connected')

        self.connections[unit1, unit1_connector] = [unit2, unit2_connector]
        self.twist_angles[unit1, unit1_connector] = twist_angle

        self.connections[unit2, unit2_connector] = [unit1, unit1_connector]
        self.twist_angles[unit2, unit2_connector] = twist_angle

    def disconnect(
            self,
            unit: int,
            unit_connector: int,
    ) -> tuple[int, int]:
        unit2, unit2_connector = self.connections[unit, unit_connector]
        if unit2 == -1:
            raise ValueError(f'Unit {unit} connector {unit_connector} is not connected')

        # twist angles don't need to be reset necessarily
        self.connections[unit, unit_connector] = [-1, -1]
        self.connections[unit2, unit2_connector] = [-1, -1]

        return unit2, unit2_connector

    def get_is_active(self):
        active_indices = np.where(self.connections[:, :, 0] != -1)

        is_active = np.zeros(self.connections.shape[:2], dtype=bool)
        is_active[active_indices] = True

        return is_active