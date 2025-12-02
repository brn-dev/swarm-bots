from dataclasses import dataclass

from swarmbots.swarm.unit_config import UnitConfig


def get_connector_suffix(connector: int):
    return f'-{connector}-connector'

@dataclass
class SwarmConfig:

    def __init__(
            self,
            num_units: int,
            unit_config: UnitConfig,
            connection_torquescale: float,
            connection_dist_threshold: float = 0.1,
            connection_angle_threshold: float = -0.5,
            disconnect_potential_threshold: float = 5,
    ):

        self.num_units = num_units
        self.unit_config = unit_config

        self.connection_torquescale = connection_torquescale
        self.connection_dist_threshold = connection_dist_threshold
        self.connection_angle_threshold = connection_angle_threshold

        self.disconnect_potential_threshold = disconnect_potential_threshold

        self.limbs_per_unit = len(unit_config)
        self.limb_name_to_idx = {
            limb_cfg.name: i
            for i, limb_cfg in enumerate(self.unit_config)
        }

        self.unit_prefixes = [f'Unit{i}-' for i in range(0, self.num_units)]

    def get_connector_name(self, unit: int, connector: int):
        return f'{self.unit_prefixes[unit]}{get_connector_suffix(connector)}'

    def get_eq_name(self, unit1: int, conn1: int, unit2: int, conn2: int):
        return f'eq_{unit1}-{conn1}_{unit2}-{conn2}'



