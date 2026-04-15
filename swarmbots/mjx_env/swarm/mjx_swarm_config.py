from dataclasses import dataclass

from swarmbots.mjx_env.swarm.mjx_unit_config import MjxUnitConfig


MJX_TWIST_BINS = 4


def mjx_get_connector_suffix(connector: int) -> str:
    return f"-{connector}-connector"


@dataclass(frozen=True)
class MjxSwarmConfig:
    num_units: int
    unit_config: MjxUnitConfig
    connection_torquescale: float
    twist_bins: int = MJX_TWIST_BINS

    def __post_init__(self) -> None:
        object.__setattr__(self, "limbs_per_unit", len(self.unit_config))
        object.__setattr__(self, "unit_prefixes", [f"Unit{i}-" for i in range(self.num_units)])

    def get_settings(self) -> dict:
        return {
            "num_units": self.num_units,
            "unit_config": [limb.to_json() for limb in self.unit_config],
            "connection_torquescale": self.connection_torquescale,
            "twist_bins": self.twist_bins,
        }

    def get_connector_name(self, unit: int, connector: int) -> str:
        return f"{self.unit_prefixes[unit]}{mjx_get_connector_suffix(connector)}"

    def get_eq_name(self, unit1: int, conn1: int, unit2: int, conn2: int, twist_bin: int) -> str:
        return f"eq_{unit1}-{conn1}_{unit2}-{conn2}_twist{twist_bin}"
