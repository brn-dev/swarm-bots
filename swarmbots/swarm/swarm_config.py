from dataclasses import dataclass


@dataclass
class SwarmConfig:
    num_units: int
    limbs_per_unit: int

