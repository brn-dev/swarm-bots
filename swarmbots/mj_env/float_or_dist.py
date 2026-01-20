from dataclasses import dataclass

import numpy as np


@dataclass
class UniformDistParams:
    low: float
    high: float

    @staticmethod
    def from_midpoint_and_width(midpoint: float, width: float) -> "UniformDistParams":
        width_half = width / 2.0
        return UniformDistParams(
            low=midpoint - width_half,
            high=midpoint + width_half
        )


@dataclass
class ClampedNormalDistParams:
    mean: float
    std: float
    high: float
    low: float


DistParams = UniformDistParams | ClampedNormalDistParams
FloatOrDist = float | DistParams


def fod_low(fod: FloatOrDist) -> float:
    if isinstance(fod, (float, int)):
        return float(fod)
    if isinstance(fod, UniformDistParams):
        return float(fod.low)
    if isinstance(fod, ClampedNormalDistParams):
        return float(fod.low)
    raise ValueError(fod)


def eval_fod(fod: FloatOrDist, rng: np.random.Generator) -> float:
    if isinstance(fod, (float, int)):
        return float(fod)
    if isinstance(fod, UniformDistParams):
        return float(rng.uniform(low=fod.low, high=fod.high))
    if isinstance(fod, ClampedNormalDistParams):
        x = rng.normal(fod.mean, fod.std)
        x = np.clip(x, fod.low, fod.high)
        return float(x)
    raise ValueError(fod)
