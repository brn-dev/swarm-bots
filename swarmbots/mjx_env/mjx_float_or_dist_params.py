import abc
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class MjxDistParams(abc.ABC):
    pass


@dataclass(frozen=True, slots=True)
class MjxBoundedDistParams(MjxDistParams, abc.ABC):
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class MjxUniformDistParams(MjxBoundedDistParams):
    @staticmethod
    def from_midpoint_and_width(midpoint: float, width: float) -> "MjxUniformDistParams":
        width_half = width / 2.0
        return MjxUniformDistParams(low=midpoint - width_half, high=midpoint + width_half)


MjxTruncatedNormalDistSamplingMode = Literal["clamp", "rejection"]


@dataclass(frozen=True, slots=True)
class MjxTruncatedNormalDistParams(MjxBoundedDistParams):
    mean: float
    std: float
    sampling_mode: MjxTruncatedNormalDistSamplingMode = "clamp"


@dataclass(frozen=True, slots=True)
class MjxNormalDistParams(MjxDistParams):
    mean: float
    std: float


MjxFloatOrDistParams = float | MjxDistParams
MjxFloatOrBoundedDistParams = float | MjxBoundedDistParams
MjxFloatOrDistParams2D = tuple[MjxFloatOrDistParams, MjxFloatOrDistParams]


def mjx_fodp_low(value: MjxFloatOrBoundedDistParams) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    return float(value.low)


def mjx_eval_fodp(value: MjxFloatOrDistParams, rng: np.random.Generator) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, MjxUniformDistParams):
        return float(rng.uniform(low=value.low, high=value.high))
    if isinstance(value, MjxTruncatedNormalDistParams):
        return _eval_truncated_normal(value, rng)
    if isinstance(value, MjxNormalDistParams):
        return float(rng.normal(value.mean, value.std))
    raise ValueError(f"{value = }")


def mjx_eval_fodp_2d(
    value: MjxFloatOrDistParams2D,
    rng: np.random.Generator,
) -> tuple[float, float]:
    return mjx_eval_fodp(value[0], rng), mjx_eval_fodp(value[1], rng)


def _eval_truncated_normal(value: MjxTruncatedNormalDistParams, rng: np.random.Generator) -> float:
    sample = rng.normal(value.mean, value.std)
    if value.sampling_mode == "clamp":
        return float(np.clip(sample, value.low, value.high))
    if value.sampling_mode == "rejection":
        for _ in range(10_000):
            if value.low <= sample <= value.high:
                return float(sample)
            sample = rng.normal(value.mean, value.std)
        raise RuntimeError(f"Failed to sample truncated normal after many retries: {value}")
    raise ValueError(f"Unknown sampling mode: {value.sampling_mode}")
