import abc
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from loguru import logger


@dataclass(frozen=True, slots=True)
class DistParams(abc.ABC):
    pass


@dataclass(frozen=True, slots=True)
class BoundedDistParams(DistParams, abc.ABC):
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class UniformDistParams(BoundedDistParams):
    @staticmethod
    def from_midpoint_and_width(midpoint: float, width: float) -> "UniformDistParams":
        width_half = width / 2.0
        return UniformDistParams(
            low=midpoint - width_half,
            high=midpoint + width_half
        )


@dataclass(frozen=True, slots=True)
class SplitUniformDistParams(BoundedDistParams):
    margin: float

    def __post_init__(self) -> None:
        if self.margin < 0.0:
            raise ValueError(f"Expected margin >= 0, got {self.margin}")
        if self.low >= -self.margin or self.high <= self.margin:
            raise ValueError(
                "Expected non-empty intervals on both sides of the margin, "
                f"got low={self.low}, high={self.high}, margin={self.margin}"
            )


TruncatedNormalDistSamplingMode = Literal['clamp', 'rejection']

@dataclass(frozen=True, slots=True)
class TruncatedNormalDistParams(BoundedDistParams):
    mean: float
    std: float
    sampling_mode: TruncatedNormalDistSamplingMode = 'clamp'
    max_rejection_attempts: int = 1_000

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.low, self.high, self.mean, self.std)):
            raise ValueError("Truncated normal parameters must be finite")
        if self.low > self.high or self.std < 0:
            raise ValueError("Truncated normal requires low <= high and std >= 0")
        if self.sampling_mode not in ("clamp", "rejection"):
            raise ValueError(f"Unsupported sampling_mode={self.sampling_mode!r}")
        if not isinstance(self.max_rejection_attempts, int) or self.max_rejection_attempts <= 0:
            raise ValueError("max_rejection_attempts must be a positive integer")
        if self.sampling_mode == "rejection":
            if self.low == self.high and self.std > 0:
                raise ValueError("Rejection sampling requires a non-empty interval when std > 0")
            if self.std == 0 and not self.low <= self.mean <= self.high:
                raise ValueError("Rejection sampling with std=0 requires mean within [low, high]")

@dataclass(frozen=True, slots=True)
class NormalDistParams(DistParams):
    mean: float
    std: float


FloatOrDistParams = float | DistParams
FloatOrBoundedDistParams = float | BoundedDistParams

REJECTION_SAMPLING_WARNING_THRESHOLD = 10


def eval_fodp(fodp: FloatOrDistParams, rng: np.random.Generator) -> float:
    if isinstance(fodp, (float, int)):
        return float(fodp)
    if isinstance(fodp, UniformDistParams):
        return float(rng.uniform(low=fodp.low, high=fodp.high))
    if isinstance(fodp, SplitUniformDistParams):
        if rng.integers(0, 2) == 0:
            return float(rng.uniform(low=fodp.low, high=-fodp.margin))
        return float(rng.uniform(low=fodp.margin, high=fodp.high))
    if isinstance(fodp, TruncatedNormalDistParams):
        return eval_truncated_normal(fodp, rng)
    if isinstance(fodp, NormalDistParams):
        return rng.normal(fodp.mean, fodp.std)
    raise ValueError(f'{fodp = }')


def eval_truncated_normal(dp: TruncatedNormalDistParams, rng: np.random.Generator):
    x = rng.normal(dp.mean, dp.std)

    if dp.sampling_mode == 'clamp':
        x = np.clip(x, dp.low, dp.high)
    elif dp.sampling_mode == 'rejection':
        counter = 0
        while x < dp.low or x > dp.high:
            if counter >= dp.max_rejection_attempts - 1:
                raise RuntimeError(
                    f"Truncated normal rejection sampling exceeded {dp.max_rejection_attempts} attempts: {dp}"
                )
            counter += 1
            x = rng.normal(dp.mean, dp.std)
        if counter > REJECTION_SAMPLING_WARNING_THRESHOLD:
            logger.warning(f'Rejected {counter} samples for {dp}')
    else:
        raise ValueError(f'{dp.sampling_mode } = ')

    return float(x)
