import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class GSDEResetMode(abc.ABC):
    pass


@dataclass(frozen=True)
class GSDEIntervalResetMode(GSDEResetMode):
    interval: int


@dataclass(frozen=True)
class GSDEProbabilityResetMode(GSDEResetMode):
    probability: float


