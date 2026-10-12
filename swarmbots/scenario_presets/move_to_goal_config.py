from __future__ import annotations

import math
from dataclasses import dataclass

from swarmbots.mjw_env.parameter_distributions import FloatOrDistParams, UniformDistParams


@dataclass(frozen=True, slots=True)
class AbsoluteGoalConfig:
    x: FloatOrDistParams = 0.0
    y: FloatOrDistParams = 3.0


@dataclass(frozen=True, slots=True)
class RelativePolarGoalConfig:
    distance: FloatOrDistParams
    angle: FloatOrDistParams = UniformDistParams(0.0, 2.0 * math.pi)


MoveToGoalConfig = AbsoluteGoalConfig | RelativePolarGoalConfig
