import enum
from dataclasses import dataclass

import numpy as np


class LimbType(int, enum.Enum):
    yx = 0
    zx = 1

@dataclass
class LimbConfig:
    name: str
    vec: np.ndarray
    type: LimbType
    rgba: tuple[float, float, float, float]

UnitConfig = list[LimbConfig]

UNIT_CONFIG_CUBE_ZX: UnitConfig = [
    LimbConfig('xp', np.array([ 1,  0,  0]), LimbType.zx, (  1,   0,   0,   1)),
    LimbConfig('xn', np.array([-1,  0,  0]), LimbType.zx, (0.7, 0.3,   0,   1)),
    LimbConfig('yp', np.array([ 0,  1,  0]), LimbType.zx, (  0,   1,   0,   1)),
    LimbConfig('yn', np.array([ 0, -1,  0]), LimbType.zx, (  0, 0.7, 0.3,   1)),
    LimbConfig('zp', np.array([ 0,  0,  1]), LimbType.zx, (  0,   0,   1,   1)),
    LimbConfig('zn', np.array([ 0,  0, -1]), LimbType.zx, (0.3,   0, 0.7,   1))
]