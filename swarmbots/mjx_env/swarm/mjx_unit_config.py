import enum
from dataclasses import dataclass

import numpy as np


def _unit_vec(vec: list[int | float]) -> np.ndarray:
    arr = np.array(vec, dtype=float)
    arr /= np.linalg.norm(arr)
    return arr


class MjxLimbType(int, enum.Enum):
    xy = 0
    zx = 1
    xyz = 2


@dataclass(frozen=True)
class MjxLimbConfig:
    name: str
    vec: np.ndarray
    type: MjxLimbType
    rgba: tuple[float, float, float, float]

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "vec": self.vec.tolist(),
            "type": f"LimbType.{self.type.name}",
            "rgba": self.rgba,
        }


MjxUnitConfig = tuple[MjxLimbConfig, ...]


MJX_UNIT_CONFIG_TETRAHEDRON_ZX: MjxUnitConfig = (
    MjxLimbConfig("ppp", _unit_vec([1, 1, 1]), MjxLimbType.zx, (1, 0, 0, 1)),
    MjxLimbConfig("pmm", _unit_vec([1, -1, -1]), MjxLimbType.zx, (0, 1, 0, 1)),
    MjxLimbConfig("mpm", _unit_vec([-1, 1, -1]), MjxLimbType.zx, (0, 0, 1, 1)),
    MjxLimbConfig("mmp", _unit_vec([-1, -1, 1]), MjxLimbType.zx, (1, 1, 0, 1)),
)

MJX_UNIT_CONFIG_TETRAHEDRON_XY: MjxUnitConfig = (
    MjxLimbConfig("ppp", _unit_vec([1, 1, 1]), MjxLimbType.xy, (1, 0, 0, 1)),
    MjxLimbConfig("pmm", _unit_vec([1, -1, -1]), MjxLimbType.xy, (0, 1, 0, 1)),
    MjxLimbConfig("mpm", _unit_vec([-1, 1, -1]), MjxLimbType.xy, (0, 0, 1, 1)),
    MjxLimbConfig("mmp", _unit_vec([-1, -1, 1]), MjxLimbType.xy, (1, 1, 0, 1)),
)

MJX_UNIT_CONFIG_TETRAHEDRON_XYZ: MjxUnitConfig = (
    MjxLimbConfig("ppp", _unit_vec([1, 1, 1]), MjxLimbType.xyz, (1, 0, 0, 1)),
    MjxLimbConfig("pmm", _unit_vec([1, -1, -1]), MjxLimbType.xyz, (0, 1, 0, 1)),
    MjxLimbConfig("mpm", _unit_vec([-1, 1, -1]), MjxLimbType.xyz, (0, 0, 1, 1)),
    MjxLimbConfig("mmp", _unit_vec([-1, -1, 1]), MjxLimbType.xyz, (1, 1, 0, 1)),
)
