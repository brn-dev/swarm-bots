import enum
import uuid
from dataclasses import dataclass
from typing import Optional, Iterable
import numpy as np
import mujoco

class LimbType(int, enum.Enum):
    yx = 0
    zx = 1

@dataclass
class LimbConfig:
    name: str
    vec: np.ndarray
    type: LimbType
    rgba: tuple[float, float, float, float]


CUBE_ZX_LIMB_CONFIGS = [
    ('xp', np.array([ 1,  0,  0]), LimbType.zx, (  1,   0,   0,   1)),
    ('xn', np.array([-1,  0,  0]), LimbType.zx, (0.7, 0.3,   0,   1)),
    ('yp', np.array([ 0,  1,  0]), LimbType.zx, (  0,   1,   0,   1)),
    ('yn', np.array([ 0, -1,  0]), LimbType.zx, (  0, 0.7, 0.3,   1)),
    ('zp', np.array([ 0,  0,  1]), LimbType.zx, (  0,   0,   1,   1)),
    ('zn', np.array([ 0,  0, -1]), LimbType.zx, (0.3,   0, 0.7,   1))
]

def init_unit(
    body_radius: float,
    leg_length: float,
    leg_radius: float,
    hinge_range: float,
    limb_configs: list[LimbConfig],
    body_rgba=(0.75, 0, 0, 0.1),
    segment_1_ratio: float = 0.1
) -> mujoco.MjsBody:
    spec = mujoco.MjSpec()
    spec.compiler.degree = False

    worldbody = spec.worldbody

    # Main Body
    body = worldbody.add_body(name='main_body', pos=[0, 0, 0])

    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[body_radius, 0, 0], # Sphere only uses first size param
        rgba=body_rgba
    )

    for limb_config in limb_configs:
        hip_pos = body_radius * limb_config.vec

        limb_root = body.add_body(
            name=f'limb_root_{limb_config.name}',
            pos=hip_pos,
            zaxis=limb_config.vec
        )

        _build_limb(
            spec=spec,
            parent_body=limb_root,
            length=leg_length,
            radius=leg_radius,
            hinge_range=hinge_range,
            limb_config=limb_config,
            segment_1_ratio=segment_1_ratio
        )

    return body

def _build_limb(
        spec: mujoco.MjSpec,
        parent_body: mujoco.MjsBody,
        limb_config: LimbConfig,
        length: float,
        radius: float,
        hinge_range: float,
        segment_1_ratio: float,
):
    rgba = tuple(limb_config.rgba)

    first_segment = parent_body.add_body(name=limb_config.name)

    if limb_config.type == LimbType.yx:
        hinge1_name = f'{limb_config.name}-hinge1y'
        hinge1 = first_segment.add_joint(
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 1, 0],
            range=[-hinge_range, hinge_range],
            name=hinge1_name
        )
    elif limb_config.type == LimbType.zx:
        hinge1_name = f'{limb_config.name}-hinge1z'
        hinge1 = first_segment.add_joint(
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 0, 1],
            name=hinge1_name
        )
    else:
        raise NotImplementedError(limb_config.type)

    length1 = length * segment_1_ratio

    first_segment.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[0, 0, 0, 0, 0, length1],
        size=[radius, 0, 0], # cylinder radius
        rgba=rgba
    )

    second_segment = first_segment.add_body(
        name=f'{limb_config.name}-seg2',
        pos=[0, 0, length1]
    )

    hinge2_name = f'{limb_config.name}-hinge2x'
    hinge2 = second_segment.add_joint(
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[1, 0, 0],
        range=[-hinge_range, hinge_range],
        name=hinge2_name
    )

    length2 = length * (1 - segment_1_ratio)
    second_segment.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[0, 0, 0, 0, 0, length2],
        size=[radius, 0, 0],
        rgba=rgba
    )

    connector: mujoco.MjsBody = second_segment.add_body(
        name=f'{limb_config.name}-connector',
        pos=[0, 0, length2]
    )
    connector.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[0, 0, 0, 0, 0, -length * 0.05],
        size=[radius * 1.1, 0, 0],
        rgba=rgba
    )

    spec.add_actuator(
        target=hinge1_name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        name=f'{limb_config.name}-actuator0'
    )
    spec.add_actuator(
        target=hinge2_name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        name=f'{limb_config.name}-actuator1'
    )
