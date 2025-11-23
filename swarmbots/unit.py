import enum
import uuid
from typing import Optional, Iterable
import numpy as np
import mujoco

class HingeType(int, enum.Enum):
    yx = 0
    zx = 1

limb_directions: dict[int, dict[str, np.ndarray]] = {
    6: {
        'xp': np.array([1, 0, 0]),
        'xn': np.array([-1, 0, 0]),
        'yp': np.array([0, 1, 0]),
        'yn': np.array([0, -1, 0]),
        'zp': np.array([0, 0, 1]),
        'zn': np.array([0, 0, -1])
    }
}

def init_unit(
    body_radius: float,
    leg_length: float,
    leg_radius: float,
    hinge_range: float,
    num_limbs: int = 6,
    body_rgba=(0.75, 0, 0, 0.1),
    leg_rgba=(0, 0, 0, 1),
    hinge_type: HingeType = HingeType.zx,
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

    for direction_name, direction in limb_directions[num_limbs].items():
        direction = np.array(direction)
        hip_pos = body_radius * direction

        limb_root = body.add_body(
            name=f'limb_root_{direction_name}',
            pos=hip_pos,
            zaxis=direction
        )

        _build_limb(
            spec=spec,
            parent_body=limb_root,
            length=leg_length,
            radius=leg_radius,
            hinge_range=hinge_range,
            rgba=leg_rgba,
            name=f'limb_{direction_name}',
            hinge_type=hinge_type,
            segment_1_ratio=segment_1_ratio
        )

    return body

def _build_limb(
        spec: mujoco.MjSpec,
        parent_body: mujoco.MjsBody,
        length: float,
        radius: float,
        hinge_range: float,
        rgba: Iterable[float],
        name: str,
        hinge_type: HingeType,
        segment_1_ratio: float,
):
    rgba = tuple(rgba)

    first_segment = parent_body.add_body(name=name)

    if hinge_type == HingeType.yx:
        hinge1_name = f'{name}-hinge1y'
        hinge1 = first_segment.add_joint(
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 1, 0],
            range=[-hinge_range, hinge_range],
            name=hinge1_name
        )
    elif hinge_type == HingeType.zx:
        hinge1_name = f'{name}-hinge1z'
        hinge1 = first_segment.add_joint(
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 0, 1],
            name=hinge1_name
        )
    else:
        raise NotImplementedError(hinge_type)

    length1 = length * segment_1_ratio

    first_segment.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[0, 0, 0, 0, 0, length1],
        size=[radius, 0, 0], # cylinder radius
        rgba=rgba
    )

    second_segment = first_segment.add_body(
        name=f'{name}-seg2',
        pos=[0, 0, length1]
    )

    hinge2_name = f'{name}-hinge2x'
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

    tip: mujoco.MjsBody = second_segment.add_body(
        name=f'{name}-tip',
        pos=[0, 0, length2]
    )
    tip.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[0, 0, 0, 0, 0, -length * 0.05],
        size=[radius * 1.1, 0, 0],
        rgba=rgba
    )

    second_segment.add_site(
        name=f'{name}-tip-site',
        pos=[0, 0, length2 * 1.]
    )

    spec.add_actuator(
        target=hinge1_name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        name=f'{name}-actuator0'
    )
    spec.add_actuator(
        target=hinge2_name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        name=f'{name}-actuator1'
    )
