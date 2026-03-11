import mujoco
from typing import TypeAlias

from swarmbots.mj_env.swarm.swarm_config import get_connector_suffix
from swarmbots.mj_env.swarm.unit_config import UnitConfig, LimbConfig, LimbType

HingeJointParam: TypeAlias = float | tuple[float, ...]


def _limb_joint_specs(limb_type: LimbType) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    if limb_type == LimbType.xy:
        return (
            ("hinge1x", (1.0, 0.0, 0.0)),
            ("hinge2y", (0.0, 1.0, 0.0)),
        )
    if limb_type == LimbType.zx:
        return (
            ("hinge1z", (0.0, 0.0, 1.0)),
            ("hinge2x", (1.0, 0.0, 0.0)),
        )
    if limb_type == LimbType.xyz:
        return (
            ("hinge1x", (1.0, 0.0, 0.0)),
            ("hinge2y", (0.0, 1.0, 0.0)),
            ("hinge3z", (0.0, 0.0, 1.0)),
        )
    raise NotImplementedError(limb_type)


def _resolve_joint_params(value: HingeJointParam, n_joints: int) -> tuple[float, ...]:
    if isinstance(value, tuple):
        if len(value) != n_joints:
            raise ValueError(len(value), n_joints)
        return value
    scalar = float(value)
    return (scalar,) * n_joints


def init_unit(
    body_radius: float,
    leg_length: float,
    leg_radius: float,
    hinge_range: tuple[float | None, ...],
    unit_config: UnitConfig,
    body_rgba=(0.75, 0, 0, 0.1),
    segment_1_ratio: float = 0.1,
    hinge_armature: HingeJointParam = 0.0,
    hinge_damping: HingeJointParam = 0.0,
    hinge_frictionloss: HingeJointParam = 0.0,
) -> mujoco.MjsBody:
    spec = mujoco.MjSpec()
    spec.compiler.degree = False

    worldbody = spec.worldbody

    # Main Body
    body = worldbody.add_body(name='-main_body', pos=[0, 0, 0])

    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[body_radius, 0, 0], # Sphere only uses first size param
        rgba=body_rgba
    )

    for i, limb_config in enumerate(unit_config):
        hip_pos = body_radius * limb_config.vec
        joint_specs = _limb_joint_specs(limb_config.type)
        joint_ranges = tuple(r if r is not None else 0.0 for r in hinge_range)
        joint_armatures = _resolve_joint_params(hinge_armature, len(joint_specs))
        joint_dampings = _resolve_joint_params(hinge_damping, len(joint_specs))
        joint_frictionlosses = _resolve_joint_params(hinge_frictionloss, len(joint_specs))

        limb_root = body.add_body(
            name=f'-{i}-limb_root',
            pos=hip_pos,
            zaxis=limb_config.vec
        )

        _build_limb(
            spec=spec,
            parent_body=limb_root,
            limb_idx=i,
            limb_config=limb_config,
            length=leg_length,
            radius=leg_radius,
            hinge_ranges=joint_ranges,
            short_segment_ratio=segment_1_ratio,
            joint_specs=joint_specs,
            joint_armatures=joint_armatures,
            joint_dampings=joint_dampings,
            joint_frictionlosses=joint_frictionlosses,
        )

    return body

def _build_limb(
        spec: mujoco.MjSpec,
        parent_body: mujoco.MjsBody,
        limb_idx: int,
        limb_config: LimbConfig,
        length: float,
        radius: float,
        hinge_ranges: tuple[float, ...],
        short_segment_ratio: float,
        joint_specs: tuple[tuple[str, tuple[float, float, float]], ...],
        joint_armatures: tuple[float, ...],
        joint_dampings: tuple[float, ...],
        joint_frictionlosses: tuple[float, ...],
):
    rgba = tuple(limb_config.rgba)
    if len(joint_specs) < 2:
        raise ValueError(f"Expected at least 2 joints per limb, got {len(joint_specs)}")

    segment_lengths = [length * short_segment_ratio]
    remaining_length = length * (1 - short_segment_ratio)
    segment_lengths.extend([remaining_length / (len(joint_specs) - 1)] * (len(joint_specs) - 1))

    hinge_names: list[str] = []
    parent_segment = parent_body
    tip_offset = 0.0

    for joint_idx, ((joint_suffix, axis), seg_length) in enumerate(zip(joint_specs, segment_lengths, strict=True)):
        segment_name = limb_config.name if joint_idx == 0 else f'-{limb_idx}-seg{joint_idx + 1}'
        segment_body = parent_segment.add_body(name=segment_name, pos=[0, 0, tip_offset])
        hinge_name = f'-{limb_idx}-{joint_suffix}'
        segment_body.add_joint(
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=list(axis),
            range=[-hinge_ranges[joint_idx], hinge_ranges[joint_idx]],
            name=hinge_name,
            armature=joint_armatures[joint_idx],
            damping=joint_dampings[joint_idx],
            frictionloss=joint_frictionlosses[joint_idx],
        )
        segment_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            fromto=[0, 0, 0, 0, 0, seg_length],
            size=[radius, 0, 0],
            rgba=rgba,
        )
        hinge_names.append(hinge_name)
        parent_segment = segment_body
        tip_offset = seg_length

    connector: mujoco.MjsBody = parent_segment.add_body(
        name=get_connector_suffix(limb_idx),
        pos=[0, 0, tip_offset]
    )
    connector.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[0, 0, 0, 0, 0, -length * 0.05],
        size=[radius * 1.1, 0, 0],
        rgba=rgba
    )

    for actuator_idx, hinge_name in enumerate(hinge_names):
        spec.add_actuator(
            target=hinge_name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            name=f'-{limb_idx}-actuator{actuator_idx}'
        )
