from typing import TypeAlias

import mujoco

from swarmbots.mjx_env.swarm.mjx_swarm_config import mjx_get_connector_suffix
from swarmbots.mjx_env.swarm.mjx_unit_config import MjxLimbConfig, MjxLimbType, MjxUnitConfig

MjxHingeJointParam: TypeAlias = float | tuple[float, ...]


def _limb_joint_specs(limb_type: MjxLimbType) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    if limb_type == MjxLimbType.xy:
        return (
            ("hinge1x", (1.0, 0.0, 0.0)),
            ("hinge2y", (0.0, 1.0, 0.0)),
        )
    if limb_type == MjxLimbType.zx:
        return (
            ("hinge1z", (0.0, 0.0, 1.0)),
            ("hinge2x", (1.0, 0.0, 0.0)),
        )
    if limb_type == MjxLimbType.xyz:
        return (
            ("hinge1x", (1.0, 0.0, 0.0)),
            ("hinge2y", (0.0, 1.0, 0.0)),
            ("hinge3z", (0.0, 0.0, 1.0)),
        )
    raise NotImplementedError(limb_type)


def _resolve_joint_params(value: MjxHingeJointParam, n_joints: int) -> tuple[float, ...]:
    if isinstance(value, tuple):
        if len(value) != n_joints:
            raise ValueError(f"Expected {n_joints} joint params, got {len(value)}")
        return tuple(float(x) for x in value)
    return (float(value),) * n_joints


def mjx_get_limb_segment_names(limb_idx: int, limb_config: MjxLimbConfig) -> tuple[str, ...]:
    joint_specs = _limb_joint_specs(limb_config.type)
    return tuple(
        limb_config.name if joint_idx == 0 else f"-{limb_idx}-seg{joint_idx + 1}"
        for joint_idx in range(len(joint_specs))
    )


def mjx_init_unit(
    body_radius: float,
    leg_length: float,
    leg_radius: float,
    hinge_range: tuple[float | None, ...],
    unit_config: MjxUnitConfig,
    body_rgba: tuple[float, float, float, float] = (0.75, 0.0, 0.0, 0.1),
    segment_1_ratio: float = 0.1,
    hinge_armature: MjxHingeJointParam = 0.0,
    hinge_damping: MjxHingeJointParam = 0.0,
    hinge_frictionloss: MjxHingeJointParam = 0.0,
) -> mujoco.MjsBody:
    spec = mujoco.MjSpec()
    spec.compiler.degree = 0
    body = spec.worldbody.add_body(name="-main_body", pos=[0, 0, 0])
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[body_radius, 0, 0],
        rgba=body_rgba,
    )

    for limb_idx, limb_config in enumerate(unit_config):
        hip_pos = body_radius * limb_config.vec
        joint_specs = _limb_joint_specs(limb_config.type)
        joint_ranges = tuple(r if r is not None else 0.0 for r in hinge_range)
        joint_armatures = _resolve_joint_params(hinge_armature, len(joint_specs))
        joint_dampings = _resolve_joint_params(hinge_damping, len(joint_specs))
        joint_frictionlosses = _resolve_joint_params(hinge_frictionloss, len(joint_specs))
        limb_root = body.add_body(name=f"-{limb_idx}-limb_root", pos=hip_pos, zaxis=limb_config.vec)
        _build_limb(
            spec=spec,
            parent_body=limb_root,
            limb_idx=limb_idx,
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
    limb_config: MjxLimbConfig,
    length: float,
    radius: float,
    hinge_ranges: tuple[float, ...],
    short_segment_ratio: float,
    joint_specs: tuple[tuple[str, tuple[float, float, float]], ...],
    joint_armatures: tuple[float, ...],
    joint_dampings: tuple[float, ...],
    joint_frictionlosses: tuple[float, ...],
) -> None:
    if len(joint_specs) < 2:
        raise ValueError(f"Expected at least 2 joints per limb, got {len(joint_specs)}")

    segment_lengths = [length * short_segment_ratio]
    segment_lengths.extend([length * (1.0 - short_segment_ratio) / (len(joint_specs) - 1)] * (len(joint_specs) - 1))
    segment_names = mjx_get_limb_segment_names(limb_idx, limb_config)

    hinge_names: list[str] = []
    parent_segment = parent_body
    tip_offset = 0.0
    for joint_idx, (((joint_suffix, axis), seg_length), segment_name) in enumerate(
        zip(zip(joint_specs, segment_lengths, strict=True), segment_names, strict=True)
    ):
        segment_body = parent_segment.add_body(name=segment_name, pos=[0, 0, tip_offset])
        hinge_name = f"-{limb_idx}-{joint_suffix}"
        segment_body.add_joint(
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=list(axis),
            range=[-hinge_ranges[joint_idx], hinge_ranges[joint_idx]],
            name=hinge_name,
            armature=joint_armatures[joint_idx],
            damping=joint_dampings[joint_idx],
            frictionloss=joint_frictionlosses[joint_idx],
        )
        geom_kwargs: dict[str, int] = {}
        if joint_idx == 0:
            geom_kwargs = {"contype": 0, "conaffinity": 0}
        segment_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, 0, 0, seg_length],
            size=[radius, 0, 0],
            rgba=tuple(limb_config.rgba),
            **geom_kwargs,
        )
        hinge_names.append(hinge_name)
        parent_segment = segment_body
        tip_offset = seg_length

    connector = parent_segment.add_body(name=mjx_get_connector_suffix(limb_idx), pos=[0, 0, tip_offset])
    connector.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, 0, 0, 0, 0, -length * 0.025],
        size=[radius * 0.6, 0, 0],
        rgba=tuple(limb_config.rgba),
        contype=0,
        conaffinity=0,
    )

    for actuator_idx, hinge_name in enumerate(hinge_names):
        spec.add_actuator(
            target=hinge_name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            name=f"-{limb_idx}-actuator{actuator_idx}",
        )
