from __future__ import annotations

import warp as wp


@wp.func
def _axis_x(xmat: wp.array(dtype=wp.float32, ndim=4), env_idx: int, body_idx: int) -> wp.vec3:
    return wp.vec3(xmat[env_idx, body_idx, 0, 0], xmat[env_idx, body_idx, 1, 0], xmat[env_idx, body_idx, 2, 0])


@wp.func
def _axis_y(xmat: wp.array(dtype=wp.float32, ndim=4), env_idx: int, body_idx: int) -> wp.vec3:
    return wp.vec3(xmat[env_idx, body_idx, 0, 1], xmat[env_idx, body_idx, 1, 1], xmat[env_idx, body_idx, 2, 1])


@wp.func
def _axis_z(xmat: wp.array(dtype=wp.float32, ndim=4), env_idx: int, body_idx: int) -> wp.vec3:
    return wp.vec3(xmat[env_idx, body_idx, 0, 2], xmat[env_idx, body_idx, 1, 2], xmat[env_idx, body_idx, 2, 2])


@wp.kernel
def apply_reset_unit_pose(
    world_idx: wp.array(dtype=wp.int32, ndim=1),
    pool_idx: wp.array(dtype=wp.int32, ndim=1),
    active_mask: wp.array(dtype=wp.bool, ndim=2),
    pool_positions: wp.array(dtype=wp.vec3, ndim=2),
    pool_quats: wp.array(dtype=wp.float32, ndim=3),
    inactive_positions: wp.array(dtype=wp.vec3, ndim=1),
    swarm_start: wp.array(dtype=wp.vec3, ndim=1),
    initial_z_rotation: wp.array(dtype=wp.float32, ndim=1),
    unit_qpos_adr: wp.array(dtype=wp.int32, ndim=1),
    qpos: wp.array(dtype=wp.float32, ndim=2),
):
    reset_slot, unit_idx = wp.tid()

    env_idx = world_idx[reset_slot]
    pool_sample_idx = pool_idx[reset_slot]
    qpos_adr = unit_qpos_adr[unit_idx]
    qpos_adr_1 = qpos_adr + 1
    qpos_adr_2 = qpos_adr + 2
    qpos_adr_3 = qpos_adr + 3
    qpos_adr_4 = qpos_adr + 4
    qpos_adr_5 = qpos_adr + 5
    qpos_adr_6 = qpos_adr + 6

    if active_mask[pool_sample_idx, unit_idx]:
        angle = initial_z_rotation[reset_slot]
        local_pos = pool_positions[pool_sample_idx, unit_idx]
        reset_start = swarm_start[reset_slot]
        if angle == 0.0:
            unit_pos = local_pos + reset_start
            qpos[env_idx, qpos_adr] = unit_pos[0]
            qpos[env_idx, qpos_adr_1] = unit_pos[1]
            qpos[env_idx, qpos_adr_2] = unit_pos[2]
            qpos[env_idx, qpos_adr_3] = pool_quats[pool_sample_idx, unit_idx, 0]
            qpos[env_idx, qpos_adr_4] = pool_quats[pool_sample_idx, unit_idx, 1]
            qpos[env_idx, qpos_adr_5] = pool_quats[pool_sample_idx, unit_idx, 2]
            qpos[env_idx, qpos_adr_6] = pool_quats[pool_sample_idx, unit_idx, 3]
        else:
            cos_angle = wp.cos(angle)
            sin_angle = wp.sin(angle)
            unit_pos = wp.vec3(
                reset_start[0] + cos_angle * local_pos[0] - sin_angle * local_pos[1],
                reset_start[1] + sin_angle * local_pos[0] + cos_angle * local_pos[1],
                reset_start[2] + local_pos[2],
            )
            qpos[env_idx, qpos_adr] = unit_pos[0]
            qpos[env_idx, qpos_adr_1] = unit_pos[1]
            qpos[env_idx, qpos_adr_2] = unit_pos[2]
            quat_w = pool_quats[pool_sample_idx, unit_idx, 0]
            quat_x = pool_quats[pool_sample_idx, unit_idx, 1]
            quat_y = pool_quats[pool_sample_idx, unit_idx, 2]
            quat_z = pool_quats[pool_sample_idx, unit_idx, 3]
            half_angle = 0.5 * angle
            yaw_w = wp.cos(half_angle)
            yaw_z = wp.sin(half_angle)
            qpos[env_idx, qpos_adr_3] = yaw_w * quat_w - yaw_z * quat_z
            qpos[env_idx, qpos_adr_4] = yaw_w * quat_x - yaw_z * quat_y
            qpos[env_idx, qpos_adr_5] = yaw_w * quat_y + yaw_z * quat_x
            qpos[env_idx, qpos_adr_6] = yaw_w * quat_z + yaw_z * quat_w
    else:
        inactive_pos = inactive_positions[unit_idx]
        qpos[env_idx, qpos_adr] = inactive_pos[0]
        qpos[env_idx, qpos_adr_1] = inactive_pos[1]
        qpos[env_idx, qpos_adr_2] = inactive_pos[2]
        qpos[env_idx, qpos_adr_3] = 1.0
        qpos[env_idx, qpos_adr_4] = 0.0
        qpos[env_idx, qpos_adr_5] = 0.0
        qpos[env_idx, qpos_adr_6] = 0.0


@wp.kernel
def gather_connector_frames(
    xpos: wp.array(dtype=wp.vec3, ndim=2),
    xmat: wp.array(dtype=wp.float32, ndim=4),
    connector_body_indices: wp.array(dtype=wp.int32, ndim=1),
    connector_positions: wp.array(dtype=wp.vec3, ndim=2),
    connector_x_axis: wp.array(dtype=wp.vec3, ndim=2),
    connector_y_axis: wp.array(dtype=wp.vec3, ndim=2),
    connector_z_axis: wp.array(dtype=wp.vec3, ndim=2),
):
    env_idx, connector_idx = wp.tid()
    body_idx = connector_body_indices[connector_idx]
    connector_positions[env_idx, connector_idx] = xpos[env_idx, body_idx]
    connector_x_axis[env_idx, connector_idx] = _axis_x(xmat, env_idx, body_idx)
    connector_y_axis[env_idx, connector_idx] = _axis_y(xmat, env_idx, body_idx)
    connector_z_axis[env_idx, connector_idx] = _axis_z(xmat, env_idx, body_idx)


@wp.kernel
def compute_best_connection_candidates(
    newly_activated: wp.array(dtype=wp.bool, ndim=2),
    connector_positions: wp.array(dtype=wp.vec3, ndim=2),
    connector_x_axis: wp.array(dtype=wp.vec3, ndim=2),
    connector_y_axis: wp.array(dtype=wp.vec3, ndim=2),
    connector_z_axis: wp.array(dtype=wp.vec3, ndim=2),
    connector_unit_idx: wp.array(dtype=wp.int32, ndim=1),
    num_total_connectors: int,
    distance_threshold_sq: float,
    angle_threshold: float,
    twist_step: float,
    twist_count: int,
    best_partner_idx: wp.array(dtype=wp.int32, ndim=2),
    best_twist_idx: wp.array(dtype=wp.int32, ndim=2),
):
    env_idx, connector_idx = wp.tid()

    if not newly_activated[env_idx, connector_idx]:
        best_partner_idx[env_idx, connector_idx] = -1
        best_twist_idx[env_idx, connector_idx] = -1
        return

    connector_unit = connector_unit_idx[connector_idx]
    pos_i = connector_positions[env_idx, connector_idx]
    x_i = connector_x_axis[env_idx, connector_idx]
    y_i = connector_y_axis[env_idx, connector_idx]
    z_i = connector_z_axis[env_idx, connector_idx]

    best_dist_sq = wp.float32(1.0e30)
    best_partner = int(-1)
    best_twist = int(-1)

    for candidate_idx in range(num_total_connectors):
        if candidate_idx == connector_idx:
            continue
        if not newly_activated[env_idx, candidate_idx]:
            continue
        if connector_unit_idx[candidate_idx] == connector_unit:
            continue

        rel_pos = connector_positions[env_idx, candidate_idx] - pos_i
        dist_sq = wp.dot(rel_pos, rel_pos)
        if dist_sq >= distance_threshold_sq:
            continue

        z_j = connector_z_axis[env_idx, candidate_idx]
        if wp.dot(z_j, z_i) > angle_threshold:
            continue
        if wp.dot(rel_pos, z_i) < 0.0:
            continue

        if dist_sq < best_dist_sq:
            x_j = connector_x_axis[env_idx, candidate_idx]
            twist = wp.atan2(wp.dot(x_j, y_i), wp.dot(x_j, x_i))
            twist_phase = twist
            if twist_phase < 0.0:
                twist_phase += 6.283185307179586
            quantized_twist = int(wp.floor((twist_phase + 0.5 * twist_step) / twist_step)) % twist_count

            best_dist_sq = dist_sq
            best_partner = candidate_idx
            best_twist = quantized_twist

    best_partner_idx[env_idx, connector_idx] = best_partner
    best_twist_idx[env_idx, connector_idx] = best_twist
