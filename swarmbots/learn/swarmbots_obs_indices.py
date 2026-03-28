from typing import Any

from swarmbots.learn.obs_indices import ObsIndices


def _hinges_per_limb(limb_type: str) -> int:
    name = limb_type.removeprefix("LimbType.")
    if name in ("xy", "zx"):
        return 2
    if name == "xyz":
        return 3
    raise ValueError(f"Unsupported limb type: {limb_type}")


def build_obs_indices(
    env_settings: dict[str, Any],
    local_obs_dim: int,
    global_obs_dim: int,
    hidden_local_vars_dim: int,
    hidden_global_vars_dim: int,
) -> ObsIndices:
    scenario_settings = env_settings["scenario"]
    quat_rot6d_representation = bool(scenario_settings.get("quat_rot6d_representation", False))

    swarm_config = scenario_settings["swarm"]["config"]
    unit_config = swarm_config["unit_config"]
    limbs_per_unit = len(unit_config)
    include_connectors_xpos_in_obs = bool(scenario_settings["include_connectors_xpos_in_obs"])
    include_connectors_xquat_in_obs = bool(scenario_settings["include_connectors_xquat_in_obs"])

    num_hinges = sum(_hinges_per_limb(limb["type"]) for limb in unit_config)
    free_joint_rot_dim = 6 if quat_rot6d_representation else 4
    qpos_obs_dim = 3 + free_joint_rot_dim + 2 * num_hinges
    qvel_dim = 6 + num_hinges
    connector_obs_dim = limbs_per_unit * 5
    connectors_xpos_dim = limbs_per_unit * 3 if include_connectors_xpos_in_obs else 0
    connectors_xquat_dim = (
        limbs_per_unit * free_joint_rot_dim if include_connectors_xquat_in_obs else 0
    )
    expected_local_dim = (
        qpos_obs_dim + qvel_dim + connector_obs_dim + connectors_xpos_dim + connectors_xquat_dim
    )
    if local_obs_dim != expected_local_dim:
        raise ValueError(
            "Unexpected local_obs_dim for obs indices. "
            f"{local_obs_dim=} {expected_local_dim=} {limbs_per_unit=} {num_hinges=}"
        )

    connector_obs_offset = qpos_obs_dim + qvel_dim
    connectors_xpos_offset = connector_obs_offset + connector_obs_dim
    connectors_xquat_offset = connectors_xpos_offset + connectors_xpos_dim

    local_scalar_indices = (
        list(range(3))
        + list(range(qpos_obs_dim, qpos_obs_dim + qvel_dim))
        + list(range(connectors_xpos_offset, connectors_xpos_offset + connectors_xpos_dim))
        + [connector_obs_offset + i * 5 + 4 for i in range(limbs_per_unit)]
    )
    local_quaternion_indices: list[int] = []
    if not quat_rot6d_representation:
        local_quaternion_indices.append(3)
        if include_connectors_xquat_in_obs:
            local_quaternion_indices.extend(
                connectors_xquat_offset + 4 * i for i in range(limbs_per_unit)
            )

    if scenario_settings.get("payload_type") is not None:
        expected_global_obs_dim = 3 + free_joint_rot_dim
        if global_obs_dim < expected_global_obs_dim:
            raise ValueError(
                "Expected global_obs to include payload pos+rotation when payload_type is set. "
                f"{global_obs_dim=}"
            )
        if quat_rot6d_representation:
            global_scalar_indices = list(range(expected_global_obs_dim))
            global_quaternion_indices: list[int] = []
        else:
            global_scalar_indices = list(range(3))
            global_quaternion_indices = [3]
    else:
        global_scalar_indices = []
        global_quaternion_indices = []

    # hidden_local_vars are currently used for per-unit binary threshold flags in wall scenarios.
    # Keep them unnormalized.
    hidden_local_vars_scalar_indices: list[int] = []
    hidden_local_vars_quaternion_indices: list[int] = []
    hidden_global_vars_scalar_indices = list(range(hidden_global_vars_dim))
    hidden_global_vars_quaternion_indices: list[int] = []

    hinge_sin_start = 3 + free_joint_rot_dim
    local_angle_indices = list(range(hinge_sin_start, hinge_sin_start + 2 * num_hinges, 2))
    local_angle_indices.extend(connector_obs_offset + i * 5 + 2 for i in range(limbs_per_unit))

    local_rot6d_indices: list[int] = []
    if quat_rot6d_representation:
        local_rot6d_indices.append(3)
        if include_connectors_xquat_in_obs:
            local_rot6d_indices.extend(connectors_xquat_offset + 6 * i for i in range(limbs_per_unit))

    local_binary_indices = [connector_obs_offset + i * 5 + 0 for i in range(limbs_per_unit)]

    return ObsIndices(
        local_scalar_indices=local_scalar_indices,
        local_angle_indices=local_angle_indices,
        local_rot6d_indices=local_rot6d_indices,
        local_binary_indices=local_binary_indices,
        local_quaternion_indices=local_quaternion_indices,
        global_scalar_indices=global_scalar_indices,
        global_quaternion_indices=global_quaternion_indices,
        hidden_local_vars_scalar_indices=hidden_local_vars_scalar_indices,
        hidden_local_vars_quaternion_indices=hidden_local_vars_quaternion_indices,
        hidden_global_vars_scalar_indices=hidden_global_vars_scalar_indices,
        hidden_global_vars_quaternion_indices=hidden_global_vars_quaternion_indices,
    )
