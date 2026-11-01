"""Public observation targets shared by on-policy and replay-based NOP presets."""

from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.obs_indices import ObsIndices


def make_next_obs_pred_config(indices: ObsIndices) -> NextObsPredConfig:
    return NextObsPredConfig(
        local_scalar_target_indices=indices.local_scalar_indices,
        local_angle_target_indices=indices.local_angle_indices,
        local_rot6d_target_indices=indices.local_rot6d_indices,
        local_binary_target_indices=indices.local_binary_indices,
        global_scalar_target_indices=indices.global_scalar_indices,
        global_rot6d_target_indices=indices.global_rot6d_indices,
        global_scalar_active_mask_indices=indices.global_scalar_active_mask_indices,
        global_rot6d_active_mask_indices=indices.global_rot6d_active_mask_indices,
    )
