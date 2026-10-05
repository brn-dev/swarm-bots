"""Compatibility exports for the shared off-policy components."""

import torch as torch

from swarmbots.learn.algos.off_policy.nop import (
    SACNOPLatentSource as SACNOPLatentSource,
    SACNOPConfig as SACNOPConfig,
    SACNOPSequenceBatch as SACNOPSequenceBatch,
    SACNOPModule as SACNOPModule,
    normalize_nop_latent_source as normalize_nop_latent_source,
    _has_next_obs_pred_targets as _has_next_obs_pred_targets,
)
