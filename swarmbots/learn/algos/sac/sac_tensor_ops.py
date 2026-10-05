"""Compatibility exports for the shared off-policy components."""

import torch as torch

from swarmbots.learn.algos.off_policy.tensor_ops import (
    BootstrapObservations as BootstrapObservations,
    OptimizerStep as OptimizerStep,
    CompiledReturn as CompiledReturn,
    _mask_terminal_observations as _mask_terminal_observations,
    _mean_agent_log_probs as _mean_agent_log_probs,
    _entropy_coefficient_loss as _entropy_coefficient_loss,
    _bellman_target as _bellman_target,
    _critic_loss as _critic_loss,
    _actor_loss as _actor_loss,
    SACTensorOperations as SACTensorOperations,
    build_sac_tensor_operations as build_sac_tensor_operations,
    build_optimizer_step as build_optimizer_step,
    _validate_compile_configuration as _validate_compile_configuration,
)
