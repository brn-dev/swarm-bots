"""Common training and recurrent replay settings for off-policy presets."""

from typing import Any

import torch


DEFAULT_OFF_POLICY_LEARNING_RATE = 5e-5


def make_off_policy_algorithm_options() -> dict[str, Any]:
    return {
        "learning_rate": DEFAULT_OFF_POLICY_LEARNING_RATE,
        "learning_rate_warmup_updates": 500,
        "learning_rate_warmup_start_factor": 0.01,
        "batch_size": 1024,
        "gradient_steps": 8,
        "tau": 0.005,
    }


def make_recurrent_replay_options() -> dict[str, Any]:
    return {
        "burn_in_steps": 32,
        "learning_steps": 64,
        "temporal_state_store_interval": 16,
        "temporal_state_storage_dtype": torch.float16,
    }
