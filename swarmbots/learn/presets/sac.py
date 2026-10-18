"""Shared SAC training settings, using TMASAC as the reference."""

from typing import Any

from swarmbots.learn.presets.off_policy import DEFAULT_OFF_POLICY_LEARNING_RATE


def make_sac_algorithm_options() -> dict[str, Any]:
    return {
        "learning_rate": DEFAULT_OFF_POLICY_LEARNING_RATE,
        "ent_coef_learning_rate": None,
        "ent_coef": "auto_0.05",
        "target_entropy": "auto_0.1",
    }
