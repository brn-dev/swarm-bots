"""Shared SAC training settings, using TMASAC as the reference."""

from typing import Any


def make_sac_algorithm_options() -> dict[str, Any]:
    return {
        "learning_rate": 5e-5,
        "ent_coef_learning_rate": None,
        "ent_coef": "auto_0.05",
        "target_entropy": "auto_0.1",
    }
