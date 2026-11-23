"""Resolve and validate loss-weight requests before applying any changes."""

import math
from collections.abc import Mapping


def validate_loss_weight(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value}")
    return value


def prepare_checkpoint_loss_weights(saved: dict[str, float], current: dict[str, float]) -> dict[str, float]:
    if not isinstance(saved, dict) or saved.keys() != current.keys():
        raise ValueError(f"Checkpoint loss weights must match the policy's weights {sorted(current)}")
    return {name: validate_loss_weight(name, value) for name, value in saved.items()}


def resolve_loss_weight_aliases(
    weights: Mapping[str, float],
    aliases: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, float], dict[str, float]]:
    remaining = dict(weights)
    resolved: dict[str, float] = {}
    for name, names in aliases.items():
        present = [alias for alias in names if alias in remaining]
        if len(present) > 1:
            raise ValueError(f"Multiple aliases for the same loss weight are not allowed: {present}")
        if present:
            alias = present[0]
            resolved[name] = validate_loss_weight(alias, remaining.pop(alias))
    return resolved, remaining
