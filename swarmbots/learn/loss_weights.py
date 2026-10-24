"""Resolve and validate loss-weight requests before applying any changes."""

import math
from collections.abc import Mapping


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
            value = float(remaining.pop(alias))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{alias} must be finite and nonnegative, got {value}")
            resolved[name] = value
    return resolved, remaining
