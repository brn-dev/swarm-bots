from __future__ import annotations

import math


def finite_positive(value: float, *, name: str, minimum: float = 0.0) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0 or resolved < minimum:
        bound = f">= {minimum:g}" if minimum > 0.0 else "> 0"
        raise ValueError(f"Expected finite {name} {bound}, got {resolved}")
    return resolved
