"""Bounded target-action smoothing shared by stochastic and deterministic learners."""

import numpy as np
import torch
from gymnasium import spaces

from swarmbots.learn.hybrid_action_space import HybridActionSpace


def bounded_action_coordinates(action_space: HybridActionSpace) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-agent centers and half-ranges, independent of replay batch axes."""
    lows, highs = [], []
    for space in action_space.sub_spaces:
        if not isinstance(space, spaces.Box):
            raise ValueError("Target-action smoothing requires continuous Box action spaces")
        if not np.isfinite(space.low).all() or not np.isfinite(space.high).all():
            raise ValueError("Target-action smoothing requires finite action bounds")
        if not (space.high > space.low).all():
            raise ValueError("Target-action smoothing requires upper bounds greater than lower bounds")
        low, high = space.low, space.high
        if low.ndim == 3:
            if not ((low == low[0]).all() and (high == high[0]).all()):
                raise ValueError("Target-action smoothing requires identical bounds across environments")
            low, high = low[0], high[0]
        lows.append(torch.as_tensor(low, dtype=torch.float32))
        highs.append(torch.as_tensor(high, dtype=torch.float32))
    low, high = torch.cat(lows, -1), torch.cat(highs, -1)
    return (high + low) / 2, (high - low) / 2


def add_normalized_action_noise(
    actions: torch.Tensor,
    *,
    center: torch.Tensor,
    scale: torch.Tensor,
    std: float,
    clip: float | None = None,
    agent_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Perturb in [-1, 1] coordinates, then clip to physical bounds and mask padding."""
    noise = torch.randn_like(actions) * std
    if clip is not None:
        noise = noise.clamp(-clip, clip)
    actions = (actions + noise * scale).clamp(center - scale, center + scale)
    return actions if agent_mask is None else actions.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
