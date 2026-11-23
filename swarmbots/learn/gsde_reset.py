import abc
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class GSDEResetMode(abc.ABC):
    pass


@dataclass(frozen=True)
class GSDEEpisodeResetMode(GSDEResetMode):
    pass


@dataclass(frozen=True)
class GSDEIntervalResetMode(GSDEResetMode):
    interval: int


@dataclass(frozen=True)
class GSDEProbabilityResetMode(GSDEResetMode):
    probability: float


def reset_gsde_noise(
    *,
    action_dist: Any,
    local_obs: torch.Tensor,
    episode_start_mask: torch.Tensor,
    rollout_step_idx: int,
    gsde_reset_mode: GSDEResetMode | None,
    initialize: bool,
) -> None:
    """Refresh noise once per selected environment/agent, immediately before sampling."""
    batch_shape = tuple(local_obs.shape[:-1])
    if gsde_reset_mode is not None and (initialize or (
        isinstance(gsde_reset_mode, GSDEIntervalResetMode) and rollout_step_idx % gsde_reset_mode.interval == 0
    )):
        action_dist.reset_temporal_correlations_on_step(batch_shape=batch_shape)
    elif isinstance(gsde_reset_mode, GSDEProbabilityResetMode):
        mask = torch.empty(batch_shape, device=local_obs.device, dtype=torch.bool).bernoulli_(gsde_reset_mode.probability)
        action_dist.reset_temporal_correlations_on_step(mask=mask | episode_start_mask.unsqueeze(-1))
    else:
        action_dist.reset_temporal_correlations_on_ep_start(episode_start_mask)


def gsde_reset_mode_from_settings(settings: Mapping[str, Any] | None) -> GSDEResetMode | None:
    if settings is None:
        return None
    mode = settings["mode"]
    if mode == "episode":
        return GSDEEpisodeResetMode()
    if mode == "interval":
        return GSDEIntervalResetMode(interval=int(settings["interval"]))
    if mode == "probability":
        return GSDEProbabilityResetMode(probability=float(settings["probability"]))
    raise ValueError(f"Unknown saved gSDE reset mode: {mode!r}")


def resolve_gsde_reset_mode(
        *,
        gsde_enabled: bool,
        reset_mode: GSDEResetMode | None,
) -> GSDEResetMode | None:
    if not gsde_enabled:
        return None
    if reset_mode is None:
        raise ValueError("gsde_reset_mode is required when gSDE is enabled.")
    if isinstance(reset_mode, GSDEEpisodeResetMode):
        return reset_mode
    if isinstance(reset_mode, GSDEIntervalResetMode):
        if reset_mode.interval <= 0:
            raise ValueError(f"GSDEIntervalResetMode.interval must be > 0, got {reset_mode.interval}")
        return reset_mode
    if isinstance(reset_mode, GSDEProbabilityResetMode):
        if not (0.0 < reset_mode.probability < 1.0):
            raise ValueError(
                f"GSDEProbabilityResetMode.probability must be in (0, 1), got {reset_mode.probability}"
            )
        return reset_mode
    raise TypeError(f"Unknown gsde_reset_mode type: {type(reset_mode)}")


