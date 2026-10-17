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


def build_gsde_step_reset_mask(
    *,
    gsde_reset_mode: GSDEResetMode | None,
    rollout_step_idx: int,
    batch_shape: tuple[int, ...],
    rollout_device: torch.device,
    initialize: bool = False,
) -> torch.Tensor | None:
    if gsde_reset_mode is None:
        return None
    if initialize:
        return torch.ones(batch_shape, device=rollout_device, dtype=torch.bool)
    if isinstance(gsde_reset_mode, GSDEEpisodeResetMode):
        return None
    if isinstance(gsde_reset_mode, GSDEIntervalResetMode):
        if rollout_step_idx % gsde_reset_mode.interval == 0:
            return torch.ones(batch_shape, device=rollout_device, dtype=torch.bool)
        return None
    if isinstance(gsde_reset_mode, GSDEProbabilityResetMode):
        return torch.empty(batch_shape, device=rollout_device, dtype=torch.bool).bernoulli_(
            gsde_reset_mode.probability
        )
    raise TypeError(f"Unknown gsde_reset_mode type: {type(gsde_reset_mode)}")


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


