from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


GOAL_SAMPLING_ATTEMPTS = 16


def _float32_bound(value: float, *, up: bool) -> float:
    rounded = np.float32(value)
    if (up and float(rounded) < value) or (not up and float(rounded) > value):
        rounded = np.nextafter(rounded, np.float32(math.inf if up else -math.inf))
    return float(rounded)


@dataclass(frozen=True, slots=True)
class PayloadGoalLayout:
    """A constructive placement of the maximum active payload population.

    Goal centres lie in the requested rectangle. At those centres, spheres and
    payloads in their reset orientation do not overlap: boxes have a 2r x 2r
    footprint and the horizontal capsules have a 4r x 2r footprint. The
    fallback grid is deliberately conservative for circles and capsules.
    """

    lower: tuple[float, float]
    extent: tuple[float, float]
    footprint: tuple[float, float]
    clearance: float
    max_active_payloads: int
    grid_offsets: tuple[tuple[float, float], ...]
    grid_room: tuple[float, float]


def make_payload_goal_layout(
    *,
    goal_rect: tuple[float, float, float, float],
    payload_shape: str,
    payload_radius: float,
    max_active_payloads: int,
) -> PayloadGoalLayout:
    if not all(math.isfinite(value) for value in goal_rect):
        raise ValueError(f"Expected finite goal_rect bounds, got {goal_rect!r}")
    if max(abs(value) for value in goal_rect) > torch.finfo(torch.float32).max / 8.0:
        raise ValueError("goal_rect bounds must be representable in float32 goal positions")
    x_min, x_max, y_min, y_max = goal_rect
    x_min, y_min = (_float32_bound(value, up=True) for value in (x_min, y_min))
    x_max, y_max = (_float32_bound(value, up=False) for value in (x_max, y_max))
    extent = tuple(_float32_bound(value, up=False) for value in (x_max - x_min, y_max - y_min))
    if not all(math.isfinite(value) and value > 0.0 for value in extent):
        raise ValueError(f"goal_rect has no positive float32 span after rounding bounds inward: {goal_rect!r}")
    # Positions use float32. Leave a physical gap that also exceeds rounding
    # error when custom rectangles are far from the origin.
    coordinate_scale = max(1.0, payload_radius, *(abs(value) for value in goal_rect))
    clearance = max(1e-5, 8.0 * torch.finfo(torch.float32).eps * coordinate_scale)
    footprint = ((4.0 if payload_shape == "capsule" else 2.0) * payload_radius, 2.0 * payload_radius)
    spacing = (footprint[0] + clearance, footprint[1] + clearance)
    options: list[tuple[float, int, int]] = []
    for columns in range(1, max_active_payloads + 1):
        rows = math.ceil(max_active_payloads / columns)
        span = tuple(_float32_bound(value, up=True) for value in ((columns - 1) * spacing[0], (rows - 1) * spacing[1]))
        if span[0] <= extent[0] and span[1] <= extent[1]:
            options.append((max(span[0] / extent[0], span[1] / extent[1]), columns, rows))
    if not options:
        raise ValueError(
            "goal_rect cannot fit a guaranteed collision-free goal grid for "
            f"{max_active_payloads} active {payload_shape} payloads; enlarge goal_rect "
            "or reduce payload_radius/active_payload_count_probs. The sampler uses "
            "conservative reset-orientation footprints, not an arbitrary packing solver."
        )
    _, columns, rows = min(options)
    offsets = tuple(
        (float(np.float32((index % columns) * spacing[0])), float(np.float32((index // columns) * spacing[1])))
        for index in range(max_active_payloads)
    )
    span = tuple(_float32_bound(value, up=True) for value in ((columns - 1) * spacing[0], (rows - 1) * spacing[1]))
    return PayloadGoalLayout(
        lower=(x_min, y_min),
        extent=extent,
        footprint=footprint,
        clearance=clearance,
        max_active_payloads=max_active_payloads,
        grid_offsets=offsets,
        # Reserve the existing rounding clearance for the fallback's two
        # additions. A grid with no translation room still sits safely at the
        # inward-rounded lower bound.
        grid_room=tuple(
            _float32_bound(max(size - used - clearance, 0.0), up=False) for size, used in zip(extent, span, strict=True)
        ),
    )


class PayloadGoalSampler:
    """Bounded, batched rejection with an always feasible randomized fallback."""

    def __init__(
        self, *, layout: PayloadGoalLayout, max_payloads: int, payload_shape: str, device: torch.device
    ) -> None:
        self.layout = layout
        self.max_payloads = max_payloads
        self.payload_shape = payload_shape
        self.lower = torch.tensor(layout.lower, device=device, dtype=torch.float32)
        self.extent = torch.tensor(layout.extent, device=device, dtype=torch.float32)
        self.grid_offsets = torch.tensor(layout.grid_offsets, device=device, dtype=torch.float32)
        self.grid_room = torch.tensor(layout.grid_room, device=device, dtype=torch.float32)
        pair_indices = torch.triu_indices(max_payloads, max_payloads, offset=1, device=device)
        self.pair_first, self.pair_second = pair_indices.unbind(dim=0)

    def sample(self, *, active_payload_mask: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
        if active_payload_mask.ndim != 2 or active_payload_mask.shape[1] != self.max_payloads:
            raise ValueError(f"Expected active_payload_mask shape (batch, {self.max_payloads})")
        n_reset = active_payload_mask.shape[0]
        torch._assert_async(
            (active_payload_mask.sum(dim=1) <= self.layout.max_active_payloads).all(),
            "Active payload count exceeds the validated goal-grid capacity",
        )
        device = active_payload_mask.device
        samples = torch.rand(
            (n_reset, GOAL_SAMPLING_ATTEMPTS, self.max_payloads, 2),
            device=device,
            dtype=torch.float32,
            generator=rng,
        )
        candidates = self.lower + samples * self.extent
        delta = candidates.index_select(2, self.pair_first) - candidates.index_select(2, self.pair_second)
        if self.payload_shape == "sphere":
            min_distance = self.layout.footprint[0] + self.layout.clearance
            overlaps = (delta / min_distance).square().sum(dim=-1) < 1.0
        else:
            overlaps = (delta[..., 0].abs() < self.layout.footprint[0] + self.layout.clearance) & (
                delta[..., 1].abs() < self.layout.footprint[1] + self.layout.clearance
            )
        active_pairs = active_payload_mask.index_select(1, self.pair_first) & active_payload_mask.index_select(
            1, self.pair_second
        )
        valid = ~(overlaps & active_pairs.unsqueeze(1)).any(dim=-1)
        first_valid = valid.to(dtype=torch.int32).argmax(dim=1)
        chosen = candidates[torch.arange(n_reset, device=device), first_valid]

        # Compute fallback for every lane, without a device-to-host branch.
        # Random permutation and translation preserve variation even in tight
        # rectangles where all bounded rejection attempts fail.
        grid_order = torch.rand(
            (n_reset, self.layout.max_active_payloads),
            device=device,
            dtype=torch.float32,
            generator=rng,
        ).argsort(dim=1)
        active_rank = active_payload_mask.to(dtype=torch.long).cumsum(dim=1) - 1
        grid_index = grid_order.gather(1, active_rank.clamp(min=0, max=self.layout.max_active_payloads - 1))
        origin = (
            self.lower + torch.rand((n_reset, 2), device=device, dtype=torch.float32, generator=rng) * self.grid_room
        )
        fallback = origin.unsqueeze(1) + self.grid_offsets[grid_index]
        positions = torch.where(valid.any(dim=1).view(-1, 1, 1), chosen, fallback)
        return torch.where(active_payload_mask.unsqueeze(-1), positions, torch.zeros_like(positions))
