import math

import torch
from torch.nn import functional as F


class TanhBijector:
    @staticmethod
    def saturation_threshold(dtype: torch.dtype) -> float:
        # atanh of the halfway point between 1 and its closest interior neighbor.
        gap = torch.finfo(dtype).eps / 4
        return 0.5 * (math.log(2 - gap) - math.log(gap))

    @staticmethod
    def forward(x: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        # One final rounding gives the same action cells as the bounded likelihood.
        return x.double().tanh().to(x.dtype if dtype is None else dtype)

    @staticmethod
    def normal_log_prob(
            distribution: torch.distributions.Normal,
            actions: torch.Tensor,
            gaussian_actions: torch.Tensor | None,
            log_scale: torch.Tensor | None = None,
            unbounded_log_scale: bool = False,
    ) -> torch.Tensor:
        if gaussian_actions is not None:
            score = distribution.log_prob(gaussian_actions) - TanhBijector.log_abs_det_jacobian(gaussian_actions)
            return score.to(distribution.loc.dtype)

        threshold = TanhBijector.saturation_threshold(actions.dtype)
        saturated = actions.abs() == 1
        magnitude = actions.abs().masked_fill(saturated, 0.0)
        # Cell boundaries are the midpoints to the neighbors in the stored action dtype.
        # At powers of two the distances on either side differ.
        center = magnitude.detach()
        upper_gap = torch.nextafter(center, torch.ones_like(center)) - center
        lower_gap = center - torch.nextafter(center, -torch.ones_like(center))
        dtype = torch.promote_types(torch.promote_types(actions.dtype, distribution.scale.dtype), torch.float32)
        magnitude = magnitude.to(dtype)
        upper_gap, lower_gap = upper_gap.to(dtype), lower_gap.to(dtype)
        cell_width = (upper_gap + lower_gap) * 0.5
        slope = (1 - magnitude) * (1 + magnitude)
        oriented_mean = torch.where(actions < 0, -distribution.loc, distribution.loc)
        # An inverse rounded in float32 can miss the mean by several standard deviations.
        offset = (torch.atanh(magnitude.double()) - oriented_mean.double()).to(dtype)
        sigma = distribution.scale.to(dtype)
        # Multiplication by exp(-log sigma) backpropagates the weighted residual before dividing
        # by sigma. Direct division can produce 0 * inf on a tail whose true gradient is zero.
        log_sigma = sigma.log() if log_scale is None else log_scale.to(dtype)
        # An unconstrained log scale may describe a subnormal std whose reciprocal overflows.
        inv_scale = (-0.5 * log_sigma if unbounded_log_scale else -log_sigma).exp()

        def standardize(value: torch.Tensor) -> torch.Tensor:
            scaled = value * inv_scale
            return scaled * inv_scale if unbounded_log_scale else scaled

        standardized = standardize(offset)
        # The point density suffices while the Jacobian and Gaussian change by
        # less than about 1% across the cell. Wider cells need their integrated mass.
        gaussian_length = sigma / (1 + standardized.abs())
        use_cell = ~saturated & (cell_width > 0.01 * slope * gaussian_length.clamp_max(1))

        # atanh(y +/- h) - atanh(y), without rounding either midpoint back to y.
        lower_half, upper_half = lower_gap * 0.5, upper_gap * 0.5
        lower = offset - torch.atanh(lower_half / (slope + magnitude * lower_half))
        upper = offset + torch.atanh(upper_half / (slope - magnitude * upper_half))
        # Reflect intervals on the right so the two CDFs never both round to one.
        reflect = lower + upper > 0
        upper_offset = torch.where(reflect, -lower, upper).masked_fill(~use_cell, 0)
        lower_offset = torch.where(reflect, -upper, lower).masked_fill(~use_cell, 0)
        tail_offset = (oriented_mean.double() - threshold).to(dtype)
        upper_offset = torch.where(saturated, tail_offset, upper_offset)
        upper_z = standardize(upper_offset)
        lower_z = standardize(lower_offset)
        if unbounded_log_scale:
            # A tail beyond the largest representable quadratic has zero gradient in an occupied cell.
            # Keep its intermediate z finite so a masked derivative cannot multiply an infinity.
            limit = math.sqrt(torch.finfo(dtype).max) * 2
            upper_z = upper_z.clamp(-limit, limit)
            lower_z = lower_z.clamp(-limit, limit)
        left_mass = TanhBijector._left_log_cdf(upper_z.clamp_max(0))
        upper_mass = torch.where(upper_z < 0, left_mass, torch.special.log_ndtr(upper_z.clamp_min(0)))
        lower_mass = TanhBijector._left_log_cdf(lower_z)
        # Inactive cells get a finite log difference, including in backward.
        resolved_cell = use_cell & (lower_mass < upper_mass)
        density_z = standardized.masked_fill(saturated | resolved_cell, 0.)
        density = -0.5 * density_z.square() - log_sigma - 0.5 * math.log(2 * math.pi) - slope.log()
        difference = (lower_mass - upper_mass).masked_fill(~resolved_cell, -1)
        log_cell_density = upper_mass + (-torch.expm1(difference)).log() - cell_width.masked_fill(~resolved_cell, 1).log()
        return torch.where(saturated, upper_mass, torch.where(resolved_cell, log_cell_density, density))

    @staticmethod
    def _left_log_cdf(z: torch.Tensor) -> torch.Tensor:
        # erfcx keeps the left-tail log CDF and its gradient finite for small standard deviations.
        return torch.special.erfcx(-z / math.sqrt(2)).log() - math.log(2) - 0.5 * z.square()

    @staticmethod
    def log_abs_det_jacobian(value: torch.Tensor) -> torch.Tensor:
        return 2.0 * (math.log(2.0) - value - F.softplus(-2.0 * value))
