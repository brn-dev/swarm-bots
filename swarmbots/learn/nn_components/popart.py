from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


class PopArtLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int = 1,
            *,
            bias: bool = True,
            beta: float = 3e-4,
            eps: float = 1e-5,
            min_std: float = 1e-4,
            init_mu: float = 0.0,
            init_sigma: float = 1.0,
            init_gain: float = 1.0,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.beta = float(beta)
        self.eps = float(eps)
        self.min_std = float(min_std)
        self.init_gain = float(init_gain)
        init_sigma = float(init_sigma)
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("PopArt beta must be in [0, 1]")
        if not all(math.isfinite(x) and x >= 0 for x in (init_sigma, self.eps, self.min_std)):
            raise ValueError("PopArt init_sigma, eps and min_std must be finite and nonnegative")
        initial_variance = init_sigma * init_sigma
        initial_scale = max(math.sqrt(max(initial_variance, self.eps)), self.min_std)
        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs))
        # Validate in the readout dtype without synchronizing an accelerator during construction.
        mean, scale = torch.tensor((init_mu, initial_scale), dtype=self.weight.dtype, device="cpu").tolist()
        if not math.isfinite(mean) or not 0.0 < scale < math.inf:
            raise ValueError("PopArt initial mean and scale must be finite in the readout dtype, with positive scale")

        if bias:
            self.bias = nn.Parameter(torch.empty((self.out_features,), **factory_kwargs))
        else:
            self.bias = None
            self.register_buffer("normalization_bias", torch.zeros((self.out_features,), **factory_kwargs))

        self.register_buffer("mu", torch.full((self.out_features,), float(init_mu), device=device, dtype=torch.float64))
        self.register_buffer("variance", torch.full_like(self.mu, initial_variance))

        self.reset_parameters()

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> PopArtLinear:
        mean, variance = self.mu, self.variance
        super()._apply(fn, recurse)
        # Module-wide precision changes may move the head too. Keep accumulated moments at full precision.
        if self.mu.dtype != torch.float64:
            self.mu = mean.to(self.mu.device)
            self.variance = variance.to(self.variance.device)
        return self

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.weight, gain=self.init_gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        else:
            self.normalization_bias.zero_()

    @classmethod
    def from_linear(
            cls,
            linear: nn.Linear,
            *,
            beta: float = 3e-4,
            eps: float = 1e-5,
            min_std: float = 1e-4,
            init_mu: float = 0.0,
            init_sigma: float = 1.0,
            init_gain: float = 1.0,
    ) -> PopArtLinear:
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            beta=beta,
            eps=eps,
            min_std=min_std,
            init_mu=init_mu,
            init_sigma=init_sigma,
            init_gain=init_gain,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                assert layer.bias is not None
                layer.bias.copy_(linear.bias)
        return layer

    def _sigma_from(self, variance: torch.Tensor) -> torch.Tensor:
        sigma = torch.sqrt(torch.clamp(variance, min=self.eps))
        return torch.clamp(sigma, min=self.min_std)

    def sigma(self) -> torch.Tensor:
        return self._sigma_from(self.variance).to(dtype=self.weight.dtype)

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mu.to(dtype=self.weight.dtype)) / self.sigma()

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.sigma() + self.mu.to(dtype=self.weight.dtype)

    @torch.no_grad()
    def update(self, targets: torch.Tensor) -> None:
        if targets.numel() == 0:
            return
        if self.beta == 0.0:
            if not torch.isfinite(targets).all():
                raise ValueError("PopArt targets must be finite")
            return

        y = targets.to(device=self.mu.device, dtype=self.mu.dtype).reshape(-1, self.out_features)
        new_variance, new_mu = torch.var_mean(y, dim=0, correction=0)
        if self.beta < 1.0:
            delta = new_mu - self.mu
            new_mu = torch.lerp(self.mu, new_mu, self.beta)
            # The mixture variance includes the distance between the two populations' means.
            new_variance = torch.lerp(self.variance, new_variance, self.beta) + (
                self.beta * (1.0 - self.beta) * delta * delta
            )
        # Preserve predictions in the same rounded frame used by normalize/denormalize.
        old_sigma = self.sigma().double()
        new_sigma = self._sigma_from(new_variance).to(dtype=self.weight.dtype).double()
        mean_shift = self.mu.to(dtype=self.weight.dtype).double() - new_mu.to(dtype=self.weight.dtype).double()
        bias = self.bias if self.bias is not None else self.normalization_bias
        new_bias = ((old_sigma * bias + mean_shift) / new_sigma).to(dtype=bias.dtype)
        new_weight = torch.empty_like(self.weight)
        torch.mul(self.weight, (old_sigma / new_sigma).unsqueeze(-1), out=new_weight)
        valid = (
            torch.isfinite(new_mu) & torch.isfinite(new_variance) & torch.isfinite(new_sigma)
            & torch.isfinite(new_bias) & torch.isfinite(new_weight).all(dim=-1)
        )
        if not valid.all():
            raise ValueError("PopArt target moments and rescaled readout must be finite")

        self.weight.copy_(new_weight)
        bias.copy_(new_bias)
        self.mu.copy_(new_mu)
        self.variance.copy_(new_variance)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if self.bias is not None else self.normalization_bias
        y_norm = F.linear(x, self.weight, bias)
        return self.denormalize(y_norm)
