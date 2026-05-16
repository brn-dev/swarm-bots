from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import torch
from torch import nn


class SquaredReLU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x).square()


def signed_square(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.square(x)


class SignedSquare(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return signed_square(x)


class NegativeSlopeMode(Enum):
    STATIC = 0
    GLOBAL = 1
    PER_FEATURE = 2

    @staticmethod
    def from_bool(is_learnable: bool) -> "NegativeSlopeMode":
        return NegativeSlopeMode.GLOBAL if is_learnable else NegativeSlopeMode.STATIC


class SignedSquaredLeakyRelu(nn.Module):

    def __init__(
            self,
            negative_slope: float = 0.2,
            negative_slope_learnable: bool | NegativeSlopeMode = False,
            num_features: int | None = None,
    ) -> None:
        super().__init__()
        self.initial_negative_slope = float(negative_slope)

        if isinstance(negative_slope_learnable, bool):
            self.negative_slope_mode = NegativeSlopeMode.from_bool(negative_slope_learnable)
        else:
            self.negative_slope_mode = negative_slope_learnable

        if self.negative_slope_mode is NegativeSlopeMode.STATIC:
            self.register_buffer("negative_slope", torch.tensor(self.initial_negative_slope))
        elif self.negative_slope_mode is NegativeSlopeMode.GLOBAL:
            self.negative_slope = nn.Parameter(torch.tensor(self.initial_negative_slope))
        elif self.negative_slope_mode is NegativeSlopeMode.PER_FEATURE:
            if num_features is None:
                raise ValueError("num_features must be provided for PER_FEATURE negative slopes")
            self.negative_slope = nn.Parameter(torch.full((num_features,), self.initial_negative_slope))
        else:
            raise ValueError(f"Unsupported negative_slope_mode: {self.negative_slope_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        negative_slope = self._get_negative_slope(x)
        positive_values = x.square()
        negative_values = -torch.square(x * negative_slope)
        return torch.where(x >= 0, positive_values, negative_values)

    def _get_negative_slope(self, x: torch.Tensor) -> torch.Tensor:
        return self.negative_slope.to(dtype=x.dtype, device=x.device)

    def extra_repr(self) -> str:
        return (
            f"negative_slope={self.initial_negative_slope}, "
            f"negative_slope_mode={self.negative_slope_mode.name}"
        )


@dataclass(frozen=True)
class SignedSquaredLeakyReluFactory:
    negative_slope: float = 0.2
    negative_slope_mode: bool | NegativeSlopeMode = NegativeSlopeMode.STATIC

    def __call__(self, *, num_features: int | None = None) -> SignedSquaredLeakyRelu:
        return SignedSquaredLeakyRelu(
            negative_slope=self.negative_slope,
            negative_slope_learnable=self.negative_slope_mode,
            num_features=num_features,
        )


ActivationFactory: TypeAlias = type[nn.Module] | Callable[[], nn.Module] | SignedSquaredLeakyReluFactory


def make_activation(factory: ActivationFactory, *, num_features: int) -> nn.Module:
    if isinstance(factory, SignedSquaredLeakyReluFactory):
        return factory(num_features=num_features)
    return factory()


def activation_factory_name(factory: ActivationFactory) -> str:
    if isinstance(factory, type):
        return f"{factory.__module__}.{factory.__qualname__}"
    if hasattr(factory, "__module__") and hasattr(factory, "__qualname__"):
        return f"{factory.__module__}.{factory.__qualname__}"
    return str(factory)
