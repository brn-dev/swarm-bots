import math

import torch
from torch.nn import functional as F


class TanhBijector:

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    @staticmethod
    def atanh(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(y.dtype).eps
        # Clip the action to avoid NaN
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    @staticmethod
    def log_abs_det_jacobian(value: torch.Tensor) -> torch.Tensor:
        return 2.0 * (math.log(2.0) - value - F.softplus(-2.0 * value))

    def log_prob_correction(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_abs_det_jacobian(x)
