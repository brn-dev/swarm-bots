from collections.abc import Sequence
from typing import Any, TypeAlias

import torch
from torch.nn import functional as F

LossDict: TypeAlias = dict[str, torch.Tensor]
LossMetrics: TypeAlias = dict[str, Any]


def mean_critic_mse_loss(values: Sequence[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    return torch.stack([F.mse_loss(value, target) for value in values]).mean()
