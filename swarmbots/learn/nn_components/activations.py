from __future__ import annotations

import torch
from torch import nn


class SquaredReLU(nn.Module):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.relu(input).square()
