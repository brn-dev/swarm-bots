import math
from typing import Optional, Self

import torch
import torch.distributions as torchdist
from torch import nn

from swarmbots import ActionNetInitialization
from swarmbots import ContinuousActionDist


class DiagGaussianActionDist(ContinuousActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            std: float,
            std_learnable: bool,
            action_net_initialization: ActionNetInitialization | None = None,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
        )
        self.std_learnable = std_learnable

        self.log_stds = nn.Parameter(
            torch.ones((self.action_dim,)) * math.log(std),
            requires_grad=std_learnable
        )

        self.distribution: Optional[torchdist.Normal] = None

    def update_distribution_params(self, means: torch.Tensor, log_stds: torch.Tensor) -> Self:
        self.distribution = torchdist.Normal(loc=means, scale=torch.exp(log_stds))
        return self

    def sample(self) -> torch.Tensor:
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean