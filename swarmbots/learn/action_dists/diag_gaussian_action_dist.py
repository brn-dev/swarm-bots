import math
from typing import Optional, Self

import torch
import torch.distributions as torchdist
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionNetInitialization
from swarmbots.learn.action_dists.continuous_action_dist import ContinuousActionDist
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


class DiagGaussianActionDist(ContinuousActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            std: float,
            std_learnable: bool,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
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

    def set_std(self, std: float) -> None:
        if std <= 0:
            raise ValueError(f"std must be > 0, got {std}")
        with torch.no_grad():
            self.log_stds[:] = math.log(std)

    def update_distribution_params(self, means: torch.Tensor, log_stds: torch.Tensor) -> Self:
        self.distribution = torchdist.Normal(loc=means, scale=torch.exp(log_stds))
        return self

    def sample(self, agent: int | None = None) -> torch.Tensor:
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean