import abc
from typing import Optional, Self

import torch
import torch.distributions as torchdist

from swarmbots import ActionDist, ActionNetInitialization, AGENT_ACTIONS_DIM


class ContinuousActionDist(ActionDist, abc.ABC):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization | None,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
        )

        self.distribution: Optional[torchdist.Distribution] = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        action_means = self.action_net(latent_pi)
        return self.update_distribution_params(action_means, self.log_stds)

    @abc.abstractmethod
    def update_distribution_params(self, means: torch.Tensor, log_stds: torch.Tensor) -> Self:
        raise NotImplementedError

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.sum_action_dim(self.distribution.log_prob(actions))

    def entropy(self) -> Optional[torch.Tensor]:
        return self.sum_action_dim(self.distribution.entropy())

    @staticmethod
    def sum_action_dim(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.sum(dim=AGENT_ACTIONS_DIM)