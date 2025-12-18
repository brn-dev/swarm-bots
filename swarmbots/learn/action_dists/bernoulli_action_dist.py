from typing import Optional, Self

import torch
import torch.distributions as torchdist

from swarmbots import ActionNetInitialization, AGENT_ACTIONS_DIM
from swarmbots import DiscreteActionDist


class BernoulliActionDist(DiscreteActionDist):

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

        self.distribution: Optional[torchdist.Bernoulli] = None

    def update_distribution_params(self, action_logits: torch.Tensor) -> Self:
        self.distribution = torchdist.Bernoulli(logits=action_logits)
        return self

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        return torch.round(self.distribution.probs)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=AGENT_ACTIONS_DIM)

    def entropy(self) -> Optional[torch.Tensor]:
        return self.distribution.entropy().sum(dim=AGENT_ACTIONS_DIM)