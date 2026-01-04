import math
from typing import Optional, Self

import torch
import torch.distributions as torchdist

from swarmbots.learn.action_dists.action_dist import AGENT_ACTIONS_DIM, ActionNetInitialization
from swarmbots.learn.action_dists.discrete_action_dist import DiscreteActionDist
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


class BernoulliActionDist(DiscreteActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
            initial_prob: float | None = None,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
        )

        self.distribution: Optional[torchdist.Bernoulli] = None
        if initial_prob is not None:
            if not 0.0 < initial_prob < 1.0:
                raise ValueError(
                    f"initial_prob must be strictly between 0 and 1, got {initial_prob}."
                )

            initial_logit = math.log(initial_prob / (1.0 - initial_prob))
            with torch.no_grad():
                self.action_net.bias.fill_(initial_logit)

    def update_distribution_params(self, action_logits: torch.Tensor) -> Self:
        self.distribution = torchdist.Bernoulli(logits=action_logits)
        return self

    def sample(self, agent: int | None = None) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        return torch.round(self.distribution.probs)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=AGENT_ACTIONS_DIM)

    def entropy(self) -> Optional[torch.Tensor]:
        return self.distribution.entropy().sum(dim=AGENT_ACTIONS_DIM)