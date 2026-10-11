import math
from dataclasses import dataclass
from typing import Optional, Self, Any

import torch
import torch.distributions as torchdist

from swarmbots.learn.action_dists.action_dist import (
    AGENT_ACTIONS_DIM,
    ActionNetInitialization,
)
from swarmbots.learn.action_dists.discrete_action_dist import DiscreteActionDist
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal
from swarmbots.learn.summary_statistics import HistogramConfig


BERNOULLI_ACTION_HISTOGRAM = HistogramConfig(bins=2, low=0.0, high=1.0)


@dataclass(frozen=True)
class BernoulliConfig:
    initial_prob: float | None = None


class BernoulliActionDist(DiscreteActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
            initial_prob: float | None = None,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
        )
        self.initial_prob = initial_prob

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
        self.distribution = torchdist.Bernoulli(logits=action_logits, validate_args=False)
        return self

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        return torch.round(self.distribution.probs)

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=AGENT_ACTIONS_DIM)

    @property
    def compile_friendly(self) -> bool:
        return True

    def _get_metrics_histogram(self) -> HistogramConfig:
        return BERNOULLI_ACTION_HISTOGRAM

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "initial_prob": self.initial_prob,
        }
