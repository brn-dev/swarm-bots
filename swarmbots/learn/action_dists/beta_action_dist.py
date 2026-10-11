import math
from dataclasses import dataclass
from typing import Any, Optional, Self

import torch
from torch import distributions as torchdist, nn
from torch.nn import functional as F

from swarmbots.learn.action_dists.action_dist import (
    AGENT_ACTIONS_DIM,
    BOUNDED_ACTION_HISTOGRAM,
    ActionDist,
    ActionMetricsSplitterInput,
    ActionNetInitialization,
    compute_action_metrics,
    resolve_action_metrics_splitter,
    validate_probability_clamp_epsilon,
)
from swarmbots.learn.action_dists.entropy_utils import compute_entropy_metrics


@dataclass(frozen=True)
class BetaConfig:
    alpha: float = 1.0 + math.log(2.0)
    beta: float = 1.0 + math.log(2.0)
    epsilon: float = 1e-6


class BetaActionDist(ActionDist):
    _OUTPUTS_PER_ACTION = 2

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization | None,
            alpha: float = 1.0 + math.log(2.0),
            beta: float = 1.0 + math.log(2.0),
            epsilon: float = 1e-6,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
            init_action_net=False,
        )

        if alpha <= 1.0:
            raise ValueError(f"alpha must be > 1.0, got {alpha}")
        if beta <= 1.0:
            raise ValueError(f"beta must be > 1.0, got {beta}")
        validate_probability_clamp_epsilon(epsilon)

        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        self.output_net = nn.Linear(latent_dim, action_dim * self._OUTPUTS_PER_ACTION)
        if action_net_initialization is not None:
            action_net_initialization(self.output_net)

        with torch.no_grad():
            bias = self.output_net.bias.view(action_dim, self._OUTPUTS_PER_ACTION)
            bias[:, 0] = _inverse_softplus(alpha - 1.0)
            bias[:, 1] = _inverse_softplus(beta - 1.0)

        self.distribution: Optional[torchdist.Beta] = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        raw = self.output_net(latent_pi).view(*latent_pi.shape[:-1], self.action_dim, self._OUTPUTS_PER_ACTION)
        alpha = 1.0 + F.softplus(raw[..., 0])
        beta = 1.0 + F.softplus(raw[..., 1])
        self.distribution = torchdist.Beta(
            concentration1=alpha,
            concentration0=beta,
            validate_args=False,
        )
        return self

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actions_01 = self.distribution.sample()
        return 2.0 * actions_01 - 1.0

    def rsample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actions_01 = self.distribution.rsample()
        return 2.0 * actions_01 - 1.0

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        return 2.0 * self.distribution.mean - 1.0

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actions_01 = ((actions + 1.0) * 0.5).clamp(self.epsilon, 1.0 - self.epsilon)
        log_prob_01 = self.distribution.log_prob(actions_01)
        return (log_prob_01 + math.log(0.5)).sum(dim=AGENT_ACTIONS_DIM)

    @torch.no_grad()
    def get_metrics(
            self,
            actions: torch.Tensor,
            action_splitter: ActionMetricsSplitterInput = None,
            *,
            agent_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        metrics = compute_action_metrics(actions, action_splitter, histogram=BOUNDED_ACTION_HISTOGRAM)
        if self.distribution is not None:
            metrics.update(compute_entropy_metrics(
                self.distribution.entropy() + math.log(2.0),
                agent_mask=agent_mask,
                action_splitter=resolve_action_metrics_splitter(action_splitter),
            ))
        return metrics

    @property
    def compile_friendly(self) -> bool:
        return True

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "alpha": self.alpha,
            "beta": self.beta,
            "epsilon": self.epsilon,
        }


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))
