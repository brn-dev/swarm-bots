import math
from dataclasses import dataclass
from typing import Optional, Self

import torch
import torch.distributions as torchdist

from swarmbots.learn.action_dists.action_dist import (
    AGENT_ACTIONS_DIM,
    ActionMetricsSplitterInput,
    ActionNetInitialization,
    compute_split_entropy_metrics,
)
from swarmbots.learn.action_dists.discrete_action_dist import DiscreteActionDist
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.masking import masked_mean
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


@dataclass(frozen=True)
class BernoulliConfig:
    initial_prob: float | None = None
    ent_loss_coef: float = 0.0


class BernoulliActionDist(DiscreteActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
            initial_prob: float | None = None,
            ent_loss_coef: float = 0.0,
    ):
        if ent_loss_coef < 0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {ent_loss_coef}")
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
        )
        self.ent_loss_coef = ent_loss_coef

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

    def compute_extra_losses(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[LossDict, LossMetrics]:
        if self.ent_loss_coef <= 0:
            return {}, {}
        entropy_per_action = self.distribution.entropy()
        entropy_per_agent = entropy_per_action.sum(dim=AGENT_ACTIONS_DIM)
        self.validate_agent_mask(agent_mask, expected_shape=tuple(entropy_per_agent.shape))
        entropy_loss = -self.ent_loss_coef * entropy_per_agent
        with torch.no_grad():
            entropy_mean = masked_mean(entropy_per_agent, agent_mask)
            metrics: LossMetrics = {
                "ent_loss": (-entropy_mean).item(),
                "ent_loss_scaled": (-self.ent_loss_coef * entropy_mean).item(),
            }
            metrics.update(
                compute_split_entropy_metrics(
                    entropy_per_action,
                    action_splitter=action_splitter,
                    agent_mask=agent_mask,
                )
            )
        return {"entropy": entropy_loss}, metrics

    def set_ent_loss_coef(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {value}")
        self.ent_loss_coef = value

    def _get_metrics_hist_bins(self) -> int:
        return 2
