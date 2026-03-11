import math
from typing import Optional, Self

import torch
from torch import nn, distributions as torchdist
from torch.nn import functional as F

from swarmbots.learn.action_dists.action_dist import ActionDist, ActionNetInitialization, AGENT_ACTIONS_DIM


class BimodalBetaActionDist(ActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization | None,
            epsilon: float = 1e-6,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
            init_action_net=False,
        )
        self.epsilon = epsilon

        self.weight_logits_net = nn.Linear(latent_dim, action_dim)
        self.alpha1_net = nn.Linear(latent_dim, action_dim)
        self.beta1_net = nn.Linear(latent_dim, action_dim)
        self.alpha2_net = nn.Linear(latent_dim, action_dim)
        self.beta2_net = nn.Linear(latent_dim, action_dim)

        if action_net_initialization is not None:
            for layer in (
                    self.weight_logits_net,
                    self.alpha1_net,
                    self.beta1_net,
                    self.alpha2_net,
                    self.beta2_net,
            ):
                action_net_initialization(layer)

        self.weight_logits: Optional[torch.Tensor] = None
        self.component1: Optional[torchdist.Beta] = None
        self.component2: Optional[torchdist.Beta] = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        self.weight_logits = self.weight_logits_net(latent_pi)

        alpha1 = 1.0 + F.softplus(self.alpha1_net(latent_pi))
        beta1 = 1.0 + F.softplus(self.beta1_net(latent_pi))
        alpha2 = 1.0 + F.softplus(self.alpha2_net(latent_pi))
        beta2 = 1.0 + F.softplus(self.beta2_net(latent_pi))

        self.component1 = torchdist.Beta(concentration1=alpha1, concentration0=beta1)
        self.component2 = torchdist.Beta(concentration1=alpha2, concentration0=beta2)
        return self

    def _assert_ready(self) -> None:
        if self.weight_logits is None or self.component1 is None or self.component2 is None:
            raise RuntimeError("Distribution parameters are not initialized. Call update_latent_features first.")

    def sample(self, agent: int | None = None) -> torch.Tensor:
        self._assert_ready()
        weights = torch.sigmoid(self.weight_logits)
        use_component1 = torch.bernoulli(weights).to(dtype=torch.bool)
        sampled_in_01 = torch.where(use_component1, self.component1.sample(), self.component2.sample())
        return 2.0 * sampled_in_01 - 1.0

    def mode(self) -> torch.Tensor:
        self._assert_ready()
        weights = torch.sigmoid(self.weight_logits)
        mean_in_01 = weights * self.component1.mean + (1.0 - weights) * self.component2.mean
        return 2.0 * mean_in_01 - 1.0

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        self._assert_ready()
        actions_in_01 = ((actions + 1.0) * 0.5).clamp(self.epsilon, 1.0 - self.epsilon)

        log_component1 = self.component1.log_prob(actions_in_01)
        log_component2 = self.component2.log_prob(actions_in_01)

        log_weight = F.logsigmoid(self.weight_logits)
        log_one_minus_weight = F.logsigmoid(-self.weight_logits)
        log_prob_in_01 = torch.logaddexp(log_weight + log_component1, log_one_minus_weight + log_component2)

        log_prob_in_m1_1 = log_prob_in_01 + math.log(0.5)
        return log_prob_in_m1_1.sum(dim=AGENT_ACTIONS_DIM)

    def entropy(self) -> Optional[torch.Tensor]:
        return None
