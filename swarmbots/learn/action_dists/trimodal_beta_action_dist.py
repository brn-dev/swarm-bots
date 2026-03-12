import math
from typing import Optional, Self

import torch
from torch import nn, distributions as torchdist
from torch.nn import functional as F

from swarmbots.learn.action_dists.action_dist import ActionDist, ActionNetInitialization, AGENT_ACTIONS_DIM


class TrimodalBetaActionDist(ActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization | None,
            epsilon: float = 1e-6,
            alphas: tuple[float, float, float] = (
                1.0 + math.log(2.0),
                1.0 + math.log(2.0),
                1.0 + math.log(2.0),
            ),
            betas: tuple[float, float, float] = (
                1.0 + math.log(2.0),
                1.0 + math.log(2.0),
                1.0 + math.log(2.0),
            ),
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
            init_action_net=False,
        )
        self.epsilon = epsilon

        for parameter_name, parameter_values in (("alphas", alphas), ("betas", betas)):
            for parameter_value in parameter_values:
                if parameter_value <= 1.0:
                    raise ValueError(
                        f"All {parameter_name} values must be > 1.0, got {parameter_values}."
                    )

        self.weight_logits_net = nn.Linear(latent_dim, action_dim * 3)
        self.alpha1_net = nn.Linear(latent_dim, action_dim)
        self.beta1_net = nn.Linear(latent_dim, action_dim)
        self.alpha2_net = nn.Linear(latent_dim, action_dim)
        self.beta2_net = nn.Linear(latent_dim, action_dim)
        self.alpha3_net = nn.Linear(latent_dim, action_dim)
        self.beta3_net = nn.Linear(latent_dim, action_dim)

        if action_net_initialization is not None:
            for layer in (
                    self.weight_logits_net,
                    self.alpha1_net,
                    self.beta1_net,
                    self.alpha2_net,
                    self.beta2_net,
                    self.alpha3_net,
                    self.beta3_net,
            ):
                action_net_initialization(layer)
        with torch.no_grad():
            self.alpha1_net.bias.fill_(_inverse_softplus(alphas[0] - 1.0))
            self.beta1_net.bias.fill_(_inverse_softplus(betas[0] - 1.0))
            self.alpha2_net.bias.fill_(_inverse_softplus(alphas[1] - 1.0))
            self.beta2_net.bias.fill_(_inverse_softplus(betas[1] - 1.0))
            self.alpha3_net.bias.fill_(_inverse_softplus(alphas[2] - 1.0))
            self.beta3_net.bias.fill_(_inverse_softplus(betas[2] - 1.0))

        self.weight_logits: Optional[torch.Tensor] = None
        self.component1: Optional[torchdist.Beta] = None
        self.component2: Optional[torchdist.Beta] = None
        self.component3: Optional[torchdist.Beta] = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        raw_weight_logits = self.weight_logits_net(latent_pi)
        self.weight_logits = raw_weight_logits.view(*raw_weight_logits.shape[:-1], self.action_dim, 3)

        alpha1 = 1.0 + F.softplus(self.alpha1_net(latent_pi))
        beta1 = 1.0 + F.softplus(self.beta1_net(latent_pi))
        alpha2 = 1.0 + F.softplus(self.alpha2_net(latent_pi))
        beta2 = 1.0 + F.softplus(self.beta2_net(latent_pi))
        alpha3 = 1.0 + F.softplus(self.alpha3_net(latent_pi))
        beta3 = 1.0 + F.softplus(self.beta3_net(latent_pi))

        self.component1 = torchdist.Beta(concentration1=alpha1, concentration0=beta1)
        self.component2 = torchdist.Beta(concentration1=alpha2, concentration0=beta2)
        self.component3 = torchdist.Beta(concentration1=alpha3, concentration0=beta3)
        return self

    def _assert_ready(self) -> None:
        if (
                self.weight_logits is None
                or self.component1 is None
                or self.component2 is None
                or self.component3 is None
        ):
            raise RuntimeError("Distribution parameters are not initialized. Call update_latent_features first.")

    def sample(self, agent: int | None = None) -> torch.Tensor:
        self._assert_ready()
        weights = F.softmax(self.weight_logits, dim=-1)
        component_indices = torch.multinomial(
            weights.reshape(-1, 3),
            num_samples=1,
        ).reshape(weights.shape[:-1])

        sampled_component1 = self.component1.sample()
        sampled_component2 = self.component2.sample()
        sampled_component3 = self.component3.sample()

        sampled_in_01 = torch.where(
            component_indices == 0,
            sampled_component1,
            torch.where(component_indices == 1, sampled_component2, sampled_component3),
        )
        return 2.0 * sampled_in_01 - 1.0

    def mode(self) -> torch.Tensor:
        self._assert_ready()
        weights = F.softmax(self.weight_logits, dim=-1)
        mean_in_01 = (
            weights[..., 0] * self.component1.mean
            + weights[..., 1] * self.component2.mean
            + weights[..., 2] * self.component3.mean
        )
        return 2.0 * mean_in_01 - 1.0

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        self._assert_ready()
        actions_in_01 = ((actions + 1.0) * 0.5).clamp(self.epsilon, 1.0 - self.epsilon)

        log_component1 = self.component1.log_prob(actions_in_01)
        log_component2 = self.component2.log_prob(actions_in_01)
        log_component3 = self.component3.log_prob(actions_in_01)

        log_weights = F.log_softmax(self.weight_logits, dim=-1)
        stacked_log_probs = torch.stack(
            (
                log_weights[..., 0] + log_component1,
                log_weights[..., 1] + log_component2,
                log_weights[..., 2] + log_component3,
            ),
            dim=-1,
        )
        log_prob_in_01 = torch.logsumexp(stacked_log_probs, dim=-1)

        log_prob_in_m1_1 = log_prob_in_01 + math.log(0.5)
        return log_prob_in_m1_1.sum(dim=AGENT_ACTIONS_DIM)

    def entropy(self) -> Optional[torch.Tensor]:
        return None


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))
