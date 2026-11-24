import math
from dataclasses import dataclass

import torch
from torch import distributions as torchdist

from swarmbots.learn.action_dists.action_dist import AGENT_ACTIONS_DIM, ActionNetInitialization
from swarmbots.learn.action_dists.beta_action_dist import promote_cpu_beta_parameters
from swarmbots.learn.action_dists.sign_magnitude_action_dist import (
    SignMagnitudeActionDist,
    shape_parameter_greater_than_one,
)


@dataclass(frozen=True)
class SignMagnitudeBetaConfig:
    initial_positive_prob: float | None = None
    epsilon: float = 1e-6
    negative_alpha: float = 1.0 + math.log(2.0)
    negative_beta: float = 1.0 + math.log(2.0)
    positive_alpha: float = 1.0 + math.log(2.0)
    positive_beta: float = 1.0 + math.log(2.0)


def _log_beta_normalizer(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)


def beta_log_density(magnitude: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    # Both the log-gamma terms and the final log density can cancel for concentrated Betas.
    dtype = torch.promote_types(magnitude.dtype, alpha.dtype)
    x, a, b = magnitude.double(), alpha.double(), beta.double()
    return ((a - 1) * x.log() + (b - 1) * torch.log1p(-x) + _log_beta_normalizer(a, b)).to(dtype)


class SignMagnitudeBetaActionDist(SignMagnitudeActionDist):
    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization | None,
            initial_positive_prob: float | None = None,
            epsilon: float = 1e-6,
            negative_alpha: float = 1.0 + math.log(2.0),
            negative_beta: float = 1.0 + math.log(2.0),
            positive_alpha: float = 1.0 + math.log(2.0),
            positive_beta: float = 1.0 + math.log(2.0),
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
            initial_positive_prob=initial_positive_prob,
            epsilon=epsilon,
            initial_magnitude_parameters=(negative_alpha, negative_beta, positive_alpha, positive_beta),
            magnitude_parameter_names=(
                "negative_alpha",
                "negative_beta",
                "positive_alpha",
                "positive_beta",
            ),
            magnitude_entropy_metric_name="ent_beta",
        )

    @property
    def negative_beta_dist(self) -> torchdist.Beta | None:
        """Beta law of the configured negative unit-interval coordinate, action + 1."""
        if self.negative_magnitude_dist is None:
            return None
        assert isinstance(self.negative_magnitude_dist, torchdist.Beta)
        return torchdist.Beta(
            self.negative_magnitude_dist.concentration0,
            self.negative_magnitude_dist.concentration1,
            validate_args=False,
        )

    @property
    def positive_beta_dist(self) -> torchdist.Beta | None:
        if self.positive_magnitude_dist is None:
            return None
        assert isinstance(self.positive_magnitude_dist, torchdist.Beta)
        return self.positive_magnitude_dist

    def _make_magnitude_distributions(
            self,
            raw_magnitude_parameters: torch.Tensor,
    ) -> tuple[torchdist.Distribution, torchdist.Distribution]:
        raw_magnitude_parameters = promote_cpu_beta_parameters(raw_magnitude_parameters)
        negative_alpha = shape_parameter_greater_than_one(raw_magnitude_parameters[..., 0])
        negative_beta = shape_parameter_greater_than_one(raw_magnitude_parameters[..., 1])
        positive_alpha = shape_parameter_greater_than_one(raw_magnitude_parameters[..., 2])
        positive_beta = shape_parameter_greater_than_one(raw_magnitude_parameters[..., 3])
        # X - 1 for X ~ Beta(a, b) has the same law as -Y for Y ~ Beta(b, a).
        # Drawing the magnitude directly keeps small negative actions and their likelihoods representable.
        return (
            torchdist.Beta(
                concentration1=negative_beta,
                concentration0=negative_alpha,
                validate_args=False,
            ),
            torchdist.Beta(
                concentration1=positive_alpha,
                concentration0=positive_beta,
                validate_args=False,
            ),
        )

    def log_prob(
            self, actions: torch.Tensor, previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._assert_ready()
        negative, positive = self.negative_magnitude_dist, self.positive_beta_dist
        assert isinstance(negative, torchdist.Beta) and positive is not None
        negative_mask = actions < 0.0
        log_weights = self._component_log_probs(previous_actions)
        log_sign = torch.where(negative_mask, log_weights[..., self._NEGATIVE_INDEX], log_weights[..., self._POSITIVE_INDEX])
        alpha = torch.where(negative_mask, negative.concentration1, positive.concentration1)
        beta = torch.where(negative_mask, negative.concentration0, positive.concentration0)
        magnitude = actions.abs()
        # External and deterministic actions can reach an endpoint; ordinary Beta draws are interior.
        limits = torch.finfo(magnitude.dtype)
        safe_magnitude = torch.where(
            (magnitude > 0) & (magnitude < 1), magnitude,
            magnitude.clamp(max(self.epsilon, limits.tiny), 1.0 - max(self.epsilon, limits.eps / 2)),
        )
        return (log_sign + beta_log_density(safe_magnitude, alpha, beta)).sum(dim=AGENT_ACTIONS_DIM)

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        self._assert_ready()
        negative = self.negative_magnitude_dist
        positive = self.positive_beta_dist
        assert isinstance(negative, torchdist.Beta) and positive is not None
        negative_mode, negative_peak = self._beta_mode_and_log_density(negative.concentration1, negative.concentration0)
        positive_mode, positive_peak = self._beta_mode_and_log_density(positive.concentration1, positive.concentration0)
        log_weights = self._component_log_probs(previous_actions)
        prefer_negative = negative_peak + log_weights[..., 0] >= positive_peak + log_weights[..., 1]
        return torch.where(prefer_negative, -negative_mode, positive_mode).to(log_weights.dtype)

    @staticmethod
    def _beta_mode_and_log_density(alpha: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Score the mathematical peak before rounding the selected action.
        alpha, beta = alpha.double(), beta.double()
        a, b = alpha - 1, beta - 1
        log_peak = (
            _log_beta_normalizer(alpha, beta)
            - a * torch.log1p(b / a) - b * torch.log1p(a / b)
        )
        return a / (a + b), log_peak
