import math
from dataclasses import dataclass

import torch
from torch import distributions as torchdist

from swarmbots.learn.action_dists.action_dist import ActionNetInitialization
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
        if self.negative_magnitude_dist is None:
            return None
        assert isinstance(self.negative_magnitude_dist, torchdist.Beta)
        return self.negative_magnitude_dist

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
        negative_alpha = shape_parameter_greater_than_one(raw_magnitude_parameters[..., 0])
        negative_beta = shape_parameter_greater_than_one(raw_magnitude_parameters[..., 1])
        positive_alpha = shape_parameter_greater_than_one(raw_magnitude_parameters[..., 2])
        positive_beta = shape_parameter_greater_than_one(raw_magnitude_parameters[..., 3])
        return (
            torchdist.Beta(
                concentration1=negative_alpha,
                concentration0=negative_beta,
                validate_args=False,
            ),
            torchdist.Beta(
                concentration1=positive_alpha,
                concentration0=positive_beta,
                validate_args=False,
            ),
        )

    def _magnitude_modes(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.negative_beta_dist is not None
        assert self.positive_beta_dist is not None

        def beta_mode(distribution: torchdist.Beta) -> torch.Tensor:
            return (
                (distribution.concentration1 - 1.0)
                / (distribution.concentration1 + distribution.concentration0 - 2.0)
            )

        return beta_mode(self.negative_beta_dist), beta_mode(self.positive_beta_dist)
