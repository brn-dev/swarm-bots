from typing import Optional, Self

import torch

from swarmbots.learn.action_dists.action_dist import ActionNetInitialization
from swarmbots.learn.action_dists.diag_gaussian_action_dist import DiagGaussianActionDist
from swarmbots.learn.action_dists.tanh_bijector import TanhBijector
from swarmbots.learn.losses import LossMetrics
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


# Inspired by
class SquashedDiagGaussianActionDist(DiagGaussianActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            std: float,
            std_learnable: bool,
            epsilon: float = 1e-6,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            std=std,
            std_learnable=std_learnable,
            action_net_initialization=action_net_initialization,
        )

        self.epsilon = epsilon
        self._last_gaussian_actions: Optional[torch.Tensor] = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        super().update_latent_features(latent_pi)
        return self

    def log_prob(self, actions: torch.Tensor, gaussian_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        if gaussian_actions is None:
            gaussian_actions = TanhBijector.inverse(actions)

        log_prob = super().log_prob(gaussian_actions)
        log_prob -= self.sum_action_dim(torch.log(1 - actions ** 2 + self.epsilon))

        return log_prob

    def compute_exploration_loss(self) -> tuple[Optional[torch.Tensor], LossMetrics]:
        return None, {}

    def sample(self, agent: int | None = None) -> torch.Tensor:
        self._last_gaussian_actions = super().sample()
        return torch.tanh(self._last_gaussian_actions)

    def mode(self) -> torch.Tensor:
        self._last_gaussian_actions = super().mode()
        return torch.tanh(self._last_gaussian_actions)

    def get_actions_with_log_probs(
            self,
            latent_pi: torch.Tensor,
            deterministic: bool = False,
            agent: int | None = None,
    ):
        # get_actions calls sample() or mode(), both of which set _last_gaussian_actions
        # --> prevents squashing and unsquashing which can lead to numerical instability
        actions = self.update_latent_features(latent_pi).get_actions(deterministic=deterministic, agent=agent)
        log_probs = self.log_prob(actions, self._last_gaussian_actions)
        return actions, log_probs
