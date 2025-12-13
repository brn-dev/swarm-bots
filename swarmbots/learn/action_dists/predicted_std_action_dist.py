import math
from typing import Optional, Self

import torch
import torch.distributions as torchdist
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionNetInitialization
from swarmbots.learn.action_dists.continuous_action_dist import ContinuousActionDist
from swarmbots.learn.action_dists.tanh_bijector import TanhBijector

LogStdNetInitialization = ActionNetInitialization


class PredictedStdActionDist(ContinuousActionDist):

    base_log_std: Optional[float]

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            base_std: float,
            squash_output: bool = False,
            epsilon: float = 1e-6,
            action_net_initialization: ActionNetInitialization | None = None,
            log_std_net_initialization: LogStdNetInitialization | None = None,
            log_std_clamp_range: tuple[float, float] = (-20.0, 2.0)
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
        )

        self.log_std_net = nn.Linear(latent_dim, action_dim)
        self.log_std_net_initialization = log_std_net_initialization
        if log_std_net_initialization is not None:
            log_std_net_initialization(self.log_std_net)

        self.base_log_std = math.log(base_std) if base_std != 1 else None
        self.log_std_clamp_range = log_std_clamp_range

        self.squash_output = squash_output
        self.epsilon = epsilon

        self.distribution: Optional[torchdist.Normal] = None

        self._last_gaussian_actions: Optional[torch.Tensor] = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        mean_actions = self.action_net(latent_pi)
        log_stds = self.log_std_net(latent_pi)
        if self.base_log_std is not None:
            log_stds = log_stds + self.base_log_std
        return self.update_distribution_params(mean_actions, log_stds)

    def update_distribution_params(
            self,
            mean_actions: torch.Tensor,
            log_stds: torch.Tensor,
    ) -> Self:
        log_stds = torch.clamp(log_stds, *self.log_std_clamp_range)
        self.distribution = torchdist.Normal(mean_actions, log_stds.exp())
        return self

    def sample(self) -> torch.Tensor:
        gaussian_actions = self.distribution.rsample()
        if self.squash_output:
            self._last_gaussian_actions = gaussian_actions
            return TanhBijector.forward(gaussian_actions)
        return gaussian_actions

    def mode(self) -> torch.Tensor:
        gaussian_actions = self.distribution.mean
        if self.squash_output:
            self._last_gaussian_actions = gaussian_actions
            return TanhBijector.forward(gaussian_actions)
        return gaussian_actions

    def log_prob(self, actions: torch.Tensor, gaussian_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.squash_output:
            return super().log_prob(actions)

        if gaussian_actions is None:
            gaussian_actions = TanhBijector.inverse(actions)

        log_prob = super().log_prob(gaussian_actions)
        log_prob -= self.sum_action_dim(torch.log(1 - actions ** 2 + self.epsilon))

        return log_prob

    def entropy(self) -> Optional[torch.Tensor]:
        if self.squash_output:
            return None
        return self.sum_action_dim(self.distribution.entropy())

    def get_actions_with_log_probs(self, latent_pi: torch.Tensor, deterministic: bool = False):
        # get_actions calls sample() or mode(), both of which set _last_gaussian_actions
        # --> prevents squashing and unsquashing which can lead to numerical instability
        actions = self.update_latent_features(latent_pi).get_actions(deterministic=deterministic)
        log_probs = self.log_prob(actions, self._last_gaussian_actions)
        return actions, log_probs

    def set_base_std(self, std: float):
        self.base_log_std = math.log(std)