import math
from dataclasses import dataclass
from typing import Optional, Self, Any

import torch
import torch.distributions as torchdist
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionNetInitialization
from swarmbots.learn.action_dists.continuous_action_dist import ContinuousActionDist
from swarmbots.learn.action_dists.tanh_bijector import TanhBijector
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal

LogStdNetInitialization = ActionNetInitialization


@dataclass(frozen=True)
class PredictedStdGaussianConfig:
    base_std: float
    epsilon: float = 1e-6
    log_std_net_initialization: LogStdNetInitialization = init_linear_orthogonal
    log_std_clamp_range: tuple[float, float] = (-20.0, 2.0)


class PredictedStdGaussianActionDist(ContinuousActionDist):
    """Diagonal Gaussian with a latent-conditioned mean and standard deviation."""

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            base_std: float,
            squash_output: bool = False,
            epsilon: float = 1e-6,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
            log_std_net_initialization: LogStdNetInitialization = init_linear_orthogonal,
            log_std_clamp_range: tuple[float, float] = (-20.0, 2.0),
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
            squash_output=squash_output,
        )

        self.log_std_net = nn.Linear(latent_dim, action_dim)
        self.log_std_net_initialization = log_std_net_initialization
        if log_std_net_initialization is not None:
            log_std_net_initialization(self.log_std_net)

        self.register_buffer("base_log_std", torch.tensor(math.log(base_std)))
        self.log_std_clamp_range = log_std_clamp_range

        self.epsilon = epsilon

        self.distribution: Optional[torchdist.Normal] = None
        self.log_stds: Optional[torch.Tensor] = None

        self._last_gaussian_actions: Optional[torch.Tensor] = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        mean_actions = self.action_net(latent_pi)
        log_stds = self.log_std_net(latent_pi) + self.base_log_std
        return self.update_distribution_params(mean_actions, log_stds)

    def update_distribution_params(
            self,
            mean_actions: torch.Tensor,
            log_stds: torch.Tensor,
    ) -> Self:
        log_stds = torch.clamp(log_stds, *self.log_std_clamp_range)
        self.log_stds = log_stds
        self.distribution = torchdist.Normal(
            mean_actions,
            log_stds.exp(),
            validate_args=False,
        )
        return self

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gaussian_actions = self.distribution.sample()
        if self.squash_output:
            self._last_gaussian_actions = gaussian_actions
            return TanhBijector.forward(gaussian_actions)
        return gaussian_actions

    def rsample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gaussian_actions = self.distribution.rsample()
        if self.squash_output:
            self._last_gaussian_actions = gaussian_actions
            return TanhBijector.forward(gaussian_actions)
        return gaussian_actions

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        gaussian_actions = self.distribution.mean
        if self.squash_output:
            self._last_gaussian_actions = gaussian_actions
            return TanhBijector.forward(gaussian_actions)
        return gaussian_actions

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
            gaussian_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.squash_output:
            return super().log_prob(actions)

        if gaussian_actions is None:
            gaussian_actions = TanhBijector.inverse(actions)

        log_prob = super().log_prob(gaussian_actions)
        log_prob -= self.sum_action_dim(TanhBijector.log_abs_det_jacobian(gaussian_actions))

        return log_prob

    def get_actions_with_log_probs(
            self,
            latent_pi: torch.Tensor,
            deterministic: bool = False,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
            use_rsample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Avoids inverse tanh in the common sample/log-prob path, which is less numerically stable.
        actions = self.update_latent_features(latent_pi).get_actions(
            deterministic=deterministic,
            agent=agent,
            previous_actions=previous_actions,
            use_rsample=use_rsample,
        )
        log_probs = self.log_prob(actions, gaussian_actions=self._last_gaussian_actions)
        return actions, log_probs

    def set_std(self, std: float) -> None:
        self.set_base_std(std)

    def _std_values_for_metrics(self, agent_mask: torch.Tensor | None) -> torch.Tensor:
        std_values = super()._std_values_for_metrics(agent_mask)
        return std_values if agent_mask is None else std_values[agent_mask]

    def set_base_std(self, std: float) -> None:
        if std <= 0:
            raise ValueError(f"std must be > 0, got {std}")
        self.base_log_std.fill_(math.log(std))

    def scale_std(self, multiplier: float) -> None:
        if multiplier <= 0:
            raise ValueError(f"multiplier must be > 0, got {multiplier}")
        self.base_log_std.add_(math.log(multiplier))

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "base_std": math.exp(self.base_log_std),
            "squash_output": self.squash_output,
            "epsilon": self.epsilon,
            "log_std_clamp_range": list(self.log_std_clamp_range),
            "log_std_net_initialization": (
                self.log_std_net_initialization.__name__
                if hasattr(self.log_std_net_initialization, "__name__")
                else str(self.log_std_net_initialization)
            ),
        }

    @property
    def compile_friendly(self) -> bool:
        return True
