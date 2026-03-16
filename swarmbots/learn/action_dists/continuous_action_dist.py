import abc
from typing import Optional, Self

import torch
import torch.distributions as torchdist

from swarmbots.learn.action_dists.action_dist import AGENT_ACTIONS_DIM, ActionNetInitialization, ActionDist
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.masking import masked_mean


class ContinuousActionDist(ActionDist, abc.ABC):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization,
            ent_loss_coef: float = 0.0,
            action_magnitude_loss_coef: float = 0.0,
            action_magnitude_loss_threshold: float = 0.0,
            action_magnitude_loss_power: int = 2,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
        )
        if ent_loss_coef < 0:
            raise ValueError(f"Expected ent_loss_coef >= 0, got {ent_loss_coef}")
        self._validate_action_magnitude_loss_params(
            coef=action_magnitude_loss_coef,
            threshold=action_magnitude_loss_threshold,
            power=action_magnitude_loss_power,
        )
        self.ent_loss_coef = ent_loss_coef
        self.action_magnitude_loss_coef = action_magnitude_loss_coef
        self.action_magnitude_loss_threshold = action_magnitude_loss_threshold
        self.action_magnitude_loss_power = action_magnitude_loss_power

        self.distribution: Optional[torchdist.Distribution] = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        action_means = self.action_net(latent_pi)
        return self.update_distribution_params(action_means, self.log_stds)

    @abc.abstractmethod
    def update_distribution_params(self, means: torch.Tensor, log_stds: torch.Tensor) -> Self:
        raise NotImplementedError

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.sum_action_dim(self.distribution.log_prob(actions))

    def compute_extra_losses(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[LossDict, LossMetrics]:
        action_magnitude_loss, action_magnitude_metrics = self.compute_action_magnitude_loss(
            agent_mask=agent_mask
        )
        if action_magnitude_loss is None:
            return {}, action_magnitude_metrics
        return {"action_magnitude": action_magnitude_loss}, action_magnitude_metrics

    def compute_action_magnitude_loss(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[Optional[torch.Tensor], LossMetrics]:
        action_magnitude_penalty, action_magnitude_metrics = self._compute_scaled_action_magnitude_penalty(
            agent_mask=agent_mask
        )
        return action_magnitude_penalty, action_magnitude_metrics

    def compute_entropy_loss(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[Optional[torch.Tensor], LossMetrics]:
        if self.ent_loss_coef <= 0:
            return None, {}
        entropy_per_agent = self.sum_action_dim(self.distribution.entropy())
        self.validate_agent_mask(agent_mask, expected_shape=tuple(entropy_per_agent.shape))
        entropy_valid_mask: torch.Tensor | None = agent_mask
        entropy_mean = masked_mean(entropy_per_agent, entropy_valid_mask)
        entropy_loss_per_agent = -self.ent_loss_coef * entropy_per_agent
        return entropy_loss_per_agent, {
            "ent_loss": (-entropy_mean).item(),
            "ent_loss_scaled": (-self.ent_loss_coef * entropy_mean).item(),
        }

    @staticmethod
    def sum_action_dim(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.sum(dim=AGENT_ACTIONS_DIM)

    def set_action_magnitude_loss_coef(self, value: float) -> None:
        self._validate_action_magnitude_loss_params(
            coef=value,
            threshold=self.action_magnitude_loss_threshold,
            power=self.action_magnitude_loss_power,
        )
        self.action_magnitude_loss_coef = value

    def set_ent_loss_coef(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"Expected ent_loss_coef >= 0, got {value}")
        self.ent_loss_coef = value

    @staticmethod
    def _validate_action_magnitude_loss_params(
            *,
            coef: float,
            threshold: float,
            power: int,
    ) -> None:
        if coef < 0:
            raise ValueError(f"Expected action_magnitude_loss_coef >= 0, got {coef}")
        if threshold < 0:
            raise ValueError(f"Expected action_magnitude_loss_threshold >= 0, got {threshold}")
        if power < 1:
            raise ValueError(f"Expected action_magnitude_loss_power >= 1, got {power}")

    def _compute_scaled_action_magnitude_penalty(
            self,
            *,
            agent_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, LossMetrics]:
        coef = self.action_magnitude_loss_coef
        if coef <= 0:
            return None, {}

        action_means = self.distribution.mean
        if action_means.ndim != 3:
            raise ValueError(f"Expected action_means shape (B, N, A), got {tuple(action_means.shape)}")

        action_valid_mask: torch.Tensor | None = None
        if agent_mask is not None:
            self.validate_agent_mask(agent_mask, expected_shape=tuple(action_means.shape[:2]))
            action_valid_mask = agent_mask.unsqueeze(-1).expand_as(action_means)

        excess_action_magnitude = torch.relu(action_means.abs() - self.action_magnitude_loss_threshold)
        action_magnitude_loss_unscaled = excess_action_magnitude.pow(self.action_magnitude_loss_power)
        action_magnitude_loss_per_agent = action_magnitude_loss_unscaled.mean(dim=AGENT_ACTIONS_DIM)
        if agent_mask is not None:
            action_magnitude_loss_per_agent = (
                action_magnitude_loss_per_agent * agent_mask.to(dtype=action_magnitude_loss_per_agent.dtype)
            )

        action_magnitude_loss = masked_mean(action_magnitude_loss_unscaled, action_valid_mask)
        action_magnitude_loss_scaled = coef * action_magnitude_loss

        return action_magnitude_loss_per_agent * coef, {
            "action_magnitude_loss": action_magnitude_loss.item(),
            "action_magnitude_loss_scaled": action_magnitude_loss_scaled.item(),
        }
