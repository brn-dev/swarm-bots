import abc
from typing import Optional, Self, Any

import torch
import torch.distributions as torchdist

from swarmbots.learn.action_dists.action_dist import (
    AGENT_ACTIONS_DIM,
    BOUNDED_ACTION_HISTOGRAM,
    ActionNetInitialization,
    ActionDist,
    ActionMetricsSplitterInput,
    compute_action_metrics,
    resolve_action_metrics_splitter,
)
from swarmbots.learn.action_dists.entropy_utils import (
    EntropyLossConfig,
    compute_ent_loss,
    compute_ent_metrics,
)
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.masking import masked_mean
from swarmbots.learn.serialization_utils import serialize_dataclass
from swarmbots.learn.summary_statistics import compute_summary_statistics


STD_HISTOGRAM_BINS = 11


class ContinuousActionDist(ActionDist, abc.ABC):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization,
            ent_loss_coef: float = 0.0,
            ent_loss_config: EntropyLossConfig | None = None,
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
        self.ent_loss_config = ent_loss_config if ent_loss_config is not None else EntropyLossConfig()
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

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.sum_action_dim(self.distribution.log_prob(actions))

    def compute_extra_losses(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[LossDict, LossMetrics]:
        _ = action_splitter
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
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[Optional[torch.Tensor], LossMetrics]:
        if self.ent_loss_coef <= 0:
            return None, {}
        entropy_per_action = self.distribution.entropy()
        entropy_loss_per_agent = self.ent_loss_coef * compute_ent_loss(
            config=self.ent_loss_config,
            entropy_per_action=entropy_per_action,
        )
        metrics = compute_ent_metrics(
            config=self.ent_loss_config,
            entropy_per_action=entropy_per_action,
            agent_mask=agent_mask,
            metrics_action_splitter=resolve_action_metrics_splitter(action_splitter),
            name="ent",
        )
        return entropy_loss_per_agent, metrics

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "ent_loss_coef": self.ent_loss_coef,
            "ent_loss_config": serialize_dataclass(self.ent_loss_config),
            "action_magnitude_loss_coef": self.action_magnitude_loss_coef,
            "action_magnitude_loss_threshold": self.action_magnitude_loss_threshold,
            "action_magnitude_loss_power": self.action_magnitude_loss_power,
        }


    def get_metrics(
            self,
            actions: torch.Tensor,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> dict[str, Any]:
        splitter = resolve_action_metrics_splitter(action_splitter)
        metrics = compute_action_metrics(actions, splitter, histogram=BOUNDED_ACTION_HISTOGRAM)

        if not hasattr(self, "log_stds"):
            return metrics
        log_stds = getattr(self, "log_stds")
        if not isinstance(log_stds, torch.Tensor):
            return metrics

        std_values = torch.exp(log_stds)

        can_split_stds = not (std_values.ndim >= 1 and std_values.shape[-1] == 1 and self.action_dim > 1)
        if splitter is not None and can_split_stds:
            for key, split_stds in splitter(std_values).items():
                metrics[f"std_{key}"] = compute_summary_statistics(
                    split_stds, find_min=True, find_max=True, make_histogram=STD_HISTOGRAM_BINS
                )
        else:
            metrics["std"] = compute_summary_statistics(
                std_values, find_min=True, find_max=True, make_histogram=STD_HISTOGRAM_BINS
            )

        return metrics

    @staticmethod
    def sum_action_dim(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.sum(dim=-1)

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
