import abc
from typing import Optional, Self, Any

import torch
import torch.distributions as torchdist

from swarmbots.learn.action_dists.action_dist import (
    BOUNDED_ACTION_HISTOGRAM,
    ActionNetInitialization,
    ActionDist,
    ActionMetricsSplitterInput,
    compute_action_metrics,
    resolve_action_metrics_splitter,
)
from swarmbots.learn.summary_statistics import compute_summary_statistics
from swarmbots.learn.action_dists.entropy_utils import compute_entropy_metrics


STD_HISTOGRAM_BINS = 11


class ContinuousActionDist(ActionDist, abc.ABC):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization,
            squash_output: bool = False,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
        )
        self.squash_output = squash_output

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

    @torch.no_grad()
    def get_metrics(
            self,
            actions: torch.Tensor,
            action_splitter: ActionMetricsSplitterInput = None,
            *,
            agent_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        splitter = resolve_action_metrics_splitter(action_splitter)
        metrics = compute_action_metrics(actions, splitter, histogram=BOUNDED_ACTION_HISTOGRAM)
        distribution = getattr(self, "distribution", None)
        if distribution is not None:
            metrics.update(compute_entropy_metrics(
                distribution.entropy(),
                agent_mask=agent_mask,
                action_splitter=splitter,
                name="ent_normal" if self.squash_output else "ent",
            ))

        if not hasattr(self, "log_stds"):
            return metrics
        log_stds = getattr(self, "log_stds")
        if not isinstance(log_stds, torch.Tensor):
            return metrics

        std_values = self._std_values_for_metrics(agent_mask)

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

    def _std_values_for_metrics(self, agent_mask: torch.Tensor | None) -> torch.Tensor:
        return self.log_stds.exp()

    @staticmethod
    def sum_action_dim(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.sum(dim=-1)
