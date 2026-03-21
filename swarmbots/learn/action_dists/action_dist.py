import abc
from typing import Callable, Optional, Self, Any

import torch
import torch.distributions as torchdist
from torch import nn

from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.masking import masked_mean
from swarmbots.learn.summary_statistics import compute_summary_statistics

ActionNetInitialization = Callable[[nn.Linear], None]
ActionMetricsSplitter = Callable[[torch.Tensor], dict[str, torch.Tensor]]
ActionMetricsSplitterInput = ActionMetricsSplitter | list[ActionMetricsSplitter | None] | None

AGENT_ACTIONS_DIM = -1



class ActionDist(nn.Module, abc.ABC):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization | None,
            init_action_net: bool = True,
    ):
        if init_action_net and action_net_initialization is None:
            raise ValueError('If init_action_net=True, then action_net_initialization must be given!')

        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        if init_action_net:
            self.action_net_initialization = action_net_initialization

            if latent_dim == action_dim:
                self.action_net = nn.Identity()
            else:
                self.action_net = nn.Linear(latent_dim, action_dim)

                action_net_initialization(self.action_net)
        else:
            self.action_net_initialization = None
            self.action_net = None

        self.distribution: Optional[torchdist.Distribution] = None

    @abc.abstractmethod
    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def compute_extra_losses(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[LossDict, LossMetrics]:
        _ = (agent_mask, action_splitter)
        return {}, {}

    def requires_previous_actions(self) -> bool:
        return False

    @abc.abstractmethod
    def get_metrics(
            self,
            actions: torch.Tensor,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def get_actions(
            self,
            deterministic: bool = False,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if deterministic:
            return self.mode(previous_actions=previous_actions)
        return self.sample(agent=agent, previous_actions=previous_actions)

    def get_actions_with_log_probs(
            self,
            latent_pi: torch.Tensor,
            deterministic: bool = False,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ):
        actions = self.update_latent_features(latent_pi).get_actions(
            deterministic=deterministic,
            agent=agent,
            previous_actions=previous_actions,
        )
        log_probs = self.log_prob(actions, previous_actions=previous_actions)
        return actions, log_probs

    @staticmethod
    def validate_agent_mask(
            agent_mask: torch.Tensor | None,
            *,
            expected_shape: tuple[int, ...],
    ) -> None:
        if agent_mask is None:
            return
        if agent_mask.dtype != torch.bool:
            raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
        if tuple(agent_mask.shape) != expected_shape:
            raise ValueError(f"Expected agent_mask shape {expected_shape}, got {tuple(agent_mask.shape)}")




def resolve_action_metrics_splitter(
        action_splitter: ActionMetricsSplitterInput,
) -> ActionMetricsSplitter | None:
    if action_splitter is None or callable(action_splitter):
        return action_splitter
    if not isinstance(action_splitter, list):
        raise TypeError(
            f"action_splitter must be callable, list of callables, or None, got {type(action_splitter)}"
        )
    if len(action_splitter) == 0:
        return None
    if len(action_splitter) > 1:
        raise ValueError("Expected one splitter for a single action distribution, got multiple entries.")
    splitter = action_splitter[0]
    if splitter is not None and not callable(splitter):
        raise TypeError(f"action_splitter list entry must be callable or None, got {type(splitter)}")
    return splitter


def compute_action_metrics(
        actions: torch.Tensor,
        action_splitter: ActionMetricsSplitterInput,
        *,
        hist_bins: int,
) -> dict[str, Any]:
    splitter = resolve_action_metrics_splitter(action_splitter)
    metrics: dict[str, Any] = {}
    if splitter is None:
        metrics["act"] = compute_summary_statistics(actions, make_histogram=hist_bins)
    else:
        for key, split_actions in splitter(actions).items():
            metrics[f"act_{key}"] = compute_summary_statistics(split_actions, make_histogram=hist_bins)
    return metrics


def compute_split_entropy_metrics(
        entropy_per_action: torch.Tensor,
        *,
        action_splitter: ActionMetricsSplitterInput,
        agent_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    splitter = resolve_action_metrics_splitter(action_splitter)
    if splitter is None:
        return {}

    metrics: dict[str, float] = {}
    with torch.no_grad():
        for key, split_entropy in splitter(entropy_per_action).items():
            if split_entropy.ndim == entropy_per_action.ndim:
                split_entropy_per_agent = split_entropy.sum(dim=AGENT_ACTIONS_DIM)
            elif split_entropy.ndim == entropy_per_action.ndim - 1:
                split_entropy_per_agent = split_entropy
            else:
                raise ValueError(
                    "split entropy must return tensors with either the same rank as entropy_per_action "
                    f"or one rank less. Got split {key!r} shape {tuple(split_entropy.shape)} "
                    f"for entropy shape {tuple(entropy_per_action.shape)}."
                )
            split_entropy_mean = masked_mean(split_entropy_per_agent, agent_mask)
            metrics[f"ent_loss_{key}"] = (-split_entropy_mean).item()
    return metrics
