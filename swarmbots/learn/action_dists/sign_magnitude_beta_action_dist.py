import math
from dataclasses import dataclass, field
from typing import Any, Optional, Self

import torch
from torch import distributions as torchdist, nn
from torch.nn import functional as F

from swarmbots.learn.action_dists.action_dist import (
    AGENT_ACTIONS_DIM,
    ActionDist,
    ActionMetricsSplitterInput,
    ActionNetInitialization,
    compute_action_metrics,
    resolve_action_metrics_splitter,
)
from swarmbots.learn.action_dists.entropy_utils import (
    AgentActionsReduction,
    EntropyLossConfig,
    compute_ent_loss,
    compute_ent_metrics,
)
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.serialization_utils import serialize_dataclass


def _default_entropy_loss_config() -> EntropyLossConfig:
    return EntropyLossConfig(
        agent_actions_reduction=AgentActionsReduction.SUM,
        metrics_reduction=AgentActionsReduction.MEAN,
    )


@dataclass(frozen=True)
class SignMagnitudeBetaConfig:
    initial_positive_prob: float | None = None
    epsilon: float = 1e-6
    negative_alpha: float = 1.0 + math.log(2.0)
    negative_beta: float = 1.0 + math.log(2.0)
    positive_alpha: float = 1.0 + math.log(2.0)
    positive_beta: float = 1.0 + math.log(2.0)
    ent_loss_coef: float = 0.0
    beta_ent_scale: float = 1.0
    categorical_ent_loss_config: EntropyLossConfig = field(default_factory=_default_entropy_loss_config)
    beta_ent_loss_config: EntropyLossConfig = field(default_factory=_default_entropy_loss_config)


class SignMagnitudeBetaActionDist(ActionDist):
    _N_MIXTURE_COMPONENTS = 2
    _OUTPUTS_PER_ACTION = 6

    _NEGATIVE_INDEX = 0
    _POSITIVE_INDEX = 1

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
            ent_loss_coef: float = 0.0,
            beta_ent_scale: float = 1.0,
            categorical_ent_loss_config: EntropyLossConfig | None = None,
            beta_ent_loss_config: EntropyLossConfig | None = None,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
            init_action_net=False,
        )

        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if ent_loss_coef < 0.0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {ent_loss_coef}")
        if beta_ent_scale < 0.0:
            raise ValueError(f"beta_ent_scale must be >= 0, got {beta_ent_scale}")
        if initial_positive_prob is not None and not (0.0 < initial_positive_prob < 1.0):
            raise ValueError(
                f"initial_positive_prob must be strictly between 0 and 1, got {initial_positive_prob}"
            )

        for name, value in (
                ("negative_alpha", negative_alpha),
                ("negative_beta", negative_beta),
                ("positive_alpha", positive_alpha),
                ("positive_beta", positive_beta),
        ):
            if value <= 1.0:
                raise ValueError(f"{name} must be > 1.0, got {value}")

        self.epsilon = epsilon
        self.ent_loss_coef = ent_loss_coef
        self.beta_ent_scale = beta_ent_scale
        self.categorical_ent_loss_config = (
            categorical_ent_loss_config if categorical_ent_loss_config is not None else _default_entropy_loss_config()
        )
        self.beta_ent_loss_config = (
            beta_ent_loss_config if beta_ent_loss_config is not None else _default_entropy_loss_config()
        )
        self.initial_positive_prob = initial_positive_prob
        self.log_interval_jacobian = 0.0

        self.output_net = nn.Linear(latent_dim, action_dim * self._OUTPUTS_PER_ACTION)
        if action_net_initialization is not None:
            action_net_initialization(self.output_net)

        with torch.no_grad():
            bias = self.output_net.bias.view(action_dim, self._OUTPUTS_PER_ACTION)
            if initial_positive_prob is not None:
                bias[:, self._NEGATIVE_INDEX] = math.log(1.0 - initial_positive_prob)
                bias[:, self._POSITIVE_INDEX] = math.log(initial_positive_prob)
            bias[:, 2] = _inverse_softplus(negative_alpha - 1.0)
            bias[:, 3] = _inverse_softplus(negative_beta - 1.0)
            bias[:, 4] = _inverse_softplus(positive_alpha - 1.0)
            bias[:, 5] = _inverse_softplus(positive_beta - 1.0)

        self.weight_logits: Optional[torch.Tensor] = None
        self.categorical_dist: Optional[torchdist.Categorical] = None
        self.negative_beta_dist: Optional[torchdist.Beta] = None
        self.positive_beta_dist: Optional[torchdist.Beta] = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        raw = self.output_net(latent_pi).view(*latent_pi.shape[:-1], self.action_dim, self._OUTPUTS_PER_ACTION)
        self.weight_logits = raw[..., :self._N_MIXTURE_COMPONENTS]
        self.categorical_dist = torchdist.Categorical(logits=self.weight_logits)

        negative_alpha = 1.0 + F.softplus(raw[..., 2])
        negative_beta = 1.0 + F.softplus(raw[..., 3])
        positive_alpha = 1.0 + F.softplus(raw[..., 4])
        positive_beta = 1.0 + F.softplus(raw[..., 5])
        self.negative_beta_dist = torchdist.Beta(concentration1=negative_alpha, concentration0=negative_beta)
        self.positive_beta_dist = torchdist.Beta(concentration1=positive_alpha, concentration0=positive_beta)
        return self

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:

        component_indices = self.categorical_dist.sample()
        negative_01 = self.negative_beta_dist.sample()
        positive_01 = self.positive_beta_dist.sample()

        negative_actions = -1.0 + negative_01
        positive_actions = positive_01

        sampled_actions = positive_actions
        sampled_actions = torch.where(component_indices == self._NEGATIVE_INDEX, negative_actions, sampled_actions)
        return sampled_actions

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        weights = F.softmax(self.weight_logits, dim=-1)

        negative_mean = -1.0 + self.negative_beta_dist.mean
        positive_mean = self.positive_beta_dist.mean
        return (
                weights[..., self._NEGATIVE_INDEX] * negative_mean
                + weights[..., self._POSITIVE_INDEX] * positive_mean
        )

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        log_weights = F.log_softmax(self.weight_logits, dim=-1)
        negative_mask = actions < 0.0

        negative_01 = (actions + 1.0).clamp(self.epsilon, 1.0 - self.epsilon)
        positive_01 = actions.clamp(self.epsilon, 1.0 - self.epsilon)

        negative_log_prob = (
                log_weights[..., self._NEGATIVE_INDEX]
                + self.negative_beta_dist.log_prob(negative_01)
                + self.log_interval_jacobian
        )
        positive_log_prob = (
                log_weights[..., self._POSITIVE_INDEX]
                + self.positive_beta_dist.log_prob(positive_01)
                + self.log_interval_jacobian
        )
        log_prob_per_action = torch.where(negative_mask, negative_log_prob, positive_log_prob)
        return log_prob_per_action.sum(dim=AGENT_ACTIONS_DIM)

    def compute_extra_losses(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[LossDict, LossMetrics]:
        if self.ent_loss_coef <= 0.0:
            return {}, {}

        weights = F.softmax(self.weight_logits, dim=-1)

        categorical_entropy_per_action = self.categorical_dist.entropy()
        weighted_beta_entropy_per_action = (
                weights[..., self._NEGATIVE_INDEX] * self.negative_beta_dist.entropy()
                + weights[..., self._POSITIVE_INDEX] * self.positive_beta_dist.entropy()
        )
        categorical_ent_loss = compute_ent_loss(
            config=self.categorical_ent_loss_config,
            entropy_per_action=categorical_entropy_per_action,
        )
        beta_ent_loss = compute_ent_loss(
            config=self.beta_ent_loss_config,
            entropy_per_action=weighted_beta_entropy_per_action,
        )
        entropy_loss = self.ent_loss_coef * (categorical_ent_loss + self.beta_ent_scale * beta_ent_loss)

        action_metrics_splitter = resolve_action_metrics_splitter(action_splitter)

        categorical_ent_metrics = compute_ent_metrics(
            config=self.categorical_ent_loss_config,
            entropy_per_action=categorical_entropy_per_action,
            agent_mask=agent_mask,
            metrics_action_splitter=action_metrics_splitter,
            name='ent_categorical'
        )
        beta_ent_metrics = compute_ent_metrics(
            config=self.beta_ent_loss_config,
            entropy_per_action=weighted_beta_entropy_per_action,
            agent_mask=agent_mask,
            metrics_action_splitter=action_metrics_splitter,
            name='ent_beta'
        )
        metrics = {
            **categorical_ent_metrics,
            **beta_ent_metrics,
        }
        return {"entropy": entropy_loss}, metrics

    def set_ent_loss_coef(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {value}")
        self.ent_loss_coef = value

    def set_beta_ent_scale(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"beta_ent_scale must be >= 0, got {value}")
        self.beta_ent_scale = value

    @property
    def compile_friendly(self) -> bool:
        return True

    def get_metrics(
            self,
            actions: torch.Tensor,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> dict[str, Any]:
        return compute_action_metrics(actions, action_splitter, hist_bins=21)

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "epsilon": self.epsilon,
            "ent_loss_coef": self.ent_loss_coef,
            "beta_ent_scale": self.beta_ent_scale,
            "initial_positive_prob": self.initial_positive_prob,
            "categorical_ent_loss_config": serialize_dataclass(self.categorical_ent_loss_config),
            "beta_ent_loss_config": serialize_dataclass(self.beta_ent_loss_config),
        }


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))
