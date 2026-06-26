import math
from dataclasses import dataclass, field
from typing import Any, Optional, Self

import torch
from torch import nn
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
class ReparameterizedSignMagnitudeKumaraswamyConfig:
    initial_positive_prob: float | None = None
    epsilon: float = 1e-6
    negative_a: float = 2.0
    negative_b: float = 2.5
    positive_a: float = 2.0
    positive_b: float = 2.5
    ent_loss_coef: float = 0.0
    kumaraswamy_ent_scale: float = 1.0
    categorical_ent_loss_config: EntropyLossConfig = field(default_factory=_default_entropy_loss_config)
    kumaraswamy_ent_loss_config: EntropyLossConfig = field(default_factory=_default_entropy_loss_config)


class ReparameterizedSignMagnitudeKumaraswamyActionDist(ActionDist):
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
            negative_a: float = 2.0,
            negative_b: float = 2.5,
            positive_a: float = 2.0,
            positive_b: float = 2.5,
            ent_loss_coef: float = 0.0,
            kumaraswamy_ent_scale: float = 1.0,
            categorical_ent_loss_config: EntropyLossConfig | None = None,
            kumaraswamy_ent_loss_config: EntropyLossConfig | None = None,
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
        if kumaraswamy_ent_scale < 0.0:
            raise ValueError(f"kumaraswamy_ent_scale must be >= 0, got {kumaraswamy_ent_scale}")
        if initial_positive_prob is not None and not (0.0 < initial_positive_prob < 1.0):
            raise ValueError(
                f"initial_positive_prob must be strictly between 0 and 1, got {initial_positive_prob}"
            )

        for name, value in (
                ("negative_a", negative_a),
                ("negative_b", negative_b),
                ("positive_a", positive_a),
                ("positive_b", positive_b),
        ):
            if value <= 1.0:
                raise ValueError(f"{name} must be > 1.0, got {value}")

        self.initial_positive_prob = initial_positive_prob
        self.epsilon = epsilon
        self.initial_negative_a = negative_a
        self.initial_negative_b = negative_b
        self.initial_positive_a = positive_a
        self.initial_positive_b = positive_b
        self.ent_loss_coef = ent_loss_coef
        self.kumaraswamy_ent_scale = kumaraswamy_ent_scale
        self.categorical_ent_loss_config = (
            categorical_ent_loss_config if categorical_ent_loss_config is not None else _default_entropy_loss_config()
        )
        self.kumaraswamy_ent_loss_config = (
            kumaraswamy_ent_loss_config
            if kumaraswamy_ent_loss_config is not None
            else _default_entropy_loss_config()
        )

        self.output_net = nn.Linear(latent_dim, action_dim * self._OUTPUTS_PER_ACTION)
        if action_net_initialization is not None:
            action_net_initialization(self.output_net)

        with torch.no_grad():
            bias = self.output_net.bias.view(action_dim, self._OUTPUTS_PER_ACTION)
            if initial_positive_prob is not None:
                bias[:, self._NEGATIVE_INDEX] = math.log(1.0 - initial_positive_prob)
                bias[:, self._POSITIVE_INDEX] = math.log(initial_positive_prob)
            bias[:, 2] = _inverse_softplus(negative_a - 1.0)
            bias[:, 3] = _inverse_softplus(negative_b - 1.0)
            bias[:, 4] = _inverse_softplus(positive_a - 1.0)
            bias[:, 5] = _inverse_softplus(positive_b - 1.0)

        self.weight_logits: Optional[torch.Tensor] = None
        self.negative_a: Optional[torch.Tensor] = None
        self.negative_b: Optional[torch.Tensor] = None
        self.positive_a: Optional[torch.Tensor] = None
        self.positive_b: Optional[torch.Tensor] = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        raw = self.output_net(latent_pi).view(*latent_pi.shape[:-1], self.action_dim, self._OUTPUTS_PER_ACTION)
        self.weight_logits = raw[..., :self._N_MIXTURE_COMPONENTS]
        self.negative_a = 1.0 + F.softplus(raw[..., 2])
        self.negative_b = 1.0 + F.softplus(raw[..., 3])
        self.positive_a = 1.0 + F.softplus(raw[..., 4])
        self.positive_b = 1.0 + F.softplus(raw[..., 5])
        return self

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = (agent, previous_actions)
        self._assert_ready()

        weights = F.softmax(self.weight_logits, dim=-1)
        negative_prob = weights[..., self._NEGATIVE_INDEX]
        positive_prob = weights[..., self._POSITIVE_INDEX]
        u = torch.rand_like(negative_prob).clamp(self.epsilon, 1.0 - self.epsilon)

        negative_mask = u < negative_prob
        negative_u = (u / negative_prob.clamp_min(self.epsilon)).clamp(self.epsilon, 1.0 - self.epsilon)
        positive_u = ((u - negative_prob) / positive_prob.clamp_min(self.epsilon)).clamp(
            self.epsilon,
            1.0 - self.epsilon,
        )

        negative_actions = -1.0 + _kumaraswamy_icdf(
            negative_u,
            self.negative_a,
            self.negative_b,
            epsilon=self.epsilon,
        )
        positive_actions = _kumaraswamy_icdf(
            positive_u,
            self.positive_a,
            self.positive_b,
            epsilon=self.epsilon,
        )
        return torch.where(negative_mask, negative_actions, positive_actions)

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        _ = previous_actions
        self._assert_ready()
        weights = F.softmax(self.weight_logits, dim=-1)

        negative_mean = -1.0 + _kumaraswamy_mean(self.negative_a, self.negative_b)
        positive_mean = _kumaraswamy_mean(self.positive_a, self.positive_b)
        return (
                weights[..., self._NEGATIVE_INDEX] * negative_mean
                + weights[..., self._POSITIVE_INDEX] * positive_mean
        )

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = previous_actions
        self._assert_ready()

        log_weights = F.log_softmax(self.weight_logits, dim=-1)
        negative_mask = actions < 0.0

        negative_01 = (actions + 1.0).clamp(self.epsilon, 1.0 - self.epsilon)
        positive_01 = actions.clamp(self.epsilon, 1.0 - self.epsilon)

        negative_log_prob = (
                log_weights[..., self._NEGATIVE_INDEX]
                + _kumaraswamy_log_prob(negative_01, self.negative_a, self.negative_b, epsilon=self.epsilon)
        )
        positive_log_prob = (
                log_weights[..., self._POSITIVE_INDEX]
                + _kumaraswamy_log_prob(positive_01, self.positive_a, self.positive_b, epsilon=self.epsilon)
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

        self._assert_ready()
        weights = F.softmax(self.weight_logits, dim=-1)

        categorical_entropy_per_action = -(weights * F.log_softmax(self.weight_logits, dim=-1)).sum(dim=-1)
        weighted_kumaraswamy_entropy_per_action = (
                weights[..., self._NEGATIVE_INDEX] * _kumaraswamy_entropy(self.negative_a, self.negative_b)
                + weights[..., self._POSITIVE_INDEX] * _kumaraswamy_entropy(self.positive_a, self.positive_b)
        )
        categorical_ent_loss = compute_ent_loss(
            config=self.categorical_ent_loss_config,
            entropy_per_action=categorical_entropy_per_action,
        )
        kumaraswamy_ent_loss = compute_ent_loss(
            config=self.kumaraswamy_ent_loss_config,
            entropy_per_action=weighted_kumaraswamy_entropy_per_action,
        )
        entropy_loss = self.ent_loss_coef * (
                categorical_ent_loss + self.kumaraswamy_ent_scale * kumaraswamy_ent_loss
        )

        action_metrics_splitter = resolve_action_metrics_splitter(action_splitter)
        categorical_ent_metrics = compute_ent_metrics(
            config=self.categorical_ent_loss_config,
            entropy_per_action=categorical_entropy_per_action,
            agent_mask=agent_mask,
            metrics_action_splitter=action_metrics_splitter,
            name="ent_categorical",
        )
        kumaraswamy_ent_metrics = compute_ent_metrics(
            config=self.kumaraswamy_ent_loss_config,
            entropy_per_action=weighted_kumaraswamy_entropy_per_action,
            agent_mask=agent_mask,
            metrics_action_splitter=action_metrics_splitter,
            name="ent_kumaraswamy",
        )
        return {"entropy": entropy_loss}, {**categorical_ent_metrics, **kumaraswamy_ent_metrics}

    def set_ent_loss_coef(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {value}")
        self.ent_loss_coef = value

    def set_kumaraswamy_ent_scale(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"kumaraswamy_ent_scale must be >= 0, got {value}")
        self.kumaraswamy_ent_scale = value

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
            "kumaraswamy_ent_scale": self.kumaraswamy_ent_scale,
            "initial_positive_prob": self.initial_positive_prob,
            "negative_a": self.initial_negative_a,
            "negative_b": self.initial_negative_b,
            "positive_a": self.initial_positive_a,
            "positive_b": self.initial_positive_b,
            "categorical_ent_loss_config": serialize_dataclass(self.categorical_ent_loss_config),
            "kumaraswamy_ent_loss_config": serialize_dataclass(self.kumaraswamy_ent_loss_config),
        }

    def _assert_ready(self) -> None:
        if (
                self.weight_logits is None
                or self.negative_a is None
                or self.negative_b is None
                or self.positive_a is None
                or self.positive_b is None
        ):
            raise RuntimeError("Distribution parameters are not initialized. Call update_latent_features first.")


def _kumaraswamy_icdf(
        u: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        epsilon: float,
) -> torch.Tensor:
    u = u.clamp(epsilon, 1.0 - epsilon)
    inner = -torch.expm1(torch.log1p(-u) / b)
    return torch.exp(torch.log(inner.clamp_min(epsilon)) / a).clamp(epsilon, 1.0 - epsilon)


def _kumaraswamy_log_prob(
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        epsilon: float,
) -> torch.Tensor:
    value = value.clamp(epsilon, 1.0 - epsilon)
    value_power = value.pow(a).clamp(max=1.0 - epsilon)
    return torch.log(a) + torch.log(b) + (a - 1.0) * torch.log(value) + (b - 1.0) * torch.log1p(-value_power)


def _kumaraswamy_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return b * torch.exp(torch.lgamma(1.0 + 1.0 / a) + torch.lgamma(b) - torch.lgamma(1.0 + 1.0 / a + b))


def _kumaraswamy_entropy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    euler_gamma = 0.5772156649015329
    return (
            -torch.log(a)
            - torch.log(b)
            + (1.0 - 1.0 / a) * (euler_gamma + torch.digamma(b + 1.0))
            + (1.0 - 1.0 / b)
    )


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))
