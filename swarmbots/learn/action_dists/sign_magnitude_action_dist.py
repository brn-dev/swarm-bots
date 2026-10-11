import abc
import math
from typing import Any, Self

import torch
from torch import distributions as torchdist
from torch import nn
from torch.nn import functional as F

from swarmbots.learn.action_dists.action_dist import (
    AGENT_ACTIONS_DIM,
    BOUNDED_ACTION_HISTOGRAM,
    ActionDist,
    ActionMetricsSplitterInput,
    ActionNetInitialization,
    compute_action_metrics,
    resolve_action_metrics_splitter,
    validate_probability_clamp_epsilon,
)
from swarmbots.learn.action_dists.entropy_utils import compute_entropy_metrics


def shape_parameter_greater_than_one(raw_parameter: torch.Tensor) -> torch.Tensor:
    minimum_offset = torch.finfo(raw_parameter.dtype).eps
    return 1.0 + F.softplus(raw_parameter).clamp_min(minimum_offset)


class SignMagnitudeActionDist(ActionDist):
    _N_MIXTURE_COMPONENTS = 2
    _OUTPUTS_PER_ACTION = 6

    _NEGATIVE_INDEX = 0
    _POSITIVE_INDEX = 1

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization | None,
            *,
            initial_positive_prob: float | None,
            epsilon: float,
            initial_magnitude_parameters: tuple[float, float, float, float],
            magnitude_parameter_names: tuple[str, str, str, str],
            magnitude_entropy_metric_name: str,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
            init_action_net=False,
        )

        validate_probability_clamp_epsilon(epsilon)
        if initial_positive_prob is not None and not (0.0 < initial_positive_prob < 1.0):
            raise ValueError(
                f"initial_positive_prob must be strictly between 0 and 1, got {initial_positive_prob}"
            )
        for name, value in zip(magnitude_parameter_names, initial_magnitude_parameters, strict=True):
            if value <= 1.0:
                raise ValueError(f"{name} must be > 1.0, got {value}")

        self.initial_positive_prob = initial_positive_prob
        self.epsilon = epsilon
        self.magnitude_entropy_metric_name = magnitude_entropy_metric_name

        self.output_net = nn.Linear(latent_dim, action_dim * self._OUTPUTS_PER_ACTION)
        if action_net_initialization is not None:
            action_net_initialization(self.output_net)

        with torch.no_grad():
            bias = self.output_net.bias.view(action_dim, self._OUTPUTS_PER_ACTION)
            if initial_positive_prob is not None:
                bias[:, self._NEGATIVE_INDEX] = math.log(1.0 - initial_positive_prob)
                bias[:, self._POSITIVE_INDEX] = math.log(initial_positive_prob)
            for output_index, parameter in enumerate(
                    initial_magnitude_parameters,
                    start=self._N_MIXTURE_COMPONENTS,
            ):
                bias[:, output_index] = inverse_softplus(parameter - 1.0)

        self.weight_logits: torch.Tensor | None = None
        self.negative_magnitude_dist: torchdist.Distribution | None = None
        self.positive_magnitude_dist: torchdist.Distribution | None = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        raw = self.output_net(latent_pi).view(*latent_pi.shape[:-1], self.action_dim, self._OUTPUTS_PER_ACTION)
        self.weight_logits = raw[..., :self._N_MIXTURE_COMPONENTS]
        self.negative_magnitude_dist, self.positive_magnitude_dist = self._make_magnitude_distributions(
            raw[..., self._N_MIXTURE_COMPONENTS:]
        )
        return self

    @abc.abstractmethod
    def _make_magnitude_distributions(
            self,
            raw_magnitude_parameters: torch.Tensor,
    ) -> tuple[torchdist.Distribution, torchdist.Distribution]:
        raise NotImplementedError

    @abc.abstractmethod
    def _magnitude_modes(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = agent
        self._assert_ready()
        component_indices = torchdist.Categorical(
            probs=self._component_probs(previous_actions),
            validate_args=False,
        ).sample()
        negative_magnitudes, positive_magnitudes = self._sample_independent_magnitudes()
        return self._select_actions(component_indices, negative_magnitudes, positive_magnitudes)

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        self._assert_ready()
        assert self.negative_magnitude_dist is not None
        assert self.positive_magnitude_dist is not None
        negative_magnitude_mode, positive_magnitude_mode = self._magnitude_modes()
        component_log_probs = self._component_log_probs(previous_actions)
        component_mode_scores = [
            component_log_probs[..., self._NEGATIVE_INDEX]
            + self.negative_magnitude_dist.log_prob(negative_magnitude_mode),
        ]
        component_mode_scores.append(
            component_log_probs[..., self._POSITIVE_INDEX]
            + self.positive_magnitude_dist.log_prob(positive_magnitude_mode)
        )
        stacked_component_mode_scores = torch.stack(component_mode_scores, dim=-1)
        component_indices = stacked_component_mode_scores.argmax(dim=-1)
        return self._select_actions(
            component_indices,
            negative_magnitude_mode,
            positive_magnitude_mode,
        )

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._assert_ready()
        log_weights = self._component_log_probs(previous_actions)
        negative_mask = actions < 0.0
        negative_log_probs, positive_log_probs = self._magnitude_log_probs(actions + 1.0, actions)
        log_prob_per_action = torch.where(
            negative_mask,
            log_weights[..., self._NEGATIVE_INDEX] + negative_log_probs,
            log_weights[..., self._POSITIVE_INDEX] + positive_log_probs,
        )
        return log_prob_per_action.sum(dim=AGENT_ACTIONS_DIM)

    def _component_probs(self, previous_actions: torch.Tensor | None) -> torch.Tensor:
        _ = previous_actions
        assert self.weight_logits is not None
        return F.softmax(self.weight_logits, dim=-1)

    def _component_log_probs(self, previous_actions: torch.Tensor | None) -> torch.Tensor:
        _ = previous_actions
        assert self.weight_logits is not None
        return F.log_softmax(self.weight_logits, dim=-1)

    def _sample_independent_magnitudes(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._assert_ready()
        assert self.negative_magnitude_dist is not None
        assert self.positive_magnitude_dist is not None
        return self.negative_magnitude_dist.sample(), self.positive_magnitude_dist.sample()

    def _rsample_independent_magnitudes(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._assert_ready()
        assert self.negative_magnitude_dist is not None
        assert self.positive_magnitude_dist is not None
        return (
            self.negative_magnitude_dist.rsample().clamp(self.epsilon, 1.0 - self.epsilon),
            self.positive_magnitude_dist.rsample().clamp(self.epsilon, 1.0 - self.epsilon),
        )

    def _magnitude_log_probs(
            self,
            negative_magnitudes: torch.Tensor,
            positive_magnitudes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._assert_ready()
        assert self.negative_magnitude_dist is not None
        assert self.positive_magnitude_dist is not None
        return (
            self.negative_magnitude_dist.log_prob(
                negative_magnitudes.clamp(self.epsilon, 1.0 - self.epsilon)
            ),
            self.positive_magnitude_dist.log_prob(
                positive_magnitudes.clamp(self.epsilon, 1.0 - self.epsilon)
            ),
        )

    def _select_actions(
            self,
            component_indices: torch.Tensor,
            negative_magnitudes: torch.Tensor,
            positive_magnitudes: torch.Tensor,
    ) -> torch.Tensor:
        negative_actions = -1.0 + negative_magnitudes
        actions = torch.where(
            component_indices == self._NEGATIVE_INDEX,
            negative_actions,
            positive_magnitudes,
        )
        return actions

    def _sampled_component_log_probs(
            self,
            negative_magnitudes: torch.Tensor,
            positive_magnitudes: torch.Tensor,
    ) -> torch.Tensor:
        negative_magnitude_log_probs, positive_magnitude_log_probs = self._magnitude_log_probs(
            negative_magnitudes,
            positive_magnitudes,
        )
        log_weights = self._component_log_probs(previous_actions=None)
        component_log_probs = [
            log_weights[..., self._NEGATIVE_INDEX] + negative_magnitude_log_probs,
        ]
        component_log_probs.append(
            log_weights[..., self._POSITIVE_INDEX] + positive_magnitude_log_probs
        )
        return torch.stack(component_log_probs, dim=-1)

    @property
    def compile_friendly(self) -> bool:
        return True

    @torch.no_grad()
    def get_metrics(
            self,
            actions: torch.Tensor,
            action_splitter: ActionMetricsSplitterInput = None,
            *,
            agent_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        metrics = compute_action_metrics(actions, action_splitter, histogram=BOUNDED_ACTION_HISTOGRAM)
        if self.weight_logits is None:
            return metrics
        assert self.negative_magnitude_dist is not None
        assert self.positive_magnitude_dist is not None
        with torch.no_grad():
            weights = F.softmax(self.weight_logits, dim=-1)
            categorical_entropy = -(weights * F.log_softmax(self.weight_logits, dim=-1)).sum(dim=-1)
            magnitude_entropy = (
                weights[..., self._NEGATIVE_INDEX] * self.negative_magnitude_dist.entropy()
                + weights[..., self._POSITIVE_INDEX] * self.positive_magnitude_dist.entropy()
            )
        splitter = resolve_action_metrics_splitter(action_splitter)
        for name, entropy in (
            ("ent", categorical_entropy + magnitude_entropy),
            ("ent_categorical", categorical_entropy),
            (self.magnitude_entropy_metric_name, magnitude_entropy),
        ):
            metrics.update(compute_entropy_metrics(
                entropy,
                agent_mask=agent_mask,
                action_splitter=splitter,
                name=name,
            ))
        return metrics

    def get_hyper_parameters(self) -> dict[str, Any]:
        hyper_parameters = {
            **super().get_hyper_parameters(),
            "epsilon": self.epsilon,
            "initial_positive_prob": self.initial_positive_prob,
        }
        return hyper_parameters

    def _assert_ready(self) -> None:
        if (
                self.weight_logits is None
                or self.negative_magnitude_dist is None
                or self.positive_magnitude_dist is None
        ):
            raise RuntimeError("Distribution parameters are not initialized. Call update_latent_features first.")


def inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))
