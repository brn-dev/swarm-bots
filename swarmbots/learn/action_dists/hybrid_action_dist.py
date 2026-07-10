from dataclasses import replace
from typing import Any, Mapping, Self, TypeAlias

import torch
from torch import nn
import numpy as np
from gymnasium import spaces

import swarmbots
from swarmbots.learn.action_dists.action_dist import (
    ActionDist,
    AGENT_ACTIONS_DIM,
    ActionNetInitialization,
    ActionMetricsSplitter,
    ActionMetricsSplitterInput,
)
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliActionDist
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.beta_action_dist import BetaActionDist, BetaConfig
from swarmbots.learn.action_dists.bang_zero_bang_action_dist import BangZeroBangActionDist, BangZeroBangConfig
from swarmbots.learn.action_dists.continuous_action_dist import ContinuousActionDist
from swarmbots.learn.action_dists.diag_gaussian_action_dist import DiagGaussianActionDist
from swarmbots.learn.action_dists.gsde_action_dist import GSDEActionDist, GSDEConfig
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import (
    SignMagnitudeBetaActionDist,
    SignMagnitudeBetaConfig,
)
from swarmbots.learn.action_dists.left_middle_right_beta_action_dist import (
    LeftMiddleRightBetaActionDist,
    LeftMiddleRightBetaConfig,
)
from swarmbots.learn.action_dists.sticky_sign_magnitude_beta_action_dist import (
    StickySignMagnitudeBetaActionDist,
    StickySignMagnitudeBetaConfig,
)
from swarmbots.learn.action_dists.sticky_left_middle_right_beta_action_dist import (
    StickyLeftMiddleRightBetaActionDist,
    StickyLeftMiddleRightBetaConfig,
)
from swarmbots.learn.action_dists.beta_mixture_action_dist import BetaMixtureActionDist, BetaMixtureConfig
from swarmbots.learn.action_dists.predicted_std_action_dist import PredictedStdActionDist, PredictedStdConfig
from swarmbots.learn.action_dists.reparameterized_sign_magnitude_kumaraswamy_action_dist import (
    ReparameterizedSignMagnitudeKumaraswamyActionDist,
    ReparameterizedSignMagnitudeKumaraswamyConfig,
)
from swarmbots.learn.action_dists.reparameterized_squashed_gaussian_mixture_action_dist import (
    ReparameterizedSquashedGaussianMixtureActionDist,
    ReparameterizedSquashedGaussianMixtureConfig,
)
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import (
    SquashedDiagGaussianActionDist,
    SquashedDiagGaussianConfig,
)
from swarmbots.learn.action_dists.sticky_bang_zero_bang_action_dist import (
    StickyBangZeroBangActionDist,
    StickyBangZeroBangConfig,
)
from swarmbots.learn.action_dists.sticky_action_dist import StickyActionDist
from swarmbots.learn.action_dists.temporally_correlated_action_dist import TemporallyCorrelatedActionDist
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal
from swarmbots.learn.serialization_utils import serialize_dataclass


ContinuousActionDistConfig: TypeAlias = (
    SquashedDiagGaussianConfig
    | PredictedStdConfig
    | GSDEConfig
    | BetaConfig
    | BetaMixtureConfig
    | ReparameterizedSignMagnitudeKumaraswamyConfig
    | ReparameterizedSquashedGaussianMixtureConfig
    | StickySignMagnitudeBetaConfig
    | StickyLeftMiddleRightBetaConfig
    | SignMagnitudeBetaConfig
    | LeftMiddleRightBetaConfig
    | BangZeroBangConfig
    | StickyBangZeroBangConfig
)

ContinuousActionDistConfigInput: TypeAlias = (
    ContinuousActionDistConfig
    | list[ContinuousActionDistConfig | None]
    | None
)


def continuous_config_to_dicts(
        continuous_config: ContinuousActionDistConfigInput,
) -> dict[str, Any] | list[dict[str, Any] | None] | None:
    if continuous_config is None:
        return None
    if isinstance(continuous_config, list):
        return [
            serialize_dataclass(cc) if cc is not None else None
            for cc in continuous_config
        ]
    return serialize_dataclass(continuous_config)


def bernoulli_config_to_dict(
        bernoulli_config: BernoulliConfig | None,
) -> dict[str, Any] | None:
    if bernoulli_config is None:
        return None
    return serialize_dataclass(bernoulli_config)


class HybridActionDistribution(ActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_space: HybridActionSpace,
            continuous_config: ContinuousActionDistConfigInput,
            bernoulli_config: BernoulliConfig | None = None,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
    ):
        self.action_space = action_space
        self.action_dims = action_space.agent_action_dims
        if isinstance(continuous_config, list):
            if len(continuous_config) != action_space.n_spaces:
                raise ValueError(
                    f"Expected {action_space.n_spaces} continuous configs (one per action sub-space), "
                    f"got {len(continuous_config)}."
                )
            self.continuous_configs = continuous_config
        else:
            self.continuous_configs = [continuous_config] * action_space.n_spaces
        self.bernoulli_config = bernoulli_config

        gsde_indices = [
            i for i, (sub_space, config) in enumerate(
                zip(action_space.sub_spaces, self.continuous_configs, strict=True)
            )
            if isinstance(sub_space, spaces.Box) and isinstance(config, GSDEConfig)
        ]
        self.has_gsde = len(gsde_indices) > 0

        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_space.total_agent_action_dim,
            action_net_initialization=None,
            init_action_net=False,
        )

        # noinspection PyTypeChecker
        self.distributions: list[ActionDist] = nn.ModuleList([
            make_proba_distribution(
                latent_dim,
                sub_space,
                sub_space_dim,
                action_net_initialization,
                cont_conf,
                bernoulli_config=bernoulli_config,
            )
            for sub_space, sub_space_dim, cont_conf
            in zip(action_space.sub_spaces, action_space.agent_action_dims, self.continuous_configs, strict=True)
        ])

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        for dist in self.distributions:
            dist.update_latent_features(latent_pi)
        return self

    def requires_previous_actions(self) -> bool:
        return any(dist.requires_previous_actions() for dist in self.distributions)

    @property
    def sampling_depends_on_agent(self) -> bool:
        return any(dist.sampling_depends_on_agent for dist in self.distributions)

    @property
    def compile_friendly(self) -> bool:
        return all(dist.compile_friendly for dist in self.distributions)

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "action_dims": list(self.action_dims),
            "has_gsde": self.has_gsde,
            "continuous_config": continuous_config_to_dicts(self.continuous_configs),
            "bernoulli_config": bernoulli_config_to_dict(self.bernoulli_config),
            "sub_distributions": [dist.get_hyper_parameters() for dist in self.distributions],
        }

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._sample_sub_distributions(
            sampler_name="sample",
            agent=agent,
            previous_actions=previous_actions,
        )

    def rsample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._sample_sub_distributions(
            sampler_name="rsample",
            agent=agent,
            previous_actions=previous_actions,
        )

    def _sample_sub_distributions(
            self,
            *,
            sampler_name: str,
            agent: int | None,
            previous_actions: torch.Tensor | None,
    ) -> torch.Tensor:
        split_previous_actions: tuple[torch.Tensor | None, ...]
        if previous_actions is None:
            split_previous_actions = (None,) * len(self.distributions)
        else:
            split_previous_actions = torch.split(previous_actions, self.action_dims, dim=AGENT_ACTIONS_DIM)
        actions: list[torch.Tensor] = []
        for idx, dist in enumerate(self.distributions):
            previous_action = split_previous_actions[idx]
            sampler = getattr(dist, sampler_name)
            if dist.sampling_depends_on_agent:
                actions.append(sampler(agent=agent, previous_actions=previous_action))
            else:
                actions.append(sampler(previous_actions=previous_action))
        return torch.cat(actions, dim=AGENT_ACTIONS_DIM)

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        split_previous_actions: tuple[torch.Tensor | None, ...]
        if previous_actions is None:
            split_previous_actions = (None,) * len(self.distributions)
        else:
            split_previous_actions = torch.split(previous_actions, self.action_dims, dim=AGENT_ACTIONS_DIM)
        actions: list[torch.Tensor] = []
        for idx, dist in enumerate(self.distributions):
            actions.append(dist.mode(previous_actions=split_previous_actions[idx]))
        return torch.cat(actions, dim=AGENT_ACTIONS_DIM)

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        split_actions = torch.split(actions, self.action_dims, dim=AGENT_ACTIONS_DIM)
        split_previous_actions: tuple[torch.Tensor | None, ...]
        if previous_actions is None:
            split_previous_actions = (None,) * len(self.distributions)
        else:
            split_previous_actions = torch.split(previous_actions, self.action_dims, dim=AGENT_ACTIONS_DIM)
        log_prob_parts: list[torch.Tensor] = []
        for idx, dist in enumerate(self.distributions):
            log_prob_parts.append(dist.log_prob(split_actions[idx], split_previous_actions[idx]))
        return torch.stack(log_prob_parts, dim=-1).sum(dim=-1)

    def get_actions_with_log_probs(
            self,
            latent_pi: torch.Tensor,
            deterministic: bool = False,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
            use_rsample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions_parts: list[torch.Tensor] = []
        log_prob_parts: list[torch.Tensor] = []
        split_previous_actions: tuple[torch.Tensor | None, ...]
        if previous_actions is None:
            split_previous_actions = (None,) * len(self.distributions)
        else:
            split_previous_actions = torch.split(previous_actions, self.action_dims, dim=AGENT_ACTIONS_DIM)

        for idx, dist in enumerate(self.distributions):
            previous_action = split_previous_actions[idx]
            if dist.sampling_depends_on_agent:
                action_part, log_prob_part = dist.get_actions_with_log_probs(
                    latent_pi=latent_pi,
                    deterministic=deterministic,
                    agent=agent,
                    previous_actions=previous_action,
                    use_rsample=use_rsample,
                )
            else:
                action_part, log_prob_part = dist.get_actions_with_log_probs(
                    latent_pi=latent_pi,
                    deterministic=deterministic,
                    previous_actions=previous_action,
                    use_rsample=use_rsample,
                )
            actions_parts.append(action_part)
            log_prob_parts.append(log_prob_part)

        actions = torch.cat(actions_parts, dim=AGENT_ACTIONS_DIM)
        log_probs = torch.stack(log_prob_parts, dim=-1).sum(dim=-1)
        return actions, log_probs

    def compute_extra_losses(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[LossDict, LossMetrics]:
        losses: LossDict = {}
        metrics: LossMetrics = {}
        splitters = self._resolve_action_splitters(action_splitter)
        for i, dist in enumerate(self.distributions):
            dist_losses, dist_metrics = dist.compute_extra_losses(
                agent_mask=agent_mask,
                action_splitter=splitters[i],
            )
            prefix = f"act{i}_"
            losses.update(self._prefix_named_values(dist_losses, prefix=prefix))
            metrics.update(self._prefix_named_values(dist_metrics, prefix=prefix))
        return losses, metrics

    def compute_extra_losses_without_metrics(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> LossDict:
        losses: LossDict = {}
        splitters = self._resolve_action_splitters(action_splitter)
        for i, dist in enumerate(self.distributions):
            compute_losses_without_metrics = getattr(dist, "compute_extra_losses_without_metrics", None)
            if callable(compute_losses_without_metrics):
                dist_losses = compute_losses_without_metrics(
                    agent_mask=agent_mask,
                    action_splitter=splitters[i],
                )
            else:
                dist_losses, _dist_metrics = dist.compute_extra_losses(
                    agent_mask=agent_mask,
                    action_splitter=splitters[i],
                )
            losses.update(self._prefix_named_values(dist_losses, prefix=f"act{i}_"))
        return losses

    def get_metrics(
            self,
            actions: torch.Tensor,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> dict[str, Any]:
        splitters = self._resolve_action_splitters(action_splitter)
        split_actions = torch.split(actions, self.action_dims, dim=AGENT_ACTIONS_DIM)

        metrics: dict[str, Any] = {}
        for i, (dist, dist_actions, dist_splitter) in enumerate(
                zip(self.distributions, split_actions, splitters, strict=True)
        ):
            for metric_name, metric_value in dist.get_metrics(dist_actions, dist_splitter).items():
                prefixed_name = self._prefix_distribution_metric_name(metric_name=metric_name, index=i)
                metrics[prefixed_name] = metric_value
        return metrics

    def reset_temporal_correlations_on_ep_start(self, mask: torch.Tensor) -> None:
        for dist in self.distributions:
            if isinstance(dist, TemporallyCorrelatedActionDist):
                dist.reset_on_ep_start(mask)

    def reset_temporal_correlations_on_step(
            self,
            mask: torch.Tensor | None = None,
            batch_shape: tuple[int, ...] | None = None,
    ) -> None:
        for dist in self.distributions:
            if isinstance(dist, TemporallyCorrelatedActionDist):
                dist.reset_on_step(mask=mask, batch_shape=batch_shape)

    def set_std(self, std: float) -> None:
        for dist in self.distributions:
            set_std = getattr(dist, "set_std", None)
            if callable(set_std):
                set_std(std)

    def scale_std(self, multiplier: float) -> None:
        for dist in self.distributions:
            scale_std = getattr(dist, "scale_std", None)
            if callable(scale_std):
                scale_std(multiplier)

    def set_action_magnitude_loss_coef(self, value: float) -> None:
        for dist in self.distributions:
            if isinstance(dist, ContinuousActionDist):
                dist.set_action_magnitude_loss_coef(value)
        for idx, config in enumerate(self.continuous_configs):
            if isinstance(config, (SquashedDiagGaussianConfig, PredictedStdConfig, GSDEConfig)):
                self.continuous_configs[idx] = replace(config, action_magnitude_loss_coef=value)

    def set_all_ent_loss_coefs(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {value}")
        for dist in self.distributions:
            set_ent_loss_coef = getattr(dist, "set_ent_loss_coef", None)
            if callable(set_ent_loss_coef):
                set_ent_loss_coef(value)
        for idx, config in enumerate(self.continuous_configs):
            if isinstance(config,
                          (SquashedDiagGaussianConfig, PredictedStdConfig, GSDEConfig,
                           BetaConfig, BangZeroBangConfig, StickyBangZeroBangConfig, SignMagnitudeBetaConfig,
                           StickySignMagnitudeBetaConfig, LeftMiddleRightBetaConfig,
                           ReparameterizedSignMagnitudeKumaraswamyConfig,
                           ReparameterizedSquashedGaussianMixtureConfig,
                           StickyLeftMiddleRightBetaConfig)
            ):
                self.continuous_configs[idx] = replace(config, ent_loss_coef=value)
        if self.bernoulli_config is not None:
            self.bernoulli_config = replace(self.bernoulli_config, ent_loss_coef=value)

    def set_sub_ent_loss_coef(self, sub_dist_idx: int, value: float) -> None:
        if value < 0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {value}")
        if not (0 <= sub_dist_idx < len(self.distributions)):
            raise IndexError(
                f"sub_dist_idx out of range [0, {len(self.distributions) - 1}], got {sub_dist_idx}"
            )

        dist = self.distributions[sub_dist_idx]
        set_ent_loss_coef = getattr(dist, "set_ent_loss_coef", None)
        if not callable(set_ent_loss_coef):
            raise ValueError(f"Action sub-dist {sub_dist_idx} does not support entropy coefficient updates")
        set_ent_loss_coef(value)

        config = self.continuous_configs[sub_dist_idx]
        if isinstance(config,
                      (SquashedDiagGaussianConfig, PredictedStdConfig, GSDEConfig,
                       BetaConfig, BangZeroBangConfig, StickyBangZeroBangConfig, SignMagnitudeBetaConfig,
                       StickySignMagnitudeBetaConfig, LeftMiddleRightBetaConfig,
                       ReparameterizedSignMagnitudeKumaraswamyConfig,
                       ReparameterizedSquashedGaussianMixtureConfig,
                       StickyLeftMiddleRightBetaConfig)
        ):
            self.continuous_configs[sub_dist_idx] = replace(config, ent_loss_coef=value)

    def set_all_stickiness(self, value: float) -> None:
        for dist in self.distributions:
            if not isinstance(dist, StickyActionDist):
                continue
            dist.set_stickiness(value)

    def set_sub_stickiness(self, sub_dist_idx: int, value: float) -> None:
        if not (0 <= sub_dist_idx < len(self.distributions)):
            raise IndexError(
                f"sub_dist_idx out of range [0, {len(self.distributions) - 1}], got {sub_dist_idx}"
            )

        dist = self.distributions[sub_dist_idx]
        if not isinstance(dist, StickyActionDist):
            raise ValueError(f"Action sub-dist {sub_dist_idx} does not support stickiness updates")

        dist.set_stickiness(value)

    @staticmethod
    def _prefix_named_values(
            values: Mapping[str, Any],
            *,
            prefix: str,
    ) -> dict[str, Any]:
        return {f"{prefix}{name}": value for name, value in values.items()}

    def _resolve_action_splitters(
            self,
            action_splitter: ActionMetricsSplitterInput,
    ) -> list[ActionMetricsSplitter | None]:
        if action_splitter is None:
            return [None] * len(self.distributions)
        if callable(action_splitter):
            return [action_splitter] * len(self.distributions)
        if not isinstance(action_splitter, list):
            raise TypeError(
                f"action_splitter must be callable, list of callables, or None, got {type(action_splitter)}"
            )
        if len(action_splitter) != len(self.distributions):
            raise ValueError(
                f"Expected {len(self.distributions)} action splitters, got {len(action_splitter)}"
            )
        for splitter in action_splitter:
            if splitter is not None and not callable(splitter):
                raise TypeError(f"action_splitter list entries must be callable or None, got {type(splitter)}")
        return action_splitter

    @staticmethod
    def _prefix_distribution_metric_name(metric_name: str, index: int) -> str:
        if metric_name == "act":
            return f"act{index}"
        if metric_name.startswith("act_"):
            return f"act{index}_{metric_name.removeprefix('act_')}"
        if metric_name == "std":
            return f"std{index}"
        if metric_name.startswith("std_"):
            return f"std{index}_{metric_name.removeprefix('std_')}"
        return f"act{index}_{metric_name}"


def make_proba_distribution(
        latent_dim: int,
        action_space: spaces.Space,
        action_space_dim: int,
        action_net_initialization: ActionNetInitialization,
        continuous_config: ContinuousActionDistConfig | None,
        bernoulli_config: BernoulliConfig | None = None,
) -> ActionDist:
    if isinstance(action_space, spaces.Box):
        _assert_unit_box_range(action_space)
        if continuous_config is None:
            raise ValueError(
                "Supply a ContinuousActionDistConfig "
                "(SquashedDiagGaussianConfig | PredictedStdConfig | GSDEConfig | "
                "BetaConfig | BetaMixtureConfig | ReparameterizedSignMagnitudeKumaraswamyConfig | "
                "ReparameterizedSquashedGaussianMixtureConfig | "
                "StickySignMagnitudeBetaConfig | StickyLeftMiddleRightBetaConfig | "
                "SignMagnitudeBetaConfig | LeftMiddleRightBetaConfig | BangZeroBangConfig | StickyBangZeroBangConfig) "
                "for continuous actions."
            )

        if isinstance(continuous_config, SquashedDiagGaussianConfig):
            return SquashedDiagGaussianActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                std=continuous_config.std,
                std_learnable=continuous_config.std_learnable,
                epsilon=continuous_config.epsilon,
                action_net_initialization=action_net_initialization,
                ent_loss_coef=continuous_config.ent_loss_coef,
                ent_loss_config=continuous_config.ent_loss_config,
                action_magnitude_loss_coef=continuous_config.action_magnitude_loss_coef,
                action_magnitude_loss_threshold=continuous_config.action_magnitude_loss_threshold,
                action_magnitude_loss_power=continuous_config.action_magnitude_loss_power,
            )
        elif isinstance(continuous_config, PredictedStdConfig):
            return PredictedStdActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                base_std=continuous_config.base_std,
                epsilon=continuous_config.epsilon,
                action_net_initialization=action_net_initialization,
                log_std_net_initialization=continuous_config.log_std_net_initialization,
                log_std_clamp_range=continuous_config.log_std_clamp_range,
                squash_output=True,
                ent_loss_coef=continuous_config.ent_loss_coef,
                ent_loss_config=continuous_config.ent_loss_config,
                action_magnitude_loss_coef=continuous_config.action_magnitude_loss_coef,
                action_magnitude_loss_threshold=continuous_config.action_magnitude_loss_threshold,
                action_magnitude_loss_power=continuous_config.action_magnitude_loss_power,
            )
        elif isinstance(continuous_config, GSDEConfig):
            return GSDEActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                base_std=continuous_config.base_std,
                latent_sde_dim=continuous_config.latent_sde_dim,
                std_learnable=continuous_config.std_learnable,
                normalize_latent_sde_by_dim=continuous_config.normalize_latent_sde_by_dim,
                squash_output=True,
                epsilon=continuous_config.epsilon,
                full_std=continuous_config.full_std,
                sde_learn_features=continuous_config.sde_learn_features,
                latent_sde_net_initialization=continuous_config.latent_sde_net_initialization,
                log_std_clamp_range=continuous_config.log_std_clamp_range,
                action_net_initialization=action_net_initialization,
                ent_loss_coef=continuous_config.ent_loss_coef,
                ent_loss_config=continuous_config.ent_loss_config,
                action_magnitude_loss_coef=continuous_config.action_magnitude_loss_coef,
                action_magnitude_loss_threshold=continuous_config.action_magnitude_loss_threshold,
                action_magnitude_loss_power=continuous_config.action_magnitude_loss_power,
            )
        elif isinstance(continuous_config, BetaConfig):
            return BetaActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                action_net_initialization=action_net_initialization,
                alpha=continuous_config.alpha,
                beta=continuous_config.beta,
                epsilon=continuous_config.epsilon,
                ent_loss_coef=continuous_config.ent_loss_coef,
                ent_loss_config=continuous_config.ent_loss_config,
            )
        elif isinstance(continuous_config, BetaMixtureConfig):
            return BetaMixtureActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                num_components=continuous_config.num_components,
                action_net_initialization=action_net_initialization,
                epsilon=continuous_config.epsilon,
                alphas=continuous_config.alphas,
                betas=continuous_config.betas,
            )
        elif isinstance(continuous_config, ReparameterizedSignMagnitudeKumaraswamyConfig):
            return ReparameterizedSignMagnitudeKumaraswamyActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                action_net_initialization=action_net_initialization,
                initial_positive_prob=continuous_config.initial_positive_prob,
                epsilon=continuous_config.epsilon,
                negative_a=continuous_config.negative_a,
                negative_b=continuous_config.negative_b,
                positive_a=continuous_config.positive_a,
                positive_b=continuous_config.positive_b,
                ent_loss_coef=continuous_config.ent_loss_coef,
                kumaraswamy_ent_scale=continuous_config.kumaraswamy_ent_scale,
                categorical_ent_loss_config=continuous_config.categorical_ent_loss_config,
                kumaraswamy_ent_loss_config=continuous_config.kumaraswamy_ent_loss_config,
            )
        elif isinstance(continuous_config, ReparameterizedSquashedGaussianMixtureConfig):
            return ReparameterizedSquashedGaussianMixtureActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                action_net_initialization=action_net_initialization,
                initial_action_modes=continuous_config.initial_action_modes,
                initial_stds=continuous_config.initial_stds,
                initial_weights=continuous_config.initial_weights,
                epsilon=continuous_config.epsilon,
                inverse_cdf_iterations=continuous_config.inverse_cdf_iterations,
                log_std_clamp_range=continuous_config.log_std_clamp_range,
                ent_loss_coef=continuous_config.ent_loss_coef,
                gaussian_ent_scale=continuous_config.gaussian_ent_scale,
                categorical_ent_loss_config=continuous_config.categorical_ent_loss_config,
                gaussian_ent_loss_config=continuous_config.gaussian_ent_loss_config,
            )
        elif isinstance(continuous_config, StickySignMagnitudeBetaConfig):
            return StickySignMagnitudeBetaActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                action_net_initialization=action_net_initialization,
                initial_positive_prob=continuous_config.initial_positive_prob,
                epsilon=continuous_config.epsilon,
                negative_alpha=continuous_config.negative_alpha,
                negative_beta=continuous_config.negative_beta,
                positive_alpha=continuous_config.positive_alpha,
                positive_beta=continuous_config.positive_beta,
                ent_loss_coef=continuous_config.ent_loss_coef,
                beta_ent_scale=continuous_config.beta_ent_scale,
                categorical_ent_loss_config=continuous_config.categorical_ent_loss_config,
                beta_ent_loss_config=continuous_config.beta_ent_loss_config,
                stickiness=continuous_config.stickiness,
            )
        elif isinstance(continuous_config, SignMagnitudeBetaConfig):
            return SignMagnitudeBetaActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                action_net_initialization=action_net_initialization,
                initial_positive_prob=continuous_config.initial_positive_prob,
                epsilon=continuous_config.epsilon,
                negative_alpha=continuous_config.negative_alpha,
                negative_beta=continuous_config.negative_beta,
                positive_alpha=continuous_config.positive_alpha,
                positive_beta=continuous_config.positive_beta,
                ent_loss_coef=continuous_config.ent_loss_coef,
                beta_ent_scale=continuous_config.beta_ent_scale,
                categorical_ent_loss_config=continuous_config.categorical_ent_loss_config,
                beta_ent_loss_config=continuous_config.beta_ent_loss_config,
            )
        elif isinstance(continuous_config, StickyLeftMiddleRightBetaConfig):
            return StickyLeftMiddleRightBetaActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                eps_c=continuous_config.eps_c,
                action_net_initialization=action_net_initialization,
                initial_middle_prob=continuous_config.initial_middle_prob,
                epsilon=continuous_config.epsilon,
                left_alpha=continuous_config.left_alpha,
                left_beta=continuous_config.left_beta,
                right_alpha=continuous_config.right_alpha,
                right_beta=continuous_config.right_beta,
                ent_loss_coef=continuous_config.ent_loss_coef,
                beta_ent_scale=continuous_config.beta_ent_scale,
                categorical_ent_loss_config=continuous_config.categorical_ent_loss_config,
                beta_ent_loss_config=continuous_config.beta_ent_loss_config,
                stickiness=continuous_config.stickiness,
                middle_sticky=continuous_config.middle_sticky,
            )
        elif isinstance(continuous_config, LeftMiddleRightBetaConfig):
            return LeftMiddleRightBetaActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                eps_c=continuous_config.eps_c,
                action_net_initialization=action_net_initialization,
                initial_middle_prob=continuous_config.initial_middle_prob,
                epsilon=continuous_config.epsilon,
                left_alpha=continuous_config.left_alpha,
                left_beta=continuous_config.left_beta,
                right_alpha=continuous_config.right_alpha,
                right_beta=continuous_config.right_beta,
                ent_loss_coef=continuous_config.ent_loss_coef,
                beta_ent_scale=continuous_config.beta_ent_scale,
                categorical_ent_loss_config=continuous_config.categorical_ent_loss_config,
                beta_ent_loss_config=continuous_config.beta_ent_loss_config,
            )
        elif isinstance(continuous_config, StickyBangZeroBangConfig):
            return StickyBangZeroBangActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                bang=continuous_config.bang,
                stickiness=continuous_config.stickiness,
                zero_sticky=continuous_config.zero_sticky,
                action_net_initialization=action_net_initialization,
                ent_loss_coef=continuous_config.ent_loss_coef,
                ent_loss_config=continuous_config.ent_loss_config,
            )
        elif isinstance(continuous_config, BangZeroBangConfig):
            return BangZeroBangActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                bang=continuous_config.bang,
                action_net_initialization=action_net_initialization,
                ent_loss_coef=continuous_config.ent_loss_coef,
                ent_loss_config=continuous_config.ent_loss_config,
            )
        raise TypeError(
            "Unsupported continuous action config type for Box action space: "
            f"{type(continuous_config)}"
        )
    elif isinstance(action_space, spaces.MultiBinary):
        return BernoulliActionDist(
            latent_dim=latent_dim,
            action_dim=action_space_dim,
            initial_prob=bernoulli_config.initial_prob if bernoulli_config is not None else None,
            action_net_initialization=action_net_initialization,
            ent_loss_coef=bernoulli_config.ent_loss_coef if bernoulli_config is not None else 0.0,
            ent_loss_config=bernoulli_config.ent_loss_config if bernoulli_config is not None else None,
        )
    else:
        raise NotImplementedError(f"Unsupported action space type: {type(action_space)}")


def _assert_unit_box_range(space: spaces.Box, atol: float = 1e-6) -> None:
    low = np.asarray(space.low, dtype=np.float64)
    high = np.asarray(space.high, dtype=np.float64)

    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        raise ValueError(f"Box action bounds must be finite, got low/high with non-finite values: {space}")

    if not (np.allclose(low, -1.0, atol=atol) and np.allclose(high, 1.0, atol=atol)):
        raise ValueError(
            "Box action space must have bounds low=-1 and high=1 for tanh-squashed policy output. "
            f"Got low in [{low.min():.6g}, {low.max():.6g}], high in [{high.min():.6g}, {high.max():.6g}]. "
        )
