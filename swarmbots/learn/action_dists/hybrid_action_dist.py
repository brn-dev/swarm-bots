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
)
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliActionDist
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.bang_zero_bang_action_dist import BangZeroBangActionDist, BangZeroBangConfig
from swarmbots.learn.action_dists.continuous_action_dist import ContinuousActionDist
from swarmbots.learn.action_dists.diag_gaussian_action_dist import DiagGaussianActionDist
from swarmbots.learn.action_dists.gsde_action_dist import GSDEActionDist, GSDEConfig
from swarmbots.learn.action_dists.beta_mixture_action_dist import BetaMixtureActionDist, BetaMixtureConfig
from swarmbots.learn.action_dists.sticky_beta_mixture_action_dist import (
    StickyBetaMixtureActionDist,
    StickyBetaMixtureConfig,
)
from swarmbots.learn.action_dists.predicted_std_action_dist import PredictedStdActionDist, PredictedStdConfig
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import (
    SquashedDiagGaussianActionDist,
    SquashedDiagGaussianConfig,
)
from swarmbots.learn.action_dists.temporally_correlated_action_dist import TemporallyCorrelatedActionDist
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.config_serialization import serialize_dataclass_config
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


ContinuousActionDistConfig: TypeAlias = (
    SquashedDiagGaussianConfig
    | PredictedStdConfig
    | GSDEConfig
    | BetaMixtureConfig
    | StickyBetaMixtureConfig
    | BangZeroBangConfig
)

ContinuousActionDistConfigInput: TypeAlias = (
    ContinuousActionDistConfig
    | list[ContinuousActionDistConfig | None]
    | None
)


def serialize_continuous_action_dist_config(config: ContinuousActionDistConfig) -> dict[str, Any]:
    return serialize_dataclass_config(config)


def serialize_continuous_action_dist_configs(
        continuous_config: ContinuousActionDistConfigInput,
) -> dict[str, Any] | list[dict[str, Any] | None] | None:
    if continuous_config is None:
        return None
    if isinstance(continuous_config, list):
        return [
            serialize_continuous_action_dist_config(cc) if cc is not None else None
            for cc in continuous_config
        ]
    return serialize_continuous_action_dist_config(continuous_config)


def serialize_bernoulli_config(
        bernoulli_config: BernoulliConfig | None,
) -> dict[str, Any] | None:
    if bernoulli_config is None:
        return None
    return serialize_dataclass_config(bernoulli_config)


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

    def sample(self, agent: int | None = None) -> torch.Tensor:
        actions: list[torch.Tensor] = []
        for dist in self.distributions:
            if isinstance(dist, GSDEActionDist):
                actions.append(dist.sample(agent=agent))
            else:
                actions.append(dist.sample())
        return torch.cat(actions, dim=AGENT_ACTIONS_DIM)

    def mode(self) -> torch.Tensor:
        return torch.cat([dist.mode() for dist in self.distributions], dim=AGENT_ACTIONS_DIM)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        split_actions = torch.split(actions, self.action_dims, dim=AGENT_ACTIONS_DIM)
        return torch.stack(
            [dist.log_prob(action) for dist, action in zip(self.distributions, split_actions, strict=True)],
            dim=-1
        ).sum(dim=-1)

    def get_actions_with_log_probs(
            self,
            latent_pi: torch.Tensor,
            deterministic: bool = False,
            agent: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions_parts: list[torch.Tensor] = []
        log_prob_parts: list[torch.Tensor] = []

        for dist in self.distributions:
            action_part, log_prob_part = dist.get_actions_with_log_probs(
                latent_pi=latent_pi,
                deterministic=deterministic,
                agent=agent,
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
    ) -> tuple[LossDict, LossMetrics]:
        losses: LossDict = {}
        metrics: LossMetrics = {}
        for i, dist in enumerate(self.distributions):
            dist_losses, dist_metrics = dist.compute_extra_losses(agent_mask=agent_mask)
            prefix = f"act{i}_"
            losses.update(self._prefix_named_values(dist_losses, prefix=prefix))
            metrics.update(self._prefix_named_values(dist_metrics, prefix=prefix))
        return losses, metrics

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
            if isinstance(config, (SquashedDiagGaussianConfig, PredictedStdConfig, GSDEConfig, BangZeroBangConfig,)):
                self.continuous_configs[idx] = replace(config, ent_loss_coef=value)
        if self.bernoulli_config is not None:
            self.bernoulli_config = replace(self.bernoulli_config, ent_loss_coef=value)


    @staticmethod
    def _prefix_named_values(
            values: Mapping[str, Any],
            *,
            prefix: str,
    ) -> dict[str, Any]:
        return {f"{prefix}{name}": value for name, value in values.items()}


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
                "BetaMixtureConfig | StickyBetaMixtureConfig | BangZeroBangConfig) "
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
                action_magnitude_loss_coef=continuous_config.action_magnitude_loss_coef,
                action_magnitude_loss_threshold=continuous_config.action_magnitude_loss_threshold,
                action_magnitude_loss_power=continuous_config.action_magnitude_loss_power,
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
        elif isinstance(continuous_config, StickyBetaMixtureConfig):
            return StickyBetaMixtureActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                num_components=continuous_config.num_components,
                action_net_initialization=action_net_initialization,
                sticky_probability=continuous_config.sticky_probability,
                epsilon=continuous_config.epsilon,
                alphas=continuous_config.alphas,
                betas=continuous_config.betas,
            )
        elif isinstance(continuous_config, BangZeroBangConfig):
            return BangZeroBangActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                bang=continuous_config.bang,
                action_net_initialization=action_net_initialization,
                ent_loss_coef=continuous_config.ent_loss_coef,
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
