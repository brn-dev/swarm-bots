from dataclasses import asdict, dataclass
from typing import Any, Optional, Self

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
from swarmbots.learn.action_dists.bang_zero_bang_action_dist import BangZeroBangActionDist
from swarmbots.learn.action_dists.continuous_action_dist import ContinuousActionDist
from swarmbots.learn.action_dists.diag_gaussian_action_dist import DiagGaussianActionDist
from swarmbots.learn.action_dists.gsde_action_dist import GSDEActionDist
from swarmbots.learn.action_dists.beta_mixture_action_dist import BetaMixtureActionDist
from swarmbots.learn.action_dists.sticky_beta_mixture_action_dist import StickyBetaMixtureActionDist
from swarmbots.learn.action_dists.predicted_std_action_dist import PredictedStdActionDist
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import SquashedDiagGaussianActionDist
from swarmbots.learn.action_dists.temporally_correlated_action_dist import TemporallyCorrelatedActionDist
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.masking import masked_mean
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal

LogStdNetInitialization = ActionNetInitialization


@dataclass(frozen=True)
class SquashedDiagParams:
    std: float
    std_learnable: bool
    epsilon: float = 1e-6

@dataclass(frozen=True)
class PredictedStdParams:
    base_std: float
    epsilon: float = 1e-6
    log_std_net_initialization: LogStdNetInitialization = init_linear_orthogonal
    log_std_clamp_range: tuple[float, float] = (-20.0, 2.0)

@dataclass(frozen=True)
class GSDEParams:
    base_std: float
    latent_sde_dim: int | None = None
    std_learnable: bool = True
    normalize_latent_sde_by_dim: bool = True
    epsilon: float = 1e-6
    full_std: bool = True
    sde_learn_features: bool = True
    latent_sde_net_initialization: ActionNetInitialization = init_linear_orthogonal
    log_std_clamp_range: tuple[float, float] = (-20.0, 2.0)

@dataclass(frozen=True)
class BetaMixtureParams:
    num_components: int
    alphas: tuple[float, ...]
    betas: tuple[float, ...]
    epsilon: float = 1e-6

@dataclass(frozen=True)
class StickyBetaMixtureParams:
    num_components: int
    sticky_probability: float
    alphas: tuple[float, ...]
    betas: tuple[float, ...]
    epsilon: float = 1e-6


@dataclass(frozen=True)
class BangZeroBangParams:
    bang: float = 1.0


ContinuousActionDistConfig = (
        SquashedDiagParams
        | PredictedStdParams
        | GSDEParams
        | BetaMixtureParams
        | StickyBetaMixtureParams
        | BangZeroBangParams
)


def serialize_continuous_action_dist_config(config: ContinuousActionDistConfig) -> dict[str, Any]:
    data = asdict(config)
    for key in ("log_std_net_initialization", "latent_sde_net_initialization"):
        if key in data and callable(data[key]):
            data[key] = data[key].__name__
    return data


def serialize_continuous_action_dist_configs(
        continuous_config: ContinuousActionDistConfig | list[ContinuousActionDistConfig | None] | None,
) -> dict[str, Any] | list[dict[str, Any] | None] | None:
    if continuous_config is None:
        return None
    if isinstance(continuous_config, list):
        return [
            serialize_continuous_action_dist_config(cc) if cc is not None else None
            for cc in continuous_config
        ]
    return serialize_continuous_action_dist_config(continuous_config)


class HybridActionDistribution(ActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_space: HybridActionSpace,
            continuous_config: ContinuousActionDistConfig | list[ContinuousActionDistConfig | None] | None,
            bernoulli_initial_prob: float | None = None,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
            action_magnitude_loss_coef: float = 0.0,
            action_magnitude_loss_threshold: float = 0.0,
            action_magnitude_loss_power: int = 2,
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

        gsde_indices = [
            i for i, (sub_space, config) in enumerate(
                zip(action_space.sub_spaces, self.continuous_configs, strict=True)
            )
            if isinstance(sub_space, spaces.Box) and isinstance(config, GSDEParams)
        ]
        self.has_gsde = len(gsde_indices) > 0
        if action_magnitude_loss_coef < 0:
            raise ValueError(f"Expected action_magnitude_loss_coef >= 0, got {action_magnitude_loss_coef}")
        if action_magnitude_loss_threshold < 0:
            raise ValueError(f"Expected action_magnitude_loss_threshold >= 0, got {action_magnitude_loss_threshold}")
        if action_magnitude_loss_power < 1:
            raise ValueError(f"Expected action_magnitude_loss_power >= 1, got {action_magnitude_loss_power}")
        self.action_magnitude_loss_coef = action_magnitude_loss_coef
        self.action_magnitude_loss_threshold = action_magnitude_loss_threshold
        self.action_magnitude_loss_power = action_magnitude_loss_power

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
                bernoulli_initial_prob=bernoulli_initial_prob,
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

    def compute_exploration_loss(self) -> tuple[Optional[torch.Tensor], LossMetrics]:
        entropies_and_metrics = [dist.compute_exploration_loss() for dist in self.distributions]
        entropies = [entropy for entropy, _ in entropies_and_metrics]
        if any(entropy is None for entropy in entropies):
            total_entropy = None
        else:
            total_entropy = torch.stack(entropies, dim=-1).sum(dim=-1)

        metrics: LossMetrics = {}
        for _, dist_metrics in entropies_and_metrics:
            metrics.update(dist_metrics)
        return total_entropy, metrics

    def compute_extra_losses(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[LossDict, LossMetrics]:
        return compute_action_magnitude_extra_losses(
            action_means=self.get_unsquashed_action_means(),
            coef=self.action_magnitude_loss_coef,
            threshold=self.action_magnitude_loss_threshold,
            power=self.action_magnitude_loss_power,
            agent_mask=agent_mask,
        )

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

    def get_unsquashed_action_means(self) -> list[torch.Tensor]:
        return [dist.distribution.mean for dist in self.distributions if isinstance(dist, ContinuousActionDist)]

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


def compute_action_magnitude_extra_losses(
        *,
        action_means: list[torch.Tensor],
        coef: float,
        threshold: float,
        power: int,
        agent_mask: torch.Tensor | None = None,
) -> tuple[LossDict, LossMetrics]:
    if coef <= 0:
        return {}, {}
    if not action_means:
        return {}, {}

    action_magnitude_loss = torch.stack(
        [
            compute_action_magnitude_loss(
                action_means=action_mean,
                threshold=threshold,
                power=power,
                agent_mask=agent_mask,
            )
            for action_mean in action_means
        ]
    ).sum()
    scaled_action_magnitude_loss = coef * action_magnitude_loss

    return (
        {"action_magnitude": scaled_action_magnitude_loss},
        {
            "action_magnitude_loss": action_magnitude_loss.item(),
            "action_magnitude_loss_scaled": scaled_action_magnitude_loss.item(),
        },
    )


def compute_action_magnitude_loss(
        action_means: torch.Tensor,
        threshold: float,
        power: int,
        agent_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if action_means.ndim != 3:
        raise ValueError(f"Expected action_means shape (B, N, A), got {tuple(action_means.shape)}")
    if threshold < 0:
        raise ValueError(f"Expected threshold >= 0, got {threshold}")
    if power < 1:
        raise ValueError(f"Expected power >= 1, got {power}")

    if agent_mask is not None:
        if agent_mask.dtype != torch.bool:
            raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
        if agent_mask.shape != action_means.shape[:2]:
            raise ValueError(
                f"Expected agent_mask shape {tuple(action_means.shape[:2])}, got {tuple(agent_mask.shape)}"
            )
        action_valid_mask = agent_mask.unsqueeze(-1).expand_as(action_means)
    else:
        action_valid_mask = None

    excess_action_magnitude = torch.relu(action_means.abs() - threshold)
    return masked_mean(excess_action_magnitude.pow(power), action_valid_mask)


def make_proba_distribution(
        latent_dim: int,
        action_space: spaces.Space,
        action_space_dim: int,
        action_net_initialization: ActionNetInitialization,
        continuous_config: ContinuousActionDistConfig | None,
        bernoulli_initial_prob: float | None = None,
) -> ActionDist:
    if isinstance(action_space, spaces.Box):
        _assert_unit_box_range(action_space)
        if continuous_config is None:
            raise ValueError(
                "Supply a ContinuousActionDistConfig "
                "(SquashedDiagParams | PredictedStdParams | GSDEParams | "
                "BetaMixtureParams | StickyBetaMixtureParams | BangZeroBangParams) "
                "for continuous actions."
            )

        if isinstance(continuous_config, SquashedDiagParams):
            return SquashedDiagGaussianActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                std=continuous_config.std,
                std_learnable=continuous_config.std_learnable,
                epsilon=continuous_config.epsilon,
                action_net_initialization=action_net_initialization,
            )
        elif isinstance(continuous_config, PredictedStdParams):
            return PredictedStdActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                base_std=continuous_config.base_std,
                epsilon=continuous_config.epsilon,
                action_net_initialization=action_net_initialization,
                log_std_net_initialization=continuous_config.log_std_net_initialization,
                log_std_clamp_range=continuous_config.log_std_clamp_range,
                squash_output=True,
            )
        elif isinstance(continuous_config, GSDEParams):
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
            )
        elif isinstance(continuous_config, BetaMixtureParams):
            return BetaMixtureActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                num_components=continuous_config.num_components,
                action_net_initialization=action_net_initialization,
                epsilon=continuous_config.epsilon,
                alphas=continuous_config.alphas,
                betas=continuous_config.betas,
            )
        elif isinstance(continuous_config, StickyBetaMixtureParams):
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
        elif isinstance(continuous_config, BangZeroBangParams):
            return BangZeroBangActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                bang=continuous_config.bang,
                action_net_initialization=action_net_initialization,
            )
        raise TypeError(
            "Unsupported continuous action config type for Box action space: "
            f"{type(continuous_config)}"
        )
    elif isinstance(action_space, spaces.MultiBinary):
        return BernoulliActionDist(
            latent_dim=latent_dim,
            action_dim=action_space_dim,
            initial_prob=bernoulli_initial_prob,
            action_net_initialization=action_net_initialization,
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
