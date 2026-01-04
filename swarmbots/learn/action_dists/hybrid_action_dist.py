from dataclasses import asdict, dataclass
from typing import Any, Optional, Self

import torch
from torch import nn
import numpy as np
from gymnasium import spaces

import swarmbots
from swarmbots.learn.action_dists.action_dist import ActionDist, AGENT_ACTIONS_DIM, ActionNetInitialization
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliActionDist
from swarmbots.learn.action_dists.diag_gaussian_action_dist import DiagGaussianActionDist
from swarmbots.learn.action_dists.gsde_action_dist import GSDEActionDist
from swarmbots.learn.action_dists.predicted_std_action_dist import PredictedStdActionDist
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import SquashedDiagGaussianActionDist
from swarmbots.learn.hybrid_action_space import HybridActionSpace
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

ContinuousActionDistConfig = SquashedDiagParams | PredictedStdParams | GSDEParams


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
    ):
        self.action_space = action_space
        self.action_dims = action_space.agent_action_dims
        if isinstance(continuous_config, list):
            self.continuous_configs = continuous_config
        else:
            self.continuous_configs = [continuous_config] * action_space.n_spaces

        self.gsde_indices = [
            i for i, (sub_space, config) in enumerate(zip(action_space.sub_spaces, self.continuous_configs))
            if isinstance(sub_space, spaces.Box) and isinstance(config, GSDEParams)
        ]
        self.has_gsde = len(self.gsde_indices) > 0

        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_space.total_agent_action_dim,
            action_net_initialization=action_net_initialization
        )

        # noinspection PyTypeChecker
        self.distributions: list[ActionDist] = nn.ModuleList([
            make_proba_distribution(
                latent_dim,
                sub_space,
                sub_space_dim,
                cont_conf,
                bernoulli_initial_prob=bernoulli_initial_prob,
            )
            for sub_space, sub_space_dim, cont_conf
            in zip(action_space.sub_spaces, action_space.agent_action_dims, self.continuous_configs)
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

    def entropy(self) -> Optional[torch.Tensor]:
        entropies = [dist.entropy() for dist in self.distributions]
        if any(e is None for e in entropies):
            return None
        return torch.stack(entropies, dim=-1).sum(dim=-1)

    def reset_noise(self, batch_shape: tuple[int, ...]) -> None:
        for idx in self.gsde_indices:
            # noinspection PyTypeChecker
            gsde_dist: GSDEActionDist = self.distributions[idx]
            gsde_dist.reset_noise(batch_shape)

    def set_std(self, std: float) -> None:
        for dist in self.distributions:
            set_std = getattr(dist, "set_std", None)
            if callable(set_std):
                set_std(std)


def make_proba_distribution(
        latent_dim: int,
        action_space: spaces.Space,
        action_space_dim: int,
        continuous_config: ContinuousActionDistConfig | None,
        bernoulli_initial_prob: float | None = None,
) -> ActionDist:
    if isinstance(action_space, spaces.Box):
        _assert_unit_box_range(action_space)
        assert continuous_config is not None, ("Supply a ContinuousActionDistConfig (SquashedDiagParams | "
                                               "PredictedStdParams | GSDEParams) for continuous actions")

        if isinstance(continuous_config, SquashedDiagParams):
            return SquashedDiagGaussianActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                std=continuous_config.std,
                std_learnable=continuous_config.std_learnable,
                epsilon=continuous_config.epsilon,
            )
        elif isinstance(continuous_config, PredictedStdParams):
            return PredictedStdActionDist(
                latent_dim=latent_dim,
                action_dim=action_space_dim,
                base_std=continuous_config.base_std,
                epsilon=continuous_config.epsilon,
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
            )
    elif isinstance(action_space, spaces.MultiBinary):
        return BernoulliActionDist(
            latent_dim=latent_dim,
            action_dim=action_space_dim,
            initial_prob=bernoulli_initial_prob,
        )
    else:
        raise NotImplementedError


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
