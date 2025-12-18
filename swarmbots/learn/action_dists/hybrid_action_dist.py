from typing import Optional, Self

import torch
from torch import nn
import numpy as np
from gymnasium import spaces

from swarmbots.learn.action_dists.action_dist import ActionDist, AGENT_ACTIONS_DIM, ActionNetInitialization
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliActionDist
from swarmbots.learn.action_dists.predicted_std_action_dist import PredictedStdActionDist
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


class HybridActionDistribution(ActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_space: HybridActionSpace,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
            base_std: float = 1.0,
    ):
        self.action_space = action_space
        self.action_dims = action_space.agent_action_dims
        self.base_std = base_std

        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_space.total_agent_action_dim,
            action_net_initialization=action_net_initialization
        )

        # noinspection PyTypeChecker
        self.distributions: list[ActionDist] = nn.ModuleList([
            make_proba_distribution(latent_dim, sub_space, sub_space_dim, base_std)
            for sub_space, sub_space_dim
            in zip(action_space.sub_spaces, action_space.agent_action_dims)
        ])

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        for dist in self.distributions:
            dist.update_latent_features(latent_pi)
        return self

    def sample(self) -> torch.Tensor:
        return torch.cat([dist.sample() for dist in self.distributions], dim=AGENT_ACTIONS_DIM)

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


def make_proba_distribution(
        latent_dim: int,
        action_space: spaces.Space,
        action_space_dim: int,
        base_std: float,
) -> ActionDist:
    if isinstance(action_space, spaces.Box):
        _assert_unit_box_range(action_space)
        return PredictedStdActionDist(
            latent_dim=latent_dim,
            action_dim=action_space_dim,
            base_std=base_std,
            squash_output=True,
        )
    elif isinstance(action_space, spaces.MultiBinary):
        return BernoulliActionDist(
            latent_dim=latent_dim,
            action_dim=action_space_dim,
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
