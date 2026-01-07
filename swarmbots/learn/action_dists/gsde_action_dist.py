import math
from typing import Optional, Self

import torch
import torch.distributions as torchdist
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionNetInitialization
from swarmbots.learn.action_dists.continuous_action_dist import ContinuousActionDist
from swarmbots.learn.action_dists.tanh_bijector import TanhBijector
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal

LogStdNetInitialization = ActionNetInitialization


class GSDEActionDist(ContinuousActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            base_std: float,
            latent_sde_dim: int | None = None,
            std_learnable: bool = True,
            normalize_latent_sde_by_dim: bool = True,
            squash_output: bool = False,
            epsilon: float = 1e-6,
            full_std: bool = True,
            sde_learn_features: bool = True,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
            latent_sde_net_initialization: ActionNetInitialization = init_linear_orthogonal,
            log_std_clamp_range: tuple[float, float] = (-20.0, 2.0),
    ):
        assert latent_sde_dim is None or latent_sde_dim == latent_dim or sde_learn_features

        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
        )

        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim
        self.std_learnable = std_learnable
        self.normalize_latent_sde_by_dim = normalize_latent_sde_by_dim
        self.full_std = full_std
        self.sde_learn_features = sde_learn_features
        self.squash_output = squash_output
        self.epsilon = epsilon
        self.log_std_clamp_range = log_std_clamp_range

        if latent_dim == self.latent_sde_dim:
            self.latent_sde_net: nn.Module = nn.Identity()
        else:
            self.latent_sde_net = nn.Linear(latent_dim, self.latent_sde_dim)
            if latent_sde_net_initialization is not None:
                latent_sde_net_initialization(self.latent_sde_net)

        log_std_cols = action_dim if self.full_std else 1
        self.log_stds = nn.Parameter(
            torch.ones((self.latent_sde_dim, log_std_cols)) * math.log(base_std),
            requires_grad=std_learnable,
        )

        self.distribution: Optional[torchdist.Normal] = None

        self._latent_sde: Optional[torch.Tensor] = None
        self._last_gaussian_actions: Optional[torch.Tensor] = None

        self._exploration_matrices: Optional[torch.Tensor] = None
        self._exploration_batch_shape: Optional[tuple[int, ...]] = None

    def set_std(self, std: float) -> None:
        if std <= 0:
            raise ValueError(f"std must be > 0, got {std}")
        with torch.no_grad():
            self.log_stds[:] = math.log(std)
        if self._exploration_batch_shape is not None:
            self.reset_noise(self._exploration_batch_shape)

    def scale_std(self, multiplier: float) -> None:
        if multiplier <= 0:
            raise ValueError(f"multiplier must be > 0, got {multiplier}")
        with torch.no_grad():
            self.log_stds *= multiplier
        if self._exploration_batch_shape is not None:
            self.reset_noise(self._exploration_batch_shape)

    def reset_noise(self, batch_shape: tuple[int, ...]) -> None:
        log_stds = torch.clamp(self.log_stds, *self.log_std_clamp_range)
        std_matrix = self._get_std_matrix(log_stds)
        noise = torch.randn((*batch_shape, self.latent_sde_dim, self.action_dim), device=std_matrix.device, dtype=std_matrix.dtype)
        self._exploration_matrices = noise * std_matrix
        self._exploration_batch_shape = batch_shape

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        action_means = self.action_net(latent_pi)
        latent_sde = self.latent_sde_net(latent_pi)
        if not self.sde_learn_features:
            latent_sde = latent_sde.detach()
        if self.normalize_latent_sde_by_dim:
            latent_sde = latent_sde / math.sqrt(self.latent_sde_dim)
        self._latent_sde = latent_sde
        return self.update_distribution_params(action_means, self.log_stds)

    def update_distribution_params(self, means: torch.Tensor, log_stds: torch.Tensor) -> Self:
        if self._latent_sde is None:
            raise RuntimeError("update_latent_features() must be called before update_distribution_params().")

        log_stds = torch.clamp(log_stds, *self.log_std_clamp_range)
        std_matrix = self._get_std_matrix(log_stds)

        latent_sde_sq = self._latent_sde ** 2
        variance = torch.matmul(latent_sde_sq, std_matrix ** 2)
        action_std = torch.sqrt(variance + self.epsilon)

        self.distribution = torchdist.Normal(loc=means, scale=action_std)
        return self

    def sample(self, agent: int | None = None) -> torch.Tensor:
        if self.squash_output:
            self._last_gaussian_actions = self._sample_gaussian_actions(agent)
            return TanhBijector.forward(self._last_gaussian_actions)

        return self._sample_gaussian_actions(agent)

    def mode(self) -> torch.Tensor:
        gaussian_actions = self.distribution.mean
        if self.squash_output:
            self._last_gaussian_actions = gaussian_actions
            return TanhBijector.forward(gaussian_actions)
        return gaussian_actions

    def log_prob(self, actions: torch.Tensor, gaussian_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.squash_output:
            return super().log_prob(actions)

        if gaussian_actions is None:
            gaussian_actions = TanhBijector.inverse(actions)

        log_prob = super().log_prob(gaussian_actions)
        log_prob -= self.sum_action_dim(torch.log(1 - actions ** 2 + self.epsilon))
        return log_prob

    def entropy(self) -> Optional[torch.Tensor]:
        if self.squash_output:
            return None
        return super().entropy()

    def get_actions_with_log_probs(
            self,
            latent_pi: torch.Tensor,
            deterministic: bool = False,
            agent: int | None = None,
    ):
        actions = self.update_latent_features(latent_pi).get_actions(deterministic=deterministic, agent=agent)
        log_probs = self.log_prob(actions, self._last_gaussian_actions)
        return actions, log_probs

    def _sample_gaussian_actions(self, agent: int | None) -> torch.Tensor:
        if self._latent_sde is None or self.distribution is None:
            raise RuntimeError("update_latent_features() must be called before sampling actions.")

        if self._exploration_matrices is None:
            raise RuntimeError("reset_noise() must be called before sampling GSDE actions.")

        if agent is not None:
            exploration_matrices = self._exploration_matrices[:, agent:agent+1]
        else:
            exploration_matrices = self._exploration_matrices

        noise = torch.einsum("...d,...da->...a", self._latent_sde, exploration_matrices)
        return self.distribution.mean + noise

    def _get_std_matrix(self, log_stds: torch.Tensor) -> torch.Tensor:
        if self.full_std:
            return torch.exp(log_stds)
        else:
            return torch.exp(log_stds).expand(self.latent_sde_dim, self.action_dim)