from typing import Optional, Self

import torch
from torch import distributions as torchdist

from swarmbots.learn.action_dists.action_dist import AGENT_ACTIONS_DIM, ActionNetInitialization
from swarmbots.learn.action_dists.discrete_action_dist import DiscreteActionDist
from swarmbots.learn.losses import LossMetrics
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


class BangZeroBangActionDist(DiscreteActionDist):
    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            bang: float = 1.0,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
    ) -> None:
        if bang <= 0.0:
            raise ValueError(f"bang must be > 0, got {bang}.")
        if bang > 1.0:
            raise ValueError(f"bang must be <= 1.0 for Box(-1, 1) actions, got {bang}.")

        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim * 3,
            action_net_initialization=action_net_initialization,
        )
        self.agent_action_dim = action_dim
        self.bang = bang
        self.distribution: Optional[torchdist.Categorical] = None

    def update_distribution_params(self, action_logits: torch.Tensor) -> Self:
        reshaped_logits = action_logits.view(*action_logits.shape[:-1], self.agent_action_dim, 3)
        self.distribution = torchdist.Categorical(logits=reshaped_logits)
        return self

    def sample(self, agent: int | None = None) -> torch.Tensor:
        sampled_indices = self.distribution.sample()
        return self._indices_to_actions(sampled_indices)

    def mode(self) -> torch.Tensor:
        greedy_indices = self.distribution.probs.argmax(dim=-1)
        return self._indices_to_actions(greedy_indices)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        action_indices = self._actions_to_indices(actions)
        return self.distribution.log_prob(action_indices).sum(dim=AGENT_ACTIONS_DIM)

    def compute_exploration_loss(self) -> tuple[Optional[torch.Tensor], LossMetrics]:
        return self.distribution.entropy().sum(dim=AGENT_ACTIONS_DIM), {}

    def _actions_to_indices(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions / self.bang).round().to(dtype=torch.long).add(1).clamp_(0, 2)

    def _indices_to_actions(self, indices: torch.Tensor) -> torch.Tensor:
        return (indices.to(dtype=torch.float32) - 1.0) * self.bang
