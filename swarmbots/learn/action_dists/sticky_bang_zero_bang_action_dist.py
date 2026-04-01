from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from swarmbots.learn.action_dists.action_dist import AGENT_ACTIONS_DIM, ActionNetInitialization
from swarmbots.learn.action_dists.bang_zero_bang_action_dist import (
    BangZeroBangActionDist,
    BangZeroBangConfig,
)
from swarmbots.learn.action_dists.entropy_utils import EntropyLossConfig
from swarmbots.learn.action_dists.sticky_action_dist import StickyActionDist
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


@dataclass(frozen=True)
class StickyBangZeroBangConfig(BangZeroBangConfig):
    stickiness: float = 0.0
    zero_sticky: bool = False


class StickyBangZeroBangActionDist(BangZeroBangActionDist, StickyActionDist):
    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            bang: float = 1.0,
            stickiness: float = 0.0,
            zero_sticky: bool = False,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
            ent_loss_coef: float = 0.0,
            ent_loss_config: EntropyLossConfig | None = None,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            bang=bang,
            action_net_initialization=action_net_initialization,
            ent_loss_coef=ent_loss_coef,
            ent_loss_config=ent_loss_config,
        )
        self.zero_sticky = zero_sticky
        self.stickiness = 0.0
        self._log_stickiness = None
        self._log_one_minus_stickiness = 0.0
        self.set_stickiness(stickiness)

    def requires_previous_actions(self) -> bool:
        return self.stickiness > 0.0

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = agent
        sampled_indices = self.distribution.sample()
        if self.stickiness <= 0.0:
            return self._indices_to_actions(sampled_indices)

        if previous_actions is None:
            raise ValueError("previous_actions is required when stickiness > 0.")

        previous_indices = self._actions_to_indices(previous_actions)
        can_stick = self._sticky_mask(previous_indices)
        should_stick = can_stick & (torch.rand_like(sampled_indices, dtype=torch.float32) < self.stickiness)
        final_indices = torch.where(should_stick, previous_indices, sampled_indices)
        return self._indices_to_actions(final_indices)

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        effective_probs = self._effective_probs(previous_actions)
        greedy_indices = effective_probs.argmax(dim=-1)
        return self._indices_to_actions(greedy_indices)

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action_indices = self._actions_to_indices(actions)
        base_log_prob = self.distribution.log_prob(action_indices)
        if self.stickiness <= 0.0:
            return base_log_prob.sum(dim=AGENT_ACTIONS_DIM)

        if previous_actions is None:
            raise ValueError("previous_actions is required when stickiness > 0.")

        previous_indices = self._actions_to_indices(previous_actions)
        can_stick = self._sticky_mask(previous_indices)
        sticky_log_prob = base_log_prob + self._log_one_minus_stickiness
        sticky_log_prob = torch.where(
            action_indices == previous_indices,
            torch.logaddexp(sticky_log_prob, sticky_log_prob.new_tensor(self._log_stickiness)),
            sticky_log_prob,
        )

        log_prob = torch.where(can_stick, sticky_log_prob, base_log_prob)
        return log_prob.sum(dim=AGENT_ACTIONS_DIM)

    def _effective_probs(self, previous_actions: torch.Tensor | None) -> torch.Tensor:
        base_probs = self.distribution.probs
        if self.stickiness <= 0.0:
            return base_probs

        if previous_actions is None:
            raise ValueError("previous_actions is required when stickiness > 0.")

        previous_indices = self._actions_to_indices(previous_actions)
        sticky_mask = self._sticky_mask(previous_indices).unsqueeze(-1)
        previous_one_hot = F.one_hot(previous_indices, num_classes=3).to(dtype=base_probs.dtype)

        sticky_probs = (1.0 - self.stickiness) * base_probs + self.stickiness * previous_one_hot
        return torch.where(sticky_mask, sticky_probs, base_probs)

    def _sticky_mask(self, previous_indices: torch.Tensor) -> torch.Tensor:
        if self.zero_sticky:
            return torch.ones_like(previous_indices, dtype=torch.bool)
        return previous_indices != 1

    def set_stickiness(self, stickiness: float) -> None:
        if not (0.0 <= stickiness < 1.0):
            raise ValueError(f"stickiness must be in [0, 1), got {stickiness}.")
        self.stickiness = float(stickiness)
        self._log_stickiness = math.log(self.stickiness) if self.stickiness > 0.0 else None
        self._log_one_minus_stickiness = math.log1p(-self.stickiness)

    def get_stickiness(self) -> float:
        return self.stickiness

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "stickiness": self.stickiness,
            "zero_sticky": self.zero_sticky,
        }
