from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import distributions as torchdist

from swarmbots.learn.action_dists.action_dist import AGENT_ACTIONS_DIM, ActionNetInitialization
from swarmbots.learn.action_dists.entropy_utils import EntropyLossConfig
from swarmbots.learn.action_dists.left_right_beta_action_dist import (
    LeftRightBetaActionDist,
    LeftRightBetaConfig,
)
from swarmbots.learn.action_dists.sticky_action_dist import StickyActionDist


@dataclass(frozen=True)
class StickyLeftRightBetaConfig(LeftRightBetaConfig):
    stickiness: float = 0.1


class StickyLeftRightBetaActionDist(LeftRightBetaActionDist, StickyActionDist):
    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization | None,
            initial_right_prob: float | None = None,
            epsilon: float = 1e-6,
            left_alpha: float = 1.0 + math.log(2.0),
            left_beta: float = 1.0 + math.log(2.0),
            right_alpha: float = 1.0 + math.log(2.0),
            right_beta: float = 1.0 + math.log(2.0),
            ent_loss_coef: float = 0.0,
            beta_ent_scale: float = 1.0,
            categorical_ent_loss_config: EntropyLossConfig | None = None,
            beta_ent_loss_config: EntropyLossConfig | None = None,
            stickiness: float = 0.0,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
            initial_right_prob=initial_right_prob,
            epsilon=epsilon,
            left_alpha=left_alpha,
            left_beta=left_beta,
            right_alpha=right_alpha,
            right_beta=right_beta,
            ent_loss_coef=ent_loss_coef,
            beta_ent_scale=beta_ent_scale,
            categorical_ent_loss_config=categorical_ent_loss_config,
            beta_ent_loss_config=beta_ent_loss_config,
        )
        self._init_stickiness_buffer()
        self.set_stickiness(stickiness)

    def requires_previous_actions(self) -> bool:
        return self.get_stickiness() > 0.0

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = agent
        left_01 = self.left_beta_dist.sample()
        right_01 = self.right_beta_dist.sample()
        effective_probs = self._effective_probs(previous_actions)
        component_indices = torchdist.Categorical(probs=effective_probs).sample()

        left_actions = -1.0 + left_01
        right_actions = right_01
        sampled_actions = torch.where(component_indices == self._LEFT_INDEX, left_actions, right_actions)
        return sampled_actions

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        weights = self._effective_probs(previous_actions)
        left_mean = -1.0 + self.left_beta_dist.mean
        right_mean = self.right_beta_dist.mean
        return (
                weights[..., self._LEFT_INDEX] * left_mean
                + weights[..., self._RIGHT_INDEX] * right_mean
        )

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        log_weights = self._effective_probs(previous_actions).log()
        left_mask = actions < 0.0

        left_01 = (actions + 1.0).clamp(self.epsilon, 1.0 - self.epsilon)
        right_01 = actions.clamp(self.epsilon, 1.0 - self.epsilon)

        left_log_prob = (
                log_weights[..., self._LEFT_INDEX]
                + self.left_beta_dist.log_prob(left_01)
                + self.log_interval_jacobian
        )
        right_log_prob = (
                log_weights[..., self._RIGHT_INDEX]
                + self.right_beta_dist.log_prob(right_01)
                + self.log_interval_jacobian
        )
        log_prob_per_action = torch.where(left_mask, left_log_prob, right_log_prob)
        return log_prob_per_action.sum(dim=AGENT_ACTIONS_DIM)

    def _effective_probs(self, previous_actions: torch.Tensor | None) -> torch.Tensor:
        base_probs = F.softmax(self.weight_logits, dim=-1)
        if previous_actions is None:
            if self.get_stickiness() > 0.0:
                raise ValueError("previous_actions is required when stickiness > 0.")
            return base_probs

        previous_indices = self._actions_to_indices(previous_actions)
        previous_one_hot = F.one_hot(previous_indices, num_classes=self._N_MIXTURE_COMPONENTS).to(dtype=base_probs.dtype)
        stickiness = self._stickiness_tensor(dtype=base_probs.dtype, device=base_probs.device)
        return torch.lerp(base_probs, previous_one_hot, stickiness)

    def _actions_to_indices(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.where(actions < 0.0, self._LEFT_INDEX, self._RIGHT_INDEX)

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "stickiness": self.get_stickiness(),
        }
