import abc
from typing import Any

import torch

from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution
from swarmbots.learn.algos.off_policy.replay_buffer import OffPolicyReplayBatch, OffPolicyReplayEpisodeSegmentBatch
from swarmbots.learn.base_policy import BasePolicy


class BaseSACPolicy(BasePolicy, abc.ABC):
    action_dist: HybridActionDistribution

    @property
    def gsde_enabled(self) -> bool:
        action_dist = getattr(self, "action_dist", None)
        return bool(getattr(action_dist, "has_gsde", False))

    @abc.abstractmethod
    def action_log_prob(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            deterministic: bool = False,
            use_rsample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def q_values(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def target_q_values(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def actor_parameters(self) -> list[torch.nn.Parameter]:
        raise NotImplementedError()

    @abc.abstractmethod
    def critic_parameters(self) -> list[torch.nn.Parameter]:
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_actor_nop_loss(
            self,
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_critic_nop_loss(
            self,
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def polyak_update_targets(self, tau: float) -> None:
        raise NotImplementedError()

    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> torch.Tensor:
        _ = previous_actions
        actions, _log_probs = self.action_log_prob(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            deterministic=deterministic,
            use_rsample=False,
        )
        return actions

    def requires_previous_actions(self) -> bool:
        return False

    def has_nop_loss(self) -> bool:
        return False

    def after_optimizer_step(self) -> None:
        pass
