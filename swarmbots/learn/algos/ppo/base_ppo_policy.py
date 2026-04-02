import abc
import re
from typing import TypeVar, Generic

import torch

from swarmbots.learn.action_dists.action_dist import ActionMetricsSplitterInput
from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisode
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples, PPOSampler, PPOSamplerConfig
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.losses import LossDict, LossMetrics

PPOSamplesType = TypeVar('PPOSamplesType', bound=PPOSamples)
PPOSamplerConfigType = TypeVar('PPOSamplerConfigType', bound=PPOSamplerConfig)


class BasePPOPolicy(BasePolicy, Generic[PPOSamplesType, PPOSamplerConfigType], abc.ABC):
    action_dist: HybridActionDistribution
    _PER_ACTION_ENTROPY_WEIGHT_PATTERN = re.compile(r"^act(?P<idx>\d+)_(?P<alias>ent_loss_coef|entropy|ent)$")

    @property
    def gsde_enabled(self) -> bool:
        return self.action_dist.has_gsde

    @abc.abstractmethod
    def forward(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :return: return actions, log_probs, values
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _evaluate_actions(
            self,
            batch: PPOSamplesType,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[torch.Tensor, torch.Tensor, LossDict, LossMetrics, torch.Tensor]:
        """
        :return: log_probs, values, extra_losses, extra_loss_metrics, local_latents
        """
        raise NotImplementedError()

    def evaluate_actions(
            self,
            batch: PPOSamplesType,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[torch.Tensor, torch.Tensor, LossDict, LossMetrics]:
        log_probs, values, extra_losses, extra_loss_metrics, _ = self._evaluate_actions(
            batch=batch,
            action_splitter=action_splitter,
        )
        return log_probs, values, extra_losses, extra_loss_metrics

    @abc.abstractmethod
    def make_sampler(
            self,
            episodes: list[PPOEpisode],
            config: PPOSamplerConfigType,
    ) -> PPOSampler[PPOSamplesType, PPOSamplerConfigType]:
        raise NotImplementedError()

    def after_optimizer_step(self) -> None:
        pass

    @property
    def has_popart(self) -> bool:
        return False

    def update_value_normalizer(self, targets: torch.Tensor) -> None:
        _ = targets

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        return values

    def get_value_normalizer_metrics(self) -> dict[str, float]:
        return {}

    def update_loss_weights(self, **weights: float) -> None:
        if weights:
            raise ValueError(f'Unknown weights given: {weights}')

    def set_action_stickiness(
            self,
            value: float,
            *,
            sub_dist_idx: int | None = None,
    ) -> None:
        if sub_dist_idx is None:
            self.action_dist.set_all_stickiness(value)
        else:
            self.action_dist.set_sub_stickiness(sub_dist_idx, value)

    @staticmethod
    def _policy_actions(actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 3:
            return actions
        if actions.ndim == 4:
            return actions[:, 0]
        raise ValueError(
            f"Expected actions shape (B, N, A) or (B, T, N, A), got {tuple(actions.shape)}"
        )

    @staticmethod
    def _pop_loss_weight_alias(
            weights: dict[str, float],
            *,
            aliases: tuple[str, ...],
    ) -> tuple[str, float] | None:
        matching_aliases = [alias for alias in aliases if alias in weights]
        if not matching_aliases:
            return None
        if len(matching_aliases) > 1:
            raise ValueError(f"Multiple aliases for the same loss weight are not allowed: {matching_aliases}")
        alias = matching_aliases[0]
        value = float(weights.pop(alias))
        return alias, value

    def _pop_per_action_entropy_weights(self, weights: dict[str, float]) -> dict[int, float]:
        updates: dict[int, float] = {}
        for key in list(weights):
            match = self._PER_ACTION_ENTROPY_WEIGHT_PATTERN.match(key)
            if match is None:
                continue

            idx = int(match.group("idx"))
            if idx in updates:
                raise ValueError(f"Multiple entropy aliases for action index {idx} are not allowed")
            updates[idx] = float(weights.pop(key))
        return updates
