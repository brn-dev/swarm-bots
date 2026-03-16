from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F

from swarmbots.learn.action_dists.action_dist import ActionNetInitialization
from swarmbots.learn.action_dists.beta_mixture_action_dist import BetaMixtureActionDist
from swarmbots.learn.action_dists.temporally_correlated_action_dist import TemporallyCorrelatedActionDist


@dataclass(frozen=True)
class StickyBetaMixtureConfig:
    num_components: int
    sticky_probability: float
    alphas: tuple[float, ...]
    betas: tuple[float, ...]
    epsilon: float = 1e-6


class StickyBetaMixtureActionDist(BetaMixtureActionDist, TemporallyCorrelatedActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            num_components: int,
            action_net_initialization: ActionNetInitialization | None,
            sticky_probability: float,
            epsilon: float = 1e-6,
            alphas: tuple[float, ...] | None = None,
            betas: tuple[float, ...] | None = None,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            num_components=num_components,
            action_net_initialization=action_net_initialization,
            epsilon=epsilon,
            alphas=alphas,
            betas=betas,
        )
        if not (0.0 <= sticky_probability <= 1.0):
            raise ValueError(f"sticky_probability must be in [0, 1], got {sticky_probability}")
        self.sticky_probability = sticky_probability
        self.previous_component_indices: Optional[torch.Tensor] = None

    def sample(self, agent: int | None = None) -> torch.Tensor:
        _ = agent
        self._assert_ready()
        weights = F.softmax(self.weight_logits, dim=-1)
        sampled_component_indices = torch.multinomial(
            weights.reshape(-1, self.num_components),
            num_samples=1,
        ).reshape(weights.shape[:-1])

        chosen_component_indices = sampled_component_indices
        if self.sticky_probability > 0.0:
            previous_component_indices = self.previous_component_indices
            if previous_component_indices is None or previous_component_indices.shape != sampled_component_indices.shape:
                previous_component_indices = torch.full_like(sampled_component_indices, fill_value=-1)
            valid_previous = previous_component_indices >= 0
            if self.sticky_probability >= 1.0:
                keep_previous_mask = valid_previous
            else:
                keep_previous_mask = (
                        torch.rand(
                            sampled_component_indices.shape,
                            device=sampled_component_indices.device,
                        ) < self.sticky_probability
                ) & valid_previous
            chosen_component_indices = torch.where(
                keep_previous_mask,
                previous_component_indices,
                sampled_component_indices,
            )

        self.previous_component_indices = chosen_component_indices.detach()

        sampled_components = self.components.sample()
        sampled_in_01 = torch.gather(
            sampled_components,
            dim=-1,
            index=chosen_component_indices.unsqueeze(-1),
        ).squeeze(-1)
        return 2.0 * sampled_in_01 - 1.0

    def reset_on_ep_start(self, mask: torch.Tensor) -> None:
        if self.previous_component_indices is None:
            return
        if mask.dtype != torch.bool:
            raise ValueError(f"Expected sticky reset mask dtype bool, got {mask.dtype}")
        if self.previous_component_indices.ndim < mask.ndim:
            raise ValueError(
                "Sticky reset mask cannot have more dimensions than previous_component_indices: "
                f"{tuple(mask.shape)} vs {tuple(self.previous_component_indices.shape)}"
            )

        mask_expanded = mask
        while mask_expanded.ndim < self.previous_component_indices.ndim:
            mask_expanded = mask_expanded.unsqueeze(-1)

        try:
            mask_expanded = torch.broadcast_to(mask_expanded, self.previous_component_indices.shape)
        except RuntimeError as err:
            raise ValueError(
                "Sticky reset mask shape is not broadcast-compatible with previous_component_indices: "
                f"{tuple(mask.shape)} vs {tuple(self.previous_component_indices.shape)}"
            ) from err

        reset_fill = torch.full_like(self.previous_component_indices, fill_value=-1)
        self.previous_component_indices = torch.where(mask_expanded, reset_fill, self.previous_component_indices)

    def reset_on_step(
            self,
            mask: torch.Tensor | None = None,
            batch_shape: tuple[int, ...] | None = None,
    ) -> None:
        _ = mask
        _ = batch_shape
