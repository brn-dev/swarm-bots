import abc
import copy
from typing import Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F

from swarmbots.learn.algos.world_modeling.transformer_transition_model import TransformerTransitionModel
from swarmbots.learn.nn_components.residual import Residual
from swarmbots.learn.polyak_update import polyak_update


class NextObsPredMixin(abc.ABC):
    transition_model: TransformerTransitionModel

    local_scalar_target_indices: Optional[list[int]]
    local_angle_target_indices: Optional[list[int]]
    local_quat_target_indices: Optional[list[int]]
    local_binary_target_indices: Optional[list[int]]

    scalar_loss_fn: Optional[nn.Module]
    binary_loss_nf: Optional[nn.Module]

    def setup_modules(
            self,
            transition_model: TransformerTransitionModel,
            local_scalar_target_indices: Optional[list[int]] = None,
            local_angle_target_indices: Optional[list[int]] = None,
            local_quat_target_indices: Optional[list[int]] = None,
            local_binary_target_indices: Optional[list[int]] = None,
            scalar_loss_fn: Optional[nn.Module] = None,
            binary_loss_fn: Optional[nn.Module] = None,
    ) -> None:

        assert local_scalar_target_indices is None or scalar_loss_fn is not None
        assert local_binary_target_indices is None or binary_loss_fn is not None

        self.transition_model = transition_model

        self.local_scalar_target_indices = local_scalar_target_indices
        self.local_angle_target_indices = local_angle_target_indices
        self.local_quat_target_indices = local_quat_target_indices
        self.local_binary_target_indices = local_binary_target_indices

        self.scalar_loss_fn = scalar_loss_fn
        self.binary_loss_nf = binary_loss_fn

        self.local_scalars_predictor = ...  # MLP from latent dim to len(local_scalar_target_indices)
        self.local_angles_predictor = ...  # MLP from latent dim to 2 * len(local_angle_target_indices)
        self.local_quats_predictor = ...  # MLP from latent dim to 4 * len(local_quat_target_indices)
        self.local_binaries_predictor = ...  # MLP from latent dim to len(local_binary_target_indices)


    def compute_next_obs_pred_loss(
            self,
            local_latents: torch.Tensor,
            next_local_obs: torch.Tensor,
            next_global_obs: torch.Tensor,
            actions: torch.Tensor,
            agent_mask: torch.Tensor | None = None,
            time_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if local_latents.ndim != 3:
            raise ValueError(f"Expected online_local_latents shape (B, N, D), got {tuple(local_latents.shape)}")

        if actions.ndim == 3:
            ...
            # insert singleton time dimension

        ...



