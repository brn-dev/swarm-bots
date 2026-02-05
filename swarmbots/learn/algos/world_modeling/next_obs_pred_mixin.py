import abc
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from swarmbots.learn.algos.world_modeling.transformer_transition_model import TransformerTransitionModel
from swarmbots.learn.masking import build_valid_mask, masked_mean


class NextObsPredMixin(abc.ABC):
    pre_transition_transform: nn.Module
    transition_model: TransformerTransitionModel

    local_scalar_target_indices: Optional[list[int]]
    local_angle_target_indices: Optional[list[int]]
    local_rot6d_target_indices: Optional[list[int]]
    local_binary_target_indices: Optional[list[int]]

    scalar_loss_fn: Optional[nn.Module]
    binary_loss_fn: Optional[nn.Module]

    local_scalars_predictor: Optional[nn.Module]
    local_angles_predictor: Optional[nn.Module]
    local_rot6ds_predictor: Optional[nn.Module]
    local_binaries_predictor: Optional[nn.Module]

    scalar_loss_weight: float
    angle_loss_weight: float
    rot6d_loss_weight: float
    binary_loss_weight: float

    def setup_modules(
            self,
            transition_model: TransformerTransitionModel,
            pre_transition_transform: Optional[nn.Module] = None,
            pre_predictors_transform: Optional[nn.Module] = None,
            local_scalar_target_indices: Optional[list[int]] = None,
            local_angle_target_indices: Optional[list[int]] = None,
            local_rot6d_target_indices: Optional[list[int]] = None,
            local_binary_target_indices: Optional[list[int]] = None,
            scalar_loss_fn: Optional[nn.Module] = None,
            binary_loss_fn: Optional[nn.Module] = None,
            local_scalars_predictor: Optional[nn.Module] = None,
            local_angles_predictor: Optional[nn.Module] = None,
            local_rot6ds_predictor: Optional[nn.Module] = None,
            local_binaries_predictor: Optional[nn.Module] = None,
            scalar_loss_weight: float = 1.0,
            angle_loss_weight: float = 1.0,
            rot6d_loss_weight: float = 1.0,
            binary_loss_weight: float = 1.0,
    ) -> None:

        local_scalar_target_indices = self._normalize_indices(local_scalar_target_indices)
        local_angle_target_indices = self._normalize_indices(local_angle_target_indices)
        local_rot6d_target_indices = self._normalize_indices(local_rot6d_target_indices)
        local_binary_target_indices = self._normalize_indices(local_binary_target_indices)

        if local_scalar_target_indices is not None and scalar_loss_fn is None:
            raise ValueError("scalar_loss_fn is required when local_scalar_target_indices is provided")
        if local_binary_target_indices is not None and binary_loss_fn is None:
            raise ValueError("binary_loss_fn is required when local_binary_target_indices is provided")
        if local_scalar_target_indices is not None and local_scalars_predictor is None:
            raise ValueError("local_scalars_predictor is required when local_scalar_target_indices is provided")
        if local_angle_target_indices is not None and local_angles_predictor is None:
            raise ValueError("local_angles_predictor is required when local_angle_target_indices is provided")
        if local_rot6d_target_indices is not None and local_rot6ds_predictor is None:
            raise ValueError("local_rot6ds_predictor is required when local_rot6d_target_indices is provided")
        if local_binary_target_indices is not None and local_binaries_predictor is None:
            raise ValueError("local_binaries_predictor is required when local_binary_target_indices is provided")

        self.pre_transition_transform = (
            pre_transition_transform if pre_transition_transform is not None else nn.Identity()
        )
        self.transition_model = transition_model
        self.pre_predictors_transform = (
            pre_predictors_transform if pre_predictors_transform is not None else nn.Identity()
        )

        self.local_scalar_target_indices = local_scalar_target_indices
        self.local_angle_target_indices = local_angle_target_indices
        self.local_rot6d_target_indices = local_rot6d_target_indices
        self.local_binary_target_indices = local_binary_target_indices

        self.scalar_loss_fn = scalar_loss_fn
        self.binary_loss_fn = binary_loss_fn

        self.local_scalars_predictor = local_scalars_predictor
        self.local_angles_predictor = local_angles_predictor
        self.local_rot6ds_predictor = local_rot6ds_predictor
        self.local_binaries_predictor = local_binaries_predictor

        self.scalar_loss_weight = scalar_loss_weight
        self.angle_loss_weight = angle_loss_weight
        self.rot6d_loss_weight = rot6d_loss_weight
        self.binary_loss_weight = binary_loss_weight


    def compute_scalar_loss(
            self,
            latent_preds: torch.Tensor,
            next_local_obs: torch.Tensor,
            valid_mask: torch.Tensor | None
    ) -> Optional[torch.Tensor]:
        if self.local_scalar_target_indices is None:
            return None

        pred_scalars = self.local_scalars_predictor(latent_preds)
        target_scalars = next_local_obs[..., self.local_scalar_target_indices]
        scalar_losses = self.scalar_loss_fn(pred_scalars, target_scalars)
        loss_per_item = self._reduce_feature_loss(scalar_losses)
        return masked_mean(loss_per_item, valid_mask)

    def compute_angle_loss(
            self,
            latent_preds: torch.Tensor,
            next_local_obs: torch.Tensor,
            valid_mask: torch.Tensor | None,
    ) -> Optional[torch.Tensor]:
        if self.local_angle_target_indices is None:
            return None

        target_angles_sin = next_local_obs[..., self.local_angle_target_indices]
        target_angles_cos = next_local_obs[..., [i + 1 for i in self.local_angle_target_indices]]
        target_angles = torch.stack((target_angles_sin, target_angles_cos), dim=-1)

        pred_angles = self.local_angles_predictor(latent_preds)

        pred_pairs = pred_angles.reshape(*pred_angles.shape[:-1], -1, 2)
        target_pairs = target_angles.reshape(*target_angles.shape[:-2], -1, 2)

        cosine_sim = F.cosine_similarity(pred_pairs, target_pairs, dim=-1, eps=1e-8)
        loss_per_angle = 1.0 - cosine_sim
        loss_per_item = loss_per_angle.mean(dim=-1)
        return masked_mean(loss_per_item, valid_mask)

    def compute_rot6d_loss(
            self,
            latent_preds: torch.Tensor,
            next_local_obs: torch.Tensor,
            valid_mask: torch.Tensor | None,
    ) -> Optional[torch.Tensor]:
        if self.local_rot6d_target_indices is None:
            return None

        pred_rot6ds = self.local_rot6ds_predictor(latent_preds)

        target_indices = self._expand_rot6d_indices(self.local_rot6d_target_indices)
        target_rot6ds = next_local_obs[..., target_indices]

        pred_rot6ds = pred_rot6ds.reshape(*pred_rot6ds.shape[:-1], -1, 6)
        target_rot6ds = target_rot6ds.reshape(*target_rot6ds.shape[:-1], -1, 6)

        pred_mats = self._rot6d_to_matrix(pred_rot6ds)
        target_mats = self._rot6d_to_matrix(target_rot6ds)

        trace_sim = (pred_mats * target_mats).sum(dim=(-1, -2))
        loss_per_rot = 1.0 - (trace_sim / 3.0)
        loss_per_item = loss_per_rot.mean(dim=-1)
        return masked_mean(loss_per_item, valid_mask)


    def compute_binary_loss(
            self,
            latent_preds: torch.Tensor,
            next_local_obs: torch.Tensor,
            valid_mask: torch.Tensor | None,
    ) -> Optional[torch.Tensor]:
        if self.local_binary_target_indices is None:
            return None

        pred_binaries = self.local_binaries_predictor(latent_preds)
        target_binaries = next_local_obs[..., self.local_binary_target_indices]
        binary_losses = self.binary_loss_fn(pred_binaries, target_binaries)
        loss_per_item = self._reduce_feature_loss(binary_losses)
        return masked_mean(loss_per_item, valid_mask)


    def compute_next_obs_pred_loss(
            self,
            local_latents: torch.Tensor,
            next_local_obs: torch.Tensor,
            actions: torch.Tensor,
            agent_mask: torch.Tensor | None = None,
            time_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if local_latents.ndim != 3:
            raise ValueError(f"Expected online_local_latents shape (B, N, D), got {tuple(local_latents.shape)}")

        local_latents = self.pre_transition_transform(local_latents)

        latent_preds, base_shape = self._predict_latents_and_base_shape(
            local_latents=local_latents,
            next_local_obs=next_local_obs,
            actions=actions,
            agent_mask=agent_mask,
            time_mask=time_mask,
        )
        latent_preds = self.pre_predictors_transform(latent_preds)

        valid_mask = build_valid_mask(
            base_shape=base_shape,
            device=latent_preds.device,
            agent_mask=agent_mask,
            time_mask=time_mask,
        )

        losses = []

        scalar_loss = self.compute_scalar_loss(
            latent_preds=latent_preds,
            next_local_obs=next_local_obs,
            valid_mask=valid_mask,
        )
        if scalar_loss is not None:
            losses.append(scalar_loss * self.scalar_loss_weight)

        angle_loss = self.compute_angle_loss(
            latent_preds=latent_preds,
            next_local_obs=next_local_obs,
            valid_mask=valid_mask,
        )
        if angle_loss is not None:
            losses.append(angle_loss * self.angle_loss_weight)

        rot6d_loss = self.compute_rot6d_loss(
            latent_preds=latent_preds,
            next_local_obs=next_local_obs,
            valid_mask=valid_mask,
        )
        if rot6d_loss is not None:
            losses.append(rot6d_loss * self.rot6d_loss_weight)

        binary_loss = self.compute_binary_loss(
            latent_preds=latent_preds,
            next_local_obs=next_local_obs,
            valid_mask=valid_mask,
        )
        if binary_loss is not None:
            losses.append(binary_loss * self.binary_loss_weight)
            
        present_losses = losses
        if not present_losses:
            raise ValueError("No next-observation prediction targets configured")

        return torch.stack(present_losses).sum()

    @staticmethod
    def _normalize_indices(indices: Optional[list[int]]) -> Optional[list[int]]:
        if indices is None:
            return None
        if len(indices) == 0:
            return None
        return list(indices)

    @staticmethod
    def _reduce_feature_loss(losses: torch.Tensor) -> torch.Tensor:
        if losses.ndim < 3:
            return losses
        return losses.mean(dim=-1)

    @staticmethod
    def _expand_rot6d_indices(indices: list[int]) -> list[int]:
        return [start + offset for start in indices for offset in range(6)]

    def _predict_latents_and_base_shape(
            self,
            local_latents: torch.Tensor,
            next_local_obs: torch.Tensor,
            actions: torch.Tensor,
            agent_mask: torch.Tensor | None,
            time_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        if actions.ndim == 3:
            if time_mask is not None:
                raise ValueError("time_mask is only supported for multi-step loss (actions shape (B, T, N, A))")
            if agent_mask is not None and agent_mask.ndim != 2:
                raise ValueError(f"Expected agent_mask shape (B, N), got {tuple(agent_mask.shape)}")
            if next_local_obs.ndim != 3:
                raise ValueError(f"Expected next_local_obs shape (B, N, F), got {tuple(next_local_obs.shape)}")
            if next_local_obs.shape[:2] != local_latents.shape[:2]:
                raise ValueError(
                    f"Expected next_local_obs shape (B, N, F)=({local_latents.shape[0]}, {local_latents.shape[1]}, F), "
                    f"got {tuple(next_local_obs.shape)}"
                )
            latent_preds = self.transition_model(local_latents, actions, agent_mask=agent_mask)
            return latent_preds, latent_preds.shape[:2]

        if actions.ndim == 4:
            b, t, n, _a = actions.shape
            if local_latents.shape[0] != b or local_latents.shape[1] != n:
                raise ValueError(
                    f"Expected local_latents shape (B, N, D)=({b}, {n}, D), got {tuple(local_latents.shape)}"
                )
            if next_local_obs.ndim != 4 or next_local_obs.shape[:3] != (b, t, n):
                raise ValueError(
                    f"Expected next_local_obs shape (B, T, N, F)=({b}, {t}, {n}, F), got {tuple(next_local_obs.shape)}"
                )
            latent_preds = self.transition_model.predict_n_steps(local_latents, actions, agent_mask=agent_mask)
            return latent_preds, latent_preds.shape[:3]

        raise ValueError(
            f"Expected actions shape (B, N, A) or (B, T, N, A), got {tuple(actions.shape)}"
        )

    @staticmethod
    def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
        a1 = rot6d[..., 0:3]
        a2 = rot6d[..., 3:6]
        b1 = F.normalize(a1, dim=-1)
        a2_proj = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = F.normalize(a2_proj, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)
