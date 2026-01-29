import abc
from typing import Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F

from swarmbots.learn.algos.world_modeling.transformer_transition_model import TransformerTransitionModel


class NextObsPredMixin(abc.ABC):
    transition_model: TransformerTransitionModel

    local_scalar_target_indices: Optional[list[int]]
    local_angle_target_indices: Optional[list[int]]
    local_quat_target_indices: Optional[list[int]]
    local_binary_target_indices: Optional[list[int]]

    scalar_loss_fn: Optional[nn.Module]
    binary_loss_fn: Optional[nn.Module]

    local_scalars_predictor: Optional[nn.Module]
    local_angles_predictor: Optional[nn.Module]
    local_quats_predictor: Optional[nn.Module]
    local_binaries_predictor: Optional[nn.Module]

    def setup_modules(
            self,
            transition_model: TransformerTransitionModel,
            local_scalar_target_indices: Optional[list[int]] = None,
            local_angle_target_indices: Optional[list[int]] = None,
            local_quat_target_indices: Optional[list[int]] = None,
            local_binary_target_indices: Optional[list[int]] = None,
            scalar_loss_fn: Optional[nn.Module] = None,
            binary_loss_fn: Optional[nn.Module] = None,
            local_scalars_predictor: Optional[nn.Module] = None,
            local_angles_predictor: Optional[nn.Module] = None,
            local_quats_predictor: Optional[nn.Module] = None,
            local_binaries_predictor: Optional[nn.Module] = None,
    ) -> None:

        local_scalar_target_indices = self._normalize_indices(local_scalar_target_indices)
        local_angle_target_indices = self._normalize_indices(local_angle_target_indices)
        local_quat_target_indices = self._normalize_indices(local_quat_target_indices)
        local_binary_target_indices = self._normalize_indices(local_binary_target_indices)

        if local_scalar_target_indices is not None and scalar_loss_fn is None:
            raise ValueError("scalar_loss_fn is required when local_scalar_target_indices is provided")
        if local_binary_target_indices is not None and binary_loss_fn is None:
            raise ValueError("binary_loss_fn is required when local_binary_target_indices is provided")
        if local_scalar_target_indices is not None and local_scalars_predictor is None:
            raise ValueError("local_scalars_predictor is required when local_scalar_target_indices is provided")
        if local_angle_target_indices is not None and local_angles_predictor is None:
            raise ValueError("local_angles_predictor is required when local_angle_target_indices is provided")
        if local_quat_target_indices is not None and local_quats_predictor is None:
            raise ValueError("local_quats_predictor is required when local_quat_target_indices is provided")
        if local_binary_target_indices is not None and local_binaries_predictor is None:
            raise ValueError("local_binaries_predictor is required when local_binary_target_indices is provided")

        self.transition_model = transition_model

        self.local_scalar_target_indices = local_scalar_target_indices
        self.local_angle_target_indices = local_angle_target_indices
        self.local_quat_target_indices = local_quat_target_indices
        self.local_binary_target_indices = local_binary_target_indices

        self.scalar_loss_fn = scalar_loss_fn
        self.binary_loss_fn = binary_loss_fn

        self.local_scalars_predictor = local_scalars_predictor
        self.local_angles_predictor = local_angles_predictor
        self.local_quats_predictor = local_quats_predictor
        self.local_binaries_predictor = local_binaries_predictor



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
        elif actions.ndim == 4:
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
        else:
            raise ValueError(
                f"Expected actions shape (B, N, A) or (B, T, N, A), got {tuple(actions.shape)}"
            )

        total_loss = None

        if self.local_scalar_target_indices is not None:
            if self.local_scalars_predictor is None or self.scalar_loss_fn is None:
                raise ValueError("Scalar predictors and loss must be set when scalar targets are configured")
            pred_scalars = self.local_scalars_predictor(latent_preds)
            targ_scalars = next_local_obs[..., self.local_scalar_target_indices]
            total_loss = self._combine_losses(
                total_loss,
                self._compute_component_loss(
                    pred_scalars, targ_scalars, self.scalar_loss_fn, agent_mask=agent_mask, time_mask=time_mask
                ),
            )

        if self.local_angle_target_indices is not None:
            if self.local_angles_predictor is None:
                raise ValueError("Angle predictors must be set when angle targets are configured")
            pred_angles = self.local_angles_predictor(latent_preds)
            targ_angles = next_local_obs[..., self.local_angle_target_indices]
            targ_angles = torch.stack([torch.sin(targ_angles), torch.cos(targ_angles)], dim=-1)
            targ_angles = targ_angles.reshape(*targ_angles.shape[:-2], -1)
            angle_loss = self._compute_component_loss(
                pred_angles, targ_angles, self._mse_loss, agent_mask=agent_mask, time_mask=time_mask
            )
            total_loss = self._combine_losses(total_loss, angle_loss)

        if self.local_quat_target_indices is not None:
            if self.local_quats_predictor is None:
                raise ValueError("Quat predictors must be set when quat targets are configured")
            quat_indices = self._expand_quat_indices(self.local_quat_target_indices)
            pred_quats = self.local_quats_predictor(latent_preds)
            targ_quats = next_local_obs[..., quat_indices]
            quat_loss = self._compute_component_loss(
                pred_quats, targ_quats, self._mse_loss, agent_mask=agent_mask, time_mask=time_mask
            )
            total_loss = self._combine_losses(total_loss, quat_loss)

        if self.local_binary_target_indices is not None:
            if self.local_binaries_predictor is None or self.binary_loss_fn is None:
                raise ValueError("Binary predictors and loss must be set when binary targets are configured")
            pred_binaries = self.local_binaries_predictor(latent_preds)
            targ_binaries = next_local_obs[..., self.local_binary_target_indices]
            total_loss = self._combine_losses(
                total_loss,
                self._compute_component_loss(
                    pred_binaries, targ_binaries, self.binary_loss_fn, agent_mask=agent_mask, time_mask=time_mask
                ),
            )

        if total_loss is None:
            raise ValueError("No target indices configured for next-obs prediction loss")

        _ = next_global_obs
        return total_loss

    @staticmethod
    def _normalize_indices(indices: Optional[list[int]]) -> Optional[list[int]]:
        if indices is None:
            return None
        if len(indices) == 0:
            return None
        return list(indices)

    @staticmethod
    def _expand_quat_indices(indices: list[int]) -> list[int]:
        return [start + offset for start in indices for offset in range(4)]

    @staticmethod
    def _combine_losses(total: Optional[torch.Tensor], component: torch.Tensor) -> torch.Tensor:
        if total is None:
            return component
        return total + component

    @staticmethod
    def _mse_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(preds, targets, reduction="none")

    def _compute_component_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        agent_mask: torch.Tensor | None,
        time_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if predictions.shape != targets.shape:
            raise ValueError(f"Predictions/targets shape mismatch: {tuple(predictions.shape)} vs {tuple(targets.shape)}")
        if predictions.ndim not in {3, 4}:
            raise ValueError(f"Unsupported predictions/targets rank: {predictions.ndim}")

        loss_raw = loss_fn(predictions, targets)
        if loss_raw.shape == predictions.shape:
            loss_per_item = loss_raw.mean(dim=-1)
            valid = self._build_valid_mask(loss_per_item, agent_mask=agent_mask, time_mask=time_mask)
            return self._masked_mean(loss_per_item, valid)

        if loss_raw.ndim == predictions.ndim - 1 and loss_raw.shape == predictions.shape[:-1]:
            valid = self._build_valid_mask(loss_raw, agent_mask=agent_mask, time_mask=time_mask)
            return self._masked_mean(loss_raw, valid)

        valid = self._build_valid_mask_for_preds(predictions, agent_mask=agent_mask, time_mask=time_mask)
        if valid is None:
            if loss_raw.ndim > 0:
                return loss_raw.mean()
            return loss_raw

        preds_valid = predictions[valid]
        targets_valid = targets[valid]
        if preds_valid.numel() == 0:
            return predictions.new_tensor(0.0)
        masked_loss = loss_fn(preds_valid, targets_valid)
        if masked_loss.ndim > 0:
            return masked_loss.mean()
        return masked_loss

    @staticmethod
    def _build_valid_mask_for_preds(
        predictions: torch.Tensor,
        *,
        agent_mask: torch.Tensor | None,
        time_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if predictions.ndim == 3:
            loss_shape = predictions.shape[:2]
        elif predictions.ndim == 4:
            loss_shape = predictions.shape[:3]
        else:
            raise ValueError(f"Unsupported predictions rank: {predictions.ndim}")
        loss_per_item = predictions.new_zeros(loss_shape)
        return NextObsPredMixin._build_valid_mask(loss_per_item, agent_mask=agent_mask, time_mask=time_mask)

    @staticmethod
    def _build_valid_mask(
        loss_per_item: torch.Tensor,
        *,
        agent_mask: torch.Tensor | None,
        time_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if agent_mask is None and time_mask is None:
            return None

        valid = torch.ones_like(loss_per_item, dtype=torch.bool)

        if agent_mask is not None:
            if loss_per_item.ndim == 2:
                if agent_mask.ndim != 2 or agent_mask.shape != loss_per_item.shape:
                    raise ValueError(f"Expected agent_mask shape {tuple(loss_per_item.shape)}, got {tuple(agent_mask.shape)}")
                valid = valid & agent_mask
            elif loss_per_item.ndim == 3:
                b, t, n = loss_per_item.shape
                if agent_mask.ndim == 2:
                    if agent_mask.shape != (b, n):
                        raise ValueError(f"Expected agent_mask shape (B, N)=({b}, {n}), got {tuple(agent_mask.shape)}")
                    valid = valid & agent_mask[:, None, :]
                elif agent_mask.ndim == 3:
                    if agent_mask.shape != (b, t, n):
                        raise ValueError(f"Expected agent_mask shape (B, T, N)=({b}, {t}, {n}), got {tuple(agent_mask.shape)}")
                    valid = valid & agent_mask
                else:
                    raise ValueError(f"Expected agent_mask ndim 2 or 3, got {agent_mask.ndim}")
            else:
                raise ValueError(f"Unsupported loss rank: {loss_per_item.ndim}")

        if time_mask is not None:
            if loss_per_item.ndim != 3:
                raise ValueError("time_mask is only supported for multi-step loss (actions shape (B, T, N, A))")
            b, t, _n = loss_per_item.shape
            if time_mask.shape != (b, t) or time_mask.dtype != torch.bool:
                raise ValueError(
                    f"Expected time_mask shape (B, T)=({b}, {t}) and dtype bool, got {tuple(time_mask.shape)} / {time_mask.dtype}"
                )
            valid = valid & time_mask[:, :, None]

        return valid

    @staticmethod
    def _masked_mean(loss_per_item: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if valid is None:
            return loss_per_item.mean()
        valid_f = valid.to(dtype=loss_per_item.dtype)
        denom = valid_f.sum().clamp_min(1.0)
        return (loss_per_item * valid_f).sum() / denom
