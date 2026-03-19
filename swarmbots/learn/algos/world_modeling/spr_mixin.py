import abc
import copy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from swarmbots.learn.algos.world_modeling.transformer_transition_model import (
    TransformerTransitionModel,
    TransformerTransitionModelConfig,
    serialize_transformer_transition_model_config,
)
from swarmbots.learn.masking import build_valid_mask, masked_mean, restrict_loss_agent_mask
from swarmbots.learn.nn_components.residual import Residual
from swarmbots.learn.polyak_update import polyak_update


def serialize_spr_world_model_config(
        *,
        transition_model_config: TransformerTransitionModelConfig,
        spr_projection_dims: list[int],
        spr_predictor_hidden_dims: list[int],
        residual_predictor: bool,
        spr_loss_weight: float,
) -> dict[str, Any]:
    return {
        "transition_model_config": serialize_transformer_transition_model_config(transition_model_config),
        "spr_projection_dims": spr_projection_dims,
        "spr_predictor_hidden_dims": spr_predictor_hidden_dims,
        "residual_predictor": residual_predictor,
        "spr_loss_weight": spr_loss_weight,
    }


class SPRMixin(abc.ABC):

    target_encoder: nn.Module
    transition_model: TransformerTransitionModel
    online_projection: nn.Module
    target_projection: nn.Module
    predictor: nn.Module

    @property
    @abc.abstractmethod
    def online_encoder(self) -> nn.Module:
        raise NotImplementedError()

    def setup_spr(
            self,
            transition_model: TransformerTransitionModel,
            projection: nn.Module,
            predictor: nn.Module,
            residual_predictor: bool = True
    ) -> None:
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

        self.transition_model = transition_model

        self.online_projection = projection
        
        self.target_projection = copy.deepcopy(projection)
        self.target_projection.requires_grad_(False)
        self.target_projection.eval()

        if residual_predictor:
            self.predictor = Residual(predictor)
        else:
            self.predictor = predictor

    @torch.no_grad()
    def update_spr_targets(self, tau: float) -> None:
        polyak_update(source=self.online_encoder, target=self.target_encoder, tau=tau)
        polyak_update(source=self.online_projection, target=self.target_projection, tau=tau)

    def _cosine_similarity_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        *,
        agent_mask: torch.Tensor | None,
        time_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        cosine_sim = F.cosine_similarity(predictions, targets, dim=-1, eps=1e-8)
        loss_per_item = 1.0 - cosine_sim
        base_shape = loss_per_item.shape
        valid = build_valid_mask(
            base_shape=base_shape,
            device=loss_per_item.device,
            agent_mask=agent_mask,
            time_mask=time_mask,
        )
        return masked_mean(loss_per_item, valid)

    def compute_spr_loss(
            self,
            online_local_latents: torch.Tensor,
            next_local_obs: torch.Tensor,
            next_global_obs: torch.Tensor,
            actions: torch.Tensor,
            agent_mask: torch.Tensor | None = None,
            loss_agent_mask: torch.Tensor | None = None,
            time_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if online_local_latents.ndim != 3:
            raise ValueError(f"Expected online_local_latents shape (B, N, D), got {tuple(online_local_latents.shape)}")

        if actions.ndim == 3:
            online_next_latents = self.transition_model(online_local_latents, actions, agent_mask=agent_mask)
            online_next_projections = self.online_projection(online_next_latents)
            predictions = self.predictor(online_next_projections)
            effective_loss_agent_mask = restrict_loss_agent_mask(
                base_shape=predictions.shape[:2],
                loss_agent_mask=loss_agent_mask,
                agent_mask=agent_mask,
            )

            with torch.no_grad():
                self.target_encoder.eval()
                self.target_projection.eval()
                target_local_latents = self.target_encoder(
                    local_obs=next_local_obs,
                    global_obs=next_global_obs,
                    agent_mask=effective_loss_agent_mask,
                )
                target_projections = self.target_projection(target_local_latents)

            return self._cosine_similarity_loss(
                predictions,
                target_projections,
                agent_mask=effective_loss_agent_mask,
                time_mask=None,
            )

        if actions.ndim != 4:
            raise ValueError(
                f"Expected actions shape (B, N, A) or (B, T, N, A), got {tuple(actions.shape)}"
            )

        b, t, n, _a = actions.shape
        if online_local_latents.shape[0] != b or online_local_latents.shape[1] != n:
            raise ValueError(
                f"Expected online_local_latents shape (B, N, D)=({b}, {n}, D), got {tuple(online_local_latents.shape)}"
            )
        if next_local_obs.ndim != 4 or next_local_obs.shape[:3] != (b, t, n):
            raise ValueError(
                f"Expected local_obs shape (B, T, N, F)=({b}, {t}, {n}, F), got {tuple(next_local_obs.shape)}"
            )

        z_preds = self.transition_model.predict_n_steps(online_local_latents, actions, agent_mask=agent_mask)
        online_next_projections = self.online_projection(z_preds)
        predictions = self.predictor(online_next_projections)
        effective_loss_agent_mask = restrict_loss_agent_mask(
            base_shape=(b, t, n),
            loss_agent_mask=loss_agent_mask,
            agent_mask=agent_mask,
        )

        if next_global_obs.ndim == 2:
            next_global_obs = next_global_obs[:, None, :].expand(b, t, -1)
        if next_global_obs.ndim != 3 or next_global_obs.shape[:2] != (b, t):
            raise ValueError(f"Expected global_obs shape (B, G) or (B, T, G), got {tuple(next_global_obs.shape)}")

        local_flat = next_local_obs.reshape(b * t, n, -1)
        global_flat = next_global_obs.reshape(b * t, -1)

        with torch.no_grad():
            self.target_encoder.eval()
            self.target_projection.eval()
            if effective_loss_agent_mask is None:
                flat_agent_mask = None
            elif effective_loss_agent_mask.ndim == 2:
                flat_agent_mask = effective_loss_agent_mask[:, None, :].expand(b, t, n).reshape(b * t, n)
            elif effective_loss_agent_mask.ndim == 3:
                flat_agent_mask = effective_loss_agent_mask.reshape(b * t, n)
            else:
                raise ValueError(
                    f"Expected loss_agent_mask ndim 2 or 3, got {effective_loss_agent_mask.ndim}"
                )

            z_targets = self.target_encoder(
                local_obs=local_flat,
                global_obs=global_flat,
                agent_mask=flat_agent_mask,
            ).reshape(b, t, n, -1)
            target_projections = self.target_projection(z_targets)

        return self._cosine_similarity_loss(
            predictions,
            target_projections,
            agent_mask=effective_loss_agent_mask,
            time_mask=time_mask,
        )



