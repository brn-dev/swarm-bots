import abc
import copy

import torch
from torch import nn
from torch.nn import functional as F

from swarmbots.learn.algos.world_modeling.transformer_transition_model import TransformerTransitionModel
from swarmbots.learn.polyak_update import polyak_update


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

    def setup_modules(
            self,
            transition_model: TransformerTransitionModel,
            projection: nn.Module,
            predictor: nn.Module
    ):
        super().__init__()
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

        self.transition_model = transition_model

        self.online_projection = projection
        
        self.target_projection = copy.deepcopy(projection)
        self.target_projection.requires_grad_(False)
        self.target_projection.eval()

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
        pred_norm = F.normalize(predictions, dim=-1)
        targ_norm = F.normalize(targets, dim=-1)
        cosine_sim = (pred_norm * targ_norm).sum(dim=-1)
        loss_per_item = 1.0 - cosine_sim

        if agent_mask is None and time_mask is None:
            return loss_per_item.mean()

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
                raise ValueError(f"Unsupported predictions/targets rank: {loss_per_item.ndim}")

        if time_mask is not None:
            if loss_per_item.ndim != 3:
                raise ValueError("time_mask is only supported for multi-step loss (actions shape (B, T, N, A))")
            b, t, _n = loss_per_item.shape
            if time_mask.shape != (b, t) or time_mask.dtype != torch.bool:
                raise ValueError(
                    f"Expected time_mask shape (B, T)=({b}, {t}) and dtype bool, got {tuple(time_mask.shape)} / {time_mask.dtype}"
                )
            valid = valid & time_mask[:, :, None]

        valid_f = valid.to(dtype=loss_per_item.dtype)
        denom = valid_f.sum().clamp_min(1.0)
        return (loss_per_item * valid_f).sum() / denom

    def compute_spr_loss(
            self,
            online_local_latents: torch.Tensor,
            next_local_obs: torch.Tensor,
            next_global_obs: torch.Tensor,
            actions: torch.Tensor,
            agent_mask: torch.Tensor | None = None,
            time_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if online_local_latents.ndim != 3:
            raise ValueError(f"Expected online_local_latents shape (B, N, D), got {tuple(online_local_latents.shape)}")

        if actions.ndim == 3:
            online_next_latents = self.transition_model(online_local_latents, actions, agent_mask=agent_mask)
            online_next_projections = self.online_projection(online_next_latents)
            predictions = self.predictor(online_next_projections)

            with torch.no_grad():
                target_local_latents = self.target_encoder(local_obs=next_local_obs, global_obs=next_global_obs)
                target_projections = self.target_projection(target_local_latents)

            return self._cosine_similarity_loss(
                predictions,
                target_projections,
                agent_mask=agent_mask,
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

        if next_global_obs.ndim == 2:
            next_global_obs = next_global_obs[:, None, :].expand(b, t, -1)
        if next_global_obs.ndim != 3 or next_global_obs.shape[:2] != (b, t):
            raise ValueError(f"Expected global_obs shape (B, G) or (B, T, G), got {tuple(next_global_obs.shape)}")

        local_flat = next_local_obs.reshape(b * t, n, -1)
        global_flat = next_global_obs.reshape(b * t, -1)

        with torch.no_grad():
            z_targets = self.target_encoder(local_obs=local_flat, global_obs=global_flat).reshape(b, t, n, -1)
            target_projections = self.target_projection(z_targets)

        return self._cosine_similarity_loss(
            predictions,
            target_projections,
            agent_mask=agent_mask,
            time_mask=time_mask,
        )



