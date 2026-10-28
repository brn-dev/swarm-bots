from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal, TypedDict

import torch
from torch import nn

from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.feed_forward import MLP
from swarmbots.learn.nn_components.nn_init import (
    DEFAULT_ORTHOGONAL_GAIN,
    LinearInitialization,
    init_linear_orthogonal,
    make_init_linear_orthogonal,
)
from swarmbots.learn.nn_components.popart import PopArtLinear

PoolMode = Literal["mean", "sum", "max"]
LocalUpdate = Literal["residual", "linear", "mlp", "residual_mlp"]


@dataclass(frozen=True)
class DeepSetInteractionConfig:
    num_layers: int = 2
    hidden_dims: tuple[int, ...] | None = None
    message_dim: int | None = None
    gated: bool = False
    gate_hidden_dim: int | None = None
    gate_input_dim_cap: int | None = None
    local_update: LocalUpdate = "residual_mlp"
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError("Deep Set interaction num_layers must be positive")
        if self.hidden_dims is not None and any(width <= 0 for width in self.hidden_dims):
            raise ValueError("Deep Set interaction hidden dimensions must be positive")
        if self.message_dim is not None and self.message_dim <= 0:
            raise ValueError("Deep Set interaction message dimension must be positive")
        if self.gate_hidden_dim is not None and self.gate_hidden_dim <= 0:
            raise ValueError("Deep Set interaction gate hidden dimension must be positive")
        if self.gate_input_dim_cap is not None and self.gate_input_dim_cap <= 0:
            raise ValueError("Deep Set interaction gate input dimension cap must be positive")
        if self.local_update not in {"residual", "linear", "mlp", "residual_mlp"}:
            raise ValueError(f"Unknown Deep Set local update: {self.local_update!r}")


def mask_numeric_elements(
    elements: torch.Tensor, mask: torch.Tensor, *, fill_value: float = 0.0,
) -> torch.Tensor:
    """Mask final-axis features with a fill value valid for the receiving encoder."""
    return elements.masked_fill(~mask.unsqueeze(-1), fill_value)


def _pool_over_set(
    x: torch.Tensor,
    *,
    set_dim: int,
    mode: PoolMode,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is None:
        if mode == "mean":
            return x.mean(dim=set_dim)
        if mode == "sum":
            return x.sum(dim=set_dim)
        if mode == "max":
            return x.max(dim=set_dim).values
        raise ValueError(f"Unknown pool mode: {mode}")

    if mask.dtype != torch.bool:
        raise ValueError(f"Expected mask dtype bool, got {mask.dtype}")
    if mask.ndim != x.ndim - 1:
        raise ValueError(f"Expected mask ndim {x.ndim - 1}, got {mask.ndim}")
    if mask.shape != x.shape[:-1]:
        raise ValueError(f"Expected mask shape {x.shape[:-1]}, got {mask.shape}")
    x = mask_numeric_elements(x, mask)
    mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
    if mode == "sum":
        return (x * mask_f).sum(dim=set_dim)
    if mode == "mean":
        denom = mask_f.sum(dim=set_dim).clamp_min(1.0)
        return (x * mask_f).sum(dim=set_dim) / denom
    if mode == "max":
        neg_inf = torch.finfo(x.dtype).min
        x_masked = x.masked_fill(~mask.unsqueeze(-1), neg_inf)
        pooled = x_masked.max(dim=set_dim).values
        any_valid = mask.any(dim=set_dim % x.ndim)
        return torch.where(any_valid.unsqueeze(-1), pooled, torch.zeros_like(pooled))
    raise ValueError(f"Unknown pool mode: {mode}")


class DeepSetInteraction(nn.Module):
    """Pool nonlinear expanded features, then broadcast a learned global update."""

    def __init__(
        self,
        features: int,
        *,
        config: DeepSetInteractionConfig,
        set_dim: int,
        pool_mode: PoolMode,
        linear_init: LinearInitialization,
        act_fn_cls: ActivationFactory,
    ) -> None:
        super().__init__()
        self.features = features
        self.message_dim = max(1, features * 5 // 4) if config.message_dim is None else config.message_dim
        hidden_dims = (features,) if config.hidden_dims is None else config.hidden_dims
        self.config = replace(config, hidden_dims=hidden_dims, message_dim=self.message_dim)
        self.set_dim = set_dim
        self.pool_mode = pool_mode
        self.local_update = config.local_update
        self.global_norm = nn.LayerNorm(features) if config.normalize else nn.Identity()
        self.local_norm = (
            nn.LayerNorm(features) if config.normalize and config.local_update != "residual" else nn.Identity()
        )
        self.message_encoder = MLP(
            input_dim=features, hidden_dims=[self.message_dim], end_with_act_fn=True,
            linear_init=linear_init, act_fn_cls=act_fn_cls,
        )
        self.global_update = MLP(
            input_dim=self.message_dim, hidden_dims=[self.message_dim], end_with_act_fn=True,
            linear_init=linear_init, act_fn_cls=act_fn_cls,
        )
        self.message_projection = nn.Linear(self.message_dim, features, bias=False)
        linear_init(self.message_projection)
        if config.local_update == "residual":
            self.local_encoder: nn.Module = nn.Identity()
        elif config.local_update == "linear":
            self.local_encoder = MLP(
                input_dim=features, hidden_dims=[features], end_with_act_fn=True,
                linear_init=linear_init, act_fn_cls=act_fn_cls,
            )
        else:
            self.local_encoder = MLP(
                input_dim=features, hidden_dims=[*hidden_dims, features], end_with_act_fn=False,
                linear_init=linear_init, act_fn_cls=act_fn_cls,
            )

    def _broadcast_update(self, normalized: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        return self.message_projection(message).unsqueeze(self.set_dim)

    def message_update(
        self, normalized_elements: torch.Tensor, element_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the communication delta for an externally normalized residual block."""
        if element_mask is not None:
            normalized_elements = normalized_elements.masked_fill(~element_mask.unsqueeze(-1), 0.0)
        message = _pool_over_set(
            self.message_encoder(normalized_elements),
            set_dim=self.set_dim, mode=self.pool_mode, mask=element_mask,
        )
        return self._broadcast_update(normalized_elements, self.global_update(message))

    def forward(self, elements: torch.Tensor, element_mask: torch.Tensor | None = None) -> torch.Tensor:
        if element_mask is not None:
            elements = elements.masked_fill(~element_mask.unsqueeze(-1), 0.0)
        normalized = self.global_norm(elements)
        updated = elements + self.message_update(normalized, element_mask)
        if self.local_update != "residual":
            local = self.local_encoder(self.local_norm(updated))
            if self.local_update == "residual_mlp":
                updated = updated + local
            else:
                updated = local
        return updated if element_mask is None else updated.masked_fill(~element_mask.unsqueeze(-1), 0.0)


class GatedDeepSetInteraction(DeepSetInteraction):
    """Let each token select global message features before their shared projection."""

    def __init__(
        self,
        features: int,
        *,
        config: DeepSetInteractionConfig,
        set_dim: int,
        pool_mode: PoolMode,
        linear_init: LinearInitialization,
        act_fn_cls: ActivationFactory,
    ) -> None:
        super().__init__(
            features, config=config, set_dim=set_dim, pool_mode=pool_mode,
            linear_init=linear_init, act_fn_cls=act_fn_cls,
        )
        self.gate_hidden_dim = max(1, features // 4) if config.gate_hidden_dim is None else config.gate_hidden_dim
        self.gate_input_dim_cap = config.gate_input_dim_cap
        gate_input_features = (
            features + self.message_dim if self.gate_input_dim_cap is None
            else min(features, self.gate_input_dim_cap) + min(self.message_dim, self.gate_input_dim_cap)
        )
        self.config = replace(self.config, gated=True, gate_hidden_dim=self.gate_hidden_dim)
        self.reception_gate = nn.Sequential(
            MLP(
                input_dim=gate_input_features,
                hidden_dims=[self.gate_hidden_dim, self.message_dim], end_with_act_fn=False,
                linear_init=linear_init, act_fn_cls=act_fn_cls,
            ),
            nn.Sigmoid(),
        )

    def _broadcast_update(self, normalized: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        expanded_message = message.unsqueeze(self.set_dim).expand(*normalized.shape[:-1], self.message_dim)
        gate = self.reception_gate(torch.cat((
            normalized[..., :self.gate_input_dim_cap], expanded_message[..., :self.gate_input_dim_cap],
        ), dim=-1))
        return self.message_projection(gate * expanded_message)


class DeepSet(nn.Module):
    """Pool encoded tokens while allowing arbitrary custom element encoders.

    Custom encoder inputs remain unchanged unless ``input_masker`` is supplied.
    The masker receives assembled element/context inputs and the encoded-token
    mask; its layout and fill value must suit the encoder. Numeric builders opt
    into zero masking of final-axis features.
    """

    def __init__(
        self,
        *,
        element_encoder: nn.Module,
        set_decoder: nn.Module,
        set_dim: int = 1,
        pool_mode: PoolMode = "mean",
        context_features: int = 0,
        context_in_elements: bool = False,
        context_after_pool: bool | None = None,
        interactions: list[DeepSetInteraction] | None = None,
        input_masker: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.element_encoder = element_encoder
        self.set_decoder = set_decoder
        self.interactions = nn.ModuleList(interactions or [])
        self.input_masker = input_masker
        self.set_dim = int(set_dim)
        self.pool_mode: PoolMode = pool_mode
        self.context_features = int(context_features)
        self.context_in_elements = bool(context_in_elements)
        # Preserve the previous placement when the decoder path is unspecified.
        self.context_after_pool = (
            not self.context_in_elements if context_after_pool is None else bool(context_after_pool)
        )

    @classmethod
    def from_dimensions(
        cls,
        *,
        element_input_features: int,
        local_projection_hidden_dims: list[int],
        set_decoder: nn.Module,
        interaction_config: DeepSetInteractionConfig,
        linear_init: LinearInitialization = init_linear_orthogonal,
        act_fn_cls: ActivationFactory = nn.Tanh,
        set_dim: int = 1,
        pool_mode: PoolMode = "mean",
        context_features: int = 0,
        context_in_elements: bool = False,
        context_after_pool: bool | None = None,
    ) -> DeepSet:
        element_encoder: nn.Module = (
            MLP(
                input_dim=element_input_features, hidden_dims=local_projection_hidden_dims,
                end_with_act_fn=True, linear_init=linear_init, act_fn_cls=act_fn_cls,
            )
            if local_projection_hidden_dims else nn.Identity()
        )
        features = local_projection_hidden_dims[-1] if local_projection_hidden_dims else element_input_features
        interaction_cls = GatedDeepSetInteraction if interaction_config.gated else DeepSetInteraction
        return cls(
            element_encoder=element_encoder, set_decoder=set_decoder,
            set_dim=set_dim, pool_mode=pool_mode, context_features=context_features,
            context_in_elements=context_in_elements, context_after_pool=context_after_pool,
            input_masker=mask_numeric_elements,
            interactions=[
                interaction_cls(
                    features, config=interaction_config, set_dim=set_dim, pool_mode=pool_mode,
                    linear_init=linear_init, act_fn_cls=act_fn_cls,
                )
                for _ in range(interaction_config.num_layers)
            ],
        )

    def encode_elements(
        self, elements: torch.Tensor, *, context: torch.Tensor | None = None,
        element_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the final equivariant features, sharing all interaction masks with the readout."""
        if self.context_in_elements:
            expanded_context = context.unsqueeze(self.set_dim).expand(*elements.shape[:-1], context.shape[-1])
            elements = torch.cat((elements, expanded_context), dim=-1)
        if element_mask is not None and self.input_masker is not None:
            elements = self.input_masker(elements, element_mask)
        encoded = self.element_encoder(elements)
        for interaction in self.interactions:
            encoded = interaction(encoded, element_mask)
        return encoded

    def forward(
        self,
        elements: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        element_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_latents(elements, context=context, element_mask=element_mask)[0]

    def forward_with_latents(
        self,
        elements: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        element_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode_elements(elements, context=context, element_mask=element_mask)
        return self.decode_elements(encoded, context=context, element_mask=element_mask), encoded

    def decode_elements(
        self,
        encoded: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        element_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Read out already encoded elements, allowing multiple heads to share an encoder."""
        pooled = _pool_over_set(encoded, set_dim=self.set_dim, mode=self.pool_mode, mask=element_mask)

        if self.context_features > 0 and self.context_after_pool:
            pooled = torch.cat((pooled, context), dim=-1)

        return self.set_decoder(pooled)


class DeepSetCriticHiddenDims(TypedDict):
    local_projection_hidden_dims: list[int]
    value_regressor_hidden_dims: list[int]


@dataclass(frozen=True)
class DeepSetCriticConfig:
    local_projection_hidden_dims: list[int] = field(default_factory=list)
    value_regressor_hidden_dims: list[int] = field(default_factory=list)
    interaction_config: DeepSetInteractionConfig = field(default_factory=DeepSetInteractionConfig)


class DeepSetCritic(nn.Module):
    """
    Convenience wrapper for a scalar-value DeepSet critic.

    Context can enter the element encoder, the value regressor after pooling, or both.
    Leaving context_after_pool unset preserves the placement selected by context_in_elements.
    """

    def __init__(
        self,
        *,
        num_local_features: int,
        local_projection_hidden_dims: list[int],
        value_regressor_hidden_dims: list[int],
        num_global_features: int = 0,
        set_dim: int = 1,
        pool_mode: PoolMode = "mean",
        linear_init: LinearInitialization = init_linear_orthogonal,
        local_projection_linear_init_gain: float = DEFAULT_ORTHOGONAL_GAIN,
        value_regressor_linear_init_gain: float = DEFAULT_ORTHOGONAL_GAIN,
        value_head_linear_init_gain: float = 0.01,
        act_fn_cls: ActivationFactory = nn.Tanh,
        context_in_elements: bool = False,
        context_after_pool: bool | None = None,
        interaction_config: DeepSetInteractionConfig = DeepSetInteractionConfig(),
        use_popart: bool = False,
        popart_beta: float = 3e-4,
        popart_eps: float = 1e-5,
        popart_min_std: float = 1e-4,
        popart_init_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_local_features = int(num_local_features)
        self.num_global_features = int(num_global_features)
        self.context_in_elements = bool(context_in_elements)
        self.context_after_pool = (
            not self.context_in_elements if context_after_pool is None else bool(context_after_pool)
        )
        self.use_popart = bool(use_popart)

        if self.context_in_elements and self.num_global_features == 0:
            raise ValueError("context_in_elements requires num_global_features > 0")

        if linear_init is init_linear_orthogonal:
            local_projection_linear_init = make_init_linear_orthogonal(local_projection_linear_init_gain)
            value_regressor_linear_init = make_init_linear_orthogonal(value_regressor_linear_init_gain)
            value_head_linear_init = make_init_linear_orthogonal(value_head_linear_init_gain)
        else:
            local_projection_linear_init = linear_init
            value_regressor_linear_init = linear_init
            value_head_linear_init = linear_init

        element_input_features = self.num_local_features + (self.num_global_features if self.context_in_elements else 0)
        context_features_after_pool = self.num_global_features if self.context_after_pool else 0

        pooled_dim = int(local_projection_hidden_dims[-1]) if local_projection_hidden_dims else element_input_features

        value_regressor_input_dim = pooled_dim + context_features_after_pool
        if value_regressor_hidden_dims:
            value_regressor_trunk: nn.Module = MLP(
                input_dim=value_regressor_input_dim,
                hidden_dims=value_regressor_hidden_dims,
                end_with_act_fn=True,
                linear_init=value_regressor_linear_init,
                act_fn_cls=act_fn_cls,
            )
            value_regressor_head_input_dim = int(value_regressor_hidden_dims[-1])
        else:
            value_regressor_trunk = nn.Identity()
            value_regressor_head_input_dim = value_regressor_input_dim

        if self.use_popart:
            self.popart_head: PopArtLinear | None = PopArtLinear(
                in_features=value_regressor_head_input_dim,
                out_features=1,
                beta=popart_beta,
                eps=popart_eps,
                min_std=popart_min_std,
                init_sigma=popart_init_sigma,
                init_gain=value_head_linear_init_gain,
            )
            value_regressor_head: nn.Module = self.popart_head
        else:
            self.popart_head = None
            value_regressor_head = nn.Linear(value_regressor_head_input_dim, 1)
            value_head_linear_init(value_regressor_head)

        value_regressor = nn.Sequential(value_regressor_trunk, value_regressor_head)

        self.deepset = DeepSet.from_dimensions(
            element_input_features=element_input_features,
            local_projection_hidden_dims=local_projection_hidden_dims,
            set_decoder=value_regressor,
            set_dim=set_dim,
            pool_mode=pool_mode,
            context_features=self.num_global_features,
            context_in_elements=self.context_in_elements,
            context_after_pool=self.context_after_pool,
            interaction_config=interaction_config,
            linear_init=local_projection_linear_init,
            act_fn_cls=act_fn_cls,
        )

    def encode_elements(
        self,
        local_features: torch.Tensor,
        global_features: torch.Tensor | None = None,
        agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = global_features if self.num_global_features > 0 else None
        return self.deepset.encode_elements(local_features, context=context, element_mask=agent_mask)

    def forward(
        self,
        local_features: torch.Tensor,
        global_features: torch.Tensor | None = None,
        agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_latents(local_features, global_features, agent_mask)[0]

    def forward_with_latents(
        self,
        local_features: torch.Tensor,
        global_features: torch.Tensor | None = None,
        agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = global_features if self.num_global_features > 0 else None
        values, latents = self.deepset.forward_with_latents(
            local_features, context=context, element_mask=agent_mask,
        )
        return values.squeeze(dim=-1), latents

    @property
    def has_popart(self) -> bool:
        return self.popart_head is not None

    def update_popart(self, targets: torch.Tensor) -> None:
        if self.popart_head is None:
            raise RuntimeError("PopArt is not enabled for this DeepSetCritic")
        self.popart_head.update(targets)

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        if self.popart_head is None:
            return values
        return self.popart_head.normalize(values)

    def get_popart_metrics(self) -> dict[str, float]:
        if self.popart_head is None:
            return {}
        sigma = self.popart_head.sigma()
        return {
            "popart_mu": self.popart_head.mu.mean().item(),
            "popart_sigma": sigma.mean().item(),
        }
