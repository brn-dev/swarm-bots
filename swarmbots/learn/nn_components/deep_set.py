from __future__ import annotations

from typing import Callable, Literal, TypedDict

import torch
from torch import nn

from swarmbots.learn.nn_components.mlp import MLP
from swarmbots.learn.nn_components.nn_init import LinearInitialization, init_linear_orthogonal

PoolMode = Literal["mean", "sum", "max"]


def _pool_over_set(x: torch.Tensor, *, set_dim: int, mode: PoolMode) -> torch.Tensor:
    if mode == "mean":
        return x.mean(dim=set_dim)
    if mode == "sum":
        return x.sum(dim=set_dim)
    if mode == "max":
        return x.max(dim=set_dim).values
    raise ValueError(f"Unknown pool mode: {mode}")


class DeepSet(nn.Module):
    def __init__(
        self,
        *,
        element_encoder: nn.Module,
        output_network: nn.Module,
        set_dim: int = 1,
        pool_mode: PoolMode = "mean",
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.element_encoder = element_encoder
        self.output_network = output_network
        self.set_dim = int(set_dim)
        self.pool_mode: PoolMode = pool_mode
        self.context_dim = int(context_dim)

    def forward(self, elements: torch.Tensor, *, context: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.element_encoder(elements)
        pooled = _pool_over_set(encoded, set_dim=self.set_dim, mode=self.pool_mode)

        if self.context_dim > 0:
            if context is None:
                raise ValueError("context must be provided when context_dim > 0")
            pooled = torch.cat((pooled, context), dim=-1)

        return self.output_network(pooled)


class DeepSetCriticHiddenDims(TypedDict):
    local_projection_hidden_dims: list[int]
    value_regressor_hidden_dims: list[int]


class DeepSetCritic(nn.Module):
    """
    Convenience wrapper for a scalar-value DeepSet critic.
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
        act_fn_cls: Callable[[], nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.num_local_features = int(num_local_features)
        self.num_global_features = int(num_global_features)

        if local_projection_hidden_dims:
            element_encoder: nn.Module = MLP(
                input_dim=self.num_local_features,
                hidden_dims=local_projection_hidden_dims,
                end_with_act_fn=False,
                linear_init=linear_init,
                act_fn_cls=act_fn_cls,
            )
            pooled_dim = int(local_projection_hidden_dims[-1])
        else:
            element_encoder = nn.Identity()
            pooled_dim = self.num_local_features

        value_regressor = MLP(
            input_dim=pooled_dim + self.num_global_features,
            hidden_dims=[*value_regressor_hidden_dims, 1],
            end_with_act_fn=False,
            linear_init=linear_init,
            act_fn_cls=act_fn_cls,
        )

        self.deepset = DeepSet(
            element_encoder=element_encoder,
            output_network=value_regressor,
            set_dim=set_dim,
            pool_mode=pool_mode,
            context_dim=self.num_global_features,
        )

    def forward(self, local_features: torch.Tensor, global_features: torch.Tensor | None = None) -> torch.Tensor:
        context = global_features if self.num_global_features > 0 else None
        return self.deepset(local_features, context=context).squeeze(dim=-1)