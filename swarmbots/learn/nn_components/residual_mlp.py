from dataclasses import dataclass

import torch
from torch import nn

from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.feed_forward import MLP
from swarmbots.learn.nn_components.nn_init import LinearInitialization, init_linear_orthogonal


@dataclass(frozen=True)
class ResidualMLPConfig:
    hidden_dim: int
    num_blocks: int


class ResidualMLPBlock(nn.Module):
    def __init__(
        self, width: int, hidden_dim: int, *, act_fn_cls: ActivationFactory,
        linear_init: LinearInitialization, output_linear_init: LinearInitialization,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.feedforward = MLP(
            input_dim=width, hidden_dims=[hidden_dim, width], end_with_act_fn=False,
            linear_init=linear_init, final_linear_init=output_linear_init, act_fn_cls=act_fn_cls,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.feedforward(self.norm(features))


class ResidualMLP(nn.Module):
    """A constant-width pre-norm residual stream with local two-layer MLP branches."""

    def __init__(
        self, input_dim: int, output_dim: int, config: ResidualMLPConfig, *,
        act_fn_cls: ActivationFactory = nn.GELU,
        linear_init: LinearInitialization = init_linear_orthogonal,
        output_linear_init: LinearInitialization | None = None,
    ) -> None:
        super().__init__()
        output_initialization = linear_init if output_linear_init is None else output_linear_init
        self.layers = nn.ModuleList(
            ResidualMLPBlock(
                input_dim, config.hidden_dim, act_fn_cls=act_fn_cls,
                linear_init=linear_init, output_linear_init=output_initialization,
            )
            for _ in range(config.num_blocks)
        )
        self.norm = nn.LayerNorm(input_dim)
        self.output_projection = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        if isinstance(self.output_projection, nn.Linear):
            output_initialization(self.output_projection)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            features = layer(features)
        return self.output_projection(self.norm(features))
