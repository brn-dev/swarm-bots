from typing import Callable

import torch
from torch import nn


class MATEncoder(nn.Module):

    def __init__(
            self,
            n_agents: int,
            num_layers: int,
            bias: bool,
            norm_first: bool,
            layer_norm_eps: float,
            activation: Callable[[torch.Tensor], torch.Tensor],
            dropout: float,
            dim_feedforward: int,
            nhead: int,
            d_model: int,
            output_norm: nn.Module,
            enable_nested_tensor: bool = True,
    ):
        super().__init__()
        self.n_agents = n_agents

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=True,
                norm_first=norm_first,
                bias=bias,
            ),
            num_layers=num_layers,
            norm=output_norm,
            enable_nested_tensor=enable_nested_tensor,
        )

    def forward(self, local_embeddings: torch.Tensor):
        augmented_observations = self.encoder(local_embeddings)
        return augmented_observations
