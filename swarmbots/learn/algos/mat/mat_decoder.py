from typing import Callable

import torch
from torch import nn

from swarmbots.learn.nn_components.custom_transformer_decoder_layer import CustomTransformerDecoderLayer


class MATDecoder(nn.Module):

    def __init__(
            self,
            n_agents: int,
            num_layers: int,
            bias: bool,
            norm_first: bool,
            cross_attn_first: bool,
            layer_norm_eps: float,
            activation: Callable[[torch.Tensor], torch.Tensor],
            dropout: float,
            dim_feedforward: int,
            nhead: int,
            d_model: int,
            memory_d_model: int | None,
            output_norm: nn.Module,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents

        self.decoder = nn.TransformerDecoder(
            decoder_layer=CustomTransformerDecoderLayer(
                d_model=d_model,
                memory_d_model=memory_d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=True,
                norm_first=norm_first,
                cross_attn_first=cross_attn_first,
                bias=bias,
            ),
            num_layers=num_layers,
            norm=output_norm,
        )


        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            self.n_agents
        )
        self.register_buffer("tgt_mask", tgt_mask)

    def forward(self, local_embeddings: torch.Tensor, augmented_observations: torch.Tensor) -> torch.Tensor:
        decoder_output = self.decoder(
            tgt=local_embeddings,
            memory=augmented_observations,
            tgt_mask=self.tgt_mask
        )
        return decoder_output
