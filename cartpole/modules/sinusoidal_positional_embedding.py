import math

import torch
from torch import nn


class SinusoidalPositionalEmbedding(nn.Module):

    pos_embedding: torch.Tensor


    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        pos_embedding = torch.zeros(max_len, d_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000.0) / d_model)

        pos_embedding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_embedding[:, 1::2] = torch.cos(positions_list * division_term)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        embedding = self.pos_embedding[positions]
        return embedding
