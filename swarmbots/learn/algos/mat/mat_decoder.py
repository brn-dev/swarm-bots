from dataclasses import dataclass

import torch
from torch import nn

from swarmbots.learn.nn_components.custom_transformer_decoder_layer import CustomTransformerDecoderLayer


@dataclass(frozen=True)
class MATDecoderConfig:
    d_model: int | None = None
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.0
    cross_attn_first: bool = False
    act_fn_cls: type[nn.Module] = nn.ReLU
    norm_first: bool = True
    layer_norm_eps: float = 1e-5
    bias: bool = True
    add_agent_embeddings: bool = True
    action_encoder_hidden_dims: list[int] | None = None
    actor_head_hidden_dims: list[int] | None = None


class MATDecoder(nn.Module):

    def __init__(
            self,
            config: MATDecoderConfig,
            *,
            max_agents: int,
            d_model: int,
            memory_d_model: int,
    ) -> None:
        super().__init__()
        self.max_agents = max_agents

        self.decoder = nn.TransformerDecoder(
            decoder_layer=CustomTransformerDecoderLayer(
                d_model=d_model,
                memory_d_model=memory_d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.act_fn_cls(),
                layer_norm_eps=config.layer_norm_eps,
                batch_first=True,
                norm_first=config.norm_first,
                cross_attn_first=config.cross_attn_first,
                bias=config.bias,
            ),
            num_layers=config.num_layers,
            norm=nn.LayerNorm(d_model),
        )


        causal_mask = torch.triu(torch.ones(self.max_agents, self.max_agents, dtype=torch.bool), diagonal=1)
        self.register_buffer("tgt_mask", causal_mask)

    def forward(
        self,
        action_embeddings: torch.Tensor,
        augmented_observations: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action_embeddings.shape[0] != augmented_observations.shape[0]:
            raise ValueError(
                "Expected local_embeddings and augmented_observations to share batch size, "
                f"got {action_embeddings.shape[0]} and {augmented_observations.shape[0]}"
            )
        n_agents = augmented_observations.shape[1]
        if n_agents > self.max_agents:
            raise ValueError(
                f"Expected augmented_observations second dim <= {self.max_agents}, got {n_agents}"
            )
        seq_len = action_embeddings.shape[1]
        if seq_len > n_agents:
            raise ValueError(f"Expected seq_len <= n_agents ({n_agents}), got {seq_len}")
        tgt_mask = self.tgt_mask[:seq_len, :seq_len]

        tgt_key_padding_mask = None
        memory_key_padding_mask = None
        if agent_mask is not None:
            if agent_mask.dtype != torch.bool:
                raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
            expected_mask_shape = (action_embeddings.shape[0], n_agents)
            if agent_mask.shape != expected_mask_shape:
                raise ValueError(
                    f"Expected agent_mask shape {expected_mask_shape}, got {tuple(agent_mask.shape)}"
                )
            tgt_key_padding_mask = ~agent_mask[:, :seq_len]
            memory_key_padding_mask = ~agent_mask

        decoder_output = self.decoder(
            tgt=action_embeddings,
            memory=augmented_observations,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return decoder_output
