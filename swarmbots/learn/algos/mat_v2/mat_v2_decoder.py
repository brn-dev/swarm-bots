from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MATv2DecoderConfig:
    d_model: int | None = None
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.0
    act_fn_cls: type[nn.Module] = nn.ReLU
    norm_first: bool = True
    layer_norm_eps: float = 1e-5
    bias: bool = True
    add_agent_embeddings: bool = True
    query_encoder_hidden_dims: list[int] | None = None
    context_encoder_hidden_dims: list[int] | None = None
    actor_head_hidden_dims: list[int] | None = None


class MATv2Decoder(nn.Module):

    def __init__(
            self,
            config: MATv2DecoderConfig,
            *,
            max_agents: int,
            d_model: int,
    ) -> None:
        super().__init__()
        self.max_agents = max_agents

        self.decoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.act_fn_cls(),
                layer_norm_eps=config.layer_norm_eps,
                batch_first=True,
                norm_first=config.norm_first,
                bias=config.bias,
            ),
            num_layers=config.num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=not config.norm_first,
        )

        self.register_buffer(
            "parallel_attention_mask",
            self._build_parallel_attention_mask(max_agents=self.max_agents),
        )
        causal_mask = torch.triu(
            torch.ones(self.max_agents + 1, self.max_agents + 1, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("step_causal_mask", causal_mask)

    @staticmethod
    def _build_parallel_attention_mask(
            *,
            max_agents: int,
    ) -> torch.Tensor:
        seq_len = max_agents * 2
        attention_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        for agent_idx in range(max_agents):
            query_idx = agent_idx * 2
            context_idx = query_idx + 1

            attention_mask[query_idx, query_idx] = False
            attention_mask[context_idx, context_idx] = False
            if agent_idx == 0:
                continue

            past_contexts = torch.arange(1, query_idx, 2)
            attention_mask[query_idx, past_contexts] = False
            attention_mask[context_idx, past_contexts] = False
        return attention_mask

    def forward(
            self,
            query_tokens: torch.Tensor,
            context_tokens: torch.Tensor,
            agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_tokens.shape != context_tokens.shape:
            raise ValueError(
                "Expected query_tokens and context_tokens to have the same shape, "
                f"got {tuple(query_tokens.shape)} and {tuple(context_tokens.shape)}"
            )
        n_agents = query_tokens.shape[1]
        if n_agents > self.max_agents:
            raise ValueError(f"Expected query_tokens second dim <= {self.max_agents}, got {n_agents}")

        interleaved_tokens = torch.stack((query_tokens, context_tokens), dim=2).reshape(
            query_tokens.shape[0], n_agents * 2, query_tokens.shape[-1]
        )
        attention_mask = self.parallel_attention_mask[: n_agents * 2, : n_agents * 2]

        key_padding_mask = None
        if agent_mask is not None:
            if agent_mask.dtype != torch.bool:
                raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
            expected_mask_shape = query_tokens.shape[:2]
            if tuple(agent_mask.shape) != tuple(expected_mask_shape):
                raise ValueError(
                    f"Expected agent_mask shape {tuple(expected_mask_shape)}, got {tuple(agent_mask.shape)}"
                )
            key_padding_mask = torch.stack((~agent_mask, ~agent_mask), dim=2).reshape(
                agent_mask.shape[0], n_agents * 2
            )

        decoder_output = self.decoder(
            src=interleaved_tokens,
            mask=attention_mask,
            src_key_padding_mask=key_padding_mask,
        )
        return decoder_output[:, 0::2, :]

    def forward_step(
            self,
            context_tokens: torch.Tensor,
            query_token: torch.Tensor,
            *,
            context_mask: torch.Tensor | None = None,
            query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_token.ndim != 3 or query_token.shape[1] != 1:
            raise ValueError(f"Expected query_token shape (B, 1, D), got {tuple(query_token.shape)}")
        if context_tokens.ndim != 3:
            raise ValueError(f"Expected context_tokens shape (B, T, D), got {tuple(context_tokens.shape)}")
        if context_tokens.shape[0] != query_token.shape[0]:
            raise ValueError(
                "Expected context_tokens and query_token to share batch size, "
                f"got {context_tokens.shape[0]} and {query_token.shape[0]}"
            )
        if context_tokens.shape[2] != query_token.shape[2]:
            raise ValueError(
                "Expected context_tokens and query_token to share hidden size, "
                f"got {context_tokens.shape[2]} and {query_token.shape[2]}"
            )
        if context_tokens.shape[1] > self.max_agents:
            raise ValueError(
                f"Expected context_tokens second dim <= {self.max_agents}, got {context_tokens.shape[1]}"
            )

        sequence = torch.cat((context_tokens, query_token), dim=1)
        seq_len = sequence.shape[1]
        attention_mask = self.step_causal_mask[:seq_len, :seq_len]

        key_padding_mask = None
        if context_mask is not None or query_mask is not None:
            if context_mask is None:
                context_mask = torch.ones(
                    context_tokens.shape[:2], dtype=torch.bool, device=context_tokens.device
                )
            else:
                if context_mask.dtype != torch.bool:
                    raise ValueError(f"Expected context_mask dtype bool, got {context_mask.dtype}")
                if tuple(context_mask.shape) != tuple(context_tokens.shape[:2]):
                    raise ValueError(
                        f"Expected context_mask shape {tuple(context_tokens.shape[:2])}, "
                        f"got {tuple(context_mask.shape)}"
                    )

            if query_mask is None:
                query_mask = torch.ones(query_token.shape[0], dtype=torch.bool, device=query_token.device)
            else:
                if query_mask.dtype != torch.bool:
                    raise ValueError(f"Expected query_mask dtype bool, got {query_mask.dtype}")
                if tuple(query_mask.shape) != (query_token.shape[0],):
                    raise ValueError(
                        f"Expected query_mask shape ({query_token.shape[0]},), got {tuple(query_mask.shape)}"
                    )

            combined_mask = torch.cat((context_mask, query_mask.unsqueeze(1)), dim=1)
            key_padding_mask = ~combined_mask

        decoder_output = self.decoder(
            src=sequence,
            mask=attention_mask,
            src_key_padding_mask=key_padding_mask,
        )
        return decoder_output[:, -1:, :]
