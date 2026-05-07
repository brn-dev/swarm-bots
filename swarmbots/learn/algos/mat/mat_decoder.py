from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn

from swarmbots.learn.nn_components.custom_transformer_decoder_layer import CustomTransformerDecoderLayer


class MATDecoderSelfAttentionMode(Enum):
    FULL_AUTOREGRESSIVE = 1
    PREVIOUS_AGENTS = 2  # c_i can not attend to q_i, only to q_<i and c_<i
    CONTEXT_TOKENS_ONLY = 3  # only context tokens can be attended to



@dataclass(frozen=True)
class MATDecoderConfig:
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
    memory_dims: list[int] | None = None
    actor_head_hidden_dims: list[int] | None = None
    self_attention_mode: MATDecoderSelfAttentionMode = MATDecoderSelfAttentionMode.FULL_AUTOREGRESSIVE


class MATDecoder(nn.Module):

    def __init__(
            self,
            config: MATDecoderConfig,
            *,
            max_agents: int,
            memory_d_model: int,
    ) -> None:
        super().__init__()
        if config.d_model is None:
            raise ValueError("MATDecoderConfig.d_model must be set before constructing MATDecoder")
        self.max_agents = max_agents
        self.d_model = config.d_model
        self.memory_d_model = memory_d_model
        self.self_attention_mode = config.self_attention_mode

        self.decoder = nn.TransformerDecoder(
            decoder_layer=CustomTransformerDecoderLayer(
                d_model=self.d_model,
                memory_d_model=memory_d_model,
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
            norm=nn.LayerNorm(self.d_model),
        )

        self.register_buffer(
            "parallel_attention_mask",
            self._build_parallel_attention_mask(
                max_agents=self.max_agents,
                self_attention_mode=config.self_attention_mode,
            ),
        )
        self.parallel_attention_mask_is_causal = (
            config.self_attention_mode is MATDecoderSelfAttentionMode.FULL_AUTOREGRESSIVE
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
            self_attention_mode: MATDecoderSelfAttentionMode,
    ) -> torch.Tensor:
        if self_attention_mode is MATDecoderSelfAttentionMode.FULL_AUTOREGRESSIVE:
            seq_len = max_agents * 2
            return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

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

            if self_attention_mode is MATDecoderSelfAttentionMode.CONTEXT_TOKENS_ONLY:
                continue
            if self_attention_mode is MATDecoderSelfAttentionMode.PREVIOUS_AGENTS:
                past_queries = torch.arange(0, query_idx, 2)
                attention_mask[query_idx, past_queries] = False
                attention_mask[context_idx, past_queries] = False
                continue
            raise ValueError(f"Unsupported self_attention_mode: {self_attention_mode}")

        if self_attention_mode is MATDecoderSelfAttentionMode.PREVIOUS_AGENTS:
            context_rows = torch.arange(1, seq_len, 2)
            same_agent_query_cols = context_rows - 1
            attention_mask[context_rows, same_agent_query_cols] = True

        return attention_mask

    def forward(
            self,
            query_tokens: torch.Tensor,
            context_tokens: torch.Tensor,
            memory_tokens: torch.Tensor,
            agent_mask: torch.Tensor | None = None,
            memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_tokens.shape[1] > self.max_agents:
            raise ValueError(f"Expected memory_tokens second dim <= {self.max_agents}, got {memory_tokens.shape[1]}")
        n_agents = query_tokens.shape[1]

        interleaved_tokens = torch.stack((query_tokens, context_tokens), dim=2).reshape(
            query_tokens.shape[0], n_agents * 2, query_tokens.shape[-1]
        )
        attention_mask = self.parallel_attention_mask[: n_agents * 2, : n_agents * 2]

        target_key_padding_mask = None
        if agent_mask is not None:
            target_key_padding_mask = torch.stack((~agent_mask, ~agent_mask), dim=2).reshape(
                agent_mask.shape[0], n_agents * 2
            )

        memory_key_padding_mask = None
        if memory_mask is not None:
            memory_key_padding_mask = ~memory_mask

        decoder_output = self.decoder(
            tgt=interleaved_tokens,
            memory=memory_tokens,
            tgt_mask=attention_mask,
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=self.parallel_attention_mask_is_causal,
        )
        return decoder_output[:, 0::2, :]

    def forward_step(
            self,
            context_tokens: torch.Tensor,
            query_token: torch.Tensor,
            memory_tokens: torch.Tensor,
            *,
            query_prefix_tokens: torch.Tensor | None = None,
            context_mask: torch.Tensor | None = None,
            query_prefix_mask: torch.Tensor | None = None,
            query_mask: torch.Tensor | None = None,
            memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_tokens.shape[1] > self.max_agents:
            raise ValueError(
                f"Expected memory_tokens second dim <= {self.max_agents}, got {memory_tokens.shape[1]}"
            )

        use_interleaved_prefix = self.self_attention_mode is not MATDecoderSelfAttentionMode.CONTEXT_TOKENS_ONLY
        if use_interleaved_prefix:
            if query_prefix_tokens is None:
                raise ValueError(
                    f"query_prefix_tokens must be provided for self_attention_mode {self.self_attention_mode}"
                )
            sequence_prefix = torch.stack((query_prefix_tokens, context_tokens), dim=2).reshape(
                query_token.shape[0], query_prefix_tokens.shape[1] * 2, query_token.shape[2]
            )
            sequence = torch.cat((sequence_prefix, query_token), dim=1)
            seq_len = sequence.shape[1]
            attention_mask = self.parallel_attention_mask[:seq_len, :seq_len]
        else:
            sequence = torch.cat((context_tokens, query_token), dim=1)
            seq_len = sequence.shape[1]
            attention_mask = self.step_causal_mask[:seq_len, :seq_len]
        attention_mask_is_causal = self.parallel_attention_mask_is_causal or not use_interleaved_prefix

        target_key_padding_mask = None
        if context_mask is not None or query_mask is not None:
            if context_mask is None:
                context_mask = torch.ones(
                    context_tokens.shape[:2], dtype=torch.bool, device=context_tokens.device
                )

            if use_interleaved_prefix:
                if query_prefix_mask is None:
                    query_prefix_mask = torch.ones(
                        context_tokens.shape[:2], dtype=torch.bool, device=context_tokens.device
                    )

            if query_mask is None:
                query_mask = torch.ones(query_token.shape[0], dtype=torch.bool, device=query_token.device)

            if use_interleaved_prefix:
                interleaved_prefix_mask = torch.stack((query_prefix_mask, context_mask), dim=2).reshape(
                    context_mask.shape[0], context_mask.shape[1] * 2
                )
                combined_mask = torch.cat((interleaved_prefix_mask, query_mask.unsqueeze(1)), dim=1)
            else:
                combined_mask = torch.cat((context_mask, query_mask.unsqueeze(1)), dim=1)
            target_key_padding_mask = ~combined_mask

        memory_key_padding_mask = None
        if memory_mask is not None:
            memory_key_padding_mask = ~memory_mask

        decoder_output = self.decoder(
            tgt=sequence,
            memory=memory_tokens,
            tgt_mask=attention_mask,
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=attention_mask_is_causal,
        )
        return decoder_output[:, -1:, :]
