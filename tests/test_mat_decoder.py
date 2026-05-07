from collections.abc import Callable

import torch

from swarmbots.learn.algos.mat.mat_decoder import MATDecoder, MATDecoderConfig, MATDecoderSelfAttentionMode


def _make_decoder(self_attention_mode: MATDecoderSelfAttentionMode) -> MATDecoder:
    return MATDecoder(
        config=MATDecoderConfig(
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            self_attention_mode=self_attention_mode,
        ),
        max_agents=4,
        memory_d_model=8,
    )


def _record_causal_hints(
        monkeypatch,
) -> list[bool | None]:
    from torch.nn.modules import transformer

    original_detect_is_causal_mask: Callable[..., bool] = transformer._detect_is_causal_mask
    causal_hints: list[bool | None] = []

    def record_detect_is_causal_mask(
            mask: torch.Tensor | None,
            is_causal: bool | None = None,
            size: int | None = None,
    ) -> bool:
        causal_hints.append(is_causal)
        return original_detect_is_causal_mask(mask, is_causal, size)

    monkeypatch.setattr(transformer, "_detect_is_causal_mask", record_detect_is_causal_mask)
    return causal_hints


def test_parallel_decoder_passes_explicit_causal_hint(monkeypatch) -> None:
    causal_hints = _record_causal_hints(monkeypatch)
    query_tokens = torch.randn(2, 4, 8)
    context_tokens = torch.randn(2, 4, 8)
    memory_tokens = torch.randn(2, 4, 8)
    agent_mask = torch.ones(2, 4, dtype=torch.bool)

    for self_attention_mode, expected_hint in (
            (MATDecoderSelfAttentionMode.FULL_AUTOREGRESSIVE, True),
            (MATDecoderSelfAttentionMode.PREVIOUS_AGENTS, False),
            (MATDecoderSelfAttentionMode.CONTEXT_TOKENS_ONLY, False),
    ):
        decoder = _make_decoder(self_attention_mode)
        decoder(
            query_tokens=query_tokens,
            context_tokens=context_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )
        assert causal_hints[-1] is expected_hint


def test_step_decoder_passes_explicit_causal_hint(monkeypatch) -> None:
    causal_hints = _record_causal_hints(monkeypatch)
    context_tokens = torch.randn(2, 2, 8)
    query_prefix_tokens = torch.randn(2, 2, 8)
    query_token = torch.randn(2, 1, 8)
    memory_tokens = torch.randn(2, 4, 8)
    agent_mask = torch.ones(2, 4, dtype=torch.bool)

    for self_attention_mode, expected_hint in (
            (MATDecoderSelfAttentionMode.FULL_AUTOREGRESSIVE, True),
            (MATDecoderSelfAttentionMode.PREVIOUS_AGENTS, False),
            (MATDecoderSelfAttentionMode.CONTEXT_TOKENS_ONLY, True),
    ):
        decoder = _make_decoder(self_attention_mode)
        decoder.forward_step(
            context_tokens=context_tokens,
            query_token=query_token,
            memory_tokens=memory_tokens,
            query_prefix_tokens=query_prefix_tokens,
            context_mask=agent_mask[:, :2],
            query_prefix_mask=agent_mask[:, :2],
            query_mask=agent_mask[:, 2],
            memory_mask=agent_mask,
        )
        assert causal_hints[-1] is expected_hint
