from collections.abc import Callable

import torch

from swarmbots.learn.algos.mat_qcs.mat_qcs_decoder import MATQCSDecoder, MATQCSDecoderConfig, MATQCSDecoderSelfAttentionMode


def _make_decoder(
        self_attention_mode: MATQCSDecoderSelfAttentionMode,
        *,
        assume_agent_mask_is_active_prefix: bool = True,
) -> MATQCSDecoder:
    return MATQCSDecoder(
        config=MATQCSDecoderConfig(
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            self_attention_mode=self_attention_mode,
            assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
        ),
        max_agents=4,
        memory_d_model=8,
    )


def _make_decoder_pair(
        self_attention_mode: MATQCSDecoderSelfAttentionMode,
) -> tuple[MATQCSDecoder, MATQCSDecoder]:
    cheap_decoder = _make_decoder(
        self_attention_mode,
        assume_agent_mask_is_active_prefix=True,
    )
    arbitrary_decoder = _make_decoder(
        self_attention_mode,
        assume_agent_mask_is_active_prefix=False,
    )
    arbitrary_decoder.load_state_dict(cheap_decoder.state_dict())
    cheap_decoder.eval()
    arbitrary_decoder.eval()
    return cheap_decoder, arbitrary_decoder


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
            (MATQCSDecoderSelfAttentionMode.FULL_AUTOREGRESSIVE, True),
            (MATQCSDecoderSelfAttentionMode.PREVIOUS_AGENTS, False),
            (MATQCSDecoderSelfAttentionMode.CONTEXT_TOKENS_ONLY, False),
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
            (MATQCSDecoderSelfAttentionMode.FULL_AUTOREGRESSIVE, True),
            (MATQCSDecoderSelfAttentionMode.PREVIOUS_AGENTS, False),
            (MATQCSDecoderSelfAttentionMode.CONTEXT_TOKENS_ONLY, True),
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


def test_arbitrary_mask_parallel_decoder_avoids_inactive_slot_zero_nan() -> None:
    decoder = _make_decoder(
        MATQCSDecoderSelfAttentionMode.FULL_AUTOREGRESSIVE,
        assume_agent_mask_is_active_prefix=False,
    )
    query_tokens = torch.randn(2, 4, 8)
    context_tokens = torch.randn(2, 4, 8)
    memory_tokens = torch.randn(2, 4, 8)
    agent_mask = torch.tensor(
        [
            [False, True, True, False],
            [False, False, True, True],
        ]
    )

    output = decoder(
        query_tokens=query_tokens,
        context_tokens=context_tokens,
        memory_tokens=memory_tokens,
        agent_mask=agent_mask,
        memory_mask=agent_mask,
    )

    assert torch.isfinite(output).all()


def test_prefix_mask_parallel_decoder_uses_cheap_causal_mask(monkeypatch) -> None:
    causal_hints = _record_causal_hints(monkeypatch)
    decoder = _make_decoder(
        MATQCSDecoderSelfAttentionMode.FULL_AUTOREGRESSIVE,
        assume_agent_mask_is_active_prefix=True,
    )
    query_tokens = torch.randn(2, 4, 8)
    context_tokens = torch.randn(2, 4, 8)
    memory_tokens = torch.randn(2, 4, 8)
    agent_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )

    decoder(
        query_tokens=query_tokens,
        context_tokens=context_tokens,
        memory_tokens=memory_tokens,
        agent_mask=agent_mask,
        memory_mask=agent_mask,
    )

    assert causal_hints[-1] is True


def test_arbitrary_parallel_decoder_matches_cheap_decoder_for_prefix_masks() -> None:
    torch.manual_seed(0)
    query_tokens = torch.randn(3, 4, 8)
    context_tokens = torch.randn(3, 4, 8)
    memory_tokens = torch.randn(3, 4, 8)
    agent_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )

    for self_attention_mode in MATQCSDecoderSelfAttentionMode:
        cheap_decoder, arbitrary_decoder = _make_decoder_pair(self_attention_mode)
        with torch.no_grad():
            cheap_output = cheap_decoder(
                query_tokens=query_tokens,
                context_tokens=context_tokens,
                memory_tokens=memory_tokens,
                agent_mask=agent_mask,
                memory_mask=agent_mask,
            )
            arbitrary_output = arbitrary_decoder(
                query_tokens=query_tokens,
                context_tokens=context_tokens,
                memory_tokens=memory_tokens,
                agent_mask=agent_mask,
                memory_mask=agent_mask,
            )

        torch.testing.assert_close(arbitrary_output, cheap_output, rtol=0.0, atol=1e-6)


def test_arbitrary_step_decoder_matches_cheap_decoder_for_prefix_masks() -> None:
    torch.manual_seed(1)
    context_tokens = torch.randn(3, 2, 8)
    query_prefix_tokens = torch.randn(3, 2, 8)
    query_token = torch.randn(3, 1, 8)
    memory_tokens = torch.randn(3, 4, 8)
    agent_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )

    for self_attention_mode in MATQCSDecoderSelfAttentionMode:
        cheap_decoder, arbitrary_decoder = _make_decoder_pair(self_attention_mode)
        with torch.no_grad():
            cheap_output = cheap_decoder.forward_step(
                context_tokens=context_tokens,
                query_token=query_token,
                memory_tokens=memory_tokens,
                query_prefix_tokens=query_prefix_tokens,
                context_mask=agent_mask[:, :2],
                query_prefix_mask=agent_mask[:, :2],
                query_mask=agent_mask[:, 2],
                memory_mask=agent_mask,
            )
            arbitrary_output = arbitrary_decoder.forward_step(
                context_tokens=context_tokens,
                query_token=query_token,
                memory_tokens=memory_tokens,
                query_prefix_tokens=query_prefix_tokens,
                context_mask=agent_mask[:, :2],
                query_prefix_mask=agent_mask[:, :2],
                query_mask=agent_mask[:, 2],
                memory_mask=agent_mask,
            )

        torch.testing.assert_close(arbitrary_output, cheap_output, rtol=0.0, atol=1e-6)


def test_arbitrary_step_decoder_avoids_inactive_current_query_nan() -> None:
    torch.manual_seed(2)
    context_tokens = torch.empty(2, 0, 8)
    query_prefix_tokens = torch.empty(2, 0, 8)
    query_token = torch.randn(2, 1, 8)
    memory_tokens = torch.randn(2, 4, 8)
    memory_mask = torch.tensor(
        [
            [False, True, True, False],
            [False, False, True, True],
        ]
    )
    query_mask = torch.tensor([False, False])

    for self_attention_mode in MATQCSDecoderSelfAttentionMode:
        decoder = _make_decoder(
            self_attention_mode,
            assume_agent_mask_is_active_prefix=False,
        )
        output = decoder.forward_step(
            context_tokens=context_tokens,
            query_token=query_token,
            memory_tokens=memory_tokens,
            query_prefix_tokens=query_prefix_tokens,
            context_mask=torch.empty(2, 0, dtype=torch.bool),
            query_prefix_mask=torch.empty(2, 0, dtype=torch.bool),
            query_mask=query_mask,
            memory_mask=memory_mask,
        )

        assert torch.isfinite(output).all()
