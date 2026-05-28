from typing import Any, cast

import torch

from swarmbots.learn.algos.mat_qcc.mat_qcc_decoder import MATQCCDecoder, MATQCCDecoderConfig


def _make_decoder(*, assume_agent_mask_is_active_prefix: bool = True) -> MATQCCDecoder:
    return MATQCCDecoder(
        config=MATQCCDecoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
        ),
        max_agents=4,
        memory_d_model=8,
    )


def test_parallel_outputs_match_step_outputs() -> None:
    torch.manual_seed(0)
    decoder = _make_decoder()
    decoder.eval()

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

    with torch.no_grad():
        parallel_output = decoder(
            query_tokens=query_tokens,
            context_tokens=context_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )
        step_output = torch.cat(
            [
                decoder.forward_step(
                    context_tokens=context_tokens[:, :agent_idx, :],
                    query_token=query_tokens[:, agent_idx:agent_idx + 1, :],
                    memory_tokens=memory_tokens,
                    context_mask=agent_mask[:, :agent_idx],
                    query_mask=agent_mask[:, agent_idx],
                    memory_mask=agent_mask,
                )
                for agent_idx in range(query_tokens.shape[1])
            ],
            dim=1,
        )

    torch.testing.assert_close(step_output, parallel_output, rtol=0.0, atol=1e-6)


def test_parallel_outputs_match_step_outputs_for_arbitrary_masks() -> None:
    torch.manual_seed(1)
    decoder = _make_decoder(assume_agent_mask_is_active_prefix=False)
    decoder.eval()

    query_tokens = torch.randn(3, 4, 8)
    context_tokens = torch.randn(3, 4, 8)
    memory_tokens = torch.randn(3, 4, 8)
    agent_mask = torch.tensor(
        [
            [False, True, True, False],
            [True, False, True, True],
            [False, False, True, True],
        ]
    )

    with torch.no_grad():
        parallel_output = decoder(
            query_tokens=query_tokens,
            context_tokens=context_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )
        step_output = torch.cat(
            [
                decoder.forward_step(
                    context_tokens=context_tokens[:, :agent_idx, :],
                    query_token=query_tokens[:, agent_idx:agent_idx + 1, :],
                    memory_tokens=memory_tokens,
                    context_mask=agent_mask[:, :agent_idx],
                    query_mask=agent_mask[:, agent_idx],
                    memory_mask=agent_mask,
                )
                for agent_idx in range(query_tokens.shape[1])
            ],
            dim=1,
        )

    torch.testing.assert_close(step_output, parallel_output, rtol=0.0, atol=1e-6)


def test_prefix_fast_path_matches_arbitrary_mask_path_for_prefix_masks() -> None:
    torch.manual_seed(2)
    prefix_decoder = _make_decoder(assume_agent_mask_is_active_prefix=True)
    arbitrary_decoder = _make_decoder(assume_agent_mask_is_active_prefix=False)
    arbitrary_decoder.load_state_dict(prefix_decoder.state_dict())
    prefix_decoder.eval()
    arbitrary_decoder.eval()

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

    with torch.no_grad():
        prefix_parallel_output = prefix_decoder(
            query_tokens=query_tokens,
            context_tokens=context_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )
        arbitrary_parallel_output = arbitrary_decoder(
            query_tokens=query_tokens,
            context_tokens=context_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )
        prefix_step_output = torch.cat(
            [
                prefix_decoder.forward_step(
                    context_tokens=context_tokens[:, :agent_idx, :],
                    query_token=query_tokens[:, agent_idx:agent_idx + 1, :],
                    memory_tokens=memory_tokens,
                    context_mask=agent_mask[:, :agent_idx],
                    query_mask=agent_mask[:, agent_idx],
                    memory_mask=agent_mask,
                )
                for agent_idx in range(query_tokens.shape[1])
            ],
            dim=1,
        )
        arbitrary_step_output = torch.cat(
            [
                arbitrary_decoder.forward_step(
                    context_tokens=context_tokens[:, :agent_idx, :],
                    query_token=query_tokens[:, agent_idx:agent_idx + 1, :],
                    memory_tokens=memory_tokens,
                    context_mask=agent_mask[:, :agent_idx],
                    query_mask=agent_mask[:, agent_idx],
                    memory_mask=agent_mask,
                )
                for agent_idx in range(query_tokens.shape[1])
            ],
            dim=1,
        )

    torch.testing.assert_close(prefix_parallel_output, arbitrary_parallel_output, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(prefix_step_output, arbitrary_step_output, rtol=0.0, atol=1e-6)


def test_context_state_is_reformed_between_layers(monkeypatch: Any) -> None:
    torch.manual_seed(3)
    decoder = _make_decoder()
    decoder.eval()

    query_tokens = torch.randn(2, 4, 8)
    context_tokens = torch.randn(2, 4, 8)
    memory_tokens = torch.randn(2, 4, 8)
    agent_mask = torch.ones(2, 4, dtype=torch.bool)
    layer_context_inputs: list[torch.Tensor] = []

    for layer in decoder.layers:
        original_forward = layer.forward

        def record_forward(
                *args: Any,
                _original_forward: Any = original_forward,
                **kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            context_input = cast(torch.Tensor, kwargs["context_tokens"])
            layer_context_inputs.append(context_input.detach().clone())
            return _original_forward(*args, **kwargs)

        monkeypatch.setattr(layer, "forward", record_forward)

    with torch.no_grad():
        decoder(
            query_tokens=query_tokens,
            context_tokens=context_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )

    assert len(layer_context_inputs) == len(decoder.layers)
    torch.testing.assert_close(layer_context_inputs[0], context_tokens, rtol=0.0, atol=0.0)
    assert not torch.allclose(layer_context_inputs[1], context_tokens, rtol=0.0, atol=1e-6)


def test_parallel_output_does_not_depend_on_future_context_tokens() -> None:
    torch.manual_seed(4)
    decoder = _make_decoder()
    decoder.eval()

    query_tokens = torch.randn(2, 4, 8)
    context_tokens = torch.randn(2, 4, 8)
    memory_tokens = torch.randn(2, 4, 8)
    agent_mask = torch.ones(2, 4, dtype=torch.bool)
    modified_context_tokens = context_tokens.clone()
    modified_context_tokens[:, 2:, :] = torch.randn_like(modified_context_tokens[:, 2:, :])

    with torch.no_grad():
        output = decoder(
            query_tokens=query_tokens,
            context_tokens=context_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )
        modified_output = decoder(
            query_tokens=query_tokens,
            context_tokens=modified_context_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )

    torch.testing.assert_close(modified_output[:, :3, :], output[:, :3, :], rtol=0.0, atol=1e-6)
    assert not torch.allclose(modified_output[:, 3, :], output[:, 3, :], rtol=0.0, atol=1e-6)


def test_arbitrary_masks_do_not_create_nan_outputs() -> None:
    torch.manual_seed(5)
    decoder = _make_decoder(assume_agent_mask_is_active_prefix=False)
    query_tokens = torch.randn(2, 4, 8)
    context_tokens = torch.randn(2, 4, 8)
    memory_tokens = torch.randn(2, 4, 8)
    agent_mask = torch.tensor(
        [
            [False, True, True, False],
            [False, False, True, True],
        ]
    )

    with torch.no_grad():
        output = decoder(
            query_tokens=query_tokens,
            context_tokens=context_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )
        first_step_output = decoder.forward_step(
            context_tokens=context_tokens[:, :0, :],
            query_token=query_tokens[:, :1, :],
            memory_tokens=memory_tokens,
            context_mask=agent_mask[:, :0],
            query_mask=agent_mask[:, 0],
            memory_mask=agent_mask,
        )

    assert torch.isfinite(output).all()
    assert torch.isfinite(first_step_output).all()
