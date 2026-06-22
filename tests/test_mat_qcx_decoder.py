import pytest
import torch

from swarmbots.learn.algos.mat_qcx.mat_qcx_decoder import MATQCXDecoder, MATQCXDecoderConfig


def _make_decoder(*, assume_agent_mask_is_active_prefix: bool = True) -> MATQCXDecoder:
    return MATQCXDecoder(
        config=MATQCXDecoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            context_encoder_hidden_dims=[8],
            assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
        ),
        max_agents=4,
        input_d_model=6,
        action_d_model=8,
        memory_d_model=7,
    )


def test_decoder_rejects_non_positive_num_layers() -> None:
    with pytest.raises(ValueError, match="num_layers must be > 0"):
        MATQCXDecoder(
            config=MATQCXDecoderConfig(
                d_model=8,
                nhead=2,
                num_layers=0,
            ),
            max_agents=4,
            input_d_model=6,
            action_d_model=8,
            memory_d_model=7,
        )


def test_parallel_outputs_match_step_outputs() -> None:
    torch.manual_seed(0)
    decoder = _make_decoder()
    decoder.eval()

    input_tokens = torch.randn(3, 4, 6)
    action_tokens = torch.randn(3, 4, 8)
    memory_tokens = torch.randn(3, 4, 7)
    agent_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )

    with torch.no_grad():
        parallel_output = decoder(
            query_tokens=input_tokens,
            action_tokens=action_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )
        step_output = torch.cat(
            [
                decoder.forward_step(
                    action_tokens=action_tokens[:, :agent_idx, :],
                    query_token=input_tokens[:, agent_idx:agent_idx + 1, :],
                    memory_tokens=memory_tokens,
                    query_prefix_tokens=input_tokens[:, :agent_idx, :],
                    context_mask=agent_mask[:, :agent_idx],
                    query_mask=agent_mask[:, agent_idx],
                    memory_mask=agent_mask,
                )
                for agent_idx in range(input_tokens.shape[1])
            ],
            dim=1,
        )

    torch.testing.assert_close(step_output, parallel_output, rtol=0.0, atol=1e-6)


def test_parallel_outputs_match_step_outputs_for_arbitrary_masks() -> None:
    torch.manual_seed(1)
    decoder = _make_decoder(assume_agent_mask_is_active_prefix=False)
    decoder.eval()

    input_tokens = torch.randn(3, 4, 6)
    action_tokens = torch.randn(3, 4, 8)
    memory_tokens = torch.randn(3, 4, 7)
    agent_mask = torch.tensor(
        [
            [False, True, True, False],
            [True, False, True, True],
            [False, False, True, True],
        ]
    )

    with torch.no_grad():
        parallel_output = decoder(
            query_tokens=input_tokens,
            action_tokens=action_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )
        step_output = torch.cat(
            [
                decoder.forward_step(
                    action_tokens=action_tokens[:, :agent_idx, :],
                    query_token=input_tokens[:, agent_idx:agent_idx + 1, :],
                    memory_tokens=memory_tokens,
                    query_prefix_tokens=input_tokens[:, :agent_idx, :],
                    context_mask=agent_mask[:, :agent_idx],
                    query_mask=agent_mask[:, agent_idx],
                    memory_mask=agent_mask,
                )
                for agent_idx in range(input_tokens.shape[1])
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

    input_tokens = torch.randn(3, 4, 6)
    action_tokens = torch.randn(3, 4, 8)
    memory_tokens = torch.randn(3, 4, 7)
    agent_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )

    with torch.no_grad():
        prefix_output = prefix_decoder(
            query_tokens=input_tokens,
            action_tokens=action_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )
        arbitrary_output = arbitrary_decoder(
            query_tokens=input_tokens,
            action_tokens=action_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )

    torch.testing.assert_close(prefix_output, arbitrary_output, rtol=0.0, atol=1e-6)


def test_configured_query_and_context_encoders_construct_and_run() -> None:
    torch.manual_seed(3)
    decoder = MATQCXDecoder(
        config=MATQCXDecoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            query_encoder_hidden_dims=[8],
            context_encoder_hidden_dims=[8],
        ),
        max_agents=4,
        input_d_model=6,
        action_d_model=8,
        memory_d_model=7,
    )
    decoder.eval()

    with torch.no_grad():
        output = decoder(
            query_tokens=torch.randn(2, 4, 6),
            action_tokens=torch.randn(2, 4, 8),
            memory_tokens=torch.randn(2, 4, 7),
            agent_mask=torch.ones(2, 4, dtype=torch.bool),
            memory_mask=torch.ones(2, 4, dtype=torch.bool),
        )

    assert output.shape == (2, 4, 8)
    assert torch.isfinite(output).all()


def test_decoder_accepts_matching_input_and_model_width_without_projection() -> None:
    torch.manual_seed(4)
    decoder = MATQCXDecoder(
        config=MATQCXDecoderConfig(
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            assume_agent_mask_is_active_prefix=False,
        ),
        max_agents=4,
        input_d_model=8,
        action_d_model=8,
        memory_d_model=7,
    )
    decoder.eval()

    with torch.no_grad():
        output = decoder(
            query_tokens=torch.randn(2, 4, 8),
            action_tokens=torch.randn(2, 4, 8),
            memory_tokens=torch.randn(2, 4, 7),
            agent_mask=torch.ones(2, 4, dtype=torch.bool),
            memory_mask=torch.ones(2, 4, dtype=torch.bool),
        )

    assert output.shape == (2, 4, 8)
    assert torch.isfinite(output).all()


def test_parallel_output_does_not_depend_on_future_query_tokens() -> None:
    torch.manual_seed(5)
    decoder = _make_decoder()
    decoder.eval()

    input_tokens = torch.randn(2, 4, 6)
    action_tokens = torch.randn(2, 4, 8)
    memory_tokens = torch.randn(2, 4, 7)
    agent_mask = torch.ones(2, 4, dtype=torch.bool)
    modified_input_tokens = input_tokens.clone()
    modified_input_tokens[:, 2:, :] = torch.randn_like(modified_input_tokens[:, 2:, :])

    with torch.no_grad():
        output = decoder(
            query_tokens=input_tokens,
            action_tokens=action_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )
        modified_output = decoder(
            query_tokens=modified_input_tokens,
            action_tokens=action_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )

    torch.testing.assert_close(modified_output[:, :2, :], output[:, :2, :], rtol=0.0, atol=1e-6)
    assert not torch.allclose(modified_output[:, 2:, :], output[:, 2:, :], rtol=0.0, atol=1e-6)


def test_parallel_output_does_not_depend_on_future_action_tokens() -> None:
    torch.manual_seed(6)
    decoder = _make_decoder()
    decoder.eval()

    input_tokens = torch.randn(2, 4, 6)
    action_tokens = torch.randn(2, 4, 8)
    memory_tokens = torch.randn(2, 4, 7)
    agent_mask = torch.ones(2, 4, dtype=torch.bool)
    modified_action_tokens = action_tokens.clone()
    modified_action_tokens[:, 2:, :] = torch.randn_like(modified_action_tokens[:, 2:, :])

    with torch.no_grad():
        output = decoder(
            query_tokens=input_tokens,
            action_tokens=action_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )
        modified_output = decoder(
            query_tokens=input_tokens,
            action_tokens=modified_action_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )

    torch.testing.assert_close(modified_output[:, :3, :], output[:, :3, :], rtol=0.0, atol=1e-6)
    assert not torch.allclose(modified_output[:, 3, :], output[:, 3, :], rtol=0.0, atol=1e-6)


def test_first_agent_output_does_not_depend_on_action_tokens() -> None:
    torch.manual_seed(7)
    decoder = _make_decoder()
    decoder.eval()

    input_tokens = torch.randn(2, 4, 6)
    action_tokens = torch.randn(2, 4, 8)
    memory_tokens = torch.randn(2, 4, 7)
    agent_mask = torch.ones(2, 4, dtype=torch.bool)
    modified_action_tokens = torch.randn_like(action_tokens)

    with torch.no_grad():
        output = decoder(
            query_tokens=input_tokens,
            action_tokens=action_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )
        modified_output = decoder(
            query_tokens=input_tokens,
            action_tokens=modified_action_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )

    torch.testing.assert_close(modified_output[:, :1, :], output[:, :1, :], rtol=0.0, atol=1e-6)


def test_inactive_context_action_tokens_do_not_affect_outputs_with_arbitrary_masks() -> None:
    torch.manual_seed(8)
    decoder = _make_decoder(assume_agent_mask_is_active_prefix=False)
    decoder.eval()

    input_tokens = torch.randn(2, 4, 6)
    action_tokens = torch.randn(2, 4, 8)
    memory_tokens = torch.randn(2, 4, 7)
    agent_mask = torch.tensor(
        [
            [False, True, True, True],
            [True, False, True, True],
        ]
    )
    modified_action_tokens = action_tokens.clone()
    modified_action_tokens[~agent_mask] = torch.randn_like(modified_action_tokens[~agent_mask])

    with torch.no_grad():
        output = decoder(
            query_tokens=input_tokens,
            action_tokens=action_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )
        modified_output = decoder(
            query_tokens=input_tokens,
            action_tokens=modified_action_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )

    torch.testing.assert_close(modified_output, output, rtol=0.0, atol=1e-6)


def test_inactive_context_input_tokens_do_not_affect_active_outputs_with_arbitrary_masks() -> None:
    torch.manual_seed(9)
    decoder = _make_decoder(assume_agent_mask_is_active_prefix=False)
    decoder.eval()

    input_tokens = torch.randn(2, 4, 6)
    action_tokens = torch.randn(2, 4, 8)
    memory_tokens = torch.randn(2, 4, 7)
    agent_mask = torch.tensor(
        [
            [False, True, True, True],
            [True, False, True, True],
        ]
    )
    modified_input_tokens = input_tokens.clone()
    modified_input_tokens[~agent_mask] = torch.randn_like(modified_input_tokens[~agent_mask])

    with torch.no_grad():
        output = decoder(
            query_tokens=input_tokens,
            action_tokens=action_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )
        modified_output = decoder(
            query_tokens=modified_input_tokens,
            action_tokens=action_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )

    torch.testing.assert_close(modified_output[agent_mask], output[agent_mask], rtol=0.0, atol=1e-6)


def test_arbitrary_masks_do_not_create_nan_outputs() -> None:
    torch.manual_seed(10)
    decoder = _make_decoder(assume_agent_mask_is_active_prefix=False)
    input_tokens = torch.randn(2, 4, 6)
    action_tokens = torch.randn(2, 4, 8)
    memory_tokens = torch.randn(2, 4, 7)
    agent_mask = torch.tensor(
        [
            [False, True, True, False],
            [False, False, True, True],
        ]
    )

    with torch.no_grad():
        output = decoder(
            query_tokens=input_tokens,
            action_tokens=action_tokens,
            memory_tokens=memory_tokens,
            agent_mask=agent_mask,
            memory_mask=agent_mask,
        )
        first_step_output = decoder.forward_step(
            action_tokens=action_tokens[:, :0, :],
            query_token=input_tokens[:, :1, :],
            memory_tokens=memory_tokens,
            query_prefix_tokens=input_tokens[:, :0, :],
            context_mask=agent_mask[:, :0],
            query_mask=agent_mask[:, 0],
            memory_mask=agent_mask,
        )

    assert torch.isfinite(output).all()
    assert torch.isfinite(first_step_output).all()
