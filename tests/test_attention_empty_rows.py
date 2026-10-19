from contextlib import nullcontext
from typing import Any

import pytest
import torch
from torch.utils._pytree import tree_leaves

from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig, MATEncoderLayer
from swarmbots.learn.algos.mat_orig.mat_orig_encoder import MATOrigSelfAttention
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig
from test_model_scale import make_policy


_DEVICES = [
    "cpu",
    pytest.param("cuda", marks=[
        pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    ]),
]


def _assert_finite(value: Any) -> None:
    assert all(torch.isfinite(tensor).all() for tensor in tree_leaves(value) if isinstance(tensor, torch.Tensor))


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("additive_mask", [False, True])
@pytest.mark.parametrize("device", _DEVICES)
def test_mat_attention_empty_rows_contribute_zero_without_masked_input_gradients(
    training: bool, grad_enabled: bool, additive_mask: bool, device: str,
) -> None:
    layer = MATEncoderLayer(MATEncoderConfig(d_model=8, nhead=2)).to(device).train(training)
    inputs = torch.randn(3, 4, 8, requires_grad=True, device=device)
    active = torch.tensor(
        [[True, False, True, False], [False, False, False, False], [True, True, True, True]], device=device,
    )
    padding_mask = ~active
    if additive_mask:
        padding_mask = torch.zeros_like(active, dtype=inputs.dtype).masked_fill(~active, -torch.inf)
    with nullcontext() if grad_enabled else torch.no_grad():
        output = layer._self_attention_block(
            inputs, attention_mask=None, key_padding_mask=padding_mask, is_causal=False,
        )
        _assert_finite(output)
        assert not output[1].count_nonzero()
        reference = layer._self_attention_block(
            inputs[:1], attention_mask=None, key_padding_mask=padding_mask[:1], is_causal=False,
        )
        torch.testing.assert_close(output[:1], reference)
        changed_inputs = inputs.detach().masked_fill(~active.unsqueeze(-1), 1234)
        changed_output = layer._self_attention_block(
            changed_inputs, attention_mask=None, key_padding_mask=padding_mask, is_causal=False,
        )
        torch.testing.assert_close(output[active], changed_output[active])
        if grad_enabled:
            (output * torch.randn_like(output)).sum().backward()
            _assert_finite([inputs.grad, [parameter.grad for parameter in layer.parameters()]])
            assert not inputs.grad[1].count_nonzero()


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("device", _DEVICES)
def test_original_mat_attention_empty_rows_keep_forward_and_backward_finite(
    causal: bool, training: bool, device: str,
) -> None:
    attention = MATOrigSelfAttention(8, 2, max_agents=4, masked=causal).to(device).train(training)
    inputs = torch.randn(3, 4, 8, requires_grad=True, device=device)
    active = torch.tensor(
        [[True, True, False, False], [False, False, False, False], [True, True, True, True]], device=device,
    )
    with torch.no_grad():
        attention.proj.bias.fill_(0.3)
        inferred = attention(inputs, inputs, inputs, key_padding_mask=active, query_padding_mask=active)
        _assert_finite(inferred)
        assert not inferred[1].count_nonzero()
        unmasked_queries = attention(inputs, inputs, inputs, key_padding_mask=active)
        assert not unmasked_queries[1].count_nonzero()
    output = attention(inputs, inputs, inputs, key_padding_mask=active, query_padding_mask=active)
    reference = attention(inputs[:1], inputs[:1], inputs[:1], key_padding_mask=active[:1], query_padding_mask=active[:1])
    torch.testing.assert_close(output[:1], reference)
    changed_inputs = inputs.detach().masked_fill(~active.unsqueeze(-1), 1234)
    changed = attention(changed_inputs, changed_inputs, changed_inputs, key_padding_mask=active, query_padding_mask=active)
    torch.testing.assert_close(output, changed)
    (output * torch.randn_like(output)).sum().backward()
    _assert_finite([inputs.grad, [parameter.grad for parameter in attention.parameters()]])
    assert not inputs.grad[~active].count_nonzero()


def test_recurrent_mat_empty_rows_remain_finite_across_inference_and_reactivation() -> None:
    encoder = RMATEncoder(
        RMATEncoderConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16),
        max_agents=3, local_obs_dim=5, global_obs_dim=2,
    ).eval()
    local_obs = torch.randn(2, 3, 3, 5)
    global_obs = torch.randn(2, 3, 2)
    active = torch.tensor([
        [[True, False, True], [False, False, False], [True, True, True]],
        [[False, False, False], [True, False, True], [True, True, False]],
    ])
    with torch.no_grad():
        sequence, final_state = encoder(local_obs, global_obs, agent_mask=active)
        _assert_finite((sequence, final_state))
        state = None
        steps = []
        for index in range(3):
            output, state = encoder(
                local_obs[:, index], global_obs[:, index], agent_mask=active[:, index], initial_state=state,
            )
            steps.append(output)
        torch.testing.assert_close(sequence, torch.stack(steps, dim=1))
        for whole, step in zip(tree_leaves(final_state), tree_leaves(state), strict=True):
            torch.testing.assert_close(whole, step)


@pytest.mark.parametrize("variant", ["mat_ind", "mat_qcx", "mat_orig", "tmasac", "tmatd3"])
def test_fixed_policy_inference_handles_empty_agent_rows(variant: str) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        policy = make_policy(variant, "2.5M", use_nop=False).eval()
        observations = dict(
            local_obs=torch.randn(3, 3, 5), global_obs=torch.randn(3, 2),
            hidden_local_vars=torch.randn(3, 3, 2), hidden_global_vars=torch.randn(3, 1),
            agent_mask=torch.tensor([[True, True, False], [False, False, False], [True, True, True]]),
        )
        with torch.no_grad():
            if variant in {"tmasac", "tmatd3"}:
                actions = policy.act(**observations)
                _assert_finite(policy.q_values(**observations, actions=actions))
            else:
                actions, log_probs, values = policy(**observations)
                _assert_finite((log_probs, values))
            _assert_finite(actions)
            if variant != "mat_qcx":
                assert not actions[1].count_nonzero()
    finally:
        torch.set_num_threads(previous_threads)
