from collections.abc import Iterator

import pytest
import torch
from torch import nn

from swarmbots.learn.algos.mat_orig.mat_orig_encoder import MATOrigSelfAttention
from swarmbots.learn.presets.transformer import MATInitGains
from test_model_scale import make_policy


@pytest.fixture(autouse=True, scope="module")
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _assert_orthogonal_gain(linear: nn.Linear, gain: float) -> None:
    weight = linear.weight.detach()
    gram = weight.T @ weight if weight.shape[0] >= weight.shape[1] else weight @ weight.T
    torch.testing.assert_close(gram, torch.eye(gram.shape[0]) * gain**2, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(linear.bias, torch.zeros_like(linear.bias), atol=1e-7, rtol=0)


@pytest.mark.parametrize("scale", [None, "2.5M"])
@pytest.mark.parametrize("gains", [
    MATInitGains(),
    MATInitGains(
        decoder_token_encoder=0.8,
        decoder_token_encoder_projection=0.7,
        decoder_transformer_ff=0.6,
        actor_head=0.5,
        action_net=0.04,
        critic_local_projection=0.9,
        critic_value_regressor=0.75,
        critic_value_head=0.03,
    ),
])
def test_original_mat_separates_feature_and_prediction_gains(scale: str | None, gains: MATInitGains) -> None:
    policy = make_policy("mat_orig", scale, use_nop=False, mat_init_gains=gains)

    _assert_orthogonal_gain(policy.decoder.action_encoder[0], gains.decoder_token_encoder)
    _assert_orthogonal_gain(policy.encoder_decoder_projection, gains.decoder_token_encoder_projection)
    for block in policy.decoder.blocks:
        _assert_orthogonal_gain(block.mlp[0], gains.decoder_transformer_ff)
        _assert_orthogonal_gain(block.mlp[2], 1.0)
    for module in policy.decoder.head:
        if isinstance(module, nn.Linear):
            _assert_orthogonal_gain(module, gains.actor_head)
    for distribution in policy.action_dist.distributions:
        _assert_orthogonal_gain(distribution.output_net, gains.action_net)

    value_linears = [module for module in policy.critic.value_head if isinstance(module, nn.Linear)]
    _assert_orthogonal_gain(value_linears[0], gains.critic_local_projection)
    for module in value_linears[1:-1]:
        _assert_orthogonal_gain(module, gains.critic_value_regressor)
    _assert_orthogonal_gain(value_linears[-1], gains.critic_value_head)


def test_original_mat_attention_uses_the_same_xavier_variance_as_packed_qkv() -> None:
    torch.manual_seed(7)
    width = 128
    attention = MATOrigSelfAttention(width, 4, max_agents=3, masked=True)

    for projection in (attention.query, attention.key, attention.value, attention.proj):
        expected_variance = (1.0 if projection is attention.proj else 0.5) / width
        assert projection.weight.detach().var().item() == pytest.approx(expected_variance, rel=0.06)
        assert projection.weight.detach().abs().max().item() <= (3 * expected_variance)**0.5
        torch.testing.assert_close(projection.bias, torch.zeros_like(projection.bias), atol=0, rtol=0)
