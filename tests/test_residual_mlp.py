import pytest
import torch
from torch import nn

from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal
from swarmbots.learn.nn_components.residual_mlp import ResidualMLP, ResidualMLPBlock, ResidualMLPConfig


def _block() -> ResidualMLPBlock:
    return ResidualMLPBlock(
        8, 16, act_fn_cls=nn.GELU,
        linear_init=init_linear_orthogonal, output_linear_init=init_linear_orthogonal,
    )


def test_zero_residual_branch_preserves_features_and_identity_gradients() -> None:
    block = _block()
    for parameter in block.feedforward.parameters():
        nn.init.zeros_(parameter)
    inputs = torch.randn(2, 3, 8, requires_grad=True)
    outputs = block(inputs)
    torch.testing.assert_close(outputs, inputs, rtol=0, atol=0)
    outputs.sum().backward()
    torch.testing.assert_close(inputs.grad, torch.ones_like(inputs), rtol=0, atol=0)


def test_pre_norm_makes_residual_updates_invariant_to_feature_shift_and_scale() -> None:
    block = _block()
    inputs = torch.randn(2, 3, 8)
    transformed = inputs * 3 + 7
    torch.testing.assert_close(block(inputs) - inputs, block(transformed) - transformed, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("output_dim", [8, 5])
def test_residual_mlp_handles_batch_and_sequence_dimensions_with_finite_gradients(output_dim: int) -> None:
    model = ResidualMLP(8, output_dim, ResidualMLPConfig(hidden_dim=16, num_blocks=3))
    inputs = torch.randn(2, 4, 3, 8, requires_grad=True)
    outputs = model(inputs)
    assert outputs.shape == (2, 4, 3, output_dim)
    (outputs * torch.randn_like(outputs)).sum().backward()
    assert torch.isfinite(inputs.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
