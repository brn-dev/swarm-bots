from collections.abc import Iterator

import pytest
import torch
from torch import nn

from swarmbots.learn.algos.world_modeling.transformer_transition_model import (
    TransformerTransitionModel,
    TransformerTransitionModelConfig,
)


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("hidden_layers", [2, 3])
def test_deep_transition_trains_every_hidden_layer_and_ignores_masked_agents(
    norm_first: bool, hidden_layers: int,
) -> None:
    model = TransformerTransitionModel(TransformerTransitionModelConfig(
        n_agents=3, latent_dim=8, action_dim=2, d_model=12, nhead=4, num_layers=2,
        dim_feedforward=16, num_feedforward_hidden_layers=hidden_layers, norm_first=norm_first,
    ))
    latents = torch.randn(2, 3, 8)
    actions = torch.randn(2, 3, 3, 2)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    predicted = model.predict_n_steps(latents, actions, mask)
    changed_latents = latents.masked_fill(~mask.unsqueeze(-1), 100)
    changed_actions = actions.masked_fill(~mask[:, None, :, None], -100)
    changed_predictions = model.predict_n_steps(changed_latents, changed_actions, mask)
    torch.testing.assert_close(predicted[mask[:, None].expand(-1, 3, -1)], changed_predictions[mask[:, None].expand(-1, 3, -1)])
    predicted[mask[:, None].expand(-1, 3, -1)].square().mean().backward()
    for layer in model.encoder.layers:
        linears = [module for module in layer.feedforward.modules() if isinstance(module, nn.Linear)]
        assert len(linears) == hidden_layers + 1
        assert all(linear.weight.grad is not None and torch.count_nonzero(linear.weight.grad) for linear in linears)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model.predict_n_steps(latents, actions, mask), predicted)
    assert model.get_hyper_parameters()["num_feedforward_hidden_layers"] == hidden_layers


def test_deep_transition_initializes_all_hidden_matrices_independently_with_requested_gains() -> None:
    model = TransformerTransitionModel(TransformerTransitionModelConfig(
        n_agents=3, latent_dim=8, action_dim=2, d_model=8, nhead=2, num_layers=2,
        dim_feedforward=16, num_feedforward_hidden_layers=2, transformer_ff_init_gain=1.5,
    ))
    first, second = model.encoder.layers
    first_linears = [module for module in first.feedforward.modules() if isinstance(module, nn.Linear)]
    second_linears = [module for module in second.feedforward.modules() if isinstance(module, nn.Linear)]
    assert not torch.equal(first.self_attn.in_proj_weight, second.self_attn.in_proj_weight)
    for index, (left, right) in enumerate(zip(first_linears, second_linears, strict=True)):
        assert left.weight is not right.weight and not torch.equal(left.weight, right.weight)
        gain = 1.0 if index == len(first_linears) - 1 else 1.5
        assert torch.linalg.vector_norm(left.weight).item() == pytest.approx(gain * min(left.weight.shape) ** 0.5)


def test_deep_transition_compilation_preserves_multi_step_predictions_and_gradients() -> None:
    config = TransformerTransitionModelConfig(
        n_agents=3, latent_dim=8, action_dim=2, d_model=8, nhead=2, num_layers=2,
        dim_feedforward=16, num_feedforward_hidden_layers=2,
    )
    eager = TransformerTransitionModel(config)
    compiled_model = TransformerTransitionModel(config)
    compiled_model.load_state_dict(eager.state_dict())
    compiled = torch.compile(compiled_model, backend="aot_eager", fullgraph=True)
    latents, actions = torch.randn(2, 3, 8), torch.randn(2, 3, 2)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    expected, actual = eager(latents, actions, mask), compiled(latents, actions, mask)
    torch.testing.assert_close(actual, expected)
    expected[mask].square().mean().backward()
    actual[mask].square().mean().backward()
    for left, right in zip(eager.parameters(), compiled_model.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad)
