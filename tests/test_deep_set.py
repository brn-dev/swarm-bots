from collections.abc import Iterator
from dataclasses import asdict, replace

import pytest
import torch
from torch import nn

from swarmbots.learn.nn_components.deep_set import (
    DeepSet,
    DeepSetCritic,
    DeepSetInteraction,
    DeepSetInteractionConfig,
    GatedDeepSetInteraction,
)
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


@pytest.fixture(scope="module", autouse=True)
def single_torch_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("set_dim", [1, 2])
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("mode", ["mean", "sum", "max"])
def test_pooling_values_and_gradients_with_partial_and_empty_sets(set_dim, use_mask, mode):
    elements = torch.tensor([
        [[1.0, -4.0], [3.0, -2.0], [50.0, 60.0]],
        [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0]],
    ])
    mask = torch.tensor([[True, True, False], [False, False, False]]) if use_mask else None
    expected_values = {
        "mean": [[2.0, -3.0], [0.0, 0.0]] if use_mask else [[18.0, 18.0], [11.0, 12.0]],
        "sum": [[4.0, -6.0], [0.0, 0.0]] if use_mask else [[54.0, 54.0], [33.0, 36.0]],
        "max": [[3.0, -2.0], [0.0, 0.0]] if use_mask else [[50.0, 60.0], [13.0, 14.0]],
    }
    expected_gradient = torch.zeros_like(elements)
    if mode == "max":
        if use_mask:
            expected_gradient[0, 1] = 1
        else:
            expected_gradient[:, 2] = 1
    elif use_mask:
        expected_gradient[0, :2] = 0.5 if mode == "mean" else 1
    else:
        expected_gradient.fill_(1 / 3 if mode == "mean" else 1)
    expected = torch.tensor(expected_values[mode])
    if set_dim == 2:
        elements = elements.unsqueeze(1)
        mask = None if mask is None else mask.unsqueeze(1)
        expected = expected.unsqueeze(1)
        expected_gradient = expected_gradient.unsqueeze(1)
    elements.requires_grad_()
    model = DeepSet(element_encoder=nn.Identity(), set_decoder=nn.Identity(), set_dim=set_dim, pool_mode=mode)

    values, latents = model.forward_with_latents(elements, element_mask=mask)

    torch.testing.assert_close(values, expected)
    torch.testing.assert_close(latents, elements)
    torch.testing.assert_close(model(elements, element_mask=mask), expected)
    values.sum().backward()
    torch.testing.assert_close(elements.grad, expected_gradient)


@pytest.mark.parametrize("set_dim", [1, 2])
@pytest.mark.parametrize("context_in_elements", [False, True])
@pytest.mark.parametrize("context_after_pool", [None, False, True])
def test_context_and_unpooled_latents_remain_differentiable(set_dim, context_in_elements, context_after_pool):
    after_pool = not context_in_elements if context_after_pool is None else context_after_pool
    elements = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]],
        [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
    ])
    context = torch.tensor([[2.0], [-1.0]])
    mask = torch.tensor([[True, True, False], [False, False, False]])
    expected_latents = torch.tensor([
        [[8.0, 3.0], [10.0, 11.0], [8.0, 696.0]],
        [[0.0, 25.0], [2.0, 33.0], [4.0, 41.0]],
    ]) if context_in_elements else torch.tensor([
        [[0.0, 7.0], [2.0, 15.0], [0.0, 700.0]],
        [[4.0, 23.0], [6.0, 31.0], [8.0, 39.0]],
    ])
    expected_values = torch.tensor(
        [[29.0], [3.0]] if context_in_elements and after_pool else
        [[25.0], [5.0]] if context_in_elements else
        [[1.0], [3.0]] if after_pool else [[-3.0], [5.0]]
    )
    if set_dim == 2:
        elements, context, mask = elements.unsqueeze(1), context.unsqueeze(1), mask.unsqueeze(1)
        expected_latents, expected_values = expected_latents.unsqueeze(1), expected_values.unsqueeze(1)
    elements.requires_grad_()
    context.requires_grad_()
    encoder = nn.Linear(3 if context_in_elements else 2, 2, bias=False)
    decoder = nn.Linear(3 if after_pool else 2, 1)
    with torch.no_grad():
        encoder.weight.copy_(torch.tensor([[2.0, -1.0, 4.0], [1.0, 3.0, -2.0]]) if context_in_elements
                             else torch.tensor([[2.0, -1.0], [1.0, 3.0]]))
        decoder.weight.copy_(torch.tensor([[3.0, -1.0, 2.0]]) if after_pool else torch.tensor([[3.0, -1.0]]))
        decoder.bias.fill_(5)
    model = DeepSet(
        element_encoder=encoder, set_decoder=decoder, set_dim=set_dim,
        context_features=1, context_in_elements=context_in_elements,
        context_after_pool=context_after_pool,
    )

    values, latents = model.forward_with_latents(elements, context=context, element_mask=mask)

    torch.testing.assert_close(values, expected_values)
    torch.testing.assert_close(latents, expected_latents)
    torch.testing.assert_close(model.encode_elements(elements, context=context), expected_latents)
    torch.testing.assert_close(model(elements, context=context, element_mask=mask), expected_values)
    # An auxiliary loss on the returned latents must still reach the element encoder.
    latents[mask].sum().backward(retain_graph=True)
    assert encoder.weight.grad is not None and encoder.weight.grad.abs().sum() > 0
    assert decoder.weight.grad is None
    assert (elements.grad[~mask] == 0).all()
    values.sum().backward()
    assert decoder.weight.grad is not None and decoder.weight.grad.abs().sum() > 0
    if context_in_elements or after_pool:
        assert context.grad is not None and torch.isfinite(context.grad).all()
    else:
        assert context.grad is None


@pytest.mark.parametrize("learned_encoder", [False, True])
@pytest.mark.parametrize("context_in_elements", [False, True])
@pytest.mark.parametrize("use_popart", [False, True])
@pytest.mark.parametrize("context_after_pool", [None, False, True])
def test_critic_latents_follow_permutations_and_share_value_gradients(
    learned_encoder, context_in_elements, use_popart, context_after_pool,
):
    torch.manual_seed(42)
    critic = DeepSetCritic(
        num_local_features=2, num_global_features=2,
        local_projection_hidden_dims=[4, 6] if learned_encoder else [],
        value_regressor_hidden_dims=[4], context_in_elements=context_in_elements, use_popart=use_popart,
        context_after_pool=context_after_pool,
    )
    local = torch.randn(2, 3, 2, requires_grad=True)
    global_features = torch.randn(2, 2, requires_grad=True)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    values, latents = critic.forward_with_latents(local, global_features, mask)
    assert values.shape == (2,)
    assert latents.shape == (2, 3, 6 if learned_encoder else 4 if context_in_elements else 2)
    torch.testing.assert_close(critic(local, global_features, mask), values)
    torch.testing.assert_close(critic.encode_elements(local, global_features, mask), latents)
    permutation = torch.tensor([2, 0, 1])
    shuffled_values, shuffled_latents = critic.forward_with_latents(
        local[:, permutation], global_features, mask[:, permutation],
    )
    torch.testing.assert_close(shuffled_values, values)
    torch.testing.assert_close(shuffled_latents, latents[:, permutation])

    (values.sum() + latents[mask].square().sum()).backward()

    assert torch.isfinite(local.grad).all() and (local.grad[~mask] == 0).all()
    assert local.grad[mask].abs().sum() > 0
    if context_in_elements or critic.context_after_pool:
        assert torch.isfinite(global_features.grad).all()
    else:
        assert global_features.grad is None
    for module in (critic.deepset.element_encoder, critic.deepset.interactions, critic.deepset.set_decoder):
        gradients = [p.grad for p in module.parameters()]
        if gradients:
            assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
            assert any(gradient.abs().sum() > 0 for gradient in gradients)


@pytest.mark.parametrize("mask,message", [
    (torch.ones(2, 3), "dtype bool"),
    (torch.ones(2, 3, 1, dtype=torch.bool), "ndim"),
    (torch.ones(2, 2, dtype=torch.bool), "shape"),
])
def test_invalid_pooling_masks_are_rejected(mask, message):
    model = DeepSet(element_encoder=nn.Identity(), set_decoder=nn.Identity())
    with pytest.raises(ValueError, match=message):
        model(torch.zeros(2, 3, 2), element_mask=mask)


@pytest.mark.parametrize("use_mask", [False, True])
def test_unknown_pooling_mode_is_rejected(use_mask):
    model = DeepSet(element_encoder=nn.Identity(), set_decoder=nn.Identity(), pool_mode="median")
    with pytest.raises(ValueError, match="Unknown pool mode"):
        model(torch.zeros(2, 3, 2), element_mask=torch.ones(2, 3, dtype=torch.bool) if use_mask else None)


@pytest.mark.parametrize("local_update", ["residual", "linear", "mlp", "residual_mlp"])
@pytest.mark.parametrize("pool_mode", ["mean", "sum", "max"])
@pytest.mark.parametrize("set_dim", [1, 2, -2])
@pytest.mark.parametrize("gated,gate_input_dim_cap", [(False, None), (True, None), (True, 3)])
def test_iterative_features_ignore_padding_and_are_equivariant(
    local_update: str, pool_mode: str, set_dim: int, gated: bool, gate_input_dim_cap: int | None,
) -> None:
    torch.manual_seed(12)
    critic = DeepSetCritic(
        num_local_features=3, num_global_features=2,
        local_projection_hidden_dims=[8], value_regressor_hidden_dims=[5],
        context_in_elements=True, context_after_pool=True, set_dim=set_dim, pool_mode=pool_mode,
        interaction_config=DeepSetInteractionConfig(
            num_layers=3, hidden_dims=(6,), local_update=local_update, gated=gated, gate_input_dim_cap=gate_input_dim_cap,
        ),
    )
    local = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, False, True, True], [False, False, False, False]])
    global_features = torch.randn(2, 2, requires_grad=True)
    local[~mask] = float("nan")
    if set_dim == 2:
        local, mask = local.unsqueeze(1), mask.unsqueeze(1)
        global_features = global_features.detach().unsqueeze(1).requires_grad_()
    local.requires_grad_()
    values, latents = critic.forward_with_latents(local, global_features, mask)
    assert torch.isfinite(values).all() and torch.isfinite(latents).all()
    assert (latents[~mask] == 0).all()
    torch.testing.assert_close(critic.encode_elements(local, global_features, mask), latents)
    permutation = torch.tensor([2, 0, 3, 1])
    shuffled_values, shuffled_latents = critic.forward_with_latents(
        local.index_select(-2, permutation), global_features, mask.index_select(-1, permutation),
    )
    torch.testing.assert_close(shuffled_values, values)
    torch.testing.assert_close(shuffled_latents, latents.index_select(-2, permutation))
    padded_local = torch.cat((local, torch.full_like(local, float("nan"))), dim=-2)
    padded_mask = torch.cat((mask, torch.zeros_like(mask)), dim=-1)
    padded_values, padded_latents = critic.forward_with_latents(padded_local, global_features, padded_mask)
    torch.testing.assert_close(padded_values, values)
    torch.testing.assert_close(padded_latents[..., :4, :], latents)
    (values.sum() + latents[mask].square().sum()).backward()
    assert torch.isfinite(local.grad).all() and (local.grad[~mask] == 0).all()
    assert torch.isfinite(global_features.grad).all()
    for parameter in critic.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_agent_features_depend_on_other_active_agents_before_final_pooling() -> None:
    torch.manual_seed(29)
    critic = DeepSetCritic(
        num_local_features=3, local_projection_hidden_dims=[8], value_regressor_hidden_dims=[4],
    )
    local = torch.randn(1, 3, 3, requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    latents = critic.encode_elements(local, agent_mask=mask)
    latents[0, 0].square().sum().backward()
    assert local.grad[0, 1].abs().sum() > 0
    assert (local.grad[0, 2] == 0).all()
    assert all(parameter.grad is not None for parameter in critic.deepset.interactions.parameters())
    assert all(parameter.grad is None for parameter in critic.deepset.set_decoder.parameters())


@pytest.mark.parametrize("local_update,expected", [
    ("residual", [97.0, 99.0]), ("linear", [1552.0, 1584.0]),
    ("mlp", [1552.0, 1584.0]), ("residual_mlp", [2425.0, 2475.0]),
])
def test_local_update_modes_process_current_broadcast_messages(local_update: str, expected: list[float]) -> None:
    critic = DeepSetCritic(
        num_local_features=1, local_projection_hidden_dims=[], value_regressor_hidden_dims=[],
        act_fn_cls=nn.Identity,
        interaction_config=DeepSetInteractionConfig(
            num_layers=2, hidden_dims=(), normalize=False, local_update=local_update,
        ),
    )
    with torch.no_grad():
        for block in critic.deepset.interactions:
            block.message_encoder[0].weight.fill_(2)
            block.message_encoder[0].bias.zero_()
            block.global_update[0].weight.fill_(3)
            block.global_update[0].bias.zero_()
            block.message_projection.weight.fill_(1)
            if local_update != "residual":
                block.local_encoder[0].weight.fill_(4)
                block.local_encoder[0].bias.zero_()
    latents = critic.encode_elements(torch.tensor([[[1.0], [3.0]]]))
    torch.testing.assert_close(latents, torch.tensor(expected).reshape(1, 2, 1))


def test_one_block_locally_interprets_the_current_global_message() -> None:
    critic = DeepSetCritic(
        num_local_features=1, local_projection_hidden_dims=[], value_regressor_hidden_dims=[],
        interaction_config=DeepSetInteractionConfig(num_layers=1, normalize=False),
    )
    block = critic.deepset.interactions[0]
    block.message_encoder = nn.Identity()
    block.global_update = nn.Identity()
    block.message_projection = nn.Identity()
    block.local_encoder = nn.ReLU()
    baseline = torch.tensor([[[-1.0], [-3.0]]])
    changed = torch.tensor([[[-1.0], [5.0]]], requires_grad=True)

    baseline_latents = critic.encode_elements(baseline)
    changed_latents = critic.encode_elements(changed)

    torch.testing.assert_close(baseline_latents[0, 0], torch.tensor([-3.0]))
    torch.testing.assert_close(changed_latents[0, 0], torch.tensor([2.0]))
    changed_latents[0, 0].sum().backward()
    torch.testing.assert_close(changed.grad[0, 1], torch.tensor([1.0]))


@pytest.mark.parametrize("gated,gate_input_dim_cap", [(False, None), (True, None), (True, 3)])
def test_iterative_critic_compiles_without_graph_breaks(gated: bool, gate_input_dim_cap: int | None) -> None:
    critic = DeepSetCritic(
        num_local_features=3, local_projection_hidden_dims=[8], value_regressor_hidden_dims=[4],
        interaction_config=DeepSetInteractionConfig(gated=gated, gate_input_dim_cap=gate_input_dim_cap),
    )
    compiled = torch.compile(critic.forward_with_latents, backend="eager", fullgraph=True)
    local = torch.randn(2, 3, 3)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    torch.testing.assert_close(compiled(local, agent_mask=mask), critic.forward_with_latents(local, agent_mask=mask))


@pytest.mark.parametrize("kwargs", [
    {"num_layers": 0}, {"num_layers": -1}, {"hidden_dims": (0,)}, {"local_update": "unknown"},
    {"message_dim": 0}, {"message_dim": -1}, {"gate_hidden_dim": 0}, {"gate_hidden_dim": -1},
    {"gate_input_dim_cap": 0}, {"gate_input_dim_cap": -1},
])
def test_invalid_interaction_configuration_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DeepSetInteractionConfig(**kwargs)


@pytest.mark.parametrize("gated", [False, True])
def test_message_and_local_widths_resolve_independently(gated: bool) -> None:
    critic = DeepSetCritic(
        num_local_features=3, local_projection_hidden_dims=[16], value_regressor_hidden_dims=[4],
        interaction_config=DeepSetInteractionConfig(
            hidden_dims=(48,), message_dim=28, gated=gated, gate_hidden_dim=7,
        ),
    )
    block = critic.deepset.interactions[0]
    assert isinstance(block, GatedDeepSetInteraction if gated else DeepSetInteraction)
    assert block.message_encoder[0].in_features == 16
    assert block.message_encoder[0].out_features == 28
    assert block.global_update[0].in_features == block.global_update[0].out_features == 28
    assert block.message_projection.in_features == 28 and block.message_projection.out_features == 16
    assert block.local_encoder[0].out_features == 48
    assert block.local_encoder[2].out_features == 16
    assert asdict(block.config)["message_dim"] == 28
    if gated:
        assert block.reception_gate[0][0].out_features == 7
        assert asdict(block.config)["gate_hidden_dim"] == 7
    assert critic.encode_elements(torch.randn(2, 5, 3)).shape == (2, 5, 16)


@pytest.mark.parametrize("features,message_dim,cap,input_width", [
    (4, 9, None, 13), (4, 9, 2, 4), (4, 9, 6, 10), (9, 4, 6, 10), (4, 9, 20, 13),
])
@pytest.mark.parametrize("hidden_dim", [None, 7])
def test_gate_input_cap_and_hidden_width_resolve_independently(
    features: int, message_dim: int, cap: int | None, input_width: int, hidden_dim: int | None,
) -> None:
    block = GatedDeepSetInteraction(
        features, config=DeepSetInteractionConfig(
            message_dim=message_dim, gate_input_dim_cap=cap, gate_hidden_dim=hidden_dim,
        ),
        set_dim=1, pool_mode="mean", linear_init=init_linear_orthogonal, act_fn_cls=nn.GELU,
    )
    assert block.reception_gate[0][0].in_features == input_width
    assert block.gate_hidden_dim == (max(1, features // 4) if hidden_dim is None else hidden_dim)
    assert block.reception_gate[0][-1].out_features == message_dim
    assert asdict(block.config)["gate_input_dim_cap"] == cap
    assert block(torch.randn(2, 3, features)).shape == (2, 3, features)


def test_gate_cap_limits_conditioning_but_preserves_all_message_features_and_gradients() -> None:
    block = GatedDeepSetInteraction(
        4, config=DeepSetInteractionConfig(message_dim=5, gate_input_dim_cap=2, gate_hidden_dim=3),
        set_dim=1, pool_mode="mean", linear_init=init_linear_orthogonal, act_fn_cls=nn.Identity,
    )
    with torch.no_grad():
        for module in block.reception_gate.modules():
            if isinstance(module, nn.Linear):
                module.weight.fill_(0.1)
                module.bias.zero_()
        block.message_projection.weight.fill_(0.2)
    local = torch.ones(2, 3, 4, requires_grad=True)
    message = torch.ones(2, 5, requires_grad=True)
    received_gates: list[torch.Tensor] = []
    handle = block.reception_gate.register_forward_hook(
        lambda _module, _inputs, output: received_gates.append(output),
    )
    baseline = block._broadcast_update(local, message)
    changed_local = local.detach().clone()
    changed_local[..., 2:] += 3
    unchanged_update = block._broadcast_update(changed_local, message)
    changed_message = message.detach().clone()
    changed_message[..., 2:] += 3
    changed_update = block._broadcast_update(local, changed_message)
    handle.remove()

    assert received_gates[0].shape == (2, 3, 5)
    torch.testing.assert_close(received_gates[0], received_gates[1])
    torch.testing.assert_close(received_gates[0], received_gates[2])
    torch.testing.assert_close(baseline, unchanged_update)
    assert (changed_update > baseline).all()
    local_gate_gradient, message_gate_gradient = torch.autograd.grad(
        received_gates[0].sum(), (local, message), retain_graph=True,
    )
    assert (local_gate_gradient[..., :2] > 0).all()
    assert (message_gate_gradient[..., :2] > 0).all()
    assert (local_gate_gradient[..., 2:] == 0).all()
    assert (message_gate_gradient[..., 2:] == 0).all()
    baseline.sum().backward()
    assert (message.grad[..., 2:] > 0).all()


def test_gate_cap_above_both_widths_matches_full_input() -> None:
    config = DeepSetInteractionConfig(message_dim=7, gate_hidden_dim=3)
    blocks = [
        GatedDeepSetInteraction(
            4, config=replace(config, gate_input_dim_cap=cap),
            set_dim=1, pool_mode="mean", linear_init=init_linear_orthogonal, act_fn_cls=nn.GELU,
        )
        for cap in (None, 10)
    ]
    blocks[1].load_state_dict(blocks[0].state_dict())
    elements = torch.randn(2, 3, 4, requires_grad=True)
    full, capped = (block(elements) for block in blocks)
    torch.testing.assert_close(full, capped, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(full.sum(), elements)[0], torch.autograd.grad(capped.sum(), elements)[0],
        rtol=0, atol=0,
    )


@pytest.mark.parametrize("features", [128, 256, 384])
def test_default_expanded_communication_budget_is_close_to_attention(features: int) -> None:
    block = DeepSetInteraction(
        features, config=DeepSetInteractionConfig(), set_dim=1, pool_mode="mean",
        linear_init=init_linear_orthogonal, act_fn_cls=nn.GELU,
    )
    attention = nn.MultiheadAttention(features, num_heads=4, batch_first=True)
    communication_parameters = sum(
        parameter.numel()
        for module in (block.message_encoder, block.global_update, block.message_projection)
        for parameter in module.parameters()
    )
    attention_parameters = sum(parameter.numel() for parameter in attention.parameters())
    assert block.message_dim == features * 5 // 4
    assert 0.98 < communication_parameters / attention_parameters < 1.02


def test_nonlinearity_before_pooling_distinguishes_equal_mean_sets() -> None:
    block = DeepSetInteraction(
        1, config=DeepSetInteractionConfig(message_dim=2, normalize=False, local_update="residual"),
        set_dim=1, pool_mode="mean", linear_init=init_linear_orthogonal, act_fn_cls=nn.GELU,
    )
    with torch.no_grad():
        block.message_encoder[0].weight.fill_(1)
        block.message_encoder[0].bias.zero_()
        block.global_update[0].weight.copy_(torch.eye(2))
        block.global_update[0].bias.zero_()
        block.message_projection.weight.fill_(1)
    spread = torch.tensor([[[-1.0], [1.0]]])
    concentrated = torch.zeros_like(spread)
    torch.testing.assert_close(spread.mean(1), concentrated.mean(1))
    spread_update = block(spread) - spread
    concentrated_update = block(concentrated) - concentrated
    assert not torch.allclose(spread_update, concentrated_update)


def test_nonlinear_global_processing_happens_before_message_compression() -> None:
    block = DeepSetInteraction(
        2, config=DeepSetInteractionConfig(message_dim=5, normalize=False, local_update="residual"),
        set_dim=1, pool_mode="mean", linear_init=init_linear_orthogonal, act_fn_cls=nn.GELU,
    )
    observed_inputs: list[torch.Tensor] = []
    handle = block.message_projection.register_forward_pre_hook(
        lambda _module, inputs: observed_inputs.append(inputs[0].detach()),
    )
    elements = torch.randn(3, 4, 2)
    block(elements)
    handle.remove()
    pooled = block.message_encoder(elements).mean(dim=1)
    torch.testing.assert_close(observed_inputs[0], block.global_update(pooled))
    assert isinstance(block.global_update[-1], nn.GELU)
    assert observed_inputs[0].shape == (3, 5)


def test_gates_select_different_message_features_for_different_tokens() -> None:
    block = GatedDeepSetInteraction(
        2, config=DeepSetInteractionConfig(
            message_dim=3, gate_hidden_dim=2, normalize=False, local_update="residual", gated=True,
        ),
        set_dim=1, pool_mode="mean", linear_init=init_linear_orthogonal, act_fn_cls=nn.GELU,
    )
    with torch.no_grad():
        block.message_encoder[0].weight.zero_()
        block.message_encoder[0].bias.fill_(1)
        block.global_update[0].weight.copy_(torch.eye(3))
        block.global_update[0].bias.zero_()
        block.message_projection.weight.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
        first, last = block.reception_gate[0][0], block.reception_gate[0][2]
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 0], first.weight[1, 0] = 1, -1
        last.weight.copy_(torch.tensor([[2., 0.], [0., 2.], [0., 0.]]))
        last.bias.zero_()
    elements = torch.tensor([[[-2., 0.], [2., 0.]]])
    updates = block(elements) - elements
    assert updates[0, 0, 0] < updates[0, 1, 0]
    assert updates[0, 0, 1] > updates[0, 1, 1]


def test_gates_condition_on_the_global_message_as_well_as_the_receiver() -> None:
    torch.manual_seed(46)
    block = GatedDeepSetInteraction(
        4, config=DeepSetInteractionConfig(gated=True, gate_hidden_dim=3),
        set_dim=1, pool_mode="mean", linear_init=init_linear_orthogonal, act_fn_cls=nn.GELU,
    )
    elements = torch.randn(1, 3, 4, requires_grad=True)
    received_gates: list[torch.Tensor] = []
    handle = block.reception_gate.register_forward_hook(
        lambda _module, _inputs, output: received_gates.append(output),
    )
    block(elements)
    received_gates[0][0, 0].square().sum().backward()
    handle.remove()
    assert elements.grad[0, 0].abs().sum() > 0
    assert elements.grad[0, 1:].abs().sum() > 0


def test_completely_closed_gates_produce_no_message_update() -> None:
    block = GatedDeepSetInteraction(
        4, config=DeepSetInteractionConfig(gated=True, local_update="residual"),
        set_dim=-2, pool_mode="mean", linear_init=init_linear_orthogonal, act_fn_cls=nn.GELU,
    )
    with torch.no_grad():
        block.reception_gate[0][-1].weight.zero_()
        block.reception_gate[0][-1].bias.fill_(-float("inf"))
    elements = torch.randn(2, 3, 5, 4)
    torch.testing.assert_close(block(elements), elements)
    assert block.message_projection.bias is None
