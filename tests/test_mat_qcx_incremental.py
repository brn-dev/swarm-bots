import copy
from collections.abc import Callable

import pytest
import torch
from torch import nn

from swarmbots.learn.algos.mat_qcx.mat_qcx_decoder import MATQCXDecoder, MATQCXDecoderConfig
from swarmbots.learn.nn_components.activations import ActivationFactory


def _decode_steps(
    decoder: MATQCXDecoder,
    queries: torch.Tensor,
    actions: torch.Tensor,
    memory: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    cached: bool,
) -> torch.Tensor:
    cache = decoder.create_step_cache(memory) if cached else None
    return torch.cat([
        decoder.forward_step(
            actions[:, :index], queries[:, index:index + 1], memory,
            query_prefix_tokens=queries[:, :index],
            context_mask=None if mask is None else mask[:, :index],
            memory_mask=mask,
            cache=cache,
        )
        for index in range(queries.shape[1])
    ], dim=1)


@pytest.mark.parametrize("norm_first", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("layers,memory_width", [(1, 8), (3, 7)])
@pytest.mark.parametrize("mask_kind", ["none", "prefix", "arbitrary"])
def test_incremental_decoder_preserves_outputs_and_gradients(
    norm_first: bool, bias: bool, layers: int, memory_width: int, mask_kind: str,
) -> None:
    torch.manual_seed(731)
    decoder = MATQCXDecoder(
        MATQCXDecoderConfig(
            d_model=8, nhead=2, num_layers=layers, dim_feedforward=13,
            bias=bias, norm_first=norm_first,
            query_encoder_hidden_dims=[11], context_encoder_hidden_dims=[9],
            normalize_context_input=True, normalize_context_tokens=True,
            assume_agent_mask_is_active_prefix=mask_kind == "prefix",
        ),
        max_agents=4, input_d_model=6, action_d_model=5, memory_d_model=memory_width,
    ).double()
    if bias:
        with torch.no_grad():
            for layer in decoder.layers:
                layer.memory_attn.out_proj.bias.uniform_(-0.3, 0.3)
                layer.query_context_attn.out_proj.bias.uniform_(-0.3, 0.3)
    reference = copy.deepcopy(decoder)
    queries = torch.randn(3, 4, 6, dtype=torch.float64, requires_grad=True)
    actions = torch.randn(3, 4, 5, dtype=torch.float64, requires_grad=True)
    memory = torch.randn(3, 4, memory_width, dtype=torch.float64, requires_grad=True)
    inputs = (queries, actions, memory)
    reference_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    mask = None
    if mask_kind == "prefix":
        mask = torch.tensor([[True, True, False, False], [True, False, False, False], [False] * 4])
    elif mask_kind == "arbitrary":
        mask = torch.tensor([[False, True, False, True], [True, False, True, False], [False] * 4])
    outputs = _decode_steps(decoder, *inputs, mask, cached=True)
    expected = _decode_steps(reference, *reference_inputs, mask, cached=False)
    torch.testing.assert_close(outputs, expected, atol=1e-9, rtol=1e-8)
    weights = torch.randn_like(outputs)
    (outputs * weights).sum().backward()
    (expected * weights).sum().backward()
    for tensor, reference_tensor in zip(inputs, reference_inputs):
        torch.testing.assert_close(tensor.grad, reference_tensor.grad, atol=1e-8, rtol=1e-7)
    torch.testing.assert_close(
        {name: parameter.grad for name, parameter in decoder.named_parameters()},
        {name: parameter.grad for name, parameter in reference.named_parameters()},
        atol=1e-8, rtol=1e-7,
    )
    assert decoder.state_dict().keys() == reference.state_dict().keys()


def _nested_stochastic_activation() -> nn.Module:
    return nn.Sequential(nn.LeakyReLU(0.2), nn.Dropout(0.25))


@pytest.mark.parametrize("activation_factory,dropout", [
    (nn.GELU, 0.2), (nn.RReLU, 0.0), (nn.Dropout, 0.0), (nn.AlphaDropout, 0.0),
    (_nested_stochastic_activation, 0.0),
])
def test_incremental_cache_preserves_stochastic_training_and_eval_behavior(
    activation_factory: ActivationFactory, dropout: float,
) -> None:
    decoder = MATQCXDecoder(
        MATQCXDecoderConfig(d_model=8, nhead=2, num_layers=2, dropout=dropout, act_fn_cls=activation_factory),
        max_agents=4, input_d_model=6, action_d_model=5, memory_d_model=7,
    )
    queries = torch.randn(2, 4, 6)
    actions = torch.randn(2, 4, 5)
    memory = torch.randn(2, 4, 7)
    assert decoder.create_step_cache(memory) is None
    torch.manual_seed(47)
    expected = _decode_steps(decoder, queries, actions, memory, None, cached=False)
    torch.manual_seed(47)
    outputs = _decode_steps(decoder, queries, actions, memory, None, cached=True)
    torch.testing.assert_close(outputs, expected, atol=0.0, rtol=0.0)
    decoder.eval()
    assert decoder.create_step_cache(memory) is not None
    torch.testing.assert_close(
        _decode_steps(decoder, queries, actions, memory, None, cached=True),
        _decode_steps(decoder, queries, actions, memory, None, cached=False),
    )


def test_incremental_cache_observes_individual_dropout_modes_and_probabilities() -> None:
    decoder = MATQCXDecoder(
        MATQCXDecoderConfig(d_model=8, nhead=2, num_layers=2, dropout=0.2),
        max_agents=4, input_d_model=6, action_d_model=5, memory_d_model=7,
    ).eval()
    memory = torch.randn(2, 4, 7)
    dropout = decoder.layers[-1].dropout
    dropout.train()
    assert decoder.create_step_cache(memory) is None
    dropout.p = 0.0
    assert decoder.create_step_cache(memory) is not None


@pytest.mark.parametrize("attention_name", ["query_context_attn", "memory_attn"])
def test_incremental_cache_observes_attention_dropout_with_other_layers_in_eval_mode(attention_name: str) -> None:
    decoder = MATQCXDecoder(
        MATQCXDecoderConfig(d_model=8, nhead=2, num_layers=2, dropout=0.2),
        max_agents=4, input_d_model=6, action_d_model=5, memory_d_model=7,
    ).eval()
    memory = torch.randn(2, 4, 7)
    attention = getattr(decoder.layers[0], attention_name)
    attention.train()
    assert decoder.create_step_cache(memory) is None
    queries = torch.randn(2, 4, 6)
    actions = torch.randn(2, 4, 5)
    torch.manual_seed(91)
    expected = _decode_steps(decoder, queries, actions, memory, None, cached=False)
    torch.manual_seed(91)
    actual = _decode_steps(decoder, queries, actions, memory, None, cached=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    attention.dropout = 0.0
    assert decoder.create_step_cache(memory) is not None


def test_incremental_joint_decoding_compiles_without_graph_breaks_or_stale_cache() -> None:
    decoder = MATQCXDecoder(
        MATQCXDecoderConfig(d_model=8, nhead=2, num_layers=2, assume_agent_mask_is_active_prefix=False),
        max_agents=4, input_d_model=6, action_d_model=5, memory_d_model=7,
    ).eval()

    def decode(queries: torch.Tensor, actions: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return _decode_steps(decoder, queries, actions, memory, mask, cached=True)

    compiled: Callable[..., torch.Tensor] = torch.compile(decode, backend="eager", fullgraph=True)
    mask = torch.tensor([[False, True, False, True], [False] * 4])
    for offset in (0.0, 0.7):
        queries = torch.randn(2, 4, 6) + offset
        actions = torch.randn(2, 4, 5)
        memory = torch.randn(2, 4, 7) - offset
        torch.testing.assert_close(
            compiled(queries, actions, memory, mask),
            _decode_steps(decoder, queries, actions, memory, mask, cached=False),
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_incremental_cuda_outputs_and_gradients_match_prefix_execution() -> None:
    torch.manual_seed(862)
    decoder = MATQCXDecoder(
        MATQCXDecoderConfig(
            d_model=16, nhead=4, num_layers=3, norm_first=False,
            assume_agent_mask_is_active_prefix=False,
        ),
        max_agents=4, input_d_model=12, action_d_model=7, memory_d_model=13,
    ).cuda()
    reference = copy.deepcopy(decoder)
    inputs = tuple(torch.randn(3, 4, width, device="cuda", requires_grad=True) for width in (12, 7, 13))
    reference_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    mask = torch.tensor([[False, True, False, True], [True, True, False, False], [False] * 4], device="cuda")
    outputs = _decode_steps(decoder, *inputs, mask, cached=True)
    expected = _decode_steps(reference, *reference_inputs, mask, cached=False)
    torch.testing.assert_close(outputs, expected, atol=1e-5, rtol=1e-5)
    weights = torch.randn_like(outputs)
    (outputs * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(
        {name: parameter.grad for name, parameter in decoder.named_parameters()},
        {name: parameter.grad for name, parameter in reference.named_parameters()},
        atol=5e-5, rtol=1e-4,
    )
