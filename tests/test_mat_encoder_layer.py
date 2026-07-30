import math

import pytest
import torch
from torch import nn

from swarmbots.learn.algos.mat import MATEncoder, MATEncoderConfig, MATEncoderLayer, MLPConfig
from swarmbots.learn.nn_components.activations import make_activation


def _make_reference_layer(config: MATEncoderConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=config.d_model,
        nhead=config.nhead,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        activation=make_activation(config.act_fn_cls, num_features=config.dim_feedforward),
        layer_norm_eps=config.layer_norm_eps,
        batch_first=True,
        norm_first=config.norm_first,
        bias=config.bias,
    )


def _make_reference_encoder(config: MATEncoderConfig) -> nn.TransformerEncoder:
    return nn.TransformerEncoder(
        _make_reference_layer(config),
        num_layers=config.num_layers,
        norm=nn.LayerNorm(config.d_model, eps=config.layer_norm_eps, bias=config.bias),
        enable_nested_tensor=not config.norm_first,
    )


def _copy_torch_layer_to_mat_layer(
        source: nn.TransformerEncoderLayer,
        target: MATEncoderLayer,
) -> None:
    target.self_attn.load_state_dict(source.self_attn.state_dict())
    target.linear1.load_state_dict(source.linear1.state_dict())
    target.linear2.load_state_dict(source.linear2.state_dict())
    target.norm1.load_state_dict(source.norm1.state_dict())
    target.norm2.load_state_dict(source.norm2.state_dict())


def _copy_mat_layer_to_torch_layer(
        source: MATEncoderLayer,
        target: nn.TransformerEncoderLayer,
) -> None:
    target.self_attn.load_state_dict(source.self_attn.state_dict())
    target.linear1.load_state_dict(source.linear1.state_dict())
    target.linear2.load_state_dict(source.linear2.state_dict())
    target.norm1.load_state_dict(source.norm1.state_dict())
    target.norm2.load_state_dict(source.norm2.state_dict())


def _feedforward_linear_layers(layer: MATEncoderLayer) -> list[nn.Linear]:
    return [module for module in layer.feedforward if isinstance(module, nn.Linear)]


def _orthogonal_weight_norm(linear: nn.Linear, gain: float) -> torch.Tensor:
    return torch.tensor(gain * math.sqrt(min(linear.weight.shape)), dtype=linear.weight.dtype)


@pytest.mark.parametrize("norm_first", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_mat_encoder_layer_matches_pytorch_default_feedforward(norm_first: bool, bias: bool) -> None:
    torch.manual_seed(123)
    config = MATEncoderConfig(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        act_fn_cls=nn.GELU,
        norm_first=norm_first,
        layer_norm_eps=1e-4,
        bias=bias,
        transformer_ff_init_gain=None,
    )
    reference_layer = _make_reference_layer(config)
    mat_layer = MATEncoderLayer(config)
    _copy_torch_layer_to_mat_layer(reference_layer, mat_layer)
    src = torch.randn(3, 4, config.d_model)
    src_key_padding_mask = torch.tensor([
        [False, False, True, False],
        [False, True, True, False],
        [False, False, False, False],
    ])
    src_mask = torch.tensor([
        [False, True, False, False],
        [False, False, False, True],
        [True, False, False, False],
        [False, False, True, False],
    ])

    expected = reference_layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    actual = mat_layer(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("norm_first", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_mat_encoder_direct_stack_matches_pytorch_encoder_for_default_feedforward(
        norm_first: bool,
        bias: bool,
) -> None:
    torch.manual_seed(123)
    config = MATEncoderConfig(
        d_model=8,
        nhead=2,
        num_layers=2,
        dim_feedforward=16,
        dropout=0.0,
        act_fn_cls=nn.GELU,
        norm_first=norm_first,
        layer_norm_eps=1e-4,
        bias=bias,
        transformer_ff_init_gain=None,
    )
    mat_encoder = MATEncoder(
        config,
        max_agents=4,
        local_obs_dim=5,
        global_obs_dim=0,
    )
    reference_encoder = _make_reference_encoder(config)
    for mat_layer, torch_layer in zip(mat_encoder.layers, reference_encoder.layers, strict=True):
        _copy_mat_layer_to_torch_layer(mat_layer, torch_layer)
    reference_encoder.norm.load_state_dict(mat_encoder.norm.state_dict())

    tokens = torch.randn(3, 4, config.d_model)
    src_key_padding_mask = torch.tensor([
        [False, False, True, False],
        [False, True, True, False],
        [False, False, False, False],
    ])
    actual = tokens
    for layer in mat_encoder.layers:
        actual = layer(actual, src_key_padding_mask=src_key_padding_mask)
    actual = mat_encoder.norm(actual)

    expected = reference_encoder(tokens, src_key_padding_mask=src_key_padding_mask)
    torch.testing.assert_close(actual, expected)


def test_mat_encoder_forward_matches_reference_stack_with_embedding_options() -> None:
    torch.manual_seed(123)
    config = MATEncoderConfig(
        d_model=8,
        nhead=2,
        num_layers=2,
        dim_feedforward=16,
        dropout=0.0,
        act_fn_cls=nn.GELU,
        norm_first=True,
        add_agent_embeddings=True,
        local_obs_encoder_config=MLPConfig(hidden_dims=[7]),
        global_obs_encoder_config=MLPConfig(hidden_dims=[6]),
        normalize_obs_inputs=True,
        normalize_tokens=True,
        transformer_ff_init_gain=None,
    )
    mat_encoder = MATEncoder(
        config,
        max_agents=4,
        local_obs_dim=5,
        global_obs_dim=3,
    )
    reference_encoder = _make_reference_encoder(config)
    for mat_layer, torch_layer in zip(mat_encoder.layers, reference_encoder.layers, strict=True):
        _copy_mat_layer_to_torch_layer(mat_layer, torch_layer)
    reference_encoder.norm.load_state_dict(mat_encoder.norm.state_dict())

    local_obs = torch.randn(3, 4, 5)
    global_obs = torch.randn(3, 3)
    agent_mask = torch.tensor([
        [True, True, True, False],
        [True, False, True, True],
        [False, True, True, True],
    ])
    tokens = mat_encoder.local_obs_encoder(mat_encoder.local_obs_input_norm(local_obs))
    assert mat_encoder.agent_embeddings is not None
    tokens = tokens + mat_encoder.agent_embeddings[:, :local_obs.shape[1], :]
    global_tokens = mat_encoder.global_obs_encoder(mat_encoder.global_obs_input_norm(global_obs))
    tokens = tokens + global_tokens.unsqueeze(1).expand(-1, local_obs.shape[1], -1)
    tokens = mat_encoder.token_norm(tokens)

    expected = reference_encoder(tokens, src_key_padding_mask=~agent_mask)
    actual = mat_encoder(local_obs, global_obs, agent_mask=agent_mask)

    torch.testing.assert_close(actual, expected)


def test_mat_encoder_uses_configurable_transformer_feedforward_hidden_dims() -> None:
    torch.manual_seed(123)
    config = MATEncoderConfig(
        d_model=8,
        nhead=2,
        num_layers=2,
        dim_feedforward=16,
        transformer_ff_config=MLPConfig(hidden_dims=[13, 11, 9]),
        transformer_ff_init_gain=1.25,
        dropout=0.0,
        act_fn_cls=nn.GELU,
    )
    encoder = MATEncoder(
        config,
        max_agents=4,
        local_obs_dim=5,
        global_obs_dim=3,
    )
    first_layer = encoder.layers[0]
    second_layer = encoder.layers[1]
    linear_layers = _feedforward_linear_layers(first_layer)

    assert [(linear.in_features, linear.out_features) for linear in linear_layers] == [
        (8, 13),
        (13, 11),
        (11, 9),
        (9, 8),
    ]
    assert sum(isinstance(module, nn.GELU) for module in first_layer.feedforward) == 3
    assert not torch.equal(first_layer.self_attn.in_proj_weight, second_layer.self_attn.in_proj_weight)
    assert not torch.equal(linear_layers[0].weight, _feedforward_linear_layers(second_layer)[0].weight)

    for hidden_linear in linear_layers[:-1]:
        torch.testing.assert_close(
            torch.linalg.vector_norm(hidden_linear.weight),
            _orthogonal_weight_norm(hidden_linear, gain=1.25),
            rtol=1e-5,
            atol=1e-6,
        )
    torch.testing.assert_close(
        torch.linalg.vector_norm(linear_layers[-1].weight),
        _orthogonal_weight_norm(linear_layers[-1], gain=1.0),
        rtol=1e-5,
        atol=1e-6,
    )

    out = encoder(
        torch.randn(3, 4, 5),
        torch.randn(3, 3),
        agent_mask=torch.tensor([
            [True, True, True, True],
            [True, False, True, False],
            [False, True, True, True],
        ]),
    )

    assert out.shape == (3, 4, config.d_model)
    assert torch.isfinite(out).all()


def test_mat_encoder_keeps_cloned_custom_feedforward_when_transformer_init_is_disabled() -> None:
    torch.manual_seed(123)
    encoder = MATEncoder(
        MATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            transformer_ff_config=MLPConfig(hidden_dims=[13, 11]),
            transformer_ff_init_gain=None,
        ),
        max_agents=4,
        local_obs_dim=5,
        global_obs_dim=0,
    )
    first_layer = encoder.layers[0]
    second_layer = encoder.layers[1]

    torch.testing.assert_close(first_layer.self_attn.in_proj_weight, second_layer.self_attn.in_proj_weight)
    for first_linear, second_linear in zip(
            _feedforward_linear_layers(first_layer),
            _feedforward_linear_layers(second_layer),
            strict=True,
    ):
        torch.testing.assert_close(first_linear.weight, second_linear.weight)
