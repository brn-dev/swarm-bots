import math
from collections.abc import Callable

import pytest
import torch
from torch import nn

from swarmbots.learn.algos.mat import MATEncoder, MATEncoderConfig, MLPConfig, SwiGLUConfig
from swarmbots.learn.algos.r_mat import RMATEncoder, RMATEncoderConfig
from swarmbots.learn.nn_components.feed_forward import (
    BilinearGLU,
    BilinearGLUConfig,
    GEGLU,
    GEGLUConfig,
    GLU,
    GLUConfig,
    GLUStackConfig,
    MLP,
    ReGLU,
    ReGLUConfig,
    StackedGLU,
    SwiGLU,
    feedforward_linear_layers,
    make_feedforward,
)
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal
from swarmbots.learn.serialization_utils import serialize_dataclass


def _orthogonal_weight_norm(linear: nn.Linear, gain: float) -> torch.Tensor:
    return torch.tensor(gain * math.sqrt(min(linear.weight.shape)), dtype=linear.weight.dtype)


def _assert_swiglu_layout(
        module: nn.Module,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
) -> StackedGLU | SwiGLU:
    if n_layers == 1:
        assert isinstance(module, SwiGLU)
        layers = [module]
    else:
        assert isinstance(module, StackedGLU)
        layers = list(module)
    assert len(layers) == n_layers
    for layer_idx, layer in enumerate(layers):
        assert isinstance(layer, SwiGLU)
        expected_input_dim = input_dim if layer_idx == 0 else output_dim
        assert (layer.gate_projection.in_features, layer.gate_projection.out_features) == (
            expected_input_dim,
            hidden_dim,
        )
        assert (layer.value_projection.in_features, layer.value_projection.out_features) == (
            expected_input_dim,
            hidden_dim,
        )
        assert (layer.output_projection.in_features, layer.output_projection.out_features) == (
            hidden_dim,
            output_dim,
        )
    return module


@pytest.mark.parametrize(
    ("glu_cls", "activation"),
    [
        (GLU, torch.sigmoid),
        (BilinearGLU, lambda inputs: inputs),
        (ReGLU, torch.relu),
        (GEGLU, nn.functional.gelu),
        (SwiGLU, nn.functional.silu),
    ],
)
def test_glu_variants_apply_the_expected_gate_activation(
        glu_cls: type[GLU],
        activation: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    torch.manual_seed(1)
    layer = glu_cls(input_dim=3, hidden_dim=5, output_dim=2, dropout=0.0)
    inputs = torch.randn(4, 3)

    expected = nn.functional.linear(
        activation(nn.functional.linear(
            inputs,
            layer.gate_projection.weight,
            layer.gate_projection.bias,
        )) * nn.functional.linear(
            inputs,
            layer.value_projection.weight,
            layer.value_projection.bias,
        ),
        layer.output_projection.weight,
        layer.output_projection.bias,
    )

    torch.testing.assert_close(layer(inputs), expected)


def test_stacked_swiglu_chains_complete_blocks_and_backpropagates() -> None:
    torch.manual_seed(2)
    swiglu = StackedGLU(
        input_dim=3,
        hidden_dim=7,
        output_dim=5,
        glu_cls=SwiGLU,
        n_layers=3,
        dropout=0.0,
    )
    inputs = torch.randn(2, 4, 3, requires_grad=True)

    expected = inputs
    for layer in swiglu:
        expected = layer(expected)
    actual = swiglu(inputs)
    actual.square().mean().backward()

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 4, 5)
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert all(parameter.grad is not None for parameter in swiglu.parameters())


def test_stacked_glu_applies_pre_norm_and_residuals_to_width_preserving_layers() -> None:
    torch.manual_seed(3)
    stacked_glu = StackedGLU(
        input_dim=3,
        hidden_dim=7,
        output_dim=5,
        glu_cls=ReGLU,
        n_layers=3,
        pre_norm=nn.LayerNorm,
        residual=True,
        dropout=0.0,
    )
    inputs = torch.randn(2, 3)

    expected = stacked_glu[0](stacked_glu.pre_norms[0](inputs))
    expected = expected + stacked_glu[1](stacked_glu.pre_norms[1](expected))
    expected = expected + stacked_glu[2](stacked_glu.pre_norms[2](expected))

    torch.testing.assert_close(stacked_glu(inputs), expected)
    assert [norm.normalized_shape for norm in stacked_glu.pre_norms] == [(3,), (5,), (5,)]


def test_stacked_glu_can_skip_first_pre_norm() -> None:
    stacked_glu = StackedGLU(
        input_dim=5,
        hidden_dim=7,
        output_dim=5,
        glu_cls=GEGLU,
        n_layers=3,
        pre_norm=nn.LayerNorm,
        norm_first_layer=False,
    )

    assert isinstance(stacked_glu.pre_norms[0], nn.Identity)
    assert all(isinstance(norm, nn.LayerNorm) for norm in stacked_glu.pre_norms[1:])


def test_stacked_glu_can_skip_first_residual() -> None:
    torch.manual_seed(4)
    stacked_glu = StackedGLU(
        input_dim=5,
        hidden_dim=7,
        output_dim=5,
        glu_cls=GEGLU,
        n_layers=3,
        residual=True,
        residual_first_layer=False,
    )
    inputs = torch.randn(2, 5)

    expected = stacked_glu[0](inputs)
    expected = expected + stacked_glu[1](expected)
    expected = expected + stacked_glu[2](expected)

    torch.testing.assert_close(stacked_glu(inputs), expected)


def test_glu_stack_config_requires_multiple_layers() -> None:
    with pytest.raises(ValueError, match="n_layers > 1"):
        GLUStackConfig(n_layers=1)


@pytest.mark.parametrize(
    ("config", "expected_type", "expected_linear_shapes"),
    [
        (MLPConfig(), nn.Linear, [(3, 5)]),
        (MLPConfig(hidden_dims=[7, 6]), MLP, [(3, 7), (7, 6), (6, 5)]),
    ],
)
def test_make_feedforward_builds_mlp_config_layouts(
        config: MLPConfig,
        expected_type: type[nn.Module],
        expected_linear_shapes: list[tuple[int, int]],
) -> None:
    module = make_feedforward(input_dim=3, output_dim=5, config=config)
    linear_layers = [child for child in module.modules() if isinstance(child, nn.Linear)]

    assert isinstance(module, expected_type)
    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == expected_linear_shapes
    assert module(torch.randn(2, 3)).shape == (2, 5)


def test_make_feedforward_builds_deep_swiglu_config() -> None:
    module = make_feedforward(
        input_dim=3,
        output_dim=5,
        config=SwiGLUConfig(
            hidden_dim=7,
            stacked=GLUStackConfig(
                n_layers=3,
                pre_norm=nn.LayerNorm,
                norm_first_layer=False,
                residual=True,
                residual_first_layer=False,
            ),
        ),
    )

    stacked_glu = _assert_swiglu_layout(
        module,
        input_dim=3,
        hidden_dim=7,
        output_dim=5,
        n_layers=3,
    )
    assert isinstance(stacked_glu, StackedGLU)
    assert stacked_glu.residual
    assert not stacked_glu.residual_first_layer
    assert isinstance(stacked_glu.pre_norms[0], nn.Identity)
    assert all(isinstance(norm, nn.LayerNorm) for norm in stacked_glu.pre_norms[1:])


def test_glu_stack_defaults_to_no_norm_or_residual() -> None:
    module = make_feedforward(
        input_dim=3,
        output_dim=5,
        config=ReGLUConfig(
            hidden_dim=7,
            stacked=GLUStackConfig(n_layers=2),
        ),
    )

    assert isinstance(module, StackedGLU)
    assert not module.residual
    assert all(isinstance(norm, nn.Identity) for norm in module.pre_norms)


def test_transformer_style_stacked_swiglu_supports_full_graph_compilation() -> None:
    module = make_feedforward(
        input_dim=8,
        output_dim=8,
        config=SwiGLUConfig(
            hidden_dim=13,
            stacked=GLUStackConfig(
                n_layers=2,
                pre_norm=nn.LayerNorm,
                norm_first_layer=False,
                residual=True,
                residual_first_layer=False,
            ),
        ),
        dropout=0.0,
    )
    compiled_module = torch.compile(module, backend="eager", fullgraph=True, dynamic=False)
    inputs = torch.randn(2, 4, 8)

    torch.testing.assert_close(compiled_module(inputs), module(inputs))


@pytest.mark.parametrize(
    ("config", "expected_type"),
    [
        (GLUConfig(hidden_dim=7), GLU),
        (BilinearGLUConfig(hidden_dim=7), BilinearGLU),
        (ReGLUConfig(hidden_dim=7), ReGLU),
        (GEGLUConfig(hidden_dim=7), GEGLU),
        (SwiGLUConfig(hidden_dim=7), SwiGLU),
    ],
)
def test_make_feedforward_builds_single_glu_variants(
        config: GLUConfig,
        expected_type: type[GLU],
) -> None:
    module = make_feedforward(input_dim=3, output_dim=5, config=config)

    assert type(module) is expected_type
    assert module(torch.randn(2, 3)).shape == (2, 5)


def test_mat_encoder_supports_swiglu_in_every_feedforward_site() -> None:
    torch.manual_seed(3)
    encoder = MATEncoder(
        MATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            dropout=0.0,
            transformer_ff_config=SwiGLUConfig(
                hidden_dim=13,
                stacked=GLUStackConfig(n_layers=2),
            ),
            local_obs_encoder_config=SwiGLUConfig(
                hidden_dim=11,
                stacked=GLUStackConfig(n_layers=3),
            ),
            global_obs_encoder_config=SwiGLUConfig(hidden_dim=9),
        ),
        max_agents=4,
        local_obs_dim=5,
        global_obs_dim=3,
    )

    _assert_swiglu_layout(
        encoder.local_obs_encoder,
        input_dim=5,
        hidden_dim=11,
        output_dim=8,
        n_layers=3,
    )
    assert encoder.global_obs_encoder is not None
    _assert_swiglu_layout(
        encoder.global_obs_encoder,
        input_dim=3,
        hidden_dim=9,
        output_dim=8,
        n_layers=1,
    )
    for layer in encoder.layers:
        _assert_swiglu_layout(
            layer.feedforward,
            input_dim=8,
            hidden_dim=13,
            output_dim=8,
            n_layers=2,
        )
        assert isinstance(layer.activation, nn.SiLU)

    output = encoder(
        local_obs=torch.randn(3, 4, 5),
        global_obs=torch.randn(3, 3),
        agent_mask=torch.tensor([
            [True, True, True, False],
            [True, False, True, True],
            [True, True, True, True],
        ]),
    )
    output.mean().backward()

    assert output.shape == (3, 4, 8)
    assert torch.isfinite(output).all()
    assert all(parameter.grad is not None for parameter in encoder.parameters())


def test_mat_swiglu_transformer_initialization_uses_hidden_and_output_gains() -> None:
    torch.manual_seed(4)
    encoder = MATEncoder(
        MATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            transformer_ff_config=SwiGLUConfig(
                hidden_dim=13,
                stacked=GLUStackConfig(n_layers=2),
            ),
            transformer_ff_init_gain=1.25,
        ),
        max_agents=4,
        local_obs_dim=5,
        global_obs_dim=0,
    )
    first_layer = encoder.layers[0]
    second_layer = encoder.layers[1]
    hidden_layers, output_layers = feedforward_linear_layers(first_layer.feedforward)

    assert len(hidden_layers) == 4
    assert len(output_layers) == 2
    for linear in hidden_layers:
        torch.testing.assert_close(
            torch.linalg.vector_norm(linear.weight),
            _orthogonal_weight_norm(linear, gain=1.25),
            rtol=1e-5,
            atol=1e-6,
        )
    for linear in output_layers:
        torch.testing.assert_close(
            torch.linalg.vector_norm(linear.weight),
            _orthogonal_weight_norm(linear, gain=1.0),
            rtol=1e-5,
            atol=1e-6,
        )
    second_hidden_layers, second_output_layers = feedforward_linear_layers(second_layer.feedforward)
    assert not torch.equal(hidden_layers[0].weight, second_hidden_layers[0].weight)
    assert not torch.equal(output_layers[0].weight, second_output_layers[0].weight)


def test_mat_swiglu_transformer_layers_remain_clones_when_reinitialization_is_disabled() -> None:
    torch.manual_seed(5)
    encoder = MATEncoder(
        MATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            transformer_ff_config=SwiGLUConfig(
                hidden_dim=13,
                stacked=GLUStackConfig(n_layers=2),
            ),
            transformer_ff_init_gain=None,
        ),
        max_agents=4,
        local_obs_dim=5,
        global_obs_dim=0,
    )
    first_linears = feedforward_linear_layers(encoder.layers[0].feedforward)
    second_linears = feedforward_linear_layers(encoder.layers[1].feedforward)

    for first_linear, second_linear in zip(
            [*first_linears[0], *first_linears[1]],
            [*second_linears[0], *second_linears[1]],
            strict=True,
    ):
        torch.testing.assert_close(first_linear.weight, second_linear.weight)


@pytest.mark.parametrize(
    "removed_config_name",
    [
        "transformer_ff_hidden_dims",
        "local_obs_encoder_hidden_dims",
        "global_obs_encoder_hidden_dims",
    ],
)
def test_mat_encoder_config_rejects_removed_hidden_dim_arguments(removed_config_name: str) -> None:
    with pytest.raises(TypeError, match=removed_config_name):
        MATEncoderConfig(**{removed_config_name: [7]})


def test_mat_encoder_config_defaults_to_direct_observation_projections() -> None:
    config = MATEncoderConfig()

    assert config.transformer_ff_config is None
    assert config.local_obs_encoder_config == MLPConfig()
    assert config.global_obs_encoder_config == MLPConfig()


def test_rmat_encoder_supports_deep_swiglu_with_inter_module_blocks_and_masks() -> None:
    torch.manual_seed(7)
    encoder = RMATEncoder(
        RMATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            dropout=0.0,
            transformer_ff_config=SwiGLUConfig(
                hidden_dim=13,
                stacked=GLUStackConfig(n_layers=2),
            ),
            local_obs_encoder_config=SwiGLUConfig(
                hidden_dim=11,
                stacked=GLUStackConfig(n_layers=2),
            ),
            global_obs_encoder_config=SwiGLUConfig(hidden_dim=9),
            inter_module_mlp=True,
        ),
        max_agents=3,
        local_obs_dim=5,
        global_obs_dim=3,
    )
    _assert_swiglu_layout(
        encoder.local_obs_encoder,
        input_dim=5,
        hidden_dim=11,
        output_dim=8,
        n_layers=2,
    )
    assert encoder.global_obs_encoder is not None
    _assert_swiglu_layout(
        encoder.global_obs_encoder,
        input_dim=3,
        hidden_dim=9,
        output_dim=8,
        n_layers=1,
    )
    for layer in encoder.layers:
        _assert_swiglu_layout(
            layer.feedforward,
            input_dim=8,
            hidden_dim=13,
            output_dim=8,
            n_layers=2,
        )
        assert layer.inter_module_feedforward is not None
        _assert_swiglu_layout(
            layer.inter_module_feedforward,
            input_dim=8,
            hidden_dim=13,
            output_dim=8,
            n_layers=2,
        )

    local_obs = torch.randn(2, 4, 3, 5)
    global_obs = torch.randn(2, 4, 3)
    agent_mask = torch.tensor([
        [
            [True, True, True],
            [True, False, True],
            [True, True, True],
            [False, True, True],
        ],
        [
            [True, True, False],
            [True, True, True],
            [False, True, True],
            [True, True, True],
        ],
    ])
    time_mask = torch.tensor([
        [True, True, False, False],
        [True, False, True, True],
    ])
    output, state = encoder(
        local_obs,
        global_obs,
        agent_mask=agent_mask,
        time_mask=time_mask,
        reset_mask=torch.tensor([
            [True, False, False, False],
            [True, False, True, False],
        ]),
    )
    output.square().mean().backward()

    valid_mask = agent_mask & time_mask.unsqueeze(-1)
    assert output.shape == (2, 4, 3, 8)
    assert len(state) == 2
    torch.testing.assert_close(output[~valid_mask], torch.zeros_like(output[~valid_mask]))
    assert torch.isfinite(output).all()
    assert all(parameter.grad is not None for parameter in encoder.parameters())


def test_feedforward_configs_serialize_with_architecture_options() -> None:
    serialized = serialize_dataclass(
        MATEncoderConfig(
            transformer_ff_config=SwiGLUConfig(
                hidden_dim=192,
                stacked=GLUStackConfig(n_layers=3),
            ),
            local_obs_encoder_config=MLPConfig(hidden_dims=[128, 64]),
        )
    )

    assert serialized["transformer_ff_config"] == {
        "hidden_dim": 192,
        "stacked": {
            "n_layers": 3,
            "pre_norm": None,
            "norm_first_layer": True,
            "residual": False,
            "residual_first_layer": True,
        },
    }
    assert serialized["local_obs_encoder_config"] == {"hidden_dims": [128, 64]}
