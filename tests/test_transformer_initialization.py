import math

import torch
from torch import nn

from swarmbots.learn.algos.mat import MATEncoder, MATEncoderConfig, MLPConfig
from swarmbots.learn.algos.r_mat import RMATEncoderLayer
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig
from swarmbots.learn.algos.world_modeling.transformer_transition_model import (
    TransformerTransitionModel,
    TransformerTransitionModelConfig,
)


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    return [child for child in module if isinstance(child, nn.Linear)]


def _orthogonal_weight_norm(linear: nn.Linear, gain: float) -> torch.Tensor:
    return torch.tensor(gain * math.sqrt(min(linear.weight.shape)), dtype=linear.weight.dtype)


def test_mat_encoder_default_reinitializes_cloned_transformer_layers() -> None:
    torch.manual_seed(123)
    encoder = MATEncoder(
        MATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
        ),
        max_agents=4,
        local_obs_dim=5,
        global_obs_dim=0,
    )

    first_layer = encoder.layers[0]
    second_layer = encoder.layers[1]
    assert not torch.equal(first_layer.self_attn.in_proj_weight, second_layer.self_attn.in_proj_weight)
    assert not torch.equal(first_layer.linear1.weight, second_layer.linear1.weight)


def test_mat_encoder_can_keep_cloned_transformer_initialization() -> None:
    torch.manual_seed(123)
    encoder = MATEncoder(
        MATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            transformer_ff_init_gain=None,
        ),
        max_agents=4,
        local_obs_dim=5,
        global_obs_dim=0,
    )

    first_layer = encoder.layers[0]
    second_layer = encoder.layers[1]
    torch.testing.assert_close(first_layer.self_attn.in_proj_weight, second_layer.self_attn.in_proj_weight)
    torch.testing.assert_close(first_layer.linear1.weight, second_layer.linear1.weight)


def test_mat_encoder_proper_init_reinitializes_cloned_transformer_layers() -> None:
    torch.manual_seed(123)
    encoder = MATEncoder(
        MATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            transformer_ff_init_gain=1.5,
        ),
        max_agents=4,
        local_obs_dim=5,
        global_obs_dim=0,
    )

    first_layer = encoder.layers[0]
    second_layer = encoder.layers[1]
    assert not torch.equal(first_layer.self_attn.in_proj_weight, second_layer.self_attn.in_proj_weight)
    assert not torch.equal(first_layer.linear1.weight, second_layer.linear1.weight)

    expected_linear1_norm = 1.5 * math.sqrt(first_layer.linear1.weight.shape[1])
    torch.testing.assert_close(
        torch.linalg.vector_norm(first_layer.linear1.weight),
        torch.tensor(expected_linear1_norm),
        rtol=1e-5,
        atol=1e-6,
    )


def test_transition_model_default_reinitializes_cloned_transformer_layers() -> None:
    torch.manual_seed(123)
    model = TransformerTransitionModel(
        TransformerTransitionModelConfig(
            n_agents=4,
            latent_dim=8,
            action_dim=3,
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            act_fn_cls=nn.GELU,
        )
    )

    first_layer = model.encoder.layers[0]
    second_layer = model.encoder.layers[1]
    assert not torch.equal(first_layer.self_attn.in_proj_weight, second_layer.self_attn.in_proj_weight)
    assert not torch.equal(first_layer.linear1.weight, second_layer.linear1.weight)


def test_transition_model_can_keep_cloned_transformer_initialization() -> None:
    torch.manual_seed(123)
    model = TransformerTransitionModel(
        TransformerTransitionModelConfig(
            n_agents=4,
            latent_dim=8,
            action_dim=3,
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            act_fn_cls=nn.GELU,
            transformer_ff_init_gain=None,
        )
    )

    first_layer = model.encoder.layers[0]
    second_layer = model.encoder.layers[1]
    torch.testing.assert_close(first_layer.self_attn.in_proj_weight, second_layer.self_attn.in_proj_weight)
    torch.testing.assert_close(first_layer.linear1.weight, second_layer.linear1.weight)


def test_transition_model_proper_init_reinitializes_cloned_transformer_layers() -> None:
    torch.manual_seed(123)
    model = TransformerTransitionModel(
        TransformerTransitionModelConfig(
            n_agents=4,
            latent_dim=8,
            action_dim=3,
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            act_fn_cls=nn.GELU,
            transformer_ff_init_gain=1.5,
        )
    )

    first_layer = model.encoder.layers[0]
    second_layer = model.encoder.layers[1]
    assert not torch.equal(first_layer.self_attn.in_proj_weight, second_layer.self_attn.in_proj_weight)
    assert not torch.equal(first_layer.linear1.weight, second_layer.linear1.weight)


def test_rmat_encoder_uses_mat_init_gains() -> None:
    torch.manual_seed(123)
    encoder = RMATEncoder(
        RMATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            linear_init_gain=1.5,
            linear_projection_init_gain=1.0,
            transformer_ff_init_gain=1.5,
        ),
        max_agents=4,
        local_obs_dim=5,
        global_obs_dim=0,
    )

    assert isinstance(encoder.local_obs_encoder, nn.Linear)
    expected_projection_norm = math.sqrt(encoder.local_obs_encoder.weight.shape[1])
    torch.testing.assert_close(
        torch.linalg.vector_norm(encoder.local_obs_encoder.weight),
        torch.tensor(expected_projection_norm),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(encoder.layers[0].temporal_output_projection.weight),
        torch.tensor(math.sqrt(encoder.layers[0].temporal_output_projection.weight.shape[1])),
        rtol=1e-5,
        atol=1e-6,
    )

    first_layer = encoder.layers[0]
    second_layer = encoder.layers[1]
    assert not torch.equal(first_layer.self_attn.in_proj_weight, second_layer.self_attn.in_proj_weight)
    assert isinstance(first_layer.feedforward[0], nn.Linear)
    assert isinstance(second_layer.feedforward[0], nn.Linear)
    assert not torch.equal(first_layer.feedforward[0].weight, second_layer.feedforward[0].weight)

    expected_feedforward_norm = 1.5 * math.sqrt(first_layer.feedforward[0].weight.shape[1])
    torch.testing.assert_close(
        torch.linalg.vector_norm(first_layer.feedforward[0].weight),
        torch.tensor(expected_feedforward_norm),
        rtol=1e-5,
        atol=1e-6,
    )


def test_rmat_encoder_uses_configurable_transformer_feedforward_hidden_dims() -> None:
    torch.manual_seed(123)
    encoder = RMATEncoder(
        RMATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            transformer_ff_config=MLPConfig(hidden_dims=[13, 11]),
            transformer_ff_init_gain=1.5,
            inter_module_mlp=True,
        ),
        max_agents=4,
        local_obs_dim=5,
        global_obs_dim=0,
    )
    first_layer = encoder.layers[0]
    second_layer = encoder.layers[1]
    feedforward_linears = _linear_layers(first_layer.feedforward)

    assert [(linear.in_features, linear.out_features) for linear in feedforward_linears] == [
        (8, 13),
        (13, 11),
        (11, 8),
    ]
    assert first_layer.inter_module_feedforward is not None
    assert [
        (linear.in_features, linear.out_features)
        for linear in _linear_layers(first_layer.inter_module_feedforward)
    ] == [
        (8, 13),
        (13, 11),
        (11, 8),
    ]
    assert not torch.equal(first_layer.self_attn.in_proj_weight, second_layer.self_attn.in_proj_weight)
    assert not torch.equal(feedforward_linears[0].weight, _linear_layers(second_layer.feedforward)[0].weight)

    for feedforward in [first_layer.feedforward, first_layer.inter_module_feedforward]:
        assert feedforward is not None
        linears = _linear_layers(feedforward)
        for hidden_linear in linears[:-1]:
            torch.testing.assert_close(
                torch.linalg.vector_norm(hidden_linear.weight),
                _orthogonal_weight_norm(hidden_linear, gain=1.5),
                rtol=1e-5,
                atol=1e-6,
            )
        torch.testing.assert_close(
            torch.linalg.vector_norm(linears[-1].weight),
            _orthogonal_weight_norm(linears[-1], gain=1.0),
            rtol=1e-5,
            atol=1e-6,
        )


def test_rmat_encoder_exposes_direct_layer_stack_like_mat_encoder() -> None:
    encoder = RMATEncoder(
        RMATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
        ),
        max_agents=4,
        local_obs_dim=5,
        global_obs_dim=0,
    )

    assert isinstance(encoder.layers, nn.ModuleList)
    assert all(isinstance(layer, RMATEncoderLayer) for layer in encoder.layers)
    assert not hasattr(encoder, "encoder")
    assert any(key.startswith("layers.0.") for key in encoder.state_dict())
    assert all(not key.startswith("encoder.") for key in encoder.state_dict())


def test_rmat_encoder_bias_false_removes_stack_biases() -> None:
    encoder = RMATEncoder(
        RMATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            bias=False,
            inter_module_mlp=True,
        ),
        max_agents=4,
        local_obs_dim=5,
        global_obs_dim=3,
    )
    layer = encoder.layers[0]

    assert isinstance(encoder.local_obs_encoder, nn.Linear)
    assert isinstance(encoder.global_obs_encoder, nn.Linear)
    assert encoder.local_obs_encoder.bias is None
    assert encoder.global_obs_encoder.bias is None
    assert layer.self_attn.in_proj_bias is None
    assert layer.self_attn.out_proj.bias is None
    assert encoder.norm.bias is None
    assert layer.temporal_norm.bias is None
    assert layer.attention_norm.bias is None
    assert layer.feedforward_norm.bias is None
    assert layer.inter_module_feedforward_norm is not None
    assert layer.inter_module_feedforward_norm.bias is None

    for feedforward in [layer.feedforward, layer.inter_module_feedforward]:
        assert feedforward is not None
        assert all(linear.bias is None for linear in _linear_layers(feedforward))
