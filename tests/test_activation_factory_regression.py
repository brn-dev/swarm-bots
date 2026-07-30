import torch
from torch import nn

from swarmbots.learn.algos.mat import MATEncoder, MATEncoderConfig, MLPConfig
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig
from swarmbots.learn.algos.world_modeling.transformer_transition_model import (
    TransformerTransitionModel,
    TransformerTransitionModelConfig,
)
from swarmbots.learn.nn_components.activations import (
    ParameterLearnMode,
    SignedSquaredLeakyRelu,
    SignedSquaredLeakyReluFactory,
)
from swarmbots.learn.nn_components.feed_forward import MLP


def _fill_linear_with_ones(linear: nn.Linear) -> nn.Linear:
    nn.init.ones_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


def test_mlp_uses_regular_activation_class_between_linear_layers() -> None:
    mlp = MLP(
        input_dim=2,
        hidden_dims=[2, 1],
        end_with_act_fn=False,
        start_with_act_fn=True,
        linear_init=_fill_linear_with_ones,
        act_fn_cls=nn.ReLU,
    )
    x = torch.tensor([[-2.0, 3.0], [4.0, -5.0]])

    output = mlp(x)

    torch.testing.assert_close(output, torch.tensor([[6.0], [8.0]]))


def test_mlp_passes_feature_counts_to_feature_aware_activation_factories() -> None:
    mlp = MLP(
        input_dim=3,
        hidden_dims=[5, 2],
        end_with_act_fn=True,
        start_with_act_fn=True,
        act_fn_cls=SignedSquaredLeakyReluFactory(negative_slope_mode=ParameterLearnMode.PER_FEATURE),
    )

    activations = [module for module in mlp.modules() if isinstance(module, SignedSquaredLeakyRelu)]

    assert [tuple(activation.negative_slope.shape) for activation in activations] == [(3,), (5,), (2,)]
    assert mlp(torch.randn(4, 3)).shape == (4, 2)


def test_mlp_supports_biasless_linear_layers_and_dropout() -> None:
    mlp = MLP(
        input_dim=3,
        hidden_dims=[5, 2],
        end_with_act_fn=False,
        bias=False,
        dropout=0.25,
    )

    linear_layers = [module for module in mlp.modules() if isinstance(module, nn.Linear)]

    assert all(linear.bias is None for linear in linear_layers)
    assert any(isinstance(module, nn.Dropout) for module in mlp.modules())
    assert mlp(torch.randn(4, 3)).shape == (4, 2)


def test_mat_encoder_accepts_regular_transformer_activation_class() -> None:
    encoder = MATEncoder(
        MATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            act_fn_cls=nn.GELU,
        ),
        max_agents=3,
        local_obs_dim=5,
        global_obs_dim=0,
    )

    out = encoder(torch.randn(2, 3, 5), torch.empty(2, 0))

    assert isinstance(encoder.layers[0].activation, nn.GELU)
    assert out.shape == (2, 3, 8)
    assert torch.isfinite(out).all()


def test_mat_encoder_custom_transformer_feedforward_passes_feature_counts_to_activation_factories() -> None:
    encoder = MATEncoder(
        MATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            transformer_ff_config=MLPConfig(hidden_dims=[13, 11]),
            act_fn_cls=SignedSquaredLeakyReluFactory(negative_slope_mode=ParameterLearnMode.PER_FEATURE),
        ),
        max_agents=3,
        local_obs_dim=5,
        global_obs_dim=0,
    )

    activations = [
        module
        for module in encoder.layers[0].feedforward.modules()
        if isinstance(module, SignedSquaredLeakyRelu)
    ]

    assert [tuple(activation.negative_slope.shape) for activation in activations] == [(13,), (11,)]
    assert encoder(torch.randn(2, 3, 5), torch.empty(2, 0)).shape == (2, 3, 8)


def test_rmat_encoder_accepts_regular_transformer_activation_class() -> None:
    encoder = RMATEncoder(
        RMATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            act_fn_cls=nn.GELU,
        ),
        max_agents=3,
        local_obs_dim=5,
        global_obs_dim=0,
    )

    out, _ = encoder(torch.randn(2, 3, 5), torch.empty(2, 0))

    assert any(isinstance(module, nn.GELU) for module in encoder.layers[0].feedforward.modules())
    assert out.shape == (2, 3, 8)
    assert torch.isfinite(out).all()


def test_rmat_encoder_custom_transformer_feedforward_passes_feature_counts_to_activation_factories() -> None:
    encoder = RMATEncoder(
        RMATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            transformer_ff_config=MLPConfig(hidden_dims=[13, 11]),
            inter_module_mlp=True,
            act_fn_cls=SignedSquaredLeakyReluFactory(negative_slope_mode=ParameterLearnMode.PER_FEATURE),
        ),
        max_agents=3,
        local_obs_dim=5,
        global_obs_dim=0,
    )

    assert encoder.layers[0].inter_module_feedforward is not None
    for feedforward in [encoder.layers[0].feedforward, encoder.layers[0].inter_module_feedforward]:
        activations = [
            module
            for module in feedforward.modules()
            if isinstance(module, SignedSquaredLeakyRelu)
        ]
        assert [tuple(activation.negative_slope.shape) for activation in activations] == [(13,), (11,)]

    out, _ = encoder(torch.randn(2, 3, 5), torch.empty(2, 0))
    assert out.shape == (2, 3, 8)


def test_transition_model_accepts_regular_transformer_and_mlp_activation_classes() -> None:
    model = TransformerTransitionModel(
        TransformerTransitionModelConfig(
            n_agents=3,
            latent_dim=8,
            action_dim=2,
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            act_fn_cls=nn.GELU,
            coembed_mlp_hidden_dims=[8],
            head_mlp_hidden_dims=[8],
        )
    )

    out = model(torch.randn(2, 3, 8), torch.randn(2, 3, 2))

    assert isinstance(model.encoder.layers[0].activation, nn.GELU)
    assert any(isinstance(module, nn.GELU) for module in model.coembed.modules())
    assert any(isinstance(module, nn.GELU) for module in model.head.modules())
    assert out.shape == (2, 3, 8)
    assert torch.isfinite(out).all()
