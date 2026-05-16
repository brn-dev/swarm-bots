import torch
from torch import nn

from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig
from swarmbots.learn.algos.world_modeling.transformer_transition_model import (
    TransformerTransitionModel,
    TransformerTransitionModelConfig,
)
from swarmbots.learn.nn_components.mlp import MLP
from swarmbots.learn.nn_components.nn_init import LinearInitialization, init_linear_orthogonal


class LegacyMLP(nn.Sequential):

    def __init__(
            self,
            input_dim: int,
            hidden_dims: list[int],
            end_with_act_fn: bool,
            start_with_act_fn: bool = False,
            linear_init: LinearInitialization = init_linear_orthogonal,
            act_fn_cls=nn.Tanh,
    ) -> None:
        dims = [input_dim, *hidden_dims]
        n_layers = len(dims) - 1

        modules: list[nn.Module] = []
        if start_with_act_fn:
            modules.append(act_fn_cls())

        for i in range(n_layers):
            linear = nn.Linear(dims[i], dims[i + 1])
            linear_init(linear)
            modules.append(linear)

            if i < n_layers - 1 or end_with_act_fn:
                modules.append(act_fn_cls())

        super().__init__(*modules)


def test_mlp_with_regular_activation_class_matches_legacy_construction() -> None:
    torch.manual_seed(123)
    current_mlp = MLP(
        input_dim=4,
        hidden_dims=[7, 3],
        end_with_act_fn=True,
        start_with_act_fn=True,
        act_fn_cls=nn.GELU,
    )

    torch.manual_seed(123)
    legacy_mlp = LegacyMLP(
        input_dim=4,
        hidden_dims=[7, 3],
        end_with_act_fn=True,
        start_with_act_fn=True,
        act_fn_cls=nn.GELU,
    )

    assert [type(module) for module in current_mlp] == [type(module) for module in legacy_mlp]
    assert current_mlp.state_dict().keys() == legacy_mlp.state_dict().keys()
    for key, current_value in current_mlp.state_dict().items():
        torch.testing.assert_close(current_value, legacy_mlp.state_dict()[key])

    x = torch.randn(5, 4)
    torch.testing.assert_close(current_mlp(x), legacy_mlp(x))


def test_mat_encoder_keeps_regular_transformer_activation_class() -> None:
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

    activation = encoder.encoder.layers[0].activation
    assert isinstance(activation, nn.GELU)

    out = encoder(torch.randn(2, 3, 5), torch.empty(2, 0))
    assert out.shape == (2, 3, 8)


def test_transition_model_keeps_regular_transformer_and_mlp_activation_classes() -> None:
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

    assert isinstance(model.encoder.layers[0].activation, nn.GELU)
    assert any(isinstance(module, nn.GELU) for module in model.coembed.modules())
    assert any(isinstance(module, nn.GELU) for module in model.head.modules())

    out = model(torch.randn(2, 3, 8), torch.randn(2, 3, 2))
    assert out.shape == (2, 3, 8)
