import torch

from swarmbots.learn.nn_components.deep_set import DeepSetCritic
from swarmbots.learn.nn_components.popart import PopArtLinear


def test_popart_linear_init_gain_controls_weight_norm() -> None:
    torch.manual_seed(0)
    layer = PopArtLinear(8, init_gain=0.01)

    assert torch.allclose(layer.weight.norm(), torch.tensor(0.01), atol=1e-6)


def test_deepset_popart_uses_value_head_init_gain() -> None:
    torch.manual_seed(0)
    critic = DeepSetCritic(
        num_local_features=4,
        local_projection_hidden_dims=[],
        value_regressor_hidden_dims=[],
        use_popart=True,
        value_head_linear_init_gain=0.01,
    )

    assert critic.popart_head is not None
    assert torch.allclose(critic.popart_head.weight.norm(), torch.tensor(0.01), atol=1e-6)
