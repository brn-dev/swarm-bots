import torch

from swarmbots.learn.nn_components.deep_set import DeepSetCritic
from swarmbots.learn.nn_components.popart import PopArtLinear


def test_popart_update_preserves_unnormalized_predictions() -> None:
    torch.manual_seed(0)
    layer = PopArtLinear(3, out_features=2, beta=1.0)
    x = torch.randn(5, 3)
    targets = torch.tensor(
        [
            [10.0, -3.0],
            [12.0, -1.0],
            [14.0, 1.0],
            [16.0, 3.0],
        ]
    )
    predictions_before = layer(x).detach().clone()

    layer.update(targets)

    torch.testing.assert_close(layer(x), predictions_before, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(layer.normalize(targets).mean(dim=0), torch.zeros(2), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(layer.normalize(targets).std(dim=0, unbiased=False), torch.ones(2), atol=1e-6, rtol=0.0)


def test_popart_empty_update_is_noop() -> None:
    torch.manual_seed(0)
    layer = PopArtLinear(3, out_features=2)
    x = torch.randn(4, 3)
    predictions_before = layer(x).detach().clone()
    mu_before = layer.mu.clone()
    sigma_before = layer.sigma().clone()

    layer.update(torch.empty(0, 2))

    torch.testing.assert_close(layer(x), predictions_before)
    torch.testing.assert_close(layer.mu, mu_before)
    torch.testing.assert_close(layer.sigma(), sigma_before)


def test_deepset_critic_popart_update_preserves_values_and_normalizes_targets() -> None:
    torch.manual_seed(0)
    critic = DeepSetCritic(
        num_local_features=4,
        local_projection_hidden_dims=[],
        value_regressor_hidden_dims=[],
        use_popart=True,
        popart_beta=1.0,
    )
    local_features = torch.randn(6, 3, 4)
    targets = torch.tensor([3.0, 4.0, 7.0, 8.0])
    values_before = critic(local_features).detach().clone()

    critic.update_popart(targets)

    assert critic.has_popart
    torch.testing.assert_close(critic(local_features), values_before, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(critic.normalize_values(targets).mean(), torch.tensor(0.0), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        critic.normalize_values(targets).std(unbiased=False),
        torch.tensor(1.0),
        atol=1e-6,
        rtol=0.0,
    )


def test_deepset_critic_value_head_init_gain_controls_prediction_scale() -> None:
    local_features = torch.randn(6, 3, 4)

    torch.manual_seed(0)
    small_gain_critic = DeepSetCritic(
        num_local_features=4,
        local_projection_hidden_dims=[],
        value_regressor_hidden_dims=[],
        use_popart=True,
        value_head_linear_init_gain=0.01,
    )
    torch.manual_seed(0)
    default_gain_critic = DeepSetCritic(
        num_local_features=4,
        local_projection_hidden_dims=[],
        value_regressor_hidden_dims=[],
        use_popart=True,
        value_head_linear_init_gain=1.0,
    )

    torch.testing.assert_close(
        small_gain_critic(local_features),
        default_gain_critic(local_features) * 0.01,
        rtol=1e-5,
        atol=1e-6,
    )
