import numpy as np
import pytest
import torch

from swarmbots.learn.nn_components.deep_set import DeepSetCritic
from swarmbots.learn.nn_components.popart import PopArtLinear


@pytest.mark.parametrize("options", [
    {"beta": -0.1}, {"beta": 1.1}, {"beta": float("nan")},
    {"eps": -0.1}, {"min_std": -0.1}, {"init_sigma": float("nan")},
    {"init_sigma": 0., "eps": 0., "min_std": 0.},
    {"init_mu": 1e40}, {"init_sigma": 1e40},
])
def test_popart_rejects_invalid_initial_statistics(options: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="PopArt"):
        PopArtLinear(1, dtype=torch.float32, **options)


def test_popart_validates_numpy_scalars_in_the_default_readout_dtype() -> None:
    with pytest.raises(ValueError, match="readout dtype"):
        PopArtLinear(1, init_mu=np.float64(1e40))
    initial_sigma = np.float32(1e20)
    layer = PopArtLinear(1, init_sigma=initial_sigma)
    assert layer.sigma().item() == initial_sigma


def test_popart_allows_frozen_statistics_and_initial_scale_at_the_floor() -> None:
    layer = PopArtLinear(1, beta=0., init_sigma=0., eps=0., min_std=0.25)
    before = {key: value.clone() for key, value in layer.state_dict().items()}
    layer.update(torch.tensor([1e200], dtype=torch.float64))
    torch.testing.assert_close(layer.state_dict(), before, rtol=0, atol=0)
    assert layer.sigma().item() == 0.25
    with pytest.raises(ValueError, match="finite"):
        layer.update(torch.tensor([float("nan")]))


def test_popart_full_replacement_uses_only_batch_statistics() -> None:
    layer = PopArtLinear(1, beta=1., dtype=torch.float64)
    layer.update(torch.tensor(1e200, dtype=torch.float64))
    assert layer.mu.item() == 1e200
    assert layer.variance.item() == 0.
    assert torch.isfinite(layer.bias).all()


@pytest.mark.parametrize("bias", [True, False])
def test_popart_update_preserves_unnormalized_predictions(bias: bool) -> None:
    torch.manual_seed(0)
    layer = PopArtLinear(3, out_features=2, beta=1.0, bias=bias)
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
    if not bias:
        assert layer.bias is None
        assert list(dict(layer.named_parameters())) == ["weight"]
        restored = PopArtLinear(3, out_features=2, beta=1.0, bias=False)
        restored.load_state_dict(layer.state_dict())
        torch.testing.assert_close(restored(x), predictions_before, rtol=1e-5, atol=1e-6)


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


def test_popart_tracks_nearly_constant_nonzero_returns_at_training_decay() -> None:
    beta = 5e-4
    initial_mean, initial_sigma = 99.99, 0.1
    layer = PopArtLinear(2, beta=beta, init_mu=initial_mean, init_sigma=initial_sigma)
    targets = torch.tensor([99.975, 100.025])
    steps = 256
    for _ in range(steps):
        layer.update(targets)

    prior_mass = (1.0 - beta) ** steps
    target_variance, target_mean = torch.var_mean(targets.double(), correction=0)
    expected_mean = prior_mass * initial_mean + (1.0 - prior_mass) * target_mean
    expected_variance = (
        prior_mass * initial_sigma ** 2 + (1.0 - prior_mass) * target_variance
        + prior_mass * (1.0 - prior_mass) * (initial_mean - target_mean).square()
    )
    torch.testing.assert_close(layer.mu, expected_mean.reshape(1), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(layer.sigma(), expected_variance.float().sqrt().reshape(1), rtol=1e-6, atol=1e-8)


def test_popart_precision_change_preserves_statistics_and_optimizer_ownership() -> None:
    layer = PopArtLinear(3, beta=0.15, init_mu=20.0123, init_sigma=.03).half()
    optimizer = torch.optim.Adam(layer.parameters())
    parameters = optimizer.param_groups[0]["params"]
    assert parameters[0] is layer.weight and parameters[1] is layer.bias
    x = torch.tensor([[1., 2., 3.]], dtype=torch.float16)
    layer.update(torch.tensor([20.01, 20.02]))
    assert parameters[0] is layer.weight and parameters[1] is layer.bias
    assert layer(x).dtype == layer.normalize(x).dtype == torch.float16

    restored = PopArtLinear(3, beta=0.15).half()
    restored.load_state_dict(layer.state_dict())
    reference = PopArtLinear(3, beta=0.15, init_mu=20.0123, init_sigma=.03)
    reference.update(torch.tensor([20.01, 20.02]))
    for head in (layer, restored, reference):
        head.update(torch.tensor([20.04, 20.06]))
    torch.testing.assert_close(restored.state_dict(), layer.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(restored.mu, reference.mu, rtol=0, atol=0)
    torch.testing.assert_close(restored.sigma(), reference.sigma().half(), rtol=0, atol=0)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("overflow", ["bias", "weight", "sigma"])
def test_popart_rejects_unrepresentable_half_precision_updates_without_mutation(bias: bool, overflow: str) -> None:
    layer = PopArtLinear(2, beta=1.0, bias=bias).half()
    if overflow == "bias":
        targets = torch.full((4,), 250.0)
    elif overflow == "weight":
        with torch.no_grad():
            layer.weight.fill_(300.0)
        targets = torch.zeros(4)
    else:
        targets = torch.tensor([-100_000., 100_000.])
    before = {name: value.clone() for name, value in layer.state_dict().items()}
    with pytest.raises(ValueError, match="PopArt.*finite"):
        layer.update(targets)
    torch.testing.assert_close(layer.state_dict(), before, rtol=0, atol=0)


def test_biasless_popart_checkpoint_requires_normalization_offset() -> None:
    layer = PopArtLinear(3, bias=False)
    state = layer.state_dict()
    del state["normalization_bias"]
    with pytest.raises(RuntimeError, match="normalization_bias"):
        layer.load_state_dict(state)


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
