from dataclasses import replace

import pytest
import torch
from torch import nn

from swarmbots.learn.algos.sac.scenario_obs_encoder import (
    ScenarioBatchedLinear,
    ScenarioFieldEncoder,
    ScenarioFieldEncoderConfig,
    ScenarioObservationSpec,
    TMASACScenarioEncoderConfig,
    TMASACScenarioObservationEncoder,
)


def _field_encoder(
        *,
        input_dims: tuple[int, ...] = (1, 3),
        padded_input_dim: int = 3,
        output_dim: int = 1,
        hidden_dims: tuple[int, ...] = (),
        normalize_input: bool = False,
) -> ScenarioFieldEncoder:
    return ScenarioFieldEncoder(
        input_dims=input_dims,
        padded_input_dim=padded_input_dim,
        config=ScenarioFieldEncoderConfig(
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            normalize_input=normalize_input,
        ),
        act_fn_cls=nn.ReLU,
    )


def _scenario_config() -> TMASACScenarioEncoderConfig:
    return TMASACScenarioEncoderConfig(
        scenarios=(
            ScenarioObservationSpec(
                scenario_id=0,
                name="empty",
                global_obs_dim=0,
                hidden_local_vars_dim=1,
                hidden_global_vars_dim=2,
            ),
            ScenarioObservationSpec(
                scenario_id=1,
                name="full",
                global_obs_dim=3,
                hidden_local_vars_dim=2,
                hidden_global_vars_dim=1,
            ),
        ),
        global_obs=ScenarioFieldEncoderConfig(output_dim=2),
        hidden_local_vars=ScenarioFieldEncoderConfig(output_dim=3),
        hidden_global_vars=ScenarioFieldEncoderConfig(output_dim=4),
        scenario_embedding_dim=2,
    )


def _observation_encoder(
        config: TMASACScenarioEncoderConfig | None = None,
        *,
        include_hidden_fields: bool = True,
) -> TMASACScenarioObservationEncoder:
    return TMASACScenarioObservationEncoder(
        config=_scenario_config() if config is None else config,
        global_obs_dim=3,
        hidden_local_vars_dim=2,
        hidden_global_vars_dim=2,
        act_fn_cls=nn.ReLU,
        include_hidden_fields=include_hidden_fields,
    )


def test_scenario_batched_linear_routes_each_row_to_its_parameter_bank() -> None:
    layer = ScenarioBatchedLinear(
        num_scenarios=2,
        input_dim=2,
        output_dim=1,
        init_gain=1.0,
    )
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[[1.0, 2.0]], [[10.0, 20.0]]]))
        layer.bias.copy_(torch.tensor([[3.0], [30.0]]))

    outputs = layer(
        torch.tensor([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([0, 1, 0]),
    )

    torch.testing.assert_close(outputs, torch.tensor([[8.0], [80.0], [14.0]]))


def test_scenario_batched_linear_supports_multiple_leading_dimensions() -> None:
    layer = ScenarioBatchedLinear(
        num_scenarios=2,
        input_dim=1,
        output_dim=1,
        init_gain=1.0,
    )
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[[2.0]], [[5.0]]]))
        layer.bias.zero_()

    outputs = layer(
        torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]),
        torch.tensor([[0, 1], [1, 0]]),
    )

    torch.testing.assert_close(
        outputs,
        torch.tensor([[[2.0], [10.0]], [[15.0], [8.0]]]),
    )


def test_scenario_field_encoder_ignores_padded_features() -> None:
    encoder = _field_encoder()
    with torch.no_grad():
        encoder.layers[0].weight.fill_(1.0)
        encoder.layers[0].bias.zero_()

    outputs = encoder(
        torch.tensor([[2.0, 100.0, -50.0], [1.0, 2.0, 3.0]]),
        torch.tensor([0, 1]),
    )

    torch.testing.assert_close(outputs, torch.tensor([[2.0], [6.0]]))


def test_scenario_field_encoder_zero_width_scenario_uses_learned_vector() -> None:
    encoder = _field_encoder(input_dims=(0, 2), padded_input_dim=2)
    assert encoder.empty_scenario_vectors is not None
    with torch.no_grad():
        encoder.layers[0].weight.fill_(9.0)
        encoder.layers[0].bias.copy_(torch.tensor([[5.0]]))
        encoder.empty_scenario_vectors.copy_(torch.tensor([[4.0]]))

    outputs = encoder(
        torch.tensor([[100.0, -100.0], [2.0, 3.0]]),
        torch.tensor([0, 1]),
    )

    assert encoder.layers[0].weight.shape[0] == 1
    torch.testing.assert_close(outputs, torch.tensor([[4.0], [50.0]]))


def test_zero_width_scenario_bypasses_configured_mlp() -> None:
    encoder = _field_encoder(
        input_dims=(0, 2),
        padded_input_dim=2,
        output_dim=3,
        hidden_dims=(4, 5),
    )
    assert encoder.empty_scenario_vectors is not None
    with torch.no_grad():
        encoder.empty_scenario_vectors.copy_(torch.tensor([[1.0, 2.0, 3.0]]))

    outputs = encoder(
        torch.tensor([[float("nan"), float("inf")], [float("-inf"), float("nan")]]),
        torch.zeros(2, dtype=torch.long),
    )
    outputs.sum().backward()

    torch.testing.assert_close(
        outputs,
        torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
    )
    assert encoder.empty_scenario_vectors.grad is not None
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for layer in encoder.layers
        for parameter in layer.parameters()
    )


def test_scenario_field_encoder_supports_all_scenarios_without_observations() -> None:
    encoder = _field_encoder(
        input_dims=(0, 0),
        padded_input_dim=0,
        output_dim=2,
        hidden_dims=(4,),
        normalize_input=True,
    )
    assert encoder.empty_scenario_vectors is not None
    with torch.no_grad():
        encoder.empty_scenario_vectors.copy_(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        )

    outputs = encoder(
        torch.empty(3, 0),
        torch.tensor([0, 1, 0]),
    )

    assert not encoder.layers
    assert encoder.input_norm_weight is None
    assert encoder.input_norm_bias is None
    torch.testing.assert_close(
        outputs,
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]]),
    )


def test_scenario_field_encoder_normalizes_only_active_features() -> None:
    encoder = _field_encoder(
        input_dims=(2, 3),
        normalize_input=True,
    )
    with torch.no_grad():
        encoder.layers[0].weight.copy_(torch.tensor([[[1.0, 2.0, 100.0]], [[1.0, 2.0, 3.0]]]))
        encoder.layers[0].bias.zero_()

    outputs = encoder(
        torch.tensor([[1.0, 3.0, 1_000.0], [1.0, 2.0, 3.0]]),
        torch.tensor([0, 1]),
    )

    assert outputs[0, 0].item() == pytest.approx(1.0, abs=1e-4)
    assert torch.isfinite(outputs).all()


def test_scenario_field_encoder_backpropagates_only_through_selected_banks() -> None:
    encoder = _field_encoder(input_dims=(2, 2), padded_input_dim=2)

    encoder(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.zeros(2, dtype=torch.long),
    ).sum().backward()

    weight_grad = encoder.layers[0].weight.grad
    bias_grad = encoder.layers[0].bias.grad
    assert weight_grad is not None
    assert bias_grad is not None
    assert torch.count_nonzero(weight_grad[0]) > 0
    assert torch.count_nonzero(bias_grad[0]) > 0
    assert torch.count_nonzero(weight_grad[1]) == 0
    assert torch.count_nonzero(bias_grad[1]) == 0


def test_scenario_field_encoder_supports_full_graph_compilation() -> None:
    encoder = _field_encoder(input_dims=(0, 3), normalize_input=True)
    compiled_encoder = torch.compile(encoder, backend="eager", fullgraph=True)
    inputs = torch.tensor([[1.0, 2.0, 99.0], [3.0, 4.0, 5.0]])
    scenario_ids = torch.tensor([0, 1])

    expected = encoder(inputs, scenario_ids)
    actual = compiled_encoder(inputs, scenario_ids)

    torch.testing.assert_close(actual, expected)


def test_scenario_field_encoder_validates_leading_dimensions() -> None:
    encoder = _field_encoder()

    with pytest.raises(ValueError, match="leading dimensions"):
        encoder(torch.zeros(2, 3), torch.zeros(2, 1, dtype=torch.long))


def test_scenario_field_encoder_validates_padded_input_dimension() -> None:
    encoder = _field_encoder()

    with pytest.raises(ValueError, match="padded input dimension 3"):
        encoder(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long))


def test_scenario_field_encoder_requires_long_scenario_ids() -> None:
    encoder = _field_encoder()

    with pytest.raises(ValueError, match="dtype torch.long"):
        encoder(torch.zeros(2, 3), torch.zeros(2, dtype=torch.int32))


@pytest.mark.parametrize("scenario_id", [-1, 2])
def test_scenario_field_encoder_rejects_out_of_range_scenario_ids(
        scenario_id: int,
) -> None:
    encoder = _field_encoder()

    with pytest.raises((AssertionError, RuntimeError), match="scenario_ids must be in"):
        encoder(
            torch.zeros(1, 3),
            torch.tensor([scenario_id], dtype=torch.long),
        )


def test_scenario_observation_encoder_appends_the_selected_embedding() -> None:
    encoder = _observation_encoder()
    assert encoder.scenario_embedding is not None
    assert encoder.global_obs_encoder.empty_scenario_vectors is not None
    with torch.no_grad():
        encoder.global_obs_encoder.layers[0].weight.zero_()
        encoder.global_obs_encoder.layers[0].bias.zero_()
        encoder.global_obs_encoder.empty_scenario_vectors.zero_()
        encoder.scenario_embedding.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    encoded = encoder.encode_global_obs(
        torch.zeros(2, 3),
        torch.tensor([0, 1]),
    )

    torch.testing.assert_close(
        encoded,
        torch.tensor([[0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 3.0, 4.0]]),
    )


def test_scenario_observation_encoder_expands_ids_across_agents() -> None:
    encoder = _observation_encoder()
    assert encoder.hidden_local_vars_encoder is not None
    with torch.no_grad():
        encoder.hidden_local_vars_encoder.layers[0].weight.zero_()
        encoder.hidden_local_vars_encoder.layers[0].bias.copy_(
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        )

    encoded = encoder.encode_hidden_local_vars(
        torch.zeros(2, 3, 2),
        torch.tensor([0, 1]),
    )

    assert encoded.shape == (2, 3, 3)
    torch.testing.assert_close(
        encoded[0],
        torch.tensor([[1.0, 2.0, 3.0]]).expand(3, -1),
    )
    torch.testing.assert_close(
        encoded[1],
        torch.tensor([[4.0, 5.0, 6.0]]).expand(3, -1),
    )


def test_scenario_observation_encoder_uses_vectors_for_each_empty_field() -> None:
    config = TMASACScenarioEncoderConfig(
        scenarios=(
            ScenarioObservationSpec(
                scenario_id=0,
                name="empty",
                global_obs_dim=0,
                hidden_local_vars_dim=0,
                hidden_global_vars_dim=0,
            ),
            ScenarioObservationSpec(
                scenario_id=1,
                name="full",
                global_obs_dim=2,
                hidden_local_vars_dim=2,
                hidden_global_vars_dim=2,
            ),
        ),
        global_obs=ScenarioFieldEncoderConfig(output_dim=2),
        hidden_local_vars=ScenarioFieldEncoderConfig(output_dim=3),
        hidden_global_vars=ScenarioFieldEncoderConfig(output_dim=4),
        scenario_embedding_dim=0,
    )
    encoder = TMASACScenarioObservationEncoder(
        config=config,
        global_obs_dim=2,
        hidden_local_vars_dim=2,
        hidden_global_vars_dim=2,
        act_fn_cls=nn.ReLU,
        include_hidden_fields=True,
    )
    assert encoder.hidden_local_vars_encoder is not None
    assert encoder.hidden_global_vars_encoder is not None
    assert encoder.global_obs_encoder.empty_scenario_vectors is not None
    assert encoder.hidden_local_vars_encoder.empty_scenario_vectors is not None
    assert encoder.hidden_global_vars_encoder.empty_scenario_vectors is not None
    with torch.no_grad():
        encoder.global_obs_encoder.empty_scenario_vectors.copy_(
            torch.tensor([[1.0, 2.0]])
        )
        encoder.hidden_local_vars_encoder.empty_scenario_vectors.copy_(
            torch.tensor([[3.0, 4.0, 5.0]])
        )
        encoder.hidden_global_vars_encoder.empty_scenario_vectors.copy_(
            torch.tensor([[6.0, 7.0, 8.0, 9.0]])
        )

    scenario_ids = torch.zeros(2, dtype=torch.long)
    global_outputs = encoder.encode_global_obs(
        torch.randn(2, 2),
        scenario_ids,
    )
    hidden_local_outputs = encoder.encode_hidden_local_vars(
        torch.randn(2, 3, 2),
        scenario_ids,
    )
    hidden_global_outputs = encoder.encode_hidden_global_vars(
        torch.randn(2, 2),
        scenario_ids,
    )

    torch.testing.assert_close(
        global_outputs,
        torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
    )
    torch.testing.assert_close(
        hidden_local_outputs,
        torch.tensor([3.0, 4.0, 5.0]).expand(2, 3, -1),
    )
    torch.testing.assert_close(
        hidden_global_outputs,
        torch.tensor([[6.0, 7.0, 8.0, 9.0]]).expand(2, -1),
    )


def test_actor_scenario_encoder_rejects_hidden_field_calls() -> None:
    encoder = _observation_encoder(include_hidden_fields=False)

    with pytest.raises(RuntimeError, match="hidden-local"):
        encoder.encode_hidden_local_vars(
            torch.zeros(1, 2, 2),
            torch.zeros(1, dtype=torch.long),
        )
    with pytest.raises(RuntimeError, match="hidden-global"):
        encoder.encode_hidden_global_vars(
            torch.zeros(1, 2),
            torch.zeros(1, dtype=torch.long),
        )


def test_scenario_config_requires_at_least_one_scenario() -> None:
    config = replace(_scenario_config(), scenarios=())

    with pytest.raises(ValueError, match="must not be empty"):
        _observation_encoder(config)


@pytest.mark.parametrize(
    "scenario_ids",
    [
        (1, 0),
        (0, 2),
    ],
)
def test_scenario_config_requires_contiguous_ordered_ids(
        scenario_ids: tuple[int, int],
) -> None:
    config = _scenario_config()
    scenarios = tuple(
        replace(scenario, scenario_id=scenario_id)
        for scenario, scenario_id in zip(config.scenarios, scenario_ids, strict=True)
    )

    with pytest.raises(ValueError, match="contiguous and ordered"):
        _observation_encoder(replace(config, scenarios=scenarios))


def test_scenario_config_requires_unique_names() -> None:
    config = _scenario_config()
    scenarios = (
        config.scenarios[0],
        replace(config.scenarios[1], name=config.scenarios[0].name),
    )

    with pytest.raises(ValueError, match="names must be unique"):
        _observation_encoder(replace(config, scenarios=scenarios))


def test_scenario_config_rejects_negative_input_dimensions() -> None:
    config = _scenario_config()
    scenarios = (
        replace(config.scenarios[0], global_obs_dim=-1),
        config.scenarios[1],
    )

    with pytest.raises(ValueError, match="must be non-negative"):
        _observation_encoder(replace(config, scenarios=scenarios))


def test_scenario_config_requires_environment_padding_to_match_maximum() -> None:
    with pytest.raises(ValueError, match="Expected padded global_obs dimension 3"):
        TMASACScenarioObservationEncoder(
            config=_scenario_config(),
            global_obs_dim=4,
            hidden_local_vars_dim=2,
            hidden_global_vars_dim=2,
            act_fn_cls=nn.ReLU,
            include_hidden_fields=True,
        )


@pytest.mark.parametrize(
    ("field_name", "field_config"),
    [
        ("global_obs", ScenarioFieldEncoderConfig(output_dim=0)),
        ("hidden_local_vars", ScenarioFieldEncoderConfig(output_dim=2, hidden_dims=(0,))),
    ],
)
def test_scenario_config_requires_positive_network_dimensions(
        field_name: str,
        field_config: ScenarioFieldEncoderConfig,
) -> None:
    config = replace(_scenario_config(), **{field_name: field_config})

    with pytest.raises(ValueError, match=field_name):
        _observation_encoder(config)


def test_scenario_config_rejects_negative_embedding_dimension() -> None:
    config = replace(_scenario_config(), scenario_embedding_dim=-1)

    with pytest.raises(ValueError, match="scenario_embedding_dim must be non-negative"):
        _observation_encoder(config)
