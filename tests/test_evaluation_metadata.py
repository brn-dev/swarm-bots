from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import torch

from swarmbots import evaluate_policy, make_scenario
from swarmbots.benchmark import evaluation
from swarmbots.benchmark.metadata import serialize_settings
from swarmbots.mjw_env.parameter_distributions import UniformDistParams


def test_settings_preserve_distribution_types_and_morphology_pool() -> None:
    scenario = make_scenario("SwarmBots-POWallMedium-v0", seed=1000)
    settings = serialize_settings(scenario.get_settings())
    assert settings["wall_distance"]["type"] == "UniformDistParams"
    assert settings["wall_distance"]["low"] < settings["wall_distance"]["high"]
    assert settings["swarm"]["unit_start_locations"]["pool_seeds"] == list(range(42_000, 42_050))
    json.dumps(settings, allow_nan=False)


@pytest.mark.parametrize(
    ("overrides", "canonical_scenario", "canonical_evaluation"),
    [
        ({}, True, True),
        ({"seed": 42}, True, False),
        ({"action_mode": "stochastic"}, True, False),
        ({"num_envs": 2, "num_episodes": 2}, True, False),
        ({"episode_length": 10}, False, False),
        ({"scenario_kwargs": {"wall_distance": UniformDistParams(0.2, 0.4)}}, False, False),
        ({"env_kwargs": {"compile_tensor_operations": False}}, False, False),
    ],
)
def test_result_identifies_protocol_deviations(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict,
    canonical_scenario: bool,
    canonical_evaluation: bool,
) -> None:
    num_envs = overrides.get("num_envs", 256)
    env = MagicMock(num_envs=num_envs, device=torch.device("cpu"))
    env.get_settings.return_value = {"episode_length": overrides.get("episode_length", 500)}
    observations = {key: torch.zeros(num_envs) for key in evaluation.POLICY_OBSERVATION_KEYS}
    env.reset.return_value = observations, {}
    env.step.return_value = (
        observations, torch.ones(num_envs), torch.zeros(num_envs, dtype=torch.bool),
        torch.ones(num_envs, dtype=torch.bool), {"success": torch.zeros(num_envs, dtype=torch.bool)},
    )
    monkeypatch.setattr(evaluation, "make_env", lambda *args, **kwargs: env)
    kwargs = {"action_mode": "deterministic", **overrides}
    result = evaluate_policy(
        lambda obs, starts: {}, "SwarmBots-WallEasy-v0",
        policy_metadata={"name": "test-policy", "checkpoint_sha256": "example"},
        source_revision="example-commit", **kwargs,
    )
    metadata = json.loads(json.dumps(result.to_dict(), allow_nan=False))["metadata"]
    assert metadata["canonical_scenario"] is canonical_scenario
    assert metadata["canonical_evaluation"] is canonical_evaluation
    assert metadata["runtime"]["packages"]["torch"]
    assert metadata["source_revision"] == "example-commit"
    assert metadata["policy"]["name"] == "test-policy"
    assert metadata["num_envs"] == num_envs
    if "scenario_kwargs" in overrides:
        assert metadata["scenario_overrides"]["wall_distance"]["type"] == "UniformDistParams"
