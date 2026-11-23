from __future__ import annotations

from collections.abc import Mapping
import json

import pytest
import torch
from gymnasium.vector import AutoresetMode

import swarmbots.benchmark.evaluation as evaluation
from swarmbots.benchmark.evaluation import EvaluationResult, policy_observation
from swarmbots.benchmark.registry import make_scenario
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig


def test_policy_observation_removes_privileged_state() -> None:
    observations = {
        "local_obs": torch.zeros(2, 3, 4),
        "global_obs": torch.zeros(2, 5),
        "agent_mask": torch.ones(2, 3, dtype=torch.bool),
        "hidden_local_vars": torch.ones(2, 3, 1),
        "hidden_global_vars": torch.ones(2, 2),
    }

    policy_obs = policy_observation(observations)

    assert tuple(policy_obs) == ("local_obs", "global_obs", "agent_mask")


def test_evaluation_result_reports_population_statistics() -> None:
    result = EvaluationResult(
        benchmark_id="SwarmBots-WallEasy-v0",
        seed=1000,
        episode_returns=(1.0, 3.0),
        episode_lengths=(10, 20),
        episode_successes=(False, True),
    )

    assert result.mean_return == 2.0
    assert result.return_std == 1.0
    assert result.mean_episode_length == 15.0
    assert result.success_rate == 0.5
    assert result.to_dict()["num_episodes"] == 2


def test_evaluate_policy_accumulates_first_episode_per_lane(monkeypatch) -> None:
    class FakeEnv:
        num_envs = 2
        device = torch.device("cpu")
        autoreset_mode = AutoresetMode.SAME_STEP

        def __init__(self) -> None:
            self.step_count = 0
            self.closed = False

        @staticmethod
        def observations() -> dict[str, torch.Tensor]:
            return {
                "local_obs": torch.zeros(2, 3, 4),
                "global_obs": torch.zeros(2, 1),
                "agent_mask": torch.ones(2, 3, dtype=torch.bool),
                "hidden_local_vars": torch.ones(2, 3, 1),
                "hidden_global_vars": torch.ones(2, 2),
            }

        def reset(self, *, seed: int):
            assert seed == 1000
            return self.observations(), {}

        def step(self, actions):
            assert actions == {}
            self.step_count += 1
            if self.step_count == 1:
                return (
                    self.observations(),
                    torch.tensor([1.0, 2.0]),
                    torch.zeros(2, dtype=torch.bool),
                    torch.zeros(2, dtype=torch.bool),
                    {"success": torch.zeros(2, dtype=torch.bool)},
                )
            return (
                self.observations(),
                torch.tensor([3.0, 4.0]),
                torch.zeros(2, dtype=torch.bool),
                torch.ones(2, dtype=torch.bool),
                {"success": torch.tensor([True, False])},
            )

        def close(self) -> None:
            self.closed = True

        def get_settings(self) -> dict[str, object]:
            return {"episode_length": 500}

    env = FakeEnv()
    monkeypatch.setattr(evaluation, "make_env", lambda *args, **kwargs: env)
    policy_episode_starts: list[torch.Tensor] = []

    def policy(observations, episode_starts):
        assert set(observations) == {"local_obs", "global_obs", "agent_mask"}
        policy_episode_starts.append(episode_starts.clone())
        return {}

    result = evaluation.evaluate_policy(
        policy,
        "SwarmBots-WallEasy-v0",
        num_envs=2,
        num_episodes=2,
        seed=1000,
        device="cpu",
    )

    assert result.episode_returns == (4.0, 6.0)
    assert result.episode_lengths == (2, 2)
    assert result.episode_successes == (True, False)
    assert torch.equal(policy_episode_starts[0], torch.ones(2, dtype=torch.bool))
    assert torch.equal(policy_episode_starts[1], torch.zeros(2, dtype=torch.bool))
    assert env.closed


@pytest.mark.parametrize("fast_lane", [0, 1])
def test_evaluation_never_replaces_a_slow_lane_with_repeated_fast_episodes(
    monkeypatch: pytest.MonkeyPatch, fast_lane: int,
) -> None:
    class StaggeredEnv:
        num_envs = 2
        device = torch.device("cpu")
        autoreset_mode = AutoresetMode.SAME_STEP

        def __init__(self) -> None:
            self.steps = 0
            self.closed = False

        def get_settings(self) -> dict[str, object]:
            return {"episode_length": 3}

        def reset(self, *, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
            return {
                "local_obs": torch.zeros(2, 1, 1),
                "global_obs": torch.zeros(2, 0),
                "agent_mask": torch.ones(2, 1, dtype=torch.bool),
            }, {}

        def step(self, actions: object) -> tuple:
            self.steps += 1
            dones = torch.tensor([False, False])
            dones[fast_lane] = True
            dones[1 - fast_lane] = self.steps == 3
            successes = torch.zeros(2, dtype=torch.bool)
            # A later success in the fast lane must not overwrite its first failure.
            successes[fast_lane] = self.steps > 1
            successes[1 - fast_lane] = self.steps == 3
            rewards = torch.full((2,), 10.0)
            rewards[fast_lane] = 1.0 if self.steps == 1 else float("nan")
            return self.reset(seed=0)[0], rewards, dones, torch.zeros(2, dtype=torch.bool), {"success": successes}

        def close(self) -> None:
            self.closed = True

    env = StaggeredEnv()
    monkeypatch.setattr(evaluation, "make_env", lambda *args, **kwargs: env)
    starts: list[torch.Tensor] = []

    def policy(observations: object, episode_starts: torch.Tensor) -> dict:
        starts.append(episode_starts.clone())
        return {}

    result = evaluation.evaluate_policy(policy, "SwarmBots-WallEasy-v0", num_envs=2, num_episodes=2, device="cpu")
    assert result.episode_returns[fast_lane] == 1.0
    assert result.episode_returns[1 - fast_lane] == 30.0
    assert result.episode_lengths[fast_lane] == 1
    assert result.episode_lengths[1 - fast_lane] == 3
    assert result.episode_successes[fast_lane] is False
    assert result.episode_successes[1 - fast_lane] is True
    assert starts[0].all()
    assert starts[1][fast_lane] and not starts[1][1 - fast_lane]
    assert env.closed
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.integration
def test_next_step_evaluation_presents_reset_observations_at_episode_starts() -> None:
    benchmark_id = "SwarmBots-MoveToGoal-v0"
    scenario_kwargs = {
        "unit_start_locations": MJWPreConnectedUnitLocationsConfig(num_units=1, pool_seeds=[42000]),
        "reset_settle_time": 0.0,
        "compile_reward_kernel": False,
    }
    action_space = make_scenario(benchmark_id, **scenario_kwargs).get_batched_action_space(num_envs=2)
    actions = {key: torch.zeros(space.shape) for key, space in action_space.items()}

    def rollout(mode: AutoresetMode) -> tuple[EvaluationResult, list[dict[str, torch.Tensor]]]:
        history: list[dict[str, torch.Tensor]] = []

        def policy(observations: Mapping[str, torch.Tensor], starts: torch.Tensor) -> dict[str, torch.Tensor]:
            snapshot = {key: value.clone() for key, value in observations.items()}
            history.append({**snapshot, "episode_starts": starts.clone()})
            return actions

        result = evaluation.evaluate_policy(
            policy, benchmark_id, num_envs=2, num_episodes=2, device="cpu", seed=1000, episode_length=3,
            scenario_kwargs=scenario_kwargs,
            env_kwargs={"autoreset_mode": mode, "first_episode_lengths": [1, 3], "compile_tensor_operations": False},
        )
        return result, history

    same_step, expected_history = rollout(AutoresetMode.SAME_STEP)
    next_step, actual_history = rollout(AutoresetMode.NEXT_STEP)
    starts = [step["episode_starts"].tolist() for step in actual_history]
    assert starts == [[True, True], [True, False], [False, False]]
    torch.testing.assert_close(actual_history, expected_history)
    assert next_step.episode_lengths == same_step.episode_lengths == (1, 3)
    assert next_step.episode_returns == pytest.approx(same_step.episode_returns)
    assert next_step.metadata["environment_overrides"]["autoreset_mode"] == "NextStep"


def test_evaluation_rejects_reusing_lanes() -> None:
    with pytest.raises(ValueError, match="first episode per lane"):
        evaluation.evaluate_policy(lambda obs, starts: {}, "SwarmBots-WallEasy-v0", num_envs=2, num_episodes=3)


def test_evaluation_closes_environment_when_policy_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    env = MagicMock(num_envs=1, device=torch.device("cpu"))
    env.get_settings.return_value = {}
    env.reset.return_value = ({key: torch.zeros(1) for key in evaluation.POLICY_OBSERVATION_KEYS}, {})
    monkeypatch.setattr(evaluation, "make_env", lambda *args, **kwargs: env)

    def broken_policy(observations: object, episode_starts: object) -> dict:
        raise RuntimeError("policy failed")

    with pytest.raises(RuntimeError, match="policy failed"):
        evaluation.evaluate_policy(broken_policy, "SwarmBots-WallEasy-v0", num_envs=1, num_episodes=1, device="cpu")
    env.close.assert_called_once()
