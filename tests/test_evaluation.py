from __future__ import annotations

import torch

import swarmbots.benchmark.evaluation as evaluation
from swarmbots.benchmark.evaluation import EvaluationResult, policy_observation


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
