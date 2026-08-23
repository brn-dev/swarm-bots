from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from swarmbots.learn.evaluation import (
    EvaluationRecordingConfig,
    FrozenEvaluationRunner,
    ScheduledEvaluationHook,
)


def _obs(num_envs: int = 2) -> dict[str, torch.Tensor]:
    return {
        "local_obs": torch.zeros((num_envs, 1, 1)),
        "global_obs": torch.zeros((num_envs, 1)),
        "hidden_local_vars": torch.empty((num_envs, 1, 0)),
        "hidden_global_vars": torch.empty((num_envs, 0)),
    }


class _StatefulActionDist:
    def __init__(self) -> None:
        self.state = torch.tensor([7.0])

    def get_temporal_correlation_state(self) -> tuple[torch.Tensor]:
        return (self.state,)

    def set_temporal_correlation_state(self, state: tuple[torch.Tensor]) -> None:
        self.state = state[0]

    def reset_temporal_correlations_on_ep_start(self, mask: torch.Tensor) -> None:
        if torch.any(mask):
            self.state = torch.tensor([-1.0])


class _EvaluationPolicy(nn.Module):
    gsde_enabled = False

    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.action_dist = _StatefulActionDist()
        self.episode_start_masks: list[torch.Tensor] = []

    def requires_previous_actions(self) -> bool:
        return True

    def initial_temporal_state(
        self,
        batch_size: int,
        n_agents: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros((batch_size, n_agents, 1), device=device, dtype=dtype)

    def act_with_temporal_state(
        self,
        *,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        hidden_local_vars: torch.Tensor,
        hidden_global_vars: torch.Tensor,
        agent_mask: torch.Tensor | None,
        previous_actions: torch.Tensor | None,
        deterministic: bool,
        temporal_state: torch.Tensor,
        episode_start_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = global_obs, hidden_local_vars, hidden_global_vars, agent_mask, previous_actions, deterministic
        self.episode_start_masks.append(episode_start_mask.clone())
        torch.rand(())
        return torch.zeros_like(local_obs), temporal_state + 1


class _EvaluationEnv:
    num_envs = 2
    n_agents = 1
    action_space = SimpleNamespace(total_agent_action_dim=1)

    def __init__(self) -> None:
        self.steps = torch.zeros((self.num_envs,), dtype=torch.int64)
        self.episodes = torch.zeros((self.num_envs,), dtype=torch.int64)
        self.prev_dones = torch.zeros((self.num_envs,), dtype=torch.bool)
        self.reset_kwargs: dict[str, object] = {}
        self.recording_kwargs: dict[str, object] | None = None
        self.closed = False

    @property
    def unwrapped(self) -> _EvaluationEnv:
        return self

    def reset(self, **kwargs: object) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        self.reset_kwargs = kwargs
        self.steps.zero_()
        self.episodes.zero_()
        self.prev_dones.zero_()
        return _obs(), {}

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
        _ = actions
        self.steps += 1
        episode_lengths = torch.tensor([2, 3])
        dones = self.steps >= episode_lengths
        self.episodes += dones
        returns = torch.where(
            dones,
            self.episodes.to(dtype=torch.float64) + torch.tensor([0.0, 100.0]),
            0.0,
        )
        lengths = torch.where(dones, self.steps, 0)
        successes = torch.where(dones, (self.episodes % 2) == 0, False)
        self.steps[dones] = 0
        self.prev_dones = dones
        infos: dict[str, object] = {}
        if torch.any(dones):
            infos = {
                "_episode": dones.clone(),
                "episode": {
                    "r": returns,
                    "l": lengths,
                    "success": successes,
                },
            }
        return _obs(), torch.zeros(2), dones, torch.zeros(2, dtype=torch.bool), infos

    def close(self) -> None:
        self.closed = True

    def start_video_recording(self, **kwargs: object) -> None:
        self.recording_kwargs = kwargs


def test_frozen_evaluation_collects_each_lane_requested_number_of_times_and_restores_state() -> None:
    training_env = object()
    evaluation_env = _EvaluationEnv()
    policy = _EvaluationPolicy()
    policy.train()
    rng_state = torch.random.get_rng_state().clone()
    runner = FrozenEvaluationRunner(
        make_env=lambda: evaluation_env,
        training_env=training_env,
        policy=policy,
        episodes_per_env=2,
        seed=123,
        deterministic=True,
        recording_config=EvaluationRecordingConfig(num_episodes=3),
        video_folder=Path("videos/eval"),
    )

    metrics = runner.evaluate(timesteps=25, milestone_percentage=25.0)

    assert metrics["eval_episodes"] == 4
    assert metrics["eval_ep_rew"].n == 4
    assert metrics["eval_ep_rew"].mean == 51.5
    assert metrics["eval_success_rate"] == 50.0
    assert metrics["timesteps"] == 25
    assert evaluation_env.reset_kwargs == {
        "seed": 123,
        "options": {"force_settled": True},
    }
    assert torch.equal(policy.action_dist.state, torch.tensor([7.0]))
    assert policy.training
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert len(policy.episode_start_masks) == 6
    assert evaluation_env.recording_kwargs is not None
    assert evaluation_env.recording_kwargs["num_episodes"] == 3
    assert evaluation_env.recording_kwargs["max_parallel_episodes"] == 2
    assert evaluation_env.recording_kwargs["frame_stride"] == 1
    assert evaluation_env.recording_kwargs["video_name_prefix"] == "eval_025pct_25_steps"

    runner.close()
    assert evaluation_env.closed


def test_frozen_evaluation_can_reuse_stateful_environment() -> None:
    evaluation_env = _EvaluationEnv()
    runner = FrozenEvaluationRunner(
        make_env=lambda: evaluation_env,
        training_env=object(),
        policy=_EvaluationPolicy(),
        episodes_per_env=1,
        deterministic=True,
    )

    first_metrics = runner.evaluate(timesteps=25, milestone_percentage=25.0)
    second_metrics = runner.evaluate(timesteps=50, milestone_percentage=50.0)

    assert first_metrics["eval_ep_rew"].mean == second_metrics["eval_ep_rew"].mean
    assert first_metrics["eval_success_rate"] == second_metrics["eval_success_rate"]
    runner.close()
    assert evaluation_env.closed


def test_scheduled_evaluation_runs_when_milestone_is_crossed() -> None:
    algorithm = SimpleNamespace(n_total_timesteps=0)
    runner = Mock()
    runner.evaluate.return_value = {"eval_success_rate": 75.0}
    metrics_logger = Mock()
    hook = ScheduledEvaluationHook(
        algorithm=algorithm,
        total_timesteps=100,
        milestones=(25, 50),
        runner=runner,
        metrics_logger=metrics_logger,
    )
    metrics: dict[str, object] = {"training_loss": 0.5}

    algorithm.n_total_timesteps = 30
    hook(algorithm, metrics, rollout_steps=8)

    assert metrics == {"training_loss": 0.5}
    runner.evaluate.assert_called_once_with(timesteps=30, milestone_percentage=25.0)
    metrics_logger.log.assert_called_once_with({"eval_success_rate": 75.0})
    metrics_logger.flush.assert_called_once_with()

    hook.close()
    runner.close.assert_called_once_with()
    metrics_logger.close.assert_called_once_with()


def test_scheduled_evaluation_only_runs_latest_crossed_milestone() -> None:
    algorithm = SimpleNamespace(n_total_timesteps=0)
    runner = Mock()
    runner.evaluate.return_value = {"eval_success_rate": 75.0}
    metrics_logger = Mock()
    hook = ScheduledEvaluationHook(
        algorithm=algorithm,
        total_timesteps=100,
        milestones=(25, 50, 75),
        runner=runner,
        metrics_logger=metrics_logger,
    )

    algorithm.n_total_timesteps = 60
    hook(algorithm, {}, rollout_steps=60)

    runner.evaluate.assert_called_once_with(timesteps=60, milestone_percentage=50.0)

    runner.evaluate.reset_mock()
    algorithm.n_total_timesteps = 80
    hook(algorithm, {}, rollout_steps=20)

    runner.evaluate.assert_called_once_with(timesteps=80, milestone_percentage=75.0)
    assert metrics_logger.log.call_count == 2
    assert metrics_logger.flush.call_count == 2


def test_scheduled_evaluation_milestones_are_relative_to_continuation_window() -> None:
    algorithm = SimpleNamespace(n_total_timesteps=100)
    runner = Mock()
    runner.evaluate.return_value = {"eval_success_rate": 75.0}
    metrics_logger = Mock()
    hook = ScheduledEvaluationHook(
        algorithm=algorithm,
        start_timesteps=100,
        total_timesteps=300,
        milestones=(50, 100),
        runner=runner,
        metrics_logger=metrics_logger,
    )

    algorithm.n_total_timesteps = 199
    hook(algorithm, {}, rollout_steps=99)
    runner.evaluate.assert_not_called()

    algorithm.n_total_timesteps = 200
    hook(algorithm, {}, rollout_steps=1)
    runner.evaluate.assert_called_once_with(timesteps=200, milestone_percentage=50.0)

    runner.evaluate.reset_mock()
    algorithm.n_total_timesteps = 300
    hook(algorithm, {}, rollout_steps=100)
    runner.evaluate.assert_called_once_with(timesteps=300, milestone_percentage=100.0)
