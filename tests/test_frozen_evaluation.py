from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from torch import nn
from gymnasium import spaces

from swarmbots.learn.action_dists.gsde_action_dist import GSDEConfig
from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution
from swarmbots.learn.evaluation import (
    EvaluationRecordingConfig,
    FrozenEvaluationRunner,
    ScheduledEvaluationHook,
    _format_percentage,
)
from swarmbots.learn.gsde_reset import (
    GSDEResetMode, GSDEEpisodeResetMode, GSDEIntervalResetMode, GSDEProbabilityResetMode,
)
from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.temporal_state import clone_detach_temporal_state


def test_recording_percentage_accepts_integer_and_fractional_milestones() -> None:
    assert _format_percentage(25) == "025"
    assert _format_percentage(25.0) == "025"
    assert _format_percentage(25.5) == "25p5"


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
        self.deterministic_modes: list[bool] = []
        self.random_values: list[torch.Tensor] = []

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
        _ = global_obs, hidden_local_vars, hidden_global_vars, agent_mask, previous_actions
        self.episode_start_masks.append(episode_start_mask.clone())
        self.deterministic_modes.append(deterministic)
        self.random_values.append(torch.rand(()))
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
                    "task/goals_completed": self.episodes.clone(),
                    "_task/goals_completed": torch.tensor([True, False]),
                },
            }
        return _obs(), torch.zeros(2), dones, torch.zeros(2, dtype=torch.bool), infos

    def close(self) -> None:
        self.closed = True

    def start_video_recording(self, **kwargs: object) -> None:
        self.recording_kwargs = kwargs

    def get_video_recording_status(self) -> dict[str, bool]:
        return {"active": False}


class _GSDEEvaluationPolicy(_EvaluationPolicy):
    gsde_enabled = True

    def __init__(self) -> None:
        super().__init__()
        self.action_dist = HybridActionDistribution(
            latent_dim=1,
            action_space=VectorHybridActionSpace({"control": spaces.Box(-1.0, 1.0, (2, 1, 1))}),
            continuous_config=GSDEConfig(base_std=0.3),
        )
        self.noise_states: list[Any] = []

    def act_with_temporal_state(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        _, next_state = super().act_with_temporal_state(**kwargs)
        self.noise_states.append(clone_detach_temporal_state(self.action_dist.get_temporal_correlation_state()))
        actions = self.action_dist.update_latent_features(kwargs["local_obs"]).get_actions(
            deterministic=kwargs["deterministic"],
        )
        return actions, next_state


@pytest.mark.parametrize("fail", [False, True])
def test_frozen_evaluation_restores_an_in_place_action_dist_state(fail: bool) -> None:
    class InPlaceActionDist(_StatefulActionDist):
        def __init__(self) -> None:
            super().__init__()
            self.installed_states: list[float] = []

        def set_temporal_correlation_state(self, state: tuple[torch.Tensor]) -> None:
            self.installed_states.append(state[0].item())
            self.state.copy_(state[0])

        def reset_temporal_correlations_on_ep_start(self, mask: torch.Tensor) -> None:
            if torch.any(mask):
                self.state.fill_(-1.0)

    policy = _EvaluationPolicy()
    action_dist = InPlaceActionDist()
    policy.action_dist = action_dist
    original_state = action_dist.state
    env = _EvaluationEnv()
    runner = FrozenEvaluationRunner(make_env=lambda: env, training_env=object(), policy=policy)
    try:
        if fail:
            with patch.object(env, "step", side_effect=RuntimeError("evaluation step failed")):
                with pytest.raises(RuntimeError, match="evaluation step failed"):
                    runner.evaluate(timesteps=25, milestone_percentage=25.0)
        else:
            runner.evaluate(timesteps=25, milestone_percentage=25.0)
        assert action_dist.state is original_state
        torch.testing.assert_close(original_state, torch.tensor([7.0]))
        assert action_dist.installed_states == [7.0] * (2 if fail else 3)
    finally:
        runner.close()


@pytest.mark.parametrize("reset_mode", [GSDEEpisodeResetMode(), GSDEIntervalResetMode(3)])
def test_frozen_gsde_noise_follows_episode_and_interval_schedules(reset_mode: GSDEResetMode) -> None:
    policy = _GSDEEvaluationPolicy()
    policy.action_dist.reset_temporal_correlations_on_step(batch_shape=(3, 2))
    original_noise = policy.action_dist.get_temporal_correlation_state()
    noise_version = original_noise[0]._version
    training_noise = clone_detach_temporal_state(original_noise)
    runner = FrozenEvaluationRunner(
        make_env=_EvaluationEnv, training_env=object(), policy=policy, episodes_per_env=4, seed=123,
    )
    rng_state = torch.random.get_rng_state().clone()
    try:
        with torch.inference_mode():
            runner.evaluate(timesteps=25, milestone_percentage=25.0, gsde_reset_mode=reset_mode)
        stochastic_steps = policy.deterministic_modes.index(True)
        for step in range(1, stochastic_steps):
            previous = policy.noise_states[step - 1][0]
            current = policy.noise_states[step][0]
            changed = (previous != current).flatten(start_dim=1).any(dim=1)
            expected = policy.episode_start_masks[step].clone()
            if isinstance(reset_mode, GSDEIntervalResetMode) and step % reset_mode.interval == 0:
                expected.fill_(True)
            torch.testing.assert_close(changed, expected)
        torch.testing.assert_close(policy.action_dist.get_temporal_correlation_state(), training_noise)
        assert policy.action_dist.get_temporal_correlation_state()[0] is original_noise[0]
        assert original_noise[0]._version == noise_version
        assert torch.equal(torch.random.get_rng_state(), rng_state)
        policy.action_dist.reset_temporal_correlations_on_ep_start(torch.tensor([True, False, False]))
    finally:
        runner.close()


def test_frozen_gsde_probability_resets_are_selective_and_repeatable() -> None:
    policy = _GSDEEvaluationPolicy()
    runner = FrozenEvaluationRunner(
        make_env=_EvaluationEnv, training_env=object(), policy=policy, episodes_per_env=8, seed=123,
    )
    mode = GSDEProbabilityResetMode(0.25)
    try:
        runner.evaluate(timesteps=25, milestone_percentage=25.0, gsde_reset_mode=mode)
        stochastic_steps = policy.deterministic_modes.index(True)
        changes_outside_episode_starts = []
        for step in range(1, stochastic_steps):
            changed = (policy.noise_states[step - 1][0] != policy.noise_states[step][0]).flatten(1).any(1)
            episode_starts = policy.episode_start_masks[step]
            assert changed[episode_starts].all()
            changes_outside_episode_starts.extend(changed[~episode_starts].tolist())
        assert any(changes_outside_episode_starts)
        assert not all(changes_outside_episode_starts)
        first_noise = clone_detach_temporal_state(policy.noise_states)
        policy.noise_states.clear()
        runner.evaluate(timesteps=50, milestone_percentage=50.0, gsde_reset_mode=mode)
        torch.testing.assert_close(policy.noise_states, first_noise)
    finally:
        runner.close()


def test_frozen_gsde_requires_an_explicit_reset_schedule() -> None:
    runner = FrozenEvaluationRunner(make_env=_EvaluationEnv, training_env=object(), policy=_GSDEEvaluationPolicy())
    with pytest.raises(ValueError, match="gsde_reset_mode is required"):
        runner.evaluate(timesteps=25, milestone_percentage=25.0)


class _LongRecordingEnv(_EvaluationEnv):
    def start_video_recording(self, **kwargs):
        super().start_video_recording(**kwargs)
        self.recording_slots = {0, 1}
        self.recordings_started = 2
        self.recordings_completed = 0

    def get_video_recording_status(self):
        return {"active": bool(getattr(self, "recording_slots", set()))}

    def reset(self, **kwargs):
        assert not self.get_video_recording_status()["active"]
        self.total_steps = 0
        return super().reset(**kwargs)

    def step(self, actions):
        self.total_steps += 1
        self.steps += 1
        lengths = torch.tensor([1 if self.episodes[0] < 2 else 50, 10])
        dones = self.steps >= lengths
        episode_lengths = self.steps.clone()
        self.episodes += dones
        self.steps[dones] = 0
        for lane in torch.nonzero(dones).flatten().tolist():
            if lane in getattr(self, "recording_slots", set()):
                self.recording_slots.remove(lane)
                self.recordings_completed += 1
            if self.get_video_recording_status()["active"] and self.recordings_started < 4:
                self.recording_slots.add(lane)
                self.recordings_started += 1
        infos = {"_episode": dones, "episode": {"r": self.episodes.float(), "l": episode_lengths}}
        return _obs(), torch.zeros(2), dones, torch.zeros(2, dtype=torch.bool), infos


def test_frozen_evaluation_finishes_recording_without_counting_extra_episodes():
    env = _LongRecordingEnv()
    runner = FrozenEvaluationRunner(
        make_env=lambda: env, training_env=object(), policy=_EvaluationPolicy(),
        episodes_per_env=2, recording_config=EvaluationRecordingConfig(num_episodes=4), video_folder=Path("videos"),
    )
    for milestone in (25., 50.):
        metrics = runner.evaluate(timesteps=int(milestone), milestone_percentage=milestone)
        assert metrics["eval_deterministic_ep_rew"].n == 4
        assert metrics["eval_stochastic_ep_rew"].n == 4
        assert env.recordings_completed == 4
        assert env.total_steps == 52
        assert not env.get_video_recording_status()["active"]
    runner.close()


def test_scheduled_recording_retries_interrupted_milestone() -> None:
    class InterruptedRecordingEnv(_LongRecordingEnv):
        def step(self, actions: torch.Tensor) -> tuple:
            if self.get_video_recording_status()["active"]:
                raise RuntimeError("interrupted recording")
            return super().step(actions)

    interrupted_env = InterruptedRecordingEnv()
    retry_env = _LongRecordingEnv()
    runner = FrozenEvaluationRunner(
        make_env=Mock(side_effect=[interrupted_env, retry_env]), training_env=object(), policy=_EvaluationPolicy(),
        episodes_per_env=2, recording_config=EvaluationRecordingConfig(num_episodes=4), video_folder=Path("videos"),
    )
    algorithm = SimpleNamespace(n_total_timesteps=0)
    metrics_logger = Mock()
    hook = ScheduledEvaluationHook(
        algorithm=algorithm, total_timesteps=100, milestones=(25, 50), runner=runner, metrics_logger=metrics_logger,
    )
    try:
        algorithm.n_total_timesteps = 50
        with pytest.raises(RuntimeError, match="interrupted recording"):
            hook(algorithm, {}, rollout_steps=50)
        assert interrupted_env.closed
        metrics_logger.log.assert_not_called()

        hook(algorithm, {}, rollout_steps=0)
        hook(algorithm, {}, rollout_steps=0)
        metrics_logger.log.assert_called_once()
        assert metrics_logger.log.call_args.args[0]["eval_milestone_pct"] == 50
        assert retry_env.recordings_completed == 4
    finally:
        hook.close()


def test_frozen_recording_reports_a_delayed_writer_failure_before_returning() -> None:
    class EncoderFailureEnv(_EvaluationEnv):
        status_checks = 0

        def get_video_recording_status(self) -> dict[str, Any]:
            self.status_checks += 1
            if self.status_checks == 3:
                raise RuntimeError("video encoder failed")
            return {"active": False, "writer_jobs_pending": 1}

    env = EncoderFailureEnv()
    runner = FrozenEvaluationRunner(
        make_env=lambda: env, training_env=object(), policy=_EvaluationPolicy(),
        recording_config=EvaluationRecordingConfig(num_episodes=1), video_folder=Path("videos"),
    )
    try:
        with pytest.raises(RuntimeError, match="video encoder failed"):
            runner.evaluate(timesteps=50, milestone_percentage=50)
        assert env.closed
    finally:
        runner.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_frozen_evaluation_collects_numpy_statistics_for_accepted_lanes(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    stats = {
        "r": np.array([1.5, 2.5, 3.5], dtype=np.float32),
        "l": np.array([6, 7, 8], dtype=np.int32),
        "success": np.array([True, False, True]),
        "_success": np.array([False, True, True]),
        "optional_vector": torch.zeros((3, 2), dtype=torch.bfloat16, device=device),
        "_optional_vector": torch.tensor([False, True, True], device=device),
    }
    metrics: dict[str, list[float]] = {}
    FrozenEvaluationRunner._append_episode_metrics(
        episode_metrics=metrics, infos={"final_info": {"episode": stats}},
        dones=torch.tensor([True, True, True], device=device),
        accepted_dones=torch.tensor([True, False, True], device=device),
    )
    assert metrics == {"r": [1.5, 3.5], "l": [6., 8.], "success": [1.]}


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
        recording_config=EvaluationRecordingConfig(num_episodes=3),
        video_folder=Path("videos/eval"),
    )

    metrics = runner.evaluate(timesteps=25, milestone_percentage=25.0)

    assert metrics["eval_episodes"] == 8
    assert metrics["eval_episodes_per_mode"] == 4
    assert metrics["eval_stochastic_ep_rew"].n == 4
    assert metrics["eval_stochastic_ep_rew"].mean == 51.5
    assert metrics["eval_deterministic_ep_rew"].n == 4
    assert metrics["eval_deterministic_ep_rew"].mean == 51.5
    assert metrics["eval_stochastic_success_rate"] == 50.0
    assert metrics["eval_deterministic_success_rate"] == 50.0
    for mode in ("stochastic", "deterministic"):
        goals = metrics[f"eval_{mode}_ep_task_goals_completed"]
        assert (goals.n, goals.mean, goals.min_value, goals.max_value) == (2, 1.5, 1., 2.)
    assert metrics["timesteps"] == 25
    assert evaluation_env.reset_kwargs == {
        "seed": 123,
    }
    assert torch.equal(policy.action_dist.state, torch.tensor([7.0]))
    assert policy.training
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert len(policy.episode_start_masks) == 12
    assert policy.deterministic_modes == [False] * 6 + [True] * 6
    torch.testing.assert_close(
        torch.stack(policy.random_values[:6]),
        torch.stack(policy.random_values[6:]),
    )
    assert evaluation_env.recording_kwargs is not None
    assert evaluation_env.recording_kwargs["num_episodes"] == 3
    assert evaluation_env.recording_kwargs["max_parallel_episodes"] == 2
    assert evaluation_env.recording_kwargs["video_name_prefix"] == "eval_deterministic_025pct_25_steps"

    runner.close()
    assert evaluation_env.closed


def test_frozen_evaluation_can_reuse_stateful_environment() -> None:
    evaluation_env = _EvaluationEnv()
    runner = FrozenEvaluationRunner(
        make_env=lambda: evaluation_env,
        training_env=object(),
        policy=_EvaluationPolicy(),
        episodes_per_env=1,
    )

    first_metrics = runner.evaluate(timesteps=25, milestone_percentage=25.0)
    second_metrics = runner.evaluate(timesteps=50, milestone_percentage=50.0)

    assert first_metrics["eval_stochastic_ep_rew"].mean == second_metrics["eval_stochastic_ep_rew"].mean
    assert first_metrics["eval_deterministic_ep_rew"].mean == second_metrics["eval_deterministic_ep_rew"].mean
    assert first_metrics["eval_stochastic_success_rate"] == second_metrics["eval_stochastic_success_rate"]
    assert first_metrics["eval_deterministic_success_rate"] == second_metrics["eval_deterministic_success_rate"]
    assert evaluation_env.recording_kwargs is None
    runner.close()
    assert evaluation_env.closed


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("env_device, next_policy_device", [("cpu", "cuda"), ("cuda", "cpu")])
def test_cached_frozen_environment_follows_the_policy_across_devices(
    env_device: str, next_policy_device: str,
) -> None:
    from tests.test_off_policy_normalization import _make_algorithm, _make_normalization_env, _set_statistics

    algorithm = _make_algorithm("tmasac__lstm", transition_observations=True, use_nop=False)
    _set_statistics(algorithm, mean=2.0, variance=4.0)
    policy = algorithm.policy
    factory = Mock(side_effect=lambda: _make_normalization_env(device=env_device, transition_observations=True))
    runner = FrozenEvaluationRunner(make_env=factory, training_env=algorithm.env, policy=policy)
    try:
        for step, policy_device in enumerate((env_device, next_policy_device)):
            policy.to(policy_device)
            with patch.object(policy, "act_with_temporal_state", wraps=policy.act_with_temporal_state) as act:
                metrics = runner.evaluate(timesteps=step, milestone_percentage=step * 100)
            assert metrics["eval_episodes"] == 2
            assert policy.training
            first = act.call_args_list[0].kwargs
            current = first["local_obs"][..., 0]
            assert current.device.type == policy_device
            assert first["episode_start_mask"].device.type == policy_device
            torch.testing.assert_close(current, torch.full_like(current, (10.0 - 2.0) / (4.0 + 1e-6) ** 0.5))
        factory.assert_called_once()
    finally:
        runner.close()
        algorithm.env.close()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_frozen_evaluation_places_observations_on_a_buffer_only_actors_device() -> None:
    class FixedPolicy(_EvaluationPolicy):
        def __init__(self) -> None:
            super().__init__()
            del self.parameter
            self.register_buffer("gain", torch.ones(1, 1, device="cuda"))

        def act_with_temporal_state(self, *, local_obs: torch.Tensor, temporal_state: torch.Tensor,
                                    **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            return local_obs @ self.gain, temporal_state

    runner = FrozenEvaluationRunner(make_env=_EvaluationEnv, training_env=object(), policy=FixedPolicy())
    try:
        assert runner.evaluate(timesteps=1, milestone_percentage=100)["eval_episodes"] == 4
    finally:
        runner.close()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_frozen_evaluation_seeds_the_physics_device_with_cpu_policy_and_wrapper() -> None:
    class CPUWrapper(_EvaluationEnv):
        device = torch.device("cpu")
        unwrapped = SimpleNamespace(device=torch.device("cuda"))

        def reset(self, **kwargs: object) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
            physics_draws.append(torch.rand(4, device=self.unwrapped.device))
            return super().reset(**kwargs)

    physics_draws: list[torch.Tensor] = []
    runner = FrozenEvaluationRunner(make_env=CPUWrapper, training_env=object(), policy=_EvaluationPolicy(), seed=92)
    try:
        prior = torch.cuda.get_rng_state()
        runner.evaluate(timesteps=1, milestone_percentage=50)
        expected = torch.rand(4, device="cuda", generator=torch.Generator(device="cuda").manual_seed(92))
        for actual in physics_draws:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), prior)
    finally:
        runner.close()


@pytest.mark.parametrize("factory_fails", [False, True])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda", marks=[pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")],
    ),
])
def test_first_evaluation_preserves_rng_through_environment_construction(
    factory_fails: bool, device: str,
) -> None:
    policy = _EvaluationPolicy().to(device)
    evaluation_env = _EvaluationEnv()

    def make_env() -> _EvaluationEnv:
        torch.manual_seed(987)
        torch.rand(3)
        if device == "cuda":
            torch.rand(3, device=device)
        if factory_fails:
            raise RuntimeError("environment construction failed")
        return evaluation_env

    runner = FrozenEvaluationRunner(make_env=make_env, training_env=object(), policy=policy)
    cpu_rng_state = torch.random.get_rng_state().clone()
    cuda_rng_states = torch.cuda.get_rng_state_all() if device == "cuda" else None
    try:
        if factory_fails:
            with pytest.raises(RuntimeError, match="environment construction failed"):
                runner.evaluate(timesteps=25, milestone_percentage=25.0)
        else:
            runner.evaluate(timesteps=25, milestone_percentage=25.0)
        assert torch.equal(torch.random.get_rng_state(), cpu_rng_state)
        if cuda_rng_states is not None:
            assert all(
                torch.equal(actual, expected)
                for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_rng_states, strict=True)
            )
    finally:
        runner.close()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cpu_evaluation_preserves_cuda_rng_across_seeded_env_resets() -> None:
    class GloballySeededEnv(_EvaluationEnv):
        def reset(self, *, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
            torch.manual_seed(seed)
            return super().reset(seed=seed)

    runner = FrozenEvaluationRunner(make_env=GloballySeededEnv, training_env=object(), policy=_EvaluationPolicy())
    try:
        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.manual_seed(89)
            states = torch.cuda.get_rng_state_all()
            for timestep in (25, 50):
                runner.evaluate(timesteps=timestep, milestone_percentage=timestep)
                for actual, expected in zip(torch.cuda.get_rng_state_all(), states, strict=True):
                    torch.testing.assert_close(actual, expected)
    finally:
        runner.close()


def _check_fresh_cuda_evaluation(first_cuda_use: str) -> None:
    assert not torch.cuda.is_initialized()
    torch.manual_seed(91)
    torch.cuda.manual_seed(92)
    enable_gpu = False

    class Env(_EvaluationEnv):
        def reset(self, *, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
            torch.manual_seed(seed)
            if enable_gpu and first_cuda_use == "reset":
                torch.rand(3, device="cuda")
            return super().reset(seed=seed)

    def make_env() -> Env:
        torch.manual_seed(1234)
        if enable_gpu and first_cuda_use == "factory":
            torch.rand(3, device="cuda")
        return Env()

    for enable_gpu in (False, torch.cuda.is_available()):
        runner = FrozenEvaluationRunner(make_env=make_env, training_env=object(), policy=_EvaluationPolicy())
        try:
            runner.evaluate(timesteps=1, milestone_percentage=100)
            assert torch.cuda.is_initialized() == enable_gpu
        finally:
            runner.close()
    if enable_gpu:
        for index in range(torch.cuda.device_count()):
            device = torch.device("cuda", index)
            expected_seed = 92 if index == torch.cuda.current_device() else 91
            expected = torch.rand(3, device=device, generator=torch.Generator(device=device).manual_seed(expected_seed))
            torch.testing.assert_close(torch.rand(3, device=device), expected)


@pytest.mark.parametrize("first_cuda_use", ["factory", "reset"])
def test_frozen_evaluation_preserves_lazy_cuda_initialization_and_seeds(first_cuda_use: str) -> None:
    # CUDA initialization is irreversible within a process shared with other tests.
    subprocess.run([
        sys.executable, "-c", "from tests.test_frozen_evaluation import _check_fresh_cuda_evaluation; "
        f"_check_fresh_cuda_evaluation({first_cuda_use!r})",
    ], check=True, timeout=90)


def test_concurrent_frozen_evaluations_restore_shared_policy() -> None:
    first_acting, second_attempting, second_finished = Event(), Event(), Event()

    class PausingEnv(_EvaluationEnv):
        def step(self, actions: torch.Tensor) -> tuple:
            if not first_acting.is_set():
                first_acting.set()
                assert second_attempting.wait(timeout=10)
                second_finished.wait(timeout=0.2)
            return super().step(actions)

    policy = _EvaluationPolicy()
    original_noise = policy.action_dist.state.clone()
    first = FrozenEvaluationRunner(make_env=PausingEnv, training_env=object(), policy=policy)
    second = FrozenEvaluationRunner(make_env=_EvaluationEnv, training_env=object(), policy=policy)
    try:
        # Use already-built environments so the second call can reach the active policy immediately.
        assert first.env is not second.env
        with ThreadPoolExecutor(1) as pool:
            task = pool.submit(first.evaluate, timesteps=1, milestone_percentage=50)
            assert first_acting.wait(timeout=10)
            second_attempting.set()
            try:
                second.evaluate(timesteps=1, milestone_percentage=50)
            finally:
                second_finished.set()
            task.result(timeout=10)
        assert policy.training
        torch.testing.assert_close(policy.action_dist.state, original_noise)
    finally:
        first.close()
        second.close()


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
    runner.evaluate.assert_called_once_with(timesteps=30, milestone_percentage=25.0, gsde_reset_mode=None)
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

    runner.evaluate.assert_called_once_with(timesteps=60, milestone_percentage=50.0, gsde_reset_mode=None)

    runner.evaluate.reset_mock()
    algorithm.n_total_timesteps = 80
    hook(algorithm, {}, rollout_steps=20)

    runner.evaluate.assert_called_once_with(timesteps=80, milestone_percentage=75.0, gsde_reset_mode=None)
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
    runner.evaluate.assert_called_once_with(timesteps=200, milestone_percentage=50.0, gsde_reset_mode=None)

    runner.evaluate.reset_mock()
    algorithm.n_total_timesteps = 300
    hook(algorithm, {}, rollout_steps=100)
    runner.evaluate.assert_called_once_with(timesteps=300, milestone_percentage=100.0, gsde_reset_mode=None)


def test_scheduled_evaluation_waits_for_fractional_targets_and_reports_latest() -> None:
    algorithm = SimpleNamespace(n_total_timesteps=100)
    runner = Mock()
    hook = ScheduledEvaluationHook(
        algorithm=algorithm, start_timesteps=100, total_timesteps=110, milestones=(28, 25, 95, 100),
        runner=runner, metrics_logger=Mock(),
    )
    for step in (102, 103, 109, 110, 111):
        algorithm.n_total_timesteps = step
        hook(algorithm, {}, rollout_steps=1)

    evaluations = [(call.kwargs["timesteps"], call.kwargs["milestone_percentage"])
                   for call in runner.evaluate.call_args_list]
    assert evaluations == [(103, 28), (110, 100)]


def test_scheduled_evaluation_handles_numpy_training_windows_and_percentages() -> None:
    algorithm = SimpleNamespace(n_total_timesteps=np.int32(0))
    runner = Mock()
    hook = ScheduledEvaluationHook(algorithm=algorithm, start_timesteps=np.int32(0),
                                  total_timesteps=np.int32(100_000_000), milestones=np.array([25, 50, 100], np.float32),
                                  runner=runner, metrics_logger=Mock())
    for step in (24_999_999, 25_000_000, 49_999_999, 50_000_000, 99_999_999, 100_000_000):
        algorithm.n_total_timesteps = np.int32(step)
        hook(algorithm, {}, rollout_steps=1)
    evaluations = [(call.kwargs["timesteps"], call.kwargs["milestone_percentage"])
                   for call in runner.evaluate.call_args_list]
    assert evaluations == [(25_000_000, 25), (50_000_000, 50), (100_000_000, 100)]


@pytest.mark.parametrize("start,total", [(0, float("nan")), (0, float("inf")), (1.5, 5), (False, 10)])
def test_scheduled_evaluation_rejects_noninteger_windows(start: object, total: object) -> None:
    with pytest.raises(ValueError, match="integers"):
        ScheduledEvaluationHook(algorithm=SimpleNamespace(n_total_timesteps=0), start_timesteps=start,
                                total_timesteps=total, milestones=[100], runner=Mock(), metrics_logger=Mock())


def test_scheduled_evaluation_uses_the_current_training_gsde_schedule() -> None:
    algorithm = SimpleNamespace(n_total_timesteps=0, gsde_reset_mode=GSDEEpisodeResetMode())
    runner = Mock()
    hook = ScheduledEvaluationHook(
        algorithm=algorithm, total_timesteps=100, milestones=(25, 50), runner=runner, metrics_logger=Mock(),
    )
    for steps, mode in ((25, GSDEIntervalResetMode(3)), (50, GSDEProbabilityResetMode(0.25))):
        algorithm.n_total_timesteps = steps
        algorithm.gsde_reset_mode = mode
        hook(algorithm, {}, rollout_steps=25)
        assert runner.evaluate.call_args.kwargs["gsde_reset_mode"] == mode
