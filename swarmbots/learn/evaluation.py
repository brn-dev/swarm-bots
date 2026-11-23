from __future__ import annotations

import time
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from swarmbots.learn.action_dists.gsde_action_dist import GSDEActionDist
from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution
from swarmbots.learn.checkpointing import apply_env_state, capture_env_state, freeze_env_normalization
from swarmbots.learn.algos.off_policy.normalization import ReplayNormalization
from swarmbots.learn.gsde_reset import GSDEResetMode, reset_gsde_noise, resolve_gsde_reset_mode
from swarmbots.learn.metrics_logger import MetricsLogger, RetryableMetricWrite
from swarmbots.learn.rollout_utils import initial_previous_actions, iter_episode_rows
from swarmbots.learn.summary_statistics import compute_summary_statistics
from swarmbots.learn.task_episode_metrics import task_episode_metric_names
from swarmbots.learn.temporal_state import clone_detach_temporal_state
from swarmbots.learn.torch_device import module_device
from swarmbots.utils.recording_resolution import DEFAULT_RECORDING_HEIGHT, DEFAULT_RECORDING_WIDTH
from swarmbots.utils.torch_utils import fork_torch_rng, seed_torch_rng


DEFAULT_EVALUATION_MILESTONES: tuple[float, ...] = (25, 50, 75, 90, 95, 100)
EVALUATION_ACTION_MODES: tuple[tuple[str, bool], ...] = (
    ("stochastic", False),
    ("deterministic", True),
)


def _snapshot_action_state(action_dist: Any, state: Any) -> Any:
    if state is None:
        return None
    setter = getattr(action_dist.set_temporal_correlation_state, "__func__", None)
    # Stock gSDE swaps buffers. Keep the training tensor and its version intact.
    if type(action_dist) is GSDEActionDist and setter is GSDEActionDist.set_temporal_correlation_state:
        return state
    if type(action_dist) is HybridActionDistribution and setter is HybridActionDistribution.set_temporal_correlation_state:
        return tuple(
            _snapshot_action_state(dist, part)
            for dist, part in zip(action_dist.distributions, state, strict=True)
        )
    return clone_detach_temporal_state(state)


@dataclass(frozen=True, slots=True)
class EvaluationRecordingConfig:
    num_episodes: int = 0
    fps: int = 30
    width: int = DEFAULT_RECORDING_WIDTH
    height: int = DEFAULT_RECORDING_HEIGHT
    camera: int | str = -1


class FrozenEvaluationRunner:
    def __init__(
        self,
        *,
        make_env: Callable[[], Any],
        training_env: Any,
        policy: Any,
        episodes_per_env: int = 1,
        seed: int = 1_000_000,
        recording_config: EvaluationRecordingConfig | None = None,
        video_folder: Path | None = None,
    ) -> None:
        if episodes_per_env <= 0:
            raise ValueError(f"episodes_per_env must be positive, got {episodes_per_env}")
        if recording_config is not None and recording_config.num_episodes < 0:
            raise ValueError(
                f"evaluation recording episode count must be non-negative, got {recording_config.num_episodes}"
            )

        self._make_env = make_env
        self._training_env = training_env
        self._policy = policy
        self.episodes_per_env = int(episodes_per_env)
        self.seed = int(seed)
        self.recording_config = recording_config
        self.video_folder = video_folder
        self._env: Any | None = None
        self._replay_normalization: ReplayNormalization | None = None

    @property
    def env(self) -> Any:
        if self._env is not None:
            return self._env
        with fork_torch_rng():
            if self._env is None:
                env = self._make_env()
                try:
                    normalization = ReplayNormalization.for_policy(env, self._policy)
                except BaseException:
                    env.close()
                    raise
                self._replay_normalization = normalization
                self._env = env
        return self._env

    def evaluate(
        self, *, timesteps: int, milestone_percentage: float, gsde_reset_mode: GSDEResetMode | None = None,
    ) -> dict[str, Any]:
        gsde_reset_mode = resolve_gsde_reset_mode(
            gsde_enabled=bool(getattr(self._policy, "gsde_enabled", False)), reset_mode=gsde_reset_mode,
        )
        policy_device = module_device(self._policy)
        metrics: dict[str, Any] = {}
        # A custom reset may seed every CUDA generator even when the policy runs on CPU.
        with fork_torch_rng():
            env = self.env
            wrapper_device = getattr(env, "device", None)
            physics_device = getattr(getattr(env, "unwrapped", env), "device", None)
            apply_env_state(env, capture_env_state(self._training_env))
            freeze_env_normalization(env)
            action_dist = getattr(self._policy, "action_dist", None)
            action_dist_state = (
                _snapshot_action_state(action_dist, action_dist.get_temporal_correlation_state())
                if action_dist is not None and hasattr(action_dist, "get_temporal_correlation_state")
                else None
            )
            module_training_modes = {module: module.training for module in self._policy.modules()}
            started_at = time.perf_counter()
            try:
                for action_mode, deterministic in EVALUATION_ACTION_MODES:
                    mode_started_at = time.perf_counter()
                    seed_torch_rng(self.seed, policy_device or "cpu", wrapper_device, physics_device)
                    if action_dist_state is not None:
                        action_dist.set_temporal_correlation_state(
                            clone_detach_temporal_state(action_dist_state)
                        )
                    obs, _info = env.reset(seed=self.seed)
                    recording_started = False
                    if deterministic:
                        recording_started = self._start_recording(
                            env=env,
                            timesteps=timesteps,
                            milestone_percentage=milestone_percentage,
                            action_mode=action_mode,
                        )
                    episode_metrics = self._collect_episodes(
                        env=env,
                        initial_obs=obs,
                        policy_device=policy_device or obs["local_obs"].device,
                        deterministic=deterministic,
                        gsde_reset_mode=gsde_reset_mode,
                        finish_recording=recording_started,
                    )
                    metrics.update(self._summarize_metrics(episode_metrics, action_mode=action_mode))
                    metrics[f"eval_{action_mode}_duration"] = time.perf_counter() - mode_started_at
            except BaseException:
                # A failed rollout may leave recording slots active even after a seeded reset.
                self.close()
                raise
            finally:
                try:
                    if action_dist is not None and hasattr(action_dist, "set_temporal_correlation_state"):
                        action_dist.set_temporal_correlation_state(action_dist_state)
                finally:
                    for module, was_training in module_training_modes.items():
                        module.training = was_training

        duration = time.perf_counter() - started_at
        episodes_per_mode = env.num_envs * self.episodes_per_env
        metrics.update(
            {
                "eval_duration": duration,
                "eval_episodes": episodes_per_mode * len(EVALUATION_ACTION_MODES),
                "eval_episodes_per_mode": episodes_per_mode,
                "eval_episodes_per_env": self.episodes_per_env,
                "eval_milestone_pct": float(milestone_percentage),
                "timesteps": int(timesteps),
            }
        )
        return metrics

    def close(self) -> None:
        env = self._env
        self._env = None
        self._replay_normalization = None
        if env is not None:
            env.close()

    def _collect_episodes(
        self,
        *,
        env: Any,
        initial_obs: dict[str, torch.Tensor],
        policy_device: torch.device,
        deterministic: bool,
        gsde_reset_mode: GSDEResetMode | None = None,
        finish_recording: bool = False,
    ) -> dict[str, list[float]]:
        self._policy.eval()
        obs = initial_obs
        env_device = obs["local_obs"].device
        previous_actions = initial_previous_actions(
            policy=self._policy,
            obs=obs,
            n_agent_actions=env.action_space.total_agent_action_dim,
        )
        if previous_actions is not None:
            previous_actions = previous_actions.to(policy_device)
        temporal_state = self._policy.initial_temporal_state(
            batch_size=env.num_envs,
            n_agents=env.n_agents,
            device=policy_device,
            dtype=obs["local_obs"].dtype,
        )
        episode_start_mask = torch.ones(
            (env.num_envs,),
            device=policy_device,
            dtype=torch.bool,
        )
        completed_per_env = torch.zeros(env.num_envs, device=env_device, dtype=torch.int64)
        episode_metrics: dict[str, list[float]] = {}
        rollout_step_idx = 0
        action_dist = getattr(self._policy, "action_dist", None)

        normalization = (
            None if self._replay_normalization is None else self._replay_normalization.snapshot(
                device=env_device, public_only=True,
            )
        )
        with torch.no_grad():
            while (
                bool(torch.any(completed_per_env < self.episodes_per_env))
                or (finish_recording and env.unwrapped.get_video_recording_status()["active"])
            ):
                policy_obs = obs if normalization is None else normalization.normalize_observations(obs)
                if policy_device != env_device:
                    policy_obs = {key: value.to(policy_device) for key, value in policy_obs.items()}
                if not deterministic and gsde_reset_mode is not None:
                    reset_gsde_noise(
                        action_dist=action_dist, local_obs=policy_obs["local_obs"], episode_start_mask=episode_start_mask,
                        rollout_step_idx=rollout_step_idx, gsde_reset_mode=gsde_reset_mode,
                        initialize=rollout_step_idx == 0,
                    )
                elif action_dist is not None and hasattr(action_dist, "reset_temporal_correlations_on_ep_start"):
                    action_dist.reset_temporal_correlations_on_ep_start(episode_start_mask)

                actions, temporal_state = self._policy.act_with_temporal_state(
                    local_obs=policy_obs["local_obs"],
                    global_obs=policy_obs["global_obs"],
                    hidden_local_vars=policy_obs["hidden_local_vars"],
                    hidden_global_vars=policy_obs["hidden_global_vars"],
                    agent_mask=policy_obs.get("agent_mask"),
                    **({} if "scenario_id" not in policy_obs else {"scenario_ids": policy_obs["scenario_id"]}),
                    previous_actions=previous_actions,
                    deterministic=deterministic,
                    temporal_state=temporal_state,
                    episode_start_mask=episode_start_mask,
                )
                obs, _rewards, terminations, truncations, infos = env.step(actions.to(env_device))
                dones = torch.logical_or(terminations, truncations)
                accepted_dones = dones & (completed_per_env < self.episodes_per_env)
                self._append_episode_metrics(
                    episode_metrics=episode_metrics,
                    infos=infos,
                    accepted_dones=accepted_dones,
                    dones=dones,
                )
                completed_per_env += accepted_dones.to(dtype=completed_per_env.dtype)
                episode_start_mask = dones.to(device=policy_device, dtype=torch.bool)
                if previous_actions is not None:
                    previous_actions = actions.detach().masked_fill(episode_start_mask[:, None, None], 0.0)
                rollout_step_idx += 1

        if finish_recording:
            # Capture can finish while MP4 writers still own frames. Surface their result in this milestone.
            while env.unwrapped.get_video_recording_status().get("writer_jobs_pending", 0):
                time.sleep(0.01)
        return episode_metrics

    @staticmethod
    def _append_episode_metrics(
        *,
        episode_metrics: dict[str, list[float]],
        infos: dict[str, Any],
        accepted_dones: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        accepted_indices = torch.nonzero(accepted_dones, as_tuple=True)[0].tolist()
        if not accepted_indices:
            return

        info_source = infos
        if "episode" not in info_source:
            final_info = infos.get("final_info")
            if isinstance(final_info, dict) and "episode" in final_info:
                info_source = final_info
            else:
                raise ValueError("Evaluation requires episode statistics in infos['episode'].")

        episode_stats = info_source["episode"]
        if not isinstance(episode_stats, Mapping):
            raise ValueError(f"Expected infos['episode'] to be a mapping, got {type(episode_stats)}")
        episode_mask = torch.as_tensor(
            info_source.get("_episode", dones),
            device=dones.device,
            dtype=torch.bool,
        ).reshape(-1)
        if not torch.equal(episode_mask, dones):
            raise ValueError("Expected infos['_episode'] to match evaluation dones.")

        for episode in iter_episode_rows(episode_stats, accepted_indices):
            for key, value in episode.items():
                if isinstance(value, (bool, int, float)):
                    episode_metrics.setdefault(key, []).append(float(value))

    @staticmethod
    def _summarize_metrics(
        episode_metrics: dict[str, list[float]],
        *,
        action_mode: str,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        summary_keys = {
            "r": "ep_rew",
            "l": "ep_len",
            "progress_reward": "progress_reward",
            "successful_connections_per_unit_mean": "successful_connections_per_unit_mean",
            "successful_connections_per_unit_std": "successful_connections_per_unit_std",
            "successful_connections_per_unit_min": "successful_connections_per_unit_min",
            "successful_connections_per_unit_max": "successful_connections_per_unit_max",
            **task_episode_metric_names(episode_metrics),
        }
        for source_key, metric_suffix in summary_keys.items():
            values = episode_metrics.get(source_key)
            if values:
                result[f"eval_{action_mode}_{metric_suffix}"] = compute_summary_statistics(
                    values,
                    find_min=True,
                    find_max=True,
                )
        successes = episode_metrics.get("success")
        if successes:
            result[f"eval_{action_mode}_success_rate"] = 100.0 * sum(successes) / len(successes)
        return result

    def _start_recording(
        self,
        *,
        env: Any,
        timesteps: int,
        milestone_percentage: float,
        action_mode: str,
    ) -> bool:
        config = self.recording_config
        if config is None or config.num_episodes == 0 or self.video_folder is None:
            return False
        start_recording = getattr(env.unwrapped, "start_video_recording", None)
        if not callable(start_recording):
            logger.warning("Evaluation recording requested, but the evaluation environment cannot record video.")
            return False
        if not callable(getattr(env.unwrapped, "get_video_recording_status", None)):
            raise ValueError("Evaluation recording requires get_video_recording_status().")

        episode_count = min(config.num_episodes, env.num_envs * self.episodes_per_env)
        start_recording(
            video_folder=str(self.video_folder),
            video_name_prefix=(
                f"eval_{action_mode}_{_format_percentage(milestone_percentage)}pct_{timesteps}_steps"
            ),
            num_episodes=episode_count,
            max_parallel_episodes=min(episode_count, env.num_envs),
            fps=config.fps,
            width=config.width,
            height=config.height,
            camera=config.camera,
        )
        return True


class ScheduledEvaluationHook:
    def __init__(
        self,
        *,
        algorithm: Any,
        total_timesteps: int,
        start_timesteps: int = 0,
        milestones: Sequence[float],
        runner: FrozenEvaluationRunner,
        metrics_logger: MetricsLogger,
    ) -> None:
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (start_timesteps, total_timesteps)):
            raise ValueError("Evaluation start_timesteps and total_timesteps must be integers")
        start_timesteps, total_timesteps = int(start_timesteps), int(total_timesteps)
        if total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be positive, got {total_timesteps}")
        if start_timesteps < 0 or start_timesteps >= total_timesteps:
            raise ValueError(
                "start_timesteps must be in [0, total_timesteps), got "
                f"{start_timesteps=} {total_timesteps=}"
            )
        self.runner = runner
        self.metrics_logger = metrics_logger
        self._start_timesteps = start_timesteps
        self._window_steps = total_timesteps - start_timesteps
        self._pending: list[float] = []
        self._pending_write: tuple[int, RetryableMetricWrite] | None = None
        for percentage in milestones:
            if not 0 <= percentage <= 100:
                raise ValueError(f"Evaluation milestone must be in [0, 100], got {percentage}")
            self._pending.append(float(percentage))
        self._pending.sort()
        del self._pending[:self._reached_count(algorithm.n_total_timesteps)]

    def _reached_count(self, timesteps: int) -> int:
        timesteps = int(timesteps)
        if timesteps <= self._start_timesteps:
            return 0
        progress = 100 * (timesteps - self._start_timesteps) / self._window_steps
        return bisect_right(self._pending, progress)

    def __call__(self, algorithm: Any, metrics: dict[str, Any], rollout_steps: int) -> None:
        if self._pending_write is not None:
            self._publish_pending()
        reached = self._reached_count(algorithm.n_total_timesteps)
        if not reached:
            return

        percentage = self._pending[reached - 1]

        logger.warning(
            f"Running frozen evaluation for {percentage:g}% milestone "
            f"at {algorithm.n_total_timesteps} timesteps."
        )
        evaluation_metrics = self.runner.evaluate(
            timesteps=int(algorithm.n_total_timesteps),
            milestone_percentage=percentage,
            gsde_reset_mode=getattr(algorithm, "gsde_reset_mode", None),
        )
        self._pending_write = reached, RetryableMetricWrite(evaluation_metrics.copy())
        self._publish_pending()

    def _publish_pending(self) -> None:
        assert self._pending_write is not None
        reached, write = self._pending_write
        metrics_logger = self.metrics_logger
        if (
            isinstance(metrics_logger, MetricsLogger)
            and getattr(metrics_logger.log, "__func__", None) is MetricsLogger.log
            and getattr(metrics_logger.flush, "__func__", None) is MetricsLogger.flush
        ):
            metrics_logger.publish_retryable(write)
        else:
            if not write.queued:
                metrics_logger.log(write.metrics)
                write.queued = True
            metrics_logger.flush()
        self._pending_write = None
        del self._pending[:reached]

    def close(self) -> None:
        try:
            self.runner.close()
        finally:
            self.metrics_logger.close()


def _format_percentage(percentage: float) -> str:
    percentage = float(percentage)
    if percentage.is_integer():
        return f"{int(percentage):03d}"
    return str(percentage).rstrip("0").rstrip(".").replace(".", "p")
