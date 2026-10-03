from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import torch

from swarmbots.benchmark.evaluation import Policy, policy_observation
from swarmbots.benchmark.registry import make_env
from swarmbots.mjw_env.mjw_swarm_bots_vector_env import MJWSwarmBotsVectorEnv
from swarmbots.utils.recording_resolution import DEFAULT_RECORDING_HEIGHT, DEFAULT_RECORDING_WIDTH


def _record_episodes(
    env: MJWSwarmBotsVectorEnv,
    policy: Policy,
    *,
    video_folder: str | Path,
    video_name_prefix: str = "policy",
    num_episodes: int = 5,
    seed: int = 1_000,
    fps: int = 30,
    fps_mode: Literal["compensate_stride", "fixed"] = "compensate_stride",
    frame_stride: int = 1,
    width: int = DEFAULT_RECORDING_WIDTH,
    height: int = DEFAULT_RECORDING_HEIGHT,
    camera: int | str = -1,
) -> None:
    torch.manual_seed(seed)
    observations, _ = env.reset(seed=seed)
    env.start_video_recording(
        video_folder=str(video_folder),
        video_name_prefix=video_name_prefix,
        num_episodes=num_episodes,
        max_parallel_episodes=env.num_envs,
        fps=fps,
        fps_mode=fps_mode,
        frame_stride=frame_stride,
        width=width,
        height=height,
        camera=camera,
    )
    episode_starts = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    with torch.inference_mode():
        while env.get_video_recording_status()["active"]:
            actions = policy(policy_observation(observations), episode_starts)
            observations, _, terminations, truncations, _ = env.step(dict(actions))
            episode_starts = terminations | truncations


def record_policy(
    policy: Policy,
    benchmark_id: str,
    *,
    video_folder: str | Path,
    video_name_prefix: str = "policy",
    num_episodes: int = 5,
    max_parallel_episodes: int = 4,
    device: str | torch.device = "auto",
    seed: int = 1_000,
    episode_length: int | None = None,
    scenario_kwargs: Mapping[str, object] | None = None,
    env_kwargs: Mapping[str, Any] | None = None,
    fps: int = 30,
    fps_mode: Literal["compensate_stride", "fixed"] = "compensate_stride",
    frame_stride: int = 1,
    width: int = DEFAULT_RECORDING_WIDTH,
    height: int = DEFAULT_RECORDING_HEIGHT,
    camera: int | str = -1,
) -> Path:
    """Record complete episodes with the benchmark callable policy interface.

    The environment is closed and all MP4 writers finish before returning the
    output directory. Configure neural-network evaluation mode and frozen
    normalization in the callable, or use ``as_benchmark_policy(trainer)``.
    """
    if num_episodes <= 0 or max_parallel_episodes <= 0:
        raise ValueError("num_episodes and max_parallel_episodes must be positive")
    env = make_env(
        benchmark_id,
        num_envs=min(num_episodes, max_parallel_episodes),
        device=device,
        seed=seed,
        episode_length=episode_length,
        scenario_kwargs=scenario_kwargs,
        **dict(env_kwargs or {}),
    )
    try:
        _record_episodes(
            env,
            policy,
            video_folder=video_folder,
            video_name_prefix=video_name_prefix,
            num_episodes=num_episodes,
            seed=seed,
            fps=fps,
            fps_mode=fps_mode,
            frame_stride=frame_stride,
            width=width,
            height=height,
            camera=camera,
        )
    finally:
        env.close()
    return Path(video_folder)
