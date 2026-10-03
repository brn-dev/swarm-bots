from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import torch

from swarmbots.benchmark.recording import _record_episodes
from swarmbots.learn.benchmark_policy import BenchmarkPolicy
from swarmbots.learn.checkpointing import (
    align_torch_compile_state_dict_keys,
    apply_env_state,
    extract_env_state,
    extract_policy_state_dict,
    load_checkpoint,
)
from swarmbots.learn.presets.policy_factory import ContinuousActionDistVariant
from swarmbots.learn.training import _make_policy_env, _variant_options
from swarmbots.utils.recording_resolution import DEFAULT_RECORDING_HEIGHT, DEFAULT_RECORDING_WIDTH


def record_checkpoint(
    checkpoint_path: str | Path,
    benchmark_id: str,
    variant: str,
    *,
    video_folder: str | Path,
    video_name_prefix: str | None = None,
    num_episodes: int = 5,
    max_parallel_episodes: int = 4,
    deterministic: bool = True,
    device: str | torch.device = "auto",
    seed: int = 1_000,
    episode_length: int | None = None,
    scenario_kwargs: Mapping[str, object] | None = None,
    env_kwargs: Mapping[str, Any] | None = None,
    continuous_action_dist: ContinuousActionDistVariant | None = None,
    use_nop: bool | None = None,
    compile_modules: bool = False,
    policy_kwargs: Mapping[str, Any] | None = None,
    fps: int = 30,
    fps_mode: Literal["compensate_stride", "fixed"] = "compensate_stride",
    frame_stride: int = 1,
    width: int = DEFAULT_RECORDING_WIDTH,
    height: int = DEFAULT_RECORDING_HEIGHT,
    camera: int | str = -1,
) -> Path:
    """Load a matching learning preset and record without constructing a trainer.

    Restores policy weights and normalization, including compiled checkpoint
    keys. The task, architecture, distribution and NOP settings must match the
    checkpoint. Replay buffers and optimizer state are not needed for inference.
    """
    if num_episodes <= 0 or max_parallel_episodes <= 0:
        raise ValueError("num_episodes and max_parallel_episodes must be positive")
    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    policy_options, _ = _variant_options(variant)
    policy_options.update(policy_kwargs or {})
    if continuous_action_dist is not None:
        policy_options["continuous_action_dist"] = continuous_action_dist
    if use_nop is not None:
        policy_options["use_nop"] = use_nop
    env, policy = _make_policy_env(
        benchmark_id,
        num_envs=min(num_episodes, max_parallel_episodes),
        device=device,
        seed=seed,
        episode_length=episode_length,
        scenario_kwargs=scenario_kwargs,
        env_kwargs=env_kwargs,
        policy_options=policy_options,
        compile_modules=compile_modules,
    )
    try:
        policy.load_state_dict(
            align_torch_compile_state_dict_keys(
                extract_policy_state_dict(checkpoint),
                target_keys=policy.state_dict().keys(),
            )
        )
        apply_env_state(env, extract_env_state(checkpoint))
        del checkpoint
        _record_episodes(
            env.unwrapped,
            BenchmarkPolicy(policy, env, deterministic=deterministic),
            video_folder=video_folder,
            video_name_prefix=checkpoint_path.stem if video_name_prefix is None else video_name_prefix,
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
