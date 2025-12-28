from __future__ import annotations

import pathlib
from typing import Any, Optional

import torch
from loguru import logger


def load_checkpoint(path: str | pathlib.Path) -> Any:
    return torch.load(pathlib.Path(path), weights_only=False)


def extract_policy_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
        return checkpoint["policy_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")


def extract_optimizer_state_dict(checkpoint: Any) -> Optional[dict[str, Any]]:
    if isinstance(checkpoint, dict):
        return checkpoint.get("optimizer_state_dict", None)
    return None


def extract_env_state(checkpoint: Any) -> Optional[list[dict[str, Any]]]:
    if isinstance(checkpoint, dict):
        env_state = checkpoint.get("env_state", None)
        if env_state is None:
            return None
        if not isinstance(env_state, list):
            raise TypeError(f"env_state must be a list, got {type(env_state)}")
        return env_state
    return None


def copy_running_mean_std(src: Any, dst: Any) -> None:
    dst.mean = src.mean.copy()
    dst.var = src.var.copy()
    dst.count = src.count


def apply_env_state(env: Any, env_state: Optional[list[dict[str, Any]]]) -> None:
    if not env_state:
        return

    current_env = env
    search_start_idx = 0

    while hasattr(current_env, "env"):
        has_rms = any(
            hasattr(current_env, attr) for attr in ("local_obs_rms", "global_obs_rms", "return_rms")
        )
        if has_rms:
            current_class = type(current_env).__name__
            match_idx = None
            for i in range(search_start_idx, len(env_state)):
                if env_state[i].get("wrapper_class") == current_class:
                    match_idx = i
                    break

            if match_idx is None:
                logger.warning(
                    f"No saved env_state entry for wrapper {current_class}; normalization stats not restored for it."
                )
            else:
                saved_state = env_state[match_idx]
                if saved_state.get("wrapper_class") != current_class:
                    logger.warning(
                        f"Wrapper type mismatch during env_state apply: "
                        f"{saved_state.get('wrapper_class')} vs {current_class}"
                    )

                if "local_obs_rms" in saved_state and hasattr(current_env, "local_obs_rms"):
                    copy_running_mean_std(saved_state["local_obs_rms"], current_env.local_obs_rms)
                if "global_obs_rms" in saved_state and hasattr(current_env, "global_obs_rms"):
                    copy_running_mean_std(saved_state["global_obs_rms"], current_env.global_obs_rms)
                if "return_rms" in saved_state and hasattr(current_env, "return_rms"):
                    copy_running_mean_std(saved_state["return_rms"], current_env.return_rms)

                search_start_idx = match_idx + 1

        current_env = current_env.env


def freeze_env_normalization(env: Any) -> None:
    current_env = env
    while hasattr(current_env, "env"):
        if hasattr(current_env, "update_running_mean"):
            current_env.update_running_mean = False
        current_env = current_env.env


