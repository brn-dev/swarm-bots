

import pathlib
from typing import Any, Optional

import torch
from loguru import logger


def load_checkpoint(path: str | pathlib.Path, map_location: Any | None = "cpu") -> Any:
    return torch.load(pathlib.Path(path), map_location=map_location, weights_only=False)


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
    if dst.mean.shape != src.mean.shape:
        raise ValueError(f'{dst.mean.shape = } is not equal {src.mean.shape = }')
    if dst.var.shape != src.var.shape:
        raise ValueError(f'{dst.mean.var = } is not equal {src.mean.var = }')
    dst.mean = src.mean.copy()
    dst.var = src.var.copy()
    dst.count = src.count


def iter_running_mean_std(env: Any) -> list[tuple[str, Any]]:
    rms_entries: list[tuple[str, Any]] = []
    for attr_name in dir(env):
        if not attr_name.endswith("_rms"):
            continue

        value = getattr(env, attr_name, None)
        if value is None:
            continue
        if all(hasattr(value, field) for field in ("mean", "var", "count")):
            rms_entries.append((attr_name, value))
    return rms_entries

def capture_env_state(env: Any) -> list[dict[str, Any]]:
    env_state: list[dict[str, Any]] = []
    current_env = env
    while hasattr(current_env, "env"):
        wrapper_state: dict[str, Any] = {}
        for attr_name, value in iter_running_mean_std(current_env):
            wrapper_state[attr_name] = value

        if wrapper_state:
            wrapper_state["wrapper_class"] = type(current_env).__name__
            env_state.append(wrapper_state)

        current_env = current_env.env
    return env_state


def apply_env_state(env: Any, env_state: Optional[list[dict[str, Any]]]) -> None:
    if not env_state:
        return

    current_env = env
    search_start_idx = 0

    while hasattr(current_env, "env"):
        rms_entries = dict(iter_running_mean_std(current_env))
        has_rms = bool(rms_entries)
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

                for attr_name, saved_value in saved_state.items():
                    if attr_name.endswith("_rms") and attr_name in rms_entries:
                        copy_running_mean_std(saved_value, rms_entries[attr_name])

                search_start_idx = match_idx + 1

        current_env = current_env.env


def freeze_env_normalization(env: Any) -> None:
    current_env = env
    while hasattr(current_env, "env"):
        if hasattr(current_env, "update_running_mean"):
            current_env.update_running_mean = False
        current_env = current_env.env


