

import pathlib
from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any

import torch

from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.torch_device import as_device


def load_checkpoint(path: str | pathlib.Path, map_location: Any | None = "cpu") -> dict[str, Any]:
    return torch.load(pathlib.Path(path), map_location=map_location, weights_only=False)


def align_torch_compile_state_dict_keys(
    state_dict: Mapping[str, Any],
    *,
    target_keys: Iterable[str],
) -> dict[str, Any]:
    """Align state-dict keys across compiled and uncompiled module wrappers."""

    def canonical_key(key: str) -> str:
        return ".".join(part for part in key.split(".") if part != "_orig_mod")

    target_by_canonical_key: dict[str, str] = {}
    for target_key in target_keys:
        canonical = canonical_key(target_key)
        previous = target_by_canonical_key.setdefault(canonical, target_key)
        if previous != target_key:
            raise ValueError(
                "Target policy has ambiguous torch.compile state-dict keys: "
                f"{previous!r} and {target_key!r}"
            )

    aligned: dict[str, Any] = {}
    source_by_target_key: dict[str, str] = {}
    for source_key, value in state_dict.items():
        target_key = target_by_canonical_key.get(canonical_key(source_key), source_key)
        previous_source = source_by_target_key.setdefault(target_key, source_key)
        if previous_source != source_key:
            raise ValueError(
                "Checkpoint has ambiguous torch.compile state-dict keys: "
                f"{previous_source!r} and {source_key!r} both map to {target_key!r}"
            )
        aligned[target_key] = value
    return aligned


def copy_running_mean_std(src: Any, dst: Any) -> None:
    if tuple(dst.mean.shape) != tuple(src.mean.shape):
        raise ValueError(f'{dst.mean.shape = } is not equal {src.mean.shape = }')
    if tuple(dst.var.shape) != tuple(src.var.shape):
        raise ValueError(f'{dst.var.shape = } is not equal {src.var.shape = }')
    dst.mean = src.mean.to(device=dst.mean.device, dtype=dst.mean.dtype).clone()
    dst.var = src.var.to(device=dst.var.device, dtype=dst.var.dtype).clone()
    dst.count = src.count.to(device=dst.count.device, dtype=dst.count.dtype).clone()


def clone_running_mean_std(src: Any) -> Any:
    return SimpleNamespace(
        mean=src.mean.detach().clone(),
        var=src.var.detach().clone(),
        count=src.count.detach().clone(),
    )


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

        if isinstance(current_env, TorchFeatureWiseObsNormWrapper):
            wrapper_state['obs_key'] = current_env.obs_key

        for attr_name, value in iter_running_mean_std(current_env):
            wrapper_state[attr_name] = clone_running_mean_std(value)

        if wrapper_state:
            wrapper_state["wrapper_class"] = type(current_env).__name__
            env_state.append(wrapper_state)

        current_env = current_env.env
    return env_state


def apply_env_state(env: Any, env_state: list[dict[str, Any]]) -> None:
    current_env = env
    search_start_idx = 0

    while hasattr(current_env, "env"):
        rms_entries = dict(iter_running_mean_std(current_env))
        has_rms = bool(rms_entries)
        if has_rms:
            current_class = type(current_env).__name__
            match_idx = None
            candidates = [
                i
                for i in range(search_start_idx, len(env_state))
                if env_state[i]["wrapper_class"] == current_class
                and any(key.endswith("_rms") for key in env_state[i])
            ]
            if candidates:
                if isinstance(current_env, TorchFeatureWiseObsNormWrapper):
                    for i in candidates:
                        if env_state[i]["obs_key"] == current_env.obs_key:
                            match_idx = i
                            break
                    if match_idx is None:
                        raise ValueError(
                            f"Obs keys not equal: {current_env.obs_key} vs "
                            f"{[env_state[i]['obs_key'] for i in candidates]}"
                        )
                else:
                    match_idx = candidates[0]

            if match_idx is None:
                raise ValueError(
                    f"No saved env_state entry for wrapper {current_class}; normalization stats not restored for it."
                )
            else:
                saved_state = env_state[match_idx]
                for attr_name, value in rms_entries.items():
                    copy_running_mean_std(saved_state[attr_name], value)

                search_start_idx = match_idx + 1

        current_env = current_env.env


def freeze_env_normalization(env: Any) -> None:
    current_env = env
    while hasattr(current_env, "env"):
        if hasattr(current_env, "update_running_mean"):
            current_env.update_running_mean = False
        current_env = current_env.env


def move_env_to_device(env: Any, device: torch.device | str) -> None:
    target_device = as_device(device)
    current_env = env
    while True:
        if hasattr(current_env, "set_device"):
            current_env.set_device(target_device)
        elif hasattr(current_env, "device"):
            current_env.device = target_device

        if not hasattr(current_env, "env"):
            break
        current_env = current_env.env
