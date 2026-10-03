from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch


def _encode_setting(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {"type": type(value).__name__, **{field.name: getattr(value, field.name) for field in fields(value)}}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if hasattr(value, "get_settings"):
        return {"type": type(value).__name__, "settings": value.get_settings()}
    raise TypeError(f"Cannot serialize benchmark setting of type {type(value).__name__}")


def serialize_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(settings), default=_encode_setting, allow_nan=False))


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    packages = {}
    for name in ("swarmbots", "torch", "gymnasium", "numpy", "mujoco", "mujoco-warp", "warp-lang"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "packages": packages,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
