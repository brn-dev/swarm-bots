from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils import dlpack


def resolve_torch_device(device: torch.device | str) -> torch.device:
    if device == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)

    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    if resolved.type == "cpu":
        return torch.device("cpu")
    return resolved


def to_numpy_array(value: Any, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False) if dtype is not None else value
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return array.astype(dtype, copy=False) if dtype is not None else array
    if hasattr(value, "__dlpack__"):
        try:
            array = dlpack.from_dlpack(value).detach().cpu().numpy()
            return array.astype(dtype, copy=False) if dtype is not None else array
        except Exception:
            pass
    return np.asarray(value, dtype=dtype)
