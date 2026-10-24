from collections.abc import Callable
import shutil
import sys
from typing import ParamSpec, TypeVar

import torch


RewardInputs = ParamSpec("RewardInputs")
RewardOutput = TypeVar("RewardOutput")


def compile_reward_kernel(
    kernel: Callable[RewardInputs, RewardOutput], *, enabled: bool, mode: str,
) -> Callable[RewardInputs, RewardOutput]:
    if not enabled:
        return kernel
    if not hasattr(torch, "compile"):
        raise RuntimeError("compile_reward_kernel=True requires torch.compile support.")
    if sys.platform == "win32" and shutil.which("cl") is None:
        raise RuntimeError(
            "compile_reward_kernel=True on this Windows setup requires cl.exe on PATH for torch.compile."
        )
    return torch.compile(kernel, mode=mode, fullgraph=False, dynamic=False)
