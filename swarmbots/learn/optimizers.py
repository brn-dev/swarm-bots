from collections.abc import Iterable
from typing import Any

import torch


def make_adam(
    parameters: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
    *,
    lr: float,
    device: torch.device,
) -> torch.optim.Adam:
    return torch.optim.Adam(parameters, lr=lr, fused=device.type == "cuda")


def move_adam_state_to_device(optimizer: torch.optim.Adam, device: torch.device) -> None:
    # Execution settings follow the training device, including after a checkpoint
    # restores parameter groups saved on a different device.
    fused = device.type == "cuda"
    optimizer.defaults.update(fused=fused, foreach=None)
    if not fused:
        optimizer.defaults["capturable"] = False
    for group in optimizer.param_groups:
        group.update(fused=fused, foreach=None)
        if not fused:
            group["capturable"] = False
        for parameter in group["params"]:
            for name, value in optimizer.state.get(parameter, {}).items():
                if torch.is_tensor(value):
                    state_device = device if name != "step" or fused or group["capturable"] else torch.device("cpu")
                    optimizer.state[parameter][name] = value.to(state_device)
