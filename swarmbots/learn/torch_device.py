from itertools import chain

import torch

from swarmbots.utils.torch_utils import resolve_torch_device


def as_device(device: torch.device | str) -> torch.device:
    return resolve_torch_device(device)


def module_device(module: torch.nn.Module) -> torch.device | None:
    tensor = next(chain(module.parameters(), module.buffers()), None)
    return None if tensor is None else tensor.device
