import torch

from swarmbots.utils.torch_utils import resolve_torch_device


def as_device(device: torch.device | str) -> torch.device:
    return resolve_torch_device(device)
