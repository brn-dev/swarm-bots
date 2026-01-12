import torch
from torch import nn


@torch.no_grad()
def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """
    Polyak / EMA update:
        target <- tau * target + (1 - tau) * source

    Typical values:
    - tau=0.0: hard copy (target = source)
    - tau close to 1.0: slow-moving target network
    """
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"{tau=} must be in [0, 1]")

    for src, tgt in zip(source.parameters(), target.parameters(), strict=True):
        tgt.mul_(tau).add_(src, alpha=1.0 - tau)

    for src_buf, tgt_buf in zip(source.buffers(), target.buffers(), strict=True):
        tgt_buf.copy_(src_buf)




