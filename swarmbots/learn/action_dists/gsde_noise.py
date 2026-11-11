"""Selective noise generation with an optional, host-read-free CUDA backend."""

from functools import lru_cache
import importlib.util
from typing import Any

import torch


@lru_cache(maxsize=1)
def _masked_normal_kernel() -> Any:
    if importlib.util.find_spec("triton") is None:
        return None
    from swarmbots.learn.action_dists._gsde_noise_triton import masked_normal_kernel

    return masked_normal_kernel


def reset_noise_masked_(noise: torch.Tensor, mask: torch.Tensor) -> None:
    """Refresh selected rows without allocating full-batch Torch temporaries."""
    if noise.numel() == 0:
        return
    mask = mask.to(noise.device)
    inference_write_allowed = not noise.is_inference() or torch.is_inference_mode_enabled()
    kernel = (
        _masked_normal_kernel()
        if noise.is_cuda and not torch.is_grad_enabled() and inference_write_allowed else None
    )
    if kernel is None:
        num_resets = int(mask.sum().item())
        if num_resets == 0:
            return
        if not inference_write_allowed:
            raise RuntimeError("Inplace update to inference tensor outside InferenceMode is not allowed.")
        with torch.no_grad():
            replacement_noise = torch.randn(
                (num_resets, *noise.shape[-2:]), device=noise.device, dtype=noise.dtype,
            )
            if noise.is_contiguous():
                noise = noise.view(-1, *noise.shape[-2:])
                mask = mask.reshape(-1)
            noise[mask] = replacement_noise
        return
    # Draw the seed on the current Torch stream so seeding and fork_rng also own these resets.
    seed = torch.randint(0, 2**63 - 1, (), device=noise.device, dtype=torch.int64)
    width = noise.shape[-2] * noise.shape[-1]
    with torch.cuda.device(noise.device):
        kernel[(mask.numel(),)](
            noise, mask.contiguous(), seed, width, noise.shape, noise.stride(), noise.is_contiguous(),
            1 << (width - 1).bit_length(),
        )
    # Direct kernel writes must retain Torch's saved-tensor mutation checks.
    torch.autograd.graph.increment_version(noise)
