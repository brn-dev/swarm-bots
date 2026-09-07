from typing import Literal

import torch

ActionSampleStrategy = Literal["iid", "stratified"]


def expand_action_samples(tensor: torch.Tensor | None, count: int) -> torch.Tensor | None:
    return None if tensor is None else tensor.unsqueeze(0).expand(count, *tensor.shape)


def expand_action_sample_batch(tensor: torch.Tensor | None, count: int) -> torch.Tensor | None:
    expanded = expand_action_samples(tensor, count)
    return None if expanded is None else expanded.flatten(0, 1)


def sample_base_uniform(
        shape: tuple[int, ...] | torch.Size,
        *,
        device: torch.device,
        dtype: torch.dtype,
        stratified_sample_dim: int | None = None,
) -> torch.Tensor:
    noise = torch.rand(shape, device=device, dtype=dtype)
    if stratified_sample_dim is None or shape[stratified_sample_dim] == 1:
        return noise
    count = shape[stratified_sample_dim]
    strata = torch.rand(shape, device=device, dtype=torch.float64).argsort(dim=stratified_sample_dim)
    strata_float = strata.to(dtype=torch.float64)
    sample = ((strata_float + noise.to(dtype=torch.float64)) / count).to(dtype=dtype)
    # Keep the cast result strictly inside its assigned stratum. Rounding across
    # either boundary would duplicate a neighbouring stratum.
    lower = (strata_float / count).to(dtype=dtype)
    lower = torch.nextafter(lower, torch.ones_like(lower))
    upper = ((strata_float + 1) / count).to(dtype=dtype)
    upper = torch.nextafter(upper, torch.zeros_like(upper))
    return torch.maximum(torch.minimum(sample, upper), lower)
