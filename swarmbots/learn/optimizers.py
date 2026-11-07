from collections.abc import Iterable
from typing import Any

import torch


@torch.no_grad()
def clip_grad_norm_(
    parameters: Iterable[torch.Tensor], max_norm: float | None,
    *, error_if_nonfinite: bool = False,
) -> float | torch.Tensor:
    """Use float64 L2 accumulation and per-device foreach gradient clipping."""
    grouped_gradients: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None:
            grouped_gradients.setdefault((gradient.device, gradient.dtype), []).append(gradient)
    if not grouped_gradients:
        return 0.0

    norms: list[torch.Tensor] = []
    for (_, dtype), gradients in grouped_gradients.items():
        accumulation_dtype = torch.complex128 if dtype.is_complex else torch.float64
        norms.extend(torch._foreach_norm(gradients, 2, dtype=accumulation_dtype))
    norm_device = norms[0].device
    total_norm = torch.linalg.vector_norm(torch.stack([norm.to(norm_device) for norm in norms]), 2)
    if error_if_nonfinite and not torch.isfinite(total_norm):
        raise RuntimeError("The total gradient norm is non-finite, so it cannot be clipped.")
    if max_norm is not None:
        coefficient = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
        for (device, dtype), gradients in grouped_gradients.items():
            # A double scalar disables the fast foreach multiply path for float gradients.
            scale_dtype = torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32
            torch._foreach_mul_(gradients, coefficient.to(device=device, dtype=scale_dtype))
    return total_norm


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
