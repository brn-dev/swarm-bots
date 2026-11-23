from collections.abc import Iterable
import math
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


def validate_adam_state(
    saved: dict[str, Any], param_groups: list[dict[str, Any]], *, lazy_shapes: dict[int, torch.Size] | None = None,
) -> None:
    """Validate Adam parameter groups and moments before installing the saved state."""
    groups = saved["param_groups"]
    if len(groups) != len(param_groups):
        raise ValueError("Checkpoint Adam parameter group count does not match the trainer")
    state = saved["state"]
    seen = set()
    for source, target in zip(groups, param_groups, strict=True):
        if len(source["params"]) != len(target["params"]):
            raise ValueError("Checkpoint Adam parameter group size does not match the trainer")
        required_keys = target.keys() - {"param_names", "lr_multiplier", "name"}
        if not required_keys.issubset(source):
            raise ValueError("Checkpoint Adam parameter group is incomplete")
        if any(not math.isfinite(source[key]) or source[key] < 0 for key in ("lr", "eps", "weight_decay")):
            raise ValueError("Invalid checkpoint Adam hyperparameters")
        beta1, beta2 = source["betas"]
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError("Invalid checkpoint Adam betas")
        for index, param in zip(source["params"], target["params"], strict=True):
            if index in seen:
                raise ValueError("Duplicate checkpoint Adam parameter")
            seen.add(index)
            slots = state.get(index, {})
            if not isinstance(slots, dict):
                raise ValueError("Checkpoint Adam parameter state must be a dictionary")
            if not slots:
                continue
            names = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") if source["amsgrad"] else ("exp_avg", "exp_avg_sq")
            if slots.keys() != {"step", *names}:
                raise ValueError("Invalid checkpoint Adam parameter state fields")
            if torch.nn.parameter.is_lazy(param):
                if lazy_shapes is None or id(param) not in lazy_shapes:
                    raise ValueError("Checkpoint Adam moments require a materialized parameter")
                expected_shape = lazy_shapes[id(param)]
            else:
                expected_shape = param.shape
            for name in names:
                moment = slots[name]
                if (not isinstance(moment, torch.Tensor) or torch.nn.parameter.is_lazy(moment) or moment.shape != expected_shape
                        or moment.layout != param.layout or moment.is_meta or moment.is_quantized
                        or moment.is_complex() != param.is_complex()):
                    raise ValueError(f"Invalid checkpoint Adam moment {name} for parameter of shape {expected_shape}")
                values = moment.to(dtype=param.dtype)
                if values.is_complex():
                    values = torch.view_as_real(values)
                valid = torch.isfinite(values)
                if name != "exp_avg":
                    valid &= values >= 0
                if not valid.all():
                    raise ValueError(f"Invalid checkpoint Adam moment values in {name}")
            step = slots.get("step")
            if (not isinstance(step, torch.Tensor) or step.ndim != 0 or not step.is_floating_point()
                    or step.layout != torch.strided or not math.isfinite(step) or step < 0):
                raise ValueError("Invalid checkpoint Adam step")
    if state.keys() - seen:
        raise ValueError("Checkpoint Adam state contains unknown parameters")


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
