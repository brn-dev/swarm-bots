"""Training input preparation shared by recurrent policies' feed-forward critics."""

from typing import cast

import torch


def sanitize_critic_sequence_inputs(
    inputs: dict[str, torch.Tensor | None],
    *,
    time_mask: torch.Tensor | None,
) -> dict[str, torch.Tensor | None]:
    if not torch.is_grad_enabled():
        return inputs

    agent_valid = inputs.get("agent_mask")
    time_valid = None
    if time_mask is not None:
        local_obs = cast(torch.Tensor, inputs["local_obs"])
        time_valid = time_mask.reshape(local_obs.shape[:-2])
        agent_valid = time_valid.unsqueeze(-1) if agent_valid is None else agent_valid & time_valid.unsqueeze(-1)
    if agent_valid is None:
        return inputs

    sanitized = dict(inputs)
    for name in ("local_obs", "hidden_local_vars", "actions", "actor_state"):
        value = inputs.get(name)
        if value is not None:
            sanitized[name] = value.masked_fill(~agent_valid.unsqueeze(-1), 0.0)
    if time_valid is not None:
        for name in ("global_obs", "hidden_global_vars"):
            value = inputs.get(name)
            if value is not None:
                sanitized[name] = value.masked_fill(~time_valid.unsqueeze(-1), 0.0)
    return sanitized
