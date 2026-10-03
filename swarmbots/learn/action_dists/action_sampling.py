import torch


def validate_action_sampling(
        *,
        actor_action_samples: int,
        target_action_samples: int,
) -> None:
    for name, value in (
        ("actor_action_samples", actor_action_samples),
        ("target_action_samples", target_action_samples),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1, got {value!r}")


def expand_action_samples(tensor: torch.Tensor | None, count: int) -> torch.Tensor | None:
    return None if tensor is None else tensor.unsqueeze(0).expand(count, *tensor.shape)


def expand_action_sample_batch(tensor: torch.Tensor | None, count: int) -> torch.Tensor | None:
    expanded = expand_action_samples(tensor, count)
    return None if expanded is None else expanded.flatten(0, 1)
