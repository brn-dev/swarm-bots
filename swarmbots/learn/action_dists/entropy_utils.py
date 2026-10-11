import torch

from swarmbots.learn.action_dists.action_dist import ActionMetricsSplitter
from swarmbots.learn.masking import masked_mean


def compute_entropy_metrics(
        entropy_per_action: torch.Tensor,
        *,
        agent_mask: torch.Tensor | None = None,
        action_splitter: ActionMetricsSplitter | None = None,
        name: str = "ent",
) -> dict[str, torch.Tensor]:
    """Report detached mean entropy per action, excluding inactive agents."""
    with torch.no_grad():
        metrics = {name: masked_mean(entropy_per_action.mean(dim=-1), agent_mask)}
        if action_splitter is not None:
            for key, split_entropy in action_splitter(entropy_per_action).items():
                if split_entropy.ndim == entropy_per_action.ndim:
                    split_entropy = split_entropy.mean(dim=-1)
                metrics[f"{name}_{key}"] = masked_mean(split_entropy, agent_mask)
        return metrics
