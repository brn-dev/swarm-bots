import torch

from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution


def distribution_entropy_metrics(
    distribution: HybridActionDistribution,
    actions: torch.Tensor,
    agent_mask: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in distribution.get_metrics(actions, agent_mask=agent_mask).items()
        if "_ent" in name and isinstance(value, torch.Tensor)
    }
