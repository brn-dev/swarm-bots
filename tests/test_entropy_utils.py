import pytest
import torch

from swarmbots.learn.action_dists.entropy_utils import EntropyLossConfig, compute_ent_loss


def test_entropy_floor_replaces_deprecated_max_entropy_alias() -> None:
    entropy_per_action = torch.tensor([[0.1, 0.4]])

    floor_loss = compute_ent_loss(
        EntropyLossConfig(entropy_floor=0.35),
        entropy_per_action,
    )
    alias_loss = compute_ent_loss(
        EntropyLossConfig(max_entropy=0.35),
        entropy_per_action,
    )

    assert torch.allclose(floor_loss, torch.tensor([0.25]))
    assert torch.allclose(alias_loss, floor_loss)


def test_entropy_floor_rejects_deprecated_alias_when_both_are_set() -> None:
    with pytest.raises(ValueError, match="either entropy_floor or deprecated max_entropy"):
        compute_ent_loss(
            EntropyLossConfig(entropy_floor=0.35, max_entropy=0.5),
            torch.tensor([[0.1]]),
        )
