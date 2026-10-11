import torch

from swarmbots.learn.action_dists.entropy_utils import compute_entropy_metrics


def test_entropy_metrics_average_actions_and_exclude_inactive_agents() -> None:
    entropy = torch.tensor([[[1., 3.], [100., 200.]], [[5., 7.], [200., 300.]]], requires_grad=True)
    mask = torch.tensor([[True, False], [True, False]])
    metrics = compute_entropy_metrics(entropy, agent_mask=mask)
    torch.testing.assert_close(metrics["ent"], torch.tensor(4.))
    assert not metrics["ent"].requires_grad


def test_entropy_splitters_support_action_slices_and_single_actions() -> None:
    entropy = torch.tensor([[[1., 3.], [5., 7.]]])

    def split_actions(values: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"first": values[..., 0], "second": values[..., 1:]}

    metrics = compute_entropy_metrics(entropy, action_splitter=split_actions, name="ent_normal")
    torch.testing.assert_close(metrics["ent_normal"], torch.tensor(4.))
    torch.testing.assert_close(metrics["ent_normal_first"], torch.tensor(3.))
    torch.testing.assert_close(metrics["ent_normal_second"], torch.tensor(5.))


def test_entropy_metrics_with_no_active_agents_are_zero() -> None:
    metrics = compute_entropy_metrics(torch.ones(2, 3, 4), agent_mask=torch.zeros(2, 3, dtype=torch.bool))
    torch.testing.assert_close(metrics["ent"], torch.tensor(0.))
