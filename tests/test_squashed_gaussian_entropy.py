from collections.abc import Callable
import math

import pytest
import torch
from torch.distributions import Normal

from swarmbots.learn.action_dists.continuous_action_dist import ContinuousActionDist
from swarmbots.learn.action_dists.diag_gaussian_action_dist import DiagGaussianActionDist
from swarmbots.learn.action_dists.gsde_action_dist import GSDEActionDist
from swarmbots.learn.action_dists.predicted_std_gaussian_action_dist import PredictedStdGaussianActionDist
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import SquashedDiagGaussianActionDist


def _predicted_std() -> ContinuousActionDist:
    return PredictedStdGaussianActionDist(latent_dim=2, action_dim=1, base_std=1., squash_output=True)


def _gsde() -> ContinuousActionDist:
    return GSDEActionDist(latent_dim=2, action_dim=1, base_std=1., squash_output=True)


def _squashed_diag() -> ContinuousActionDist:
    return SquashedDiagGaussianActionDist(latent_dim=2, action_dim=1, std=1., std_learnable=True)


@pytest.mark.parametrize("factory", [_predicted_std, _gsde, _squashed_diag])
def test_squashed_gaussian_entropy_diagnostic_is_detached_and_does_not_consume_rng(
    factory: Callable[[], ContinuousActionDist],
) -> None:
    action_dist = factory()
    means = torch.tensor([[[0.]], [[5.]], [[100.]]], requires_grad=True)
    stds = torch.ones_like(means)
    action_dist.distribution = Normal(means, stds, validate_args=False)
    rng_state = torch.random.get_rng_state().clone()
    metrics = action_dist.get_metrics(torch.tanh(means))
    torch.testing.assert_close(metrics["ent_normal"], torch.tensor(.5 * math.log(2. * math.pi * math.e)))
    torch.testing.assert_close(torch.random.get_rng_state(), rng_state)
    assert "ent" not in metrics
    assert not metrics["ent_normal"].requires_grad


@pytest.mark.parametrize("factory", [_predicted_std, _gsde, _squashed_diag])
def test_squashed_gaussian_state_dict_preserves_parameter_layout(factory: Callable[[], ContinuousActionDist]) -> None:
    action_dist = factory()
    expected_keys = {"action_net.weight", "action_net.bias"}
    if isinstance(action_dist, PredictedStdGaussianActionDist):
        expected_keys.update({"base_log_std", "log_std_net.weight", "log_std_net.bias"})
    else:
        expected_keys.add("log_stds")
    assert set(action_dist.state_dict()) == expected_keys
    assert all("entropy" not in name for name, _ in action_dist.named_buffers())


def test_unsquashed_normal_reports_analytic_entropy() -> None:
    action_dist = PredictedStdGaussianActionDist(latent_dim=2, action_dim=1, base_std=1.)
    means = torch.tensor([[[0.]], [[5.]]])
    action_dist.update_distribution_params(means, torch.zeros_like(means))
    metrics = action_dist.get_metrics(means)
    torch.testing.assert_close(metrics["ent"], torch.tensor(.5 * math.log(2. * math.pi * math.e)))
    assert "ent_normal" not in metrics


@pytest.mark.parametrize("batch_shape", [(2,), (2, 3), (4, 2, 3)])
@pytest.mark.parametrize("split", [False, True])
def test_conditional_std_diagnostics_exclude_inactive_agents_and_padding(
    batch_shape: tuple[int, ...], split: bool,
) -> None:
    action_dist = PredictedStdGaussianActionDist(latent_dim=2, action_dim=2, base_std=1.)
    mask = torch.ones(*batch_shape, 3, dtype=torch.bool)
    mask[..., 1] = False
    if len(batch_shape) > 1:
        mask[..., -1, :] = False
    stds = torch.full((*mask.shape, 2), .5)
    stds[~mask] = math.exp(2.)
    means = torch.zeros_like(stds)
    action_dist.update_distribution_params(means, stds.log())

    def split_actions(values: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"first": values[..., :1], "second": values[..., 1:]}

    metrics = action_dist.get_metrics(means[mask], split_actions if split else None, agent_mask=mask)
    for name in (("std_first", "std_second") if split else ("std",)):
        summary = metrics[name]
        assert summary.mean == pytest.approx(.5)
        assert summary.min_value == pytest.approx(.5)
        assert summary.max_value == pytest.approx(.5)


@pytest.mark.parametrize("gsde", [False, True])
def test_global_std_parameter_diagnostics_do_not_use_agent_masks(gsde: bool) -> None:
    action_dist = (
        GSDEActionDist(latent_dim=3, action_dim=2, base_std=.5)
        if gsde else DiagGaussianActionDist(latent_dim=3, action_dim=2, std=.5, std_learnable=True)
    )
    mask = torch.tensor([[True, False, True], [False, True, True]])
    means = torch.zeros(*mask.shape, 2)
    action_dist.update_latent_features(torch.ones(*mask.shape, 3))
    metrics = action_dist.get_metrics(means[mask], agent_mask=mask)
    assert metrics["std"].n == action_dist.log_stds.numel()
    assert metrics["std"].mean == pytest.approx(.5)
