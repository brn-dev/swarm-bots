import math

import pytest
import torch

from swarmbots import make_scenario
from swarmbots.mjw_env.scenarios.mjw_payload_goal_runtime import (
    _compute_payload_goal_approach_progress,
    _compute_payload_goal_approach_reward_kernel,
)


def _potential(positions: torch.Tensor, mask: torch.Tensor, softness: float = .5) -> torch.Tensor:
    return _compute_payload_goal_approach_progress(
        positions, positions.new_tensor([[0., 0., .3]]).expand(positions.shape[0], -1),
        positions.new_tensor([[0., 1.]]).expand(positions.shape[0], -1), mask, .3, 1.25,
        "pseudo_huber", softness,
    )


@pytest.mark.parametrize("softness", [.25, .5, .75])
def test_potential_matches_pseudo_huber_with_signal_inside_the_old_zone(softness: float) -> None:
    distances = torch.tensor([0., 1e-5, .1, .3, .5, 1., 1.25, 2.])
    positions = torch.stack((distances, torch.full_like(distances, -.3)), dim=-1).unsqueeze(1)
    progress = _potential(positions, torch.ones((len(distances), 1), dtype=torch.bool), softness)
    expected = -(torch.sqrt(distances.double().square() + softness**2) - softness)
    torch.testing.assert_close(progress.double(), expected, rtol=2e-6, atol=1e-12)
    assert progress[0] == 0.
    assert (progress[:-1] > progress[1:]).all()
    positions.requires_grad_()
    _potential(positions, torch.ones((len(distances), 1), dtype=torch.bool), softness).sum().backward()
    assert torch.isfinite(positions.grad).all()
    assert not positions.grad[0].any()


@pytest.mark.parametrize("agent", [0, 4])
def test_closest_and_farthest_agents_both_contribute_and_stationary_reward_is_zero(agent: int) -> None:
    positions = torch.tensor([[[.2, -.3], [.5, -.3], [.8, -.3], [1.1, -.3], [1.6, -.3], [99., 99.]]])
    active = torch.tensor([[True, True, True, True, True, False]])
    previous = _potential(positions, active)
    moved = positions.clone()
    moved[0, agent, 0] -= .1
    progress, reward = _compute_payload_goal_approach_reward_kernel(
        moved, torch.tensor([[0., 0., .3]]), torch.tensor([[0., 1.]]), torch.tensor([True]),
        active, previous, .3, 1.25, 1., 1., 1., "pseudo_huber", .5,
    )
    distance = float(positions[0, agent, 0])
    expected = (math.sqrt(distance**2 + .25) - math.sqrt((distance - .1)**2 + .25)) / 5
    torch.testing.assert_close(reward, torch.tensor([expected]))
    assert reward.item() > 0.
    _, stationary = _compute_payload_goal_approach_reward_kernel(
        moved, torch.tensor([[0., 0., .3]]), torch.tensor([[0., 1.]]), torch.tensor([True]),
        active, progress, .3, 1.25, 1., 1., 1., "pseudo_huber", .5,
    )
    _, retreat = _compute_payload_goal_approach_reward_kernel(
        positions, torch.tensor([[0., 0., .3]]), torch.tensor([[0., 1.]]), torch.tensor([True]),
        active, progress, .3, 1.25, 1., 1., 1., "pseudo_huber", .5,
    )
    torch.testing.assert_close(stationary, torch.zeros(1))
    torch.testing.assert_close(reward + retreat, torch.zeros(1))


def test_smooth_potential_is_rigid_transform_invariant_and_fullgraph_traceable() -> None:
    positions = torch.tensor([[[.3, -.3], [.7, .1], [99., 99.]]])
    active = torch.tensor([[True, True, False]])
    expected = _potential(positions, active)
    rotation = torch.tensor([[0., -1.], [1., 0.]])
    offset = torch.tensor([7., -4.])
    transformed = _compute_payload_goal_approach_progress(
        positions @ rotation.T + offset,
        torch.tensor([[7., -4., .3]]), torch.tensor([[6., -4.]]), active, .3, 1.25,
        "pseudo_huber", .5,
    )
    torch.testing.assert_close(transformed, expected)
    compiled = torch.compile(_potential, backend="eager", fullgraph=True)
    try:
        torch.testing.assert_close(compiled(positions, active), expected)
    finally:
        torch.compiler.reset()


@pytest.mark.parametrize("task", ["SwarmBots-PayloadGoal-v0", "SwarmBots-PayloadGoalSequence-v0"])
@pytest.mark.parametrize("override", [
    {"towards_payload_softness": 0.}, {"towards_payload_softness": -1.},
    {"towards_payload_softness": float("nan")}, {"towards_payload_softness": float("inf")},
    {"towards_payload_potential": "unknown"},
])
def test_invalid_smooth_potential_configuration_is_rejected(task: str, override: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="towards_payload"):
        make_scenario(task, **override)


def test_smooth_potential_settings_are_recorded() -> None:
    scenario = make_scenario(
        "SwarmBots-PayloadGoalSequence-v0", towards_payload_potential="pseudo_huber", towards_payload_softness=.5,
    )
    settings = scenario.get_settings()
    assert settings["towards_payload_potential"] == "pseudo_huber"
    assert settings["towards_payload_softness"] == .5


@pytest.mark.cuda
def test_smooth_reward_compiles_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    positions = torch.tensor([[[.3, -.3], [1., -.3], [99., 99.]]], device="cuda")
    active = torch.tensor([[True, True, False]], device="cuda")
    previous = _potential(positions, active)
    moved = positions.clone()
    moved[0, 1, 0] -= .1
    arguments = (
        moved, torch.tensor([[0., 0., .3]], device="cuda"), torch.tensor([[0., 1.]], device="cuda"),
        torch.tensor([True], device="cuda"), active, previous, .3, 1.25, 1., 1., 1., "pseudo_huber", .5,
    )
    expected_progress, expected_reward = _compute_payload_goal_approach_reward_kernel(*arguments)
    compiled = torch.compile(_compute_payload_goal_approach_reward_kernel, fullgraph=True)
    try:
        progress, reward = compiled(*arguments)
        torch.testing.assert_close(progress, expected_progress)
        torch.testing.assert_close(reward, expected_reward)
        assert reward.item() > 0.
    finally:
        torch.compiler.reset()
