from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from swarmbots.mjw_env.scenarios.mjw_obstacle_street_runtime import (
    ObstacleStreetMJWScenarioRuntime,
    _build_wall_pass_rank_weights,
    _compute_obstacle_street_reward_kernel,
)


def _make_runtime(
    *,
    num_units: int = 1,
    wall_pass_reward_skew: float = 0.0,
    wall_pass_reward_weight: float = 4.0,
    wall_pass_absolute_thresholds: list[float] | None = None,
    units_active_mask: list[bool] | None = None,
) -> ObstacleStreetMJWScenarioRuntime:
    runtime = object.__new__(ObstacleStreetMJWScenarioRuntime)
    device = torch.device("cpu")
    thresholds = [0.5] if wall_pass_absolute_thresholds is None else wall_pass_absolute_thresholds
    runtime.bindings = SimpleNamespace(
        num_envs=1,
        device=device,
        units_active_mask=torch.ones((1, num_units), device=device, dtype=torch.bool)
        if units_active_mask is None
        else torch.tensor([units_active_mask], device=device, dtype=torch.bool),
        partner_unit=torch.full((1, num_units), -1, device=device, dtype=torch.long),
    )
    runtime.scenario = SimpleNamespace(
        progress_reward_weight=0.5,
        forward_reward_weight=0.0,
        forward_reward_max_y=None,
        forward_reward_wall_boost_factor=1.0,
        forward_reward_wall_boost_distance=None,
        forward_reward_wall_boost_height_margin=None,
        wall_pass_reward_weight=wall_pass_reward_weight,
        wall_pass_reward_skew=wall_pass_reward_skew,
        wall_climb_reward_weight=0.0,
        wall_climb_reward_distance=0.5,
        potential_reward_discount_factor=1.0,
        units_without_connections_reward_weight=0.0,
        guidance_reward_weight=1.0,
    )
    runtime.progress = torch.zeros((1,), device=device, dtype=torch.float32)
    runtime.forward_progress_unit_y = torch.zeros((1, num_units), device=device, dtype=torch.float32)
    runtime.wall_pass_absolute_thresholds = torch.tensor([thresholds], device=device, dtype=torch.float32)
    runtime.next_threshold_for_unit = torch.zeros((1, num_units), device=device, dtype=torch.long)
    runtime.wall_climb_potential = torch.zeros((1, num_units, 0), device=device, dtype=torch.float32)
    runtime.wall_climb_done_mask = torch.zeros((1, num_units, 0), device=device, dtype=torch.bool)
    runtime.passed_thresholds_mask = torch.zeros((1, num_units, len(thresholds)), device=device, dtype=torch.bool)
    runtime._hidden_local_obs = torch.zeros((1, num_units, len(thresholds)), device=device, dtype=torch.float32)
    runtime._threshold_index_torch = torch.arange(len(thresholds), device=device, dtype=torch.long)
    runtime._unit_rank_torch = torch.arange(1, num_units + 1, device=device, dtype=torch.long)
    runtime._wall_pass_rank_weights = torch.zeros((num_units + 1, num_units), device=device, dtype=torch.float32)
    runtime._wall_pass_rank_weights_skew = None
    runtime._wall_thresholds_per_wall = len(thresholds)
    runtime._reward_kernel = _compute_obstacle_street_reward_kernel
    return runtime


def test_wall_pass_reward_is_only_issued_once_per_threshold() -> None:
    runtime = _make_runtime()
    stable_mask = torch.tensor([True], dtype=torch.bool)

    runtime._get_unit_y = lambda: torch.tensor([[0.6]], dtype=torch.float32)
    first = runtime.compute_step_rewards(stable_mask=stable_mask)
    assert float(first.info["wall_pass_reward"][0]) == 2.0
    assert float(first.info["progress_reward"][0]) == 2.0
    assert float(first.info["forward_reward"][0]) == 0.0
    assert float(first.info["forward_progress_reward"][0]) == 0.0
    assert int(runtime.next_threshold_for_unit[0, 0]) == 1

    runtime._get_unit_y = lambda: torch.tensor([[0.4]], dtype=torch.float32)
    backtrack = runtime.compute_step_rewards(stable_mask=stable_mask)
    assert float(backtrack.info["wall_pass_reward"][0]) == 0.0
    assert int(runtime.next_threshold_for_unit[0, 0]) == 1

    runtime._get_unit_y = lambda: torch.tensor([[0.6]], dtype=torch.float32)
    recross = runtime.compute_step_rewards(stable_mask=stable_mask)
    assert float(recross.info["wall_pass_reward"][0]) == 0.0
    assert int(runtime.next_threshold_for_unit[0, 0]) == 1


def test_forward_reward_cap_stops_progress_reward_beyond_max_y() -> None:
    runtime = _make_runtime()
    runtime.scenario.progress_reward_weight = 1.0
    runtime.scenario.forward_reward_weight = 1.0
    runtime.scenario.forward_reward_max_y = 0.5
    runtime.scenario.wall_pass_reward_weight = 0.0
    runtime.progress[0] = 0.45
    stable_mask = torch.tensor([True], dtype=torch.bool)

    runtime._get_unit_y = lambda: torch.tensor([[0.7]], dtype=torch.float32)
    first = runtime.compute_step_rewards(stable_mask=stable_mask)
    assert float(first.info["forward_reward"][0]) == pytest.approx(0.05)

    runtime._get_unit_y = lambda: torch.tensor([[0.9]], dtype=torch.float32)
    second = runtime.compute_step_rewards(stable_mask=stable_mask)
    assert float(second.info["forward_reward"][0]) == pytest.approx(0.0)


def test_wall_height_forward_reward_boost_applies_per_high_unit_in_mjw() -> None:
    runtime = _make_runtime(num_units=2, wall_pass_reward_weight=0.0)
    runtime.scenario.progress_reward_weight = 1.0
    runtime.scenario.forward_reward_weight = 1.0
    runtime.scenario.forward_reward_wall_boost_factor = 3.0
    runtime.scenario.forward_reward_wall_boost_distance = 0.5
    runtime.scenario.forward_reward_wall_boost_height_margin = 0.1
    runtime.scenario.swarm = SimpleNamespace(body_radius=0.1)
    runtime.progress[0] = 1.2
    runtime.forward_progress_unit_y[0] = torch.tensor([1.8, 0.6], dtype=torch.float32)
    runtime.wall_y_by_wall = torch.tensor([[1.0]], dtype=torch.float32)
    runtime._wall_heights = torch.tensor([0.4], dtype=torch.float32)
    stable_mask = torch.tensor([True], dtype=torch.bool)

    runtime._get_unit_y = lambda: torch.tensor([[0.7, 0.7]], dtype=torch.float32)
    runtime._get_unit_z = lambda: torch.tensor([[0.5, 0.49]], dtype=torch.float32)
    result = runtime.compute_step_rewards(stable_mask=stable_mask)

    assert float(result.info["forward_reward"][0]) == pytest.approx(0.2)


def test_wall_height_forward_reward_boost_penalizes_backtracking_symmetrically_in_mjw() -> None:
    runtime = _make_runtime(num_units=1, wall_pass_reward_weight=0.0)
    runtime.scenario.progress_reward_weight = 1.0
    runtime.scenario.forward_reward_weight = 1.0
    runtime.scenario.forward_reward_wall_boost_factor = 3.0
    runtime.scenario.forward_reward_wall_boost_distance = 0.5
    runtime.scenario.forward_reward_wall_boost_height_margin = 0.1
    runtime.scenario.swarm = SimpleNamespace(body_radius=0.1)
    runtime.progress[0] = 2.1
    runtime.forward_progress_unit_y[0] = torch.tensor([2.1], dtype=torch.float32)
    runtime.wall_y_by_wall = torch.tensor([[1.0]], dtype=torch.float32)
    runtime._wall_heights = torch.tensor([0.4], dtype=torch.float32)
    stable_mask = torch.tensor([True], dtype=torch.bool)

    runtime._get_unit_y = lambda: torch.tensor([[0.6]], dtype=torch.float32)
    runtime._get_unit_z = lambda: torch.tensor([[0.5]], dtype=torch.float32)
    backtrack = runtime.compute_step_rewards(stable_mask=stable_mask)

    runtime._get_unit_y = lambda: torch.tensor([[0.7]], dtype=torch.float32)
    retry = runtime.compute_step_rewards(stable_mask=stable_mask)

    assert float(backtrack.info["forward_reward"][0]) == pytest.approx(-0.3)
    assert float(retry.info["forward_reward"][0]) == pytest.approx(0.3)
    assert float(backtrack.info["forward_reward"][0] + retry.info["forward_reward"][0]) == pytest.approx(0.0)


def test_forward_reward_applies_discount_factor_to_current_potential_in_mjw() -> None:
    runtime = _make_runtime(num_units=1, wall_pass_reward_weight=0.0)
    runtime.scenario.progress_reward_weight = 1.0
    runtime.scenario.forward_reward_weight = 1.0
    runtime.scenario.potential_reward_discount_factor = 0.9
    runtime.progress[0] = 1.0
    runtime.forward_progress_unit_y[0] = torch.tensor([1.0], dtype=torch.float32)
    stable_mask = torch.tensor([True], dtype=torch.bool)

    runtime._get_unit_y = lambda: torch.tensor([[2.0]], dtype=torch.float32)
    result = runtime.compute_step_rewards(stable_mask=stable_mask)

    assert float(result.info["forward_reward"][0]) == pytest.approx(0.8)
    assert float(result.info["progress_reward"][0]) == pytest.approx(0.8)
    assert float(runtime.progress[0]) == pytest.approx(2.0)


def test_units_without_connections_reward_is_named_explicitly_in_mjw_info() -> None:
    runtime = _make_runtime(num_units=2)
    runtime.scenario.units_without_connections_reward_weight = -0.25
    stable_mask = torch.tensor([True], dtype=torch.bool)

    runtime._get_unit_y = lambda: torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    result = runtime.compute_step_rewards(stable_mask=stable_mask)

    assert float(result.info["units_without_connections_reward"][0]) == pytest.approx(-0.25)
    assert float(result.info["guidance_reward"][0]) == pytest.approx(-0.25)
    assert float(result.info["reward_terms"]["units_without_connections"][0]) == pytest.approx(-0.25)
    assert "guidance" not in result.info["reward_terms"]


def test_default_wall_pass_reward_skew_keeps_equal_rank_rewards() -> None:
    runtime = _make_runtime(num_units=3)
    runtime.scenario.progress_reward_weight = 1.0
    runtime.scenario.wall_pass_reward_weight = 3.0
    stable_mask = torch.tensor([True], dtype=torch.bool)

    runtime._get_unit_y = lambda: torch.tensor([[0.6, 0.0, 0.0]], dtype=torch.float32)
    first = runtime.compute_step_rewards(stable_mask=stable_mask)
    assert float(first.info["wall_pass_reward"][0]) == pytest.approx(1.0)

    runtime._get_unit_y = lambda: torch.tensor([[0.6, 0.6, 0.6]], dtype=torch.float32)
    remaining = runtime.compute_step_rewards(stable_mask=stable_mask)
    assert float(remaining.info["wall_pass_reward"][0]) == pytest.approx(2.0)


def test_wall_pass_reward_skew_pays_later_crossing_ranks_more_without_changing_total() -> None:
    runtime = _make_runtime(num_units=3, wall_pass_reward_skew=1.0, wall_pass_reward_weight=3.0)
    runtime.scenario.progress_reward_weight = 1.0
    stable_mask = torch.tensor([True], dtype=torch.bool)

    runtime._get_unit_y = lambda: torch.tensor([[0.6, 0.0, 0.0]], dtype=torch.float32)
    first = runtime.compute_step_rewards(stable_mask=stable_mask)
    assert float(first.info["wall_pass_reward"][0]) == pytest.approx(0.5)

    runtime._get_unit_y = lambda: torch.tensor([[0.6, 0.6, 0.0]], dtype=torch.float32)
    second = runtime.compute_step_rewards(stable_mask=stable_mask)
    assert float(second.info["wall_pass_reward"][0]) == pytest.approx(1.0)

    runtime._get_unit_y = lambda: torch.tensor([[0.6, 0.6, 0.6]], dtype=torch.float32)
    third = runtime.compute_step_rewards(stable_mask=stable_mask)
    assert float(third.info["wall_pass_reward"][0]) == pytest.approx(1.5)
    total_reward = first.info["wall_pass_reward"][0] + second.info["wall_pass_reward"][0] + third.info["wall_pass_reward"][0]
    assert float(total_reward) == pytest.approx(3.0)


def test_wall_pass_reward_skew_handles_simultaneous_crossings() -> None:
    runtime = _make_runtime(num_units=3, wall_pass_reward_skew=1.0, wall_pass_reward_weight=3.0)
    runtime.scenario.progress_reward_weight = 1.0
    stable_mask = torch.tensor([True], dtype=torch.bool)

    runtime._get_unit_y = lambda: torch.tensor([[0.6, 0.6, 0.0]], dtype=torch.float32)
    first_two = runtime.compute_step_rewards(stable_mask=stable_mask)
    assert float(first_two.info["wall_pass_reward"][0]) == pytest.approx(1.5)

    runtime._get_unit_y = lambda: torch.tensor([[0.6, 0.6, 0.6]], dtype=torch.float32)
    last = runtime.compute_step_rewards(stable_mask=stable_mask)
    assert float(last.info["wall_pass_reward"][0]) == pytest.approx(1.5)
    assert float(first_two.info["wall_pass_reward"][0] + last.info["wall_pass_reward"][0]) == pytest.approx(3.0)


def test_wall_pass_reward_skew_handles_multiple_thresholds() -> None:
    runtime = _make_runtime(
        num_units=3,
        wall_pass_reward_skew=1.0,
        wall_pass_reward_weight=3.0,
        wall_pass_absolute_thresholds=[0.5, 1.0],
    )
    runtime.scenario.progress_reward_weight = 1.0
    stable_mask = torch.tensor([True], dtype=torch.bool)

    runtime._get_unit_y = lambda: torch.tensor([[1.1, 0.0, 0.0]], dtype=torch.float32)
    first_unit_both_thresholds = runtime.compute_step_rewards(stable_mask=stable_mask)
    assert float(first_unit_both_thresholds.info["wall_pass_reward"][0]) == pytest.approx(0.5)

    runtime._get_unit_y = lambda: torch.tensor([[1.1, 0.6, 0.0]], dtype=torch.float32)
    second_unit_first_threshold = runtime.compute_step_rewards(stable_mask=stable_mask)
    assert float(second_unit_first_threshold.info["wall_pass_reward"][0]) == pytest.approx(0.5)

    runtime._get_unit_y = lambda: torch.tensor([[1.1, 1.1, 1.1]], dtype=torch.float32)
    remaining_crossings = runtime.compute_step_rewards(stable_mask=stable_mask)
    assert float(remaining_crossings.info["wall_pass_reward"][0]) == pytest.approx(2.0)

    total_reward = (
        first_unit_both_thresholds.info["wall_pass_reward"][0]
        + second_unit_first_threshold.info["wall_pass_reward"][0]
        + remaining_crossings.info["wall_pass_reward"][0]
    )
    assert float(total_reward) == pytest.approx(3.0)


def test_wall_pass_reward_skew_ignores_inactive_units() -> None:
    runtime = _make_runtime(
        num_units=3,
        wall_pass_reward_skew=1.0,
        wall_pass_reward_weight=2.0,
        units_active_mask=[True, False, True],
    )
    runtime.scenario.progress_reward_weight = 1.0
    stable_mask = torch.tensor([True], dtype=torch.bool)

    runtime._get_unit_y = lambda: torch.tensor([[0.6, 0.6, 0.0]], dtype=torch.float32)
    inactive_and_first_active = runtime.compute_step_rewards(stable_mask=stable_mask)
    assert float(inactive_and_first_active.info["wall_pass_reward"][0]) == pytest.approx(2.0 / 3.0)

    runtime._get_unit_y = lambda: torch.tensor([[0.6, 0.6, 0.6]], dtype=torch.float32)
    last_active = runtime.compute_step_rewards(stable_mask=stable_mask)
    assert float(last_active.info["wall_pass_reward"][0]) == pytest.approx(4.0 / 3.0)
    assert float(inactive_and_first_active.info["wall_pass_reward"][0] + last_active.info["wall_pass_reward"][0]) == pytest.approx(2.0)
    assert runtime.next_threshold_for_unit.tolist() == [[1, 0, 1]]


def test_wall_pass_rank_weights_are_precomputed_by_active_count() -> None:
    weights = _build_wall_pass_rank_weights(max_units=3, skew=1.0, device=torch.device("cpu"))

    assert weights[0].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert weights[1].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert weights[2].tolist() == pytest.approx([2.0 / 3.0, 4.0 / 3.0, 0.0])
    assert weights[3].tolist() == pytest.approx([0.5, 1.0, 1.5])


def test_wall_pass_rank_weight_cache_rebuilds_only_when_skew_changes() -> None:
    runtime = _make_runtime(num_units=3, wall_pass_reward_skew=1.0)

    first = runtime._get_wall_pass_rank_weights(1.0).clone()
    assert runtime._wall_pass_rank_weights_skew == 1.0

    second = runtime._get_wall_pass_rank_weights(1.0).clone()
    assert torch.allclose(second, first)
    assert runtime._wall_pass_rank_weights_skew == 1.0

    third = runtime._get_wall_pass_rank_weights(2.0).clone()
    assert runtime._wall_pass_rank_weights_skew == 2.0
    assert third[3].tolist() == pytest.approx([3.0 / 14.0, 12.0 / 14.0, 27.0 / 14.0])
    assert not torch.allclose(third, first)


def test_wall_climb_reward_uses_signed_potential_delta_so_retry_is_rewarded_in_mjw() -> None:
    runtime = _make_runtime()
    runtime.scenario.progress_reward_weight = 1.0
    runtime.scenario.wall_pass_reward_weight = 0.0
    runtime.scenario.wall_climb_reward_weight = 5.0
    runtime.scenario.wall_climb_reward_distance = 0.5
    runtime.scenario.swarm = SimpleNamespace(body_radius=0.1)
    runtime.wall_y_by_wall = torch.tensor([[1.0]], dtype=torch.float32)
    runtime._wall_heights = torch.tensor([0.4], dtype=torch.float32)
    runtime.wall_climb_potential = torch.zeros((1, 1, 1), dtype=torch.float32)
    runtime.wall_climb_done_mask = torch.zeros((1, 1, 1), dtype=torch.bool)
    stable_mask = torch.tensor([True], dtype=torch.bool)

    runtime._get_unit_y = lambda: torch.tensor([[0.75]], dtype=torch.float32)
    runtime._get_unit_z = lambda: torch.tensor([[0.25]], dtype=torch.float32)
    climb = runtime.compute_step_rewards(stable_mask=stable_mask)

    runtime._get_unit_z = lambda: torch.tensor([[0.1]], dtype=torch.float32)
    fall = runtime.compute_step_rewards(stable_mask=stable_mask)

    runtime._get_unit_z = lambda: torch.tensor([[0.25]], dtype=torch.float32)
    retry = runtime.compute_step_rewards(stable_mask=stable_mask)

    expected_reward = 5.0 * (0.5 ** 0.5) * 0.5
    assert float(climb.info["wall_climb_reward"][0]) == pytest.approx(expected_reward)
    assert float(fall.info["wall_climb_reward"][0]) == pytest.approx(-expected_reward)
    assert float(retry.info["wall_climb_reward"][0]) == pytest.approx(expected_reward)


def test_wall_climb_reward_applies_discount_factor_to_current_potential_in_mjw() -> None:
    runtime = _make_runtime()
    runtime.scenario.progress_reward_weight = 1.0
    runtime.scenario.wall_pass_reward_weight = 0.0
    runtime.scenario.wall_climb_reward_weight = 5.0
    runtime.scenario.wall_climb_reward_distance = 0.5
    runtime.scenario.potential_reward_discount_factor = 0.9
    runtime.scenario.swarm = SimpleNamespace(body_radius=0.1)
    runtime.wall_y_by_wall = torch.tensor([[1.0]], dtype=torch.float32)
    runtime._wall_heights = torch.tensor([0.4], dtype=torch.float32)
    runtime.wall_climb_potential = torch.zeros((1, 1, 1), dtype=torch.float32)
    runtime.wall_climb_done_mask = torch.zeros((1, 1, 1), dtype=torch.bool)
    stable_mask = torch.tensor([True], dtype=torch.bool)

    runtime._get_unit_y = lambda: torch.tensor([[0.75]], dtype=torch.float32)
    runtime._get_unit_z = lambda: torch.tensor([[0.25]], dtype=torch.float32)
    climb = runtime.compute_step_rewards(stable_mask=stable_mask)

    runtime._get_unit_z = lambda: torch.tensor([[0.1]], dtype=torch.float32)
    fall = runtime.compute_step_rewards(stable_mask=stable_mask)

    runtime._get_unit_z = lambda: torch.tensor([[0.25]], dtype=torch.float32)
    retry = runtime.compute_step_rewards(stable_mask=stable_mask)

    potential = (0.5 ** 0.5) * 0.5
    assert float(climb.info["wall_climb_reward"][0]) == pytest.approx(5.0 * 0.9 * potential)
    assert float(fall.info["wall_climb_reward"][0]) == pytest.approx(-5.0 * potential)
    assert float(retry.info["wall_climb_reward"][0]) == pytest.approx(5.0 * 0.9 * potential)


def test_wall_climb_reward_latches_after_crossing_wall_y_without_penalty_or_retry_reward_in_mjw() -> None:
    runtime = _make_runtime()
    runtime.scenario.progress_reward_weight = 1.0
    runtime.scenario.wall_pass_reward_weight = 0.0
    runtime.scenario.wall_climb_reward_weight = 5.0
    runtime.scenario.wall_climb_reward_distance = 0.5
    runtime.scenario.swarm = SimpleNamespace(body_radius=0.1)
    runtime.wall_y_by_wall = torch.tensor([[1.0]], dtype=torch.float32)
    runtime._wall_heights = torch.tensor([0.4], dtype=torch.float32)
    runtime.wall_climb_potential = torch.zeros((1, 1, 1), dtype=torch.float32)
    runtime.wall_climb_done_mask = torch.zeros((1, 1, 1), dtype=torch.bool)
    stable_mask = torch.tensor([True], dtype=torch.bool)

    runtime._get_unit_y = lambda: torch.tensor([[0.75]], dtype=torch.float32)
    runtime._get_unit_z = lambda: torch.tensor([[0.25]], dtype=torch.float32)
    climb = runtime.compute_step_rewards(stable_mask=stable_mask)

    runtime._get_unit_y = lambda: torch.tensor([[1.01]], dtype=torch.float32)
    cross_wall_y = runtime.compute_step_rewards(stable_mask=stable_mask)

    runtime._get_unit_y = lambda: torch.tensor([[0.75]], dtype=torch.float32)
    retry_after_backtracking = runtime.compute_step_rewards(stable_mask=stable_mask)

    expected_reward = 5.0 * (0.5 ** 0.5) * 0.5
    assert float(climb.info["wall_climb_reward"][0]) == pytest.approx(expected_reward)
    assert float(cross_wall_y.info["wall_climb_reward"][0]) == pytest.approx(0.0)
    assert float(retry_after_backtracking.info["wall_climb_reward"][0]) == pytest.approx(0.0)
    assert runtime.wall_climb_done_mask.tolist() == [[[True]]]
    assert float(runtime.wall_climb_potential[0, 0, 0]) == pytest.approx(0.0)
