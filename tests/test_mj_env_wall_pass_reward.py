from __future__ import annotations

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from swarmbots.mj_env.scenarios.obstacle_street_scenario import ObstacleStreetScenario


def _make_scenario(
    *,
    wall_pass_reward_skew: float,
    wall_pass_thresholds: list[float] | None = None,
    wall_pass_reward_weight: float = 3.0,
    num_units: int = 3,
) -> ObstacleStreetScenario:
    scenario = object.__new__(ObstacleStreetScenario)
    scenario.num_units = num_units
    scenario.wall_pass_thresholds = np.asarray([0.0] if wall_pass_thresholds is None else wall_pass_thresholds, dtype=float)
    scenario.wall_pass_reward_weight = wall_pass_reward_weight
    scenario.wall_pass_reward_skew = wall_pass_reward_skew
    scenario._qpos_indices = np.asarray([[2 * unit_idx, (2 * unit_idx) + 1] for unit_idx in range(num_units)], dtype=np.int64)
    return scenario


def _make_state(
    *,
    wall_pass_absolute_thresholds: list[float] | None = None,
    units_active_mask: list[bool] | None = None,
    num_units: int = 3,
) -> dict[str, object]:
    thresholds = np.asarray([0.5] if wall_pass_absolute_thresholds is None else wall_pass_absolute_thresholds, dtype=float)
    return {
        "wall_pass_absolute_thresholds": thresholds,
        "next_threshold_for_unit": np.zeros((num_units,), dtype=int),
        "units_active_mask": np.ones((num_units,), dtype=bool) if units_active_mask is None else np.asarray(units_active_mask, dtype=bool),
    }


def _make_data(unit_y: list[float]) -> SimpleNamespace:
    qpos = np.zeros((len(unit_y) * 2,), dtype=float)
    qpos[np.arange(1, len(unit_y) * 2, 2)] = unit_y
    return SimpleNamespace(qpos=qpos)


def _make_wall_climb_scenario() -> ObstacleStreetScenario:
    scenario = object.__new__(ObstacleStreetScenario)
    scenario.num_units = 1
    scenario.num_walls = 1
    scenario.wall_climb_reward_weight = 5.0
    scenario.wall_climb_reward_distance = 0.5
    scenario.wall_heights = [0.4]
    scenario.swarm = SimpleNamespace(body_radius=0.1)
    scenario._qpos_indices = np.asarray([[0, 1, 2]], dtype=np.int64)
    return scenario


def _make_wall_climb_data(*, unit_y: float, unit_z: float) -> SimpleNamespace:
    return SimpleNamespace(qpos=np.asarray([0.0, unit_y, unit_z], dtype=float))


def _unit_y_for_passed_count(*, passed_count: int, thresholds: np.ndarray) -> float:
    if passed_count <= 0:
        return float(thresholds[0] - 0.25)
    if passed_count >= thresholds.size:
        return float(thresholds[-1] + 0.25)
    return float((thresholds[passed_count - 1] + thresholds[passed_count]) / 2.0)


def _oracle_rank_weights(*, active_units_count: int, skew: float) -> np.ndarray:
    if active_units_count <= 0:
        return np.zeros((0,), dtype=float)
    ranks = np.arange(1, active_units_count + 1, dtype=float)
    if skew == 0.0:
        return np.ones_like(ranks)
    unnormalized = (ranks / float(active_units_count)) ** skew
    return unnormalized * (float(active_units_count) / float(unnormalized.sum()))


def _oracle_wall_pass_step(
    *,
    unit_y: list[float],
    thresholds: np.ndarray,
    next_threshold_for_unit: np.ndarray,
    active_mask: np.ndarray,
    thresholds_per_wall: int,
    reward_weight: float,
    skew: float,
) -> tuple[float, int, np.ndarray, np.ndarray]:
    active_units_count = int(active_mask.sum())
    rank_weights = _oracle_rank_weights(active_units_count=active_units_count, skew=skew)
    passed_by_active_units = np.zeros((thresholds.size,), dtype=int)
    for unit_idx, next_threshold_idx in enumerate(next_threshold_for_unit):
        if active_mask[unit_idx]:
            passed_by_active_units[: int(next_threshold_idx)] += 1

    reward_units = 0.0
    num_passed = 0
    new_next_thresholds = next_threshold_for_unit.copy()
    for unit_idx, y_pos in enumerate(unit_y):
        if not active_mask[unit_idx]:
            continue
        total_passed = int((float(y_pos) > thresholds).sum())
        for threshold_idx in range(int(next_threshold_for_unit[unit_idx]), total_passed):
            next_rank = passed_by_active_units[threshold_idx]
            reward_units += float(rank_weights[next_rank])
            passed_by_active_units[threshold_idx] = next_rank + 1
            num_passed += 1
        new_next_thresholds[unit_idx] = max(int(next_threshold_for_unit[unit_idx]), total_passed)

    if active_units_count > 0 and thresholds_per_wall > 0:
        reward = (reward_units / float(active_units_count * thresholds_per_wall)) * reward_weight
    else:
        reward = 0.0
    passed_thresholds_mask = np.arange(thresholds.size)[np.newaxis, :] < new_next_thresholds[:, np.newaxis]
    return reward, num_passed, new_next_thresholds, passed_thresholds_mask


def _schedule_to_unit_y(schedule: list[list[int]], thresholds: np.ndarray) -> list[list[float]]:
    return [
        [_unit_y_for_passed_count(passed_count=passed_count, thresholds=thresholds) for passed_count in step]
        for step in schedule
    ]


def _make_count_schedules(*, num_units: int, total_thresholds: int) -> list[list[list[int]]]:
    schedules = [
        [[0] * num_units, [total_thresholds] * num_units],
        [[total_thresholds if unit_idx <= step_idx else 0 for unit_idx in range(num_units)] for step_idx in range(num_units)],
        [[total_thresholds if unit_idx < 2 else 0 for unit_idx in range(num_units)], [total_thresholds] * num_units],
        [[total_thresholds if unit_idx == 0 else 0 for unit_idx in range(num_units)], [0] * num_units, [total_thresholds] * num_units],
    ]
    if total_thresholds > 1 and num_units > 1:
        schedules.append(
            [
                [1 if unit_idx == 0 else 0 for unit_idx in range(num_units)],
                [total_thresholds if unit_idx == 0 else 1 if unit_idx == 1 else 0 for unit_idx in range(num_units)],
                [total_thresholds] * num_units,
            ]
        )
    return schedules


def test_default_wall_pass_reward_skew_keeps_equal_cpu_rank_rewards() -> None:
    scenario = _make_scenario(wall_pass_reward_skew=0.0)
    state = _make_state()

    first = scenario._compute_wall_pass_reward(_make_data([0.6, 0.0, 0.0]), state)
    assert first == pytest.approx(1.0)

    remaining = scenario._compute_wall_pass_reward(_make_data([0.6, 0.6, 0.6]), state)
    assert remaining == pytest.approx(2.0)


def test_wall_pass_reward_skew_pays_later_cpu_ranks_more_without_changing_total() -> None:
    scenario = _make_scenario(wall_pass_reward_skew=1.0)
    state = _make_state()

    first = scenario._compute_wall_pass_reward(_make_data([0.6, 0.0, 0.0]), state)
    second = scenario._compute_wall_pass_reward(_make_data([0.6, 0.6, 0.0]), state)
    third = scenario._compute_wall_pass_reward(_make_data([0.6, 0.6, 0.6]), state)

    assert first == pytest.approx(0.5)
    assert second == pytest.approx(1.0)
    assert third == pytest.approx(1.5)
    assert first + second + third == pytest.approx(3.0)


def test_wall_pass_reward_skew_handles_simultaneous_cpu_crossings() -> None:
    scenario = _make_scenario(wall_pass_reward_skew=1.0)
    state = _make_state()

    first_two = scenario._compute_wall_pass_reward(_make_data([0.6, 0.6, 0.0]), state)
    last = scenario._compute_wall_pass_reward(_make_data([0.6, 0.6, 0.6]), state)

    assert first_two == pytest.approx(1.5)
    assert last == pytest.approx(1.5)
    assert first_two + last == pytest.approx(3.0)


def test_wall_pass_reward_skew_handles_multiple_cpu_thresholds() -> None:
    scenario = _make_scenario(
        wall_pass_reward_skew=1.0,
        wall_pass_thresholds=[0.0, 0.5],
        wall_pass_reward_weight=3.0,
    )
    state = _make_state(wall_pass_absolute_thresholds=[0.5, 1.0])

    first_unit_both_thresholds = scenario._compute_wall_pass_reward(_make_data([1.1, 0.0, 0.0]), state)
    second_unit_first_threshold = scenario._compute_wall_pass_reward(_make_data([1.1, 0.6, 0.0]), state)
    remaining_crossings = scenario._compute_wall_pass_reward(_make_data([1.1, 1.1, 1.1]), state)

    assert first_unit_both_thresholds == pytest.approx(0.5)
    assert second_unit_first_threshold == pytest.approx(0.5)
    assert remaining_crossings == pytest.approx(2.0)
    assert first_unit_both_thresholds + second_unit_first_threshold + remaining_crossings == pytest.approx(3.0)


def test_wall_pass_reward_skew_ignores_inactive_cpu_units() -> None:
    scenario = _make_scenario(wall_pass_reward_skew=1.0, wall_pass_reward_weight=2.0)
    state = _make_state(units_active_mask=[True, False, True])

    inactive_and_first_active = scenario._compute_wall_pass_reward(_make_data([0.6, 0.6, 0.0]), state)
    last_active = scenario._compute_wall_pass_reward(_make_data([0.6, 0.6, 0.6]), state)

    assert inactive_and_first_active == pytest.approx(2.0 / 3.0)
    assert last_active == pytest.approx(4.0 / 3.0)
    assert inactive_and_first_active + last_active == pytest.approx(2.0)
    assert state["next_threshold_for_unit"].tolist() == [1, 0, 1]


def test_wall_pass_reward_rank_weights_sum_to_active_unit_count() -> None:
    for active_units_count, skew in product(range(1, 9), [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]):
        weights = _oracle_rank_weights(active_units_count=active_units_count, skew=skew)

        assert weights.sum() == pytest.approx(active_units_count)
        assert weights.shape == (active_units_count,)
        if skew == 0.0:
            assert weights.tolist() == pytest.approx([1.0] * active_units_count)
        else:
            assert np.all(np.diff(weights) >= 0.0)


def test_wall_pass_reward_matches_independent_oracle_exhaustively() -> None:
    reward_weight = 7.5
    skews = [0.0, 0.25, 1.0, 2.0]
    thresholds_per_wall_options = [1, 2, 3]

    for num_units in range(1, 5):
        active_masks = [
            np.asarray(mask, dtype=bool)
            for mask in product([False, True], repeat=num_units)
            if any(mask)
        ]
        for active_mask, thresholds_per_wall, num_walls, skew in product(active_masks, thresholds_per_wall_options, [1, 2], skews):
            thresholds = np.arange(thresholds_per_wall * num_walls, dtype=float) + 0.5
            scenario = _make_scenario(
                wall_pass_reward_skew=skew,
                wall_pass_thresholds=[0.0] * thresholds_per_wall,
                wall_pass_reward_weight=reward_weight,
                num_units=num_units,
            )
            for schedule in _make_count_schedules(num_units=num_units, total_thresholds=thresholds.size):
                state = _make_state(
                    wall_pass_absolute_thresholds=thresholds.tolist(),
                    units_active_mask=active_mask.tolist(),
                    num_units=num_units,
                )
                oracle_next_thresholds = np.zeros((num_units,), dtype=int)
                total_reward = 0.0
                total_oracle_reward = 0.0

                for unit_y in _schedule_to_unit_y(schedule, thresholds):
                    actual_reward = scenario._compute_wall_pass_reward(_make_data(unit_y), state)
                    oracle_reward, oracle_num_passed, oracle_next_thresholds, oracle_passed_mask = _oracle_wall_pass_step(
                        unit_y=unit_y,
                        thresholds=thresholds,
                        next_threshold_for_unit=oracle_next_thresholds,
                        active_mask=active_mask,
                        thresholds_per_wall=thresholds_per_wall,
                        reward_weight=reward_weight,
                        skew=skew,
                    )
                    total_reward += actual_reward
                    total_oracle_reward += oracle_reward

                    assert actual_reward == pytest.approx(oracle_reward)
                    assert state["num_walls_passed"] == oracle_num_passed
                    assert state["next_threshold_for_unit"].tolist() == oracle_next_thresholds.tolist()
                    assert state["passed_thresholds_mask"].tolist() == oracle_passed_mask.tolist()

                assert total_reward == pytest.approx(total_oracle_reward)
                max_reward = reward_weight * num_walls
                if all(step_active >= thresholds.size for unit_idx, step_active in enumerate(schedule[-1]) if active_mask[unit_idx]):
                    assert total_reward == pytest.approx(max_reward)


def test_wall_pass_reward_handles_no_active_units() -> None:
    scenario = _make_scenario(wall_pass_reward_skew=1.0, wall_pass_reward_weight=3.0)
    state = _make_state(units_active_mask=[False, False, False])

    reward = scenario._compute_wall_pass_reward(_make_data([0.6, 0.6, 0.6]), state)

    assert reward == 0.0
    assert state["num_walls_passed"] == 0
    assert state["next_threshold_for_unit"].tolist() == [0, 0, 0]
    assert state["passed_thresholds_mask"].tolist() == [[False], [False], [False]]


def test_wall_climb_reward_uses_signed_potential_delta_so_retry_is_rewarded() -> None:
    scenario = _make_wall_climb_scenario()
    state: dict[str, object] = {
        "wall_y": np.asarray([1.0], dtype=float),
        "wall_climb_potential": np.zeros((1, 1), dtype=np.float32),
        "units_active_mask": np.asarray([True], dtype=bool),
    }

    climb = scenario._compute_wall_climb_reward(_make_wall_climb_data(unit_y=0.75, unit_z=0.25), state)
    fall = scenario._compute_wall_climb_reward(_make_wall_climb_data(unit_y=0.75, unit_z=0.1), state)
    retry = scenario._compute_wall_climb_reward(_make_wall_climb_data(unit_y=0.75, unit_z=0.25), state)

    expected_reward = 5.0 * np.sqrt(0.5) * 0.5
    assert climb == pytest.approx(expected_reward)
    assert fall == pytest.approx(-expected_reward)
    assert retry == pytest.approx(expected_reward)


def test_wall_climb_reward_latches_after_crossing_wall_y_without_penalty_or_retry_reward() -> None:
    scenario = _make_wall_climb_scenario()
    state: dict[str, object] = {
        "wall_y": np.asarray([1.0], dtype=float),
        "wall_climb_potential": np.zeros((1, 1), dtype=np.float32),
        "wall_climb_done_mask": np.zeros((1, 1), dtype=bool),
        "units_active_mask": np.asarray([True], dtype=bool),
    }

    climb = scenario._compute_wall_climb_reward(_make_wall_climb_data(unit_y=0.75, unit_z=0.25), state)
    cross_wall_y = scenario._compute_wall_climb_reward(_make_wall_climb_data(unit_y=1.01, unit_z=0.25), state)
    retry_after_backtracking = scenario._compute_wall_climb_reward(_make_wall_climb_data(unit_y=0.75, unit_z=0.25), state)

    expected_reward = 5.0 * np.sqrt(0.5) * 0.5
    assert climb == pytest.approx(expected_reward)
    assert cross_wall_y == pytest.approx(0.0)
    assert retry_after_backtracking == pytest.approx(0.0)
    assert state["wall_climb_done_mask"].tolist() == [[True]]
    assert state["wall_climb_potential"].tolist() == [[0.0]]


def test_units_without_connections_reward_is_named_explicitly_in_cpu_scenario_state() -> None:
    scenario = _make_scenario(wall_pass_reward_skew=0.0, wall_pass_reward_weight=0.0, num_units=2)
    scenario.forward_reward_weight = 0.0
    scenario.forward_reward_max_y = None
    scenario.wall_climb_reward_weight = 0.0
    scenario.reward_weights = {
        "progress_reward_weight": 1.0,
        "guidance_reward_weight": 2.0,
        "units_without_connections_reward_weight": -0.25,
    }
    state: dict[str, object] = {
        "progress": 0.0,
        "wall_pass_absolute_thresholds": np.asarray([], dtype=float),
        "next_threshold_for_unit": np.zeros((2,), dtype=int),
    }
    connections = SimpleNamespace(get_is_active_mask=lambda: np.zeros((2, 1), dtype=bool))

    reward, done = scenario.evaluate_step(
        action={},
        model=None,
        data=_make_data([0.0, 0.0]),
        state=state,
        connections=connections,
    )

    assert done is False
    assert reward == pytest.approx(-0.5)
    assert state["units_without_connections_reward"] == pytest.approx(-0.25)
    assert state["guidance_reward"] == pytest.approx(-0.25)
    assert state["weighted_units_without_connections_reward"] == pytest.approx(-0.5)
    assert state["weighted_guidance_reward"] == pytest.approx(-0.5)
    assert state["reward_terms"]["units_without_connections"] == pytest.approx(-0.5)
    assert "guidance" not in state["reward_terms"]
