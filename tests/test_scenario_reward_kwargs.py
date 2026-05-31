from __future__ import annotations

from swarmbots.mj_env.scenarios import scenario_presets
from swarmbots.mjw_env.scenarios import mjw_scenario_presets
from swarmbots.scenario_presets.scenario_presets_kwargs import (
    BRIDGE_SCENARIO_KWARGS,
    CLIMB_SCENARIO_KWARGS,
    COMMON_SCENARIO_KWARGS,
    MOVE_TO_SCENARIO_KWARGS,
    PAYLOAD_PLANE_SCENARIO_KWARGS,
    EASY_WALL_SCENARIO_KWARGS,
    HARD_WALL_SCENARIO_KWARGS,
    MEDIUM_WALL_SCENARIO_KWARGS,
    WALL_SCENARIO_KWARGS,
)


def test_mj_and_mjw_common_scenario_kwargs_use_shared_defaults() -> None:
    mj_common_kwargs = {key: scenario_presets.DEFAULT_KWARGS[key] for key in COMMON_SCENARIO_KWARGS}

    assert mj_common_kwargs == COMMON_SCENARIO_KWARGS
    assert scenario_presets.DEFAULT_KWARGS["force_elliptic_cone"] is False
    assert mjw_scenario_presets.DEFAULT_KWARGS == COMMON_SCENARIO_KWARGS


def test_mj_and_mjw_wall_scenario_kwargs_use_shared_defaults() -> None:
    assert scenario_presets.WALL_SCENARIO_KWARGS == WALL_SCENARIO_KWARGS
    assert mjw_scenario_presets.WALL_SCENARIO_KWARGS == WALL_SCENARIO_KWARGS


def test_mj_and_mjw_wall_difficulty_kwargs_use_shared_defaults() -> None:
    assert scenario_presets.EASY_WALL_SCENARIO_KWARGS == EASY_WALL_SCENARIO_KWARGS
    assert scenario_presets.MEDIUM_WALL_SCENARIO_KWARGS == MEDIUM_WALL_SCENARIO_KWARGS
    assert scenario_presets.HARD_WALL_SCENARIO_KWARGS == HARD_WALL_SCENARIO_KWARGS
    assert mjw_scenario_presets.EASY_WALL_SCENARIO_KWARGS == EASY_WALL_SCENARIO_KWARGS
    assert mjw_scenario_presets.MEDIUM_WALL_SCENARIO_KWARGS == MEDIUM_WALL_SCENARIO_KWARGS
    assert mjw_scenario_presets.HARD_WALL_SCENARIO_KWARGS == HARD_WALL_SCENARIO_KWARGS
    assert COMMON_SCENARIO_KWARGS["potential_reward_discount_factor"] == 1.0
    assert MEDIUM_WALL_SCENARIO_KWARGS["potential_reward_discount_factor"] == 0.99
    assert MEDIUM_WALL_SCENARIO_KWARGS["forward_reward_wall_boost_factor"] == 2.0
    assert HARD_WALL_SCENARIO_KWARGS["forward_reward_wall_boost_factor"] == 2.0


def test_mj_and_mjw_bridge_scenario_kwargs_use_shared_defaults() -> None:
    assert scenario_presets.BRIDGE_SCENARIO_KWARGS == BRIDGE_SCENARIO_KWARGS
    assert mjw_scenario_presets.BRIDGE_SCENARIO_KWARGS == BRIDGE_SCENARIO_KWARGS


def test_mj_and_mjw_climb_scenario_kwargs_use_shared_defaults() -> None:
    assert scenario_presets.CLIMB_SCENARIO_KWARGS == CLIMB_SCENARIO_KWARGS
    assert mjw_scenario_presets.CLIMB_SCENARIO_KWARGS == CLIMB_SCENARIO_KWARGS


def test_mj_and_mjw_payload_scenario_kwargs_use_shared_defaults() -> None:
    assert scenario_presets.PAYLOAD_PLANE_SCENARIO_KWARGS == PAYLOAD_PLANE_SCENARIO_KWARGS
    assert mjw_scenario_presets.PAYLOAD_PLANE_SCENARIO_KWARGS == PAYLOAD_PLANE_SCENARIO_KWARGS


def test_mj_and_mjw_move_to_scenario_kwargs_use_shared_defaults() -> None:
    assert scenario_presets.MOVE_TO_SCENARIO_KWARGS == MOVE_TO_SCENARIO_KWARGS
    assert mjw_scenario_presets.MOVE_TO_SCENARIO_KWARGS == MOVE_TO_SCENARIO_KWARGS
