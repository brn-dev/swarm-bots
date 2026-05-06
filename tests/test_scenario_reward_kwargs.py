from __future__ import annotations

from swarmbots.mj_env.scenarios import scenario_presets
from swarmbots.mjw_env.scenarios import mjw_scenario_presets
from swarmbots.scenario_presets_kwargs import (
    BRIDGE_REWARD_KWARGS,
    COMMON_SCENARIO_KWARGS,
    MOVE_TO_REWARD_KWARGS,
    PAYLOAD_PLANE_REWARD_KWARGS,
    WALL_PASS_REWARD_KWARGS,
)


def test_mj_and_mjw_wall_reward_kwargs_use_shared_defaults() -> None:
    assert scenario_presets.WALL_PASS_REWARD_KWARGS == WALL_PASS_REWARD_KWARGS
    assert mjw_scenario_presets.WALL_PASS_REWARD_KWARGS == WALL_PASS_REWARD_KWARGS


def test_mj_and_mjw_bridge_reward_kwargs_use_shared_defaults() -> None:
    assert scenario_presets.BRIDGE_REWARD_KWARGS == BRIDGE_REWARD_KWARGS
    assert mjw_scenario_presets.BRIDGE_REWARD_KWARGS == BRIDGE_REWARD_KWARGS


def test_mj_and_mjw_payload_reward_kwargs_use_shared_defaults() -> None:
    assert scenario_presets.PAYLOAD_PLANE_REWARD_KWARGS == PAYLOAD_PLANE_REWARD_KWARGS
    assert mjw_scenario_presets.PAYLOAD_PLANE_REWARD_KWARGS == PAYLOAD_PLANE_REWARD_KWARGS


def test_mj_and_mjw_move_to_reward_kwargs_use_shared_defaults() -> None:
    assert scenario_presets.MOVE_TO_REWARD_KWARGS == MOVE_TO_REWARD_KWARGS
    assert mjw_scenario_presets.MOVE_TO_REWARD_KWARGS == MOVE_TO_REWARD_KWARGS


def test_mj_and_mjw_common_scenario_kwargs_use_shared_defaults() -> None:
    mj_common_kwargs = {key: scenario_presets.DEFAULT_KWARGS[key] for key in COMMON_SCENARIO_KWARGS}

    assert mj_common_kwargs == COMMON_SCENARIO_KWARGS
    assert scenario_presets.DEFAULT_KWARGS["force_elliptic_cone"] is False
    assert mjw_scenario_presets.DEFAULT_KWARGS == COMMON_SCENARIO_KWARGS
