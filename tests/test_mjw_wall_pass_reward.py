from __future__ import annotations

from types import SimpleNamespace

import torch

from swarmbots.mjw_env.scenarios.mjw_obstacle_street_runtime import (
    ObstacleStreetMJWScenarioRuntime,
    _compute_obstacle_street_reward_kernel,
)


def _make_runtime() -> ObstacleStreetMJWScenarioRuntime:
    runtime = object.__new__(ObstacleStreetMJWScenarioRuntime)
    device = torch.device("cpu")
    runtime.bindings = SimpleNamespace(
        num_envs=1,
        device=device,
        units_active_mask=torch.tensor([[True]], device=device, dtype=torch.bool),
        partner_unit=torch.tensor([[-1]], device=device, dtype=torch.long),
    )
    runtime.scenario = SimpleNamespace(
        progress_reward_weight=0.5,
        forward_reward_weight=0.0,
        wall_pass_reward_weight=4.0,
        units_without_connections_reward_weight=0.0,
        guidance_reward_weight=1.0,
    )
    runtime.progress = torch.zeros((1,), device=device, dtype=torch.float32)
    runtime.wall_pass_absolute_thresholds = torch.tensor([[0.5]], device=device, dtype=torch.float32)
    runtime.next_threshold_for_unit = torch.zeros((1, 1), device=device, dtype=torch.long)
    runtime.passed_thresholds_mask = torch.zeros((1, 1, 1), device=device, dtype=torch.bool)
    runtime._hidden_local_obs = torch.zeros((1, 1, 1), device=device, dtype=torch.float32)
    runtime._threshold_index_torch = torch.arange(1, device=device, dtype=torch.long)
    runtime._wall_thresholds_per_wall = 1
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
