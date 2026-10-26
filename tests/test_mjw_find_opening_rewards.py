from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from swarmbots.mjw_env.scenarios.mjw_find_opening_runtime import (
    FindOpeningMJWScenarioRuntime,
    _compute_find_opening_reward_kernel,
)


CUDA = pytest.param("cuda", marks=(
    pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
))


def _make_runtime(device: str, cell_count: int) -> FindOpeningMJWScenarioRuntime:
    runtime = object.__new__(FindOpeningMJWScenarioRuntime)
    runtime.bindings = SimpleNamespace(units_active_mask=torch.ones((2, 2), device=device, dtype=torch.bool))
    runtime.scenario = SimpleNamespace(
        wall_y=1.0, opening_y_margin=0.2, opening_distance_reward_weight=0.0,
        opening_distance_reward_falloff_distance=None, success_reward=0.0, progress_reward_weight=1.0,
        potential_reward_discount_factor=1.0, wall_exploration_cell_count=cell_count,
        wall_exploration_cell_reward=1.0, wall_exploration_cell_depth=0.5, side_wall_x=2.0, wall_thickness=0.2,
    )
    runtime.opening_x = torch.tensor([[0.5], [-0.5]], device=device)
    runtime.opening_potential = torch.tensor([0.0, torch.nan], device=device)
    runtime.wall_exploration_visited_cells = torch.zeros((2, cell_count), device=device, dtype=torch.bool)
    runtime.wall_exploration_visited_cells[1] = True
    runtime._hidden_global_obs = torch.full((2, cell_count + 1), 7.0, device=device)
    runtime._hidden_global_obs[:, :1] = runtime.opening_x
    unit_x = torch.tensor([[-1.5, 0.5], [torch.nan, torch.nan]], device=device)
    unit_y = torch.tensor([[0.8, 0.8], [torch.nan, torch.nan]], device=device)
    runtime._get_unit_x = lambda: unit_x
    runtime._get_unit_y = lambda: unit_y
    runtime._reward_kernel = _compute_find_opening_reward_kernel
    return runtime


@pytest.mark.parametrize("device", ["cpu", CUDA])
@pytest.mark.parametrize("cell_count", [0, 4])
def test_exploration_observations_follow_stable_world_history_and_preserve_unstable_rows(
    device: str, cell_count: int,
) -> None:
    runtime = _make_runtime(device, cell_count)
    state_names = ("opening_potential", "wall_exploration_visited_cells", "_hidden_global_obs")
    original_buffers = {name: getattr(runtime, name) for name in state_names}
    original_states = {name: value.clone() for name, value in original_buffers.items()}

    result = runtime.compute_step_rewards(stable_mask=torch.tensor([True, False], device=device))

    expected_cells = torch.tensor([True, False, True, False], device=device)[:cell_count]
    torch.testing.assert_close(runtime.wall_exploration_visited_cells[0], expected_cells)
    torch.testing.assert_close(runtime._hidden_global_obs[0, 1:], expected_cells.float())
    torch.testing.assert_close(runtime._hidden_global_obs[:, :1], runtime.opening_x)
    assert result.info["wall_exploration_reward"][0] == (2.0 if cell_count else 0.0)
    for name, original in original_states.items():
        assert getattr(runtime, name) is original_buffers[name]
        torch.testing.assert_close(getattr(runtime, name)[1], original[1], equal_nan=True)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_exploration_reward_and_observation_updates_can_replay_in_a_cuda_graph() -> None:
    runtime = _make_runtime("cuda", cell_count=4)
    mask = torch.tensor([True, False], device="cuda")
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        runtime.compute_step_rewards(stable_mask=mask)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = runtime.compute_step_rewards(stable_mask=mask)

    runtime.wall_exploration_visited_cells[0] = False
    runtime._hidden_global_obs[0, 1:] = -1.0
    graph.replay()

    torch.testing.assert_close(runtime._hidden_global_obs.cpu(), torch.tensor([
        [0.5, 1.0, 0.0, 1.0, 0.0], [-0.5, 7.0, 7.0, 7.0, 7.0],
    ]))
    assert result.info["wall_exploration_reward"][0] == 2.0
