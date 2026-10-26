from types import SimpleNamespace

import pytest
import torch

from swarmbots import make_scenario
from swarmbots.mjw_env.mjw_swarm_bots_vector_env import _build_inactive_unit_positions
from swarmbots.mjw_env.scenarios.base_mjw_scenario import MJWCommonResetBatch
from swarmbots.mjw_env.scenarios.mjw_multi_payload_goal_runtime import MultiPayloadGoalMJWScenarioRuntime
from swarmbots.mjw_env.scenarios.mjw_payload_goal_sampling import PayloadGoalSampler
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig


DEVICES = [
    "cpu",
    pytest.param("cuda", marks=[pytest.mark.cuda, pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable",
    )]),
]


def _make_runtime(
    device: str, *, shape: str = "box", max_payloads: int = 4,
) -> MultiPayloadGoalMJWScenarioRuntime:
    scenario = make_scenario(
        "SwarmBots-MultiPayloadGoal-v0", payload_shape=shape, max_payloads=max_payloads,
        active_payload_count_probs={max_payloads: 1.0}, compile_reward_kernel=False,
        unit_start_locations=MJWPreConnectedUnitLocationsConfig(num_units=5, pool_seeds=(42000,)),
    )
    runtime = object.__new__(MultiPayloadGoalMJWScenarioRuntime)
    runtime.scenario = scenario
    runtime.bindings = SimpleNamespace(
        device=torch.device(device),
        inactive_unit_positions=_build_inactive_unit_positions(
            num_units=5, max_unit_extent=scenario.swarm.max_unit_extent,
            inactive_area_location=scenario.inactive_area_location, device=torch.device(device),
        ),
    )
    runtime._goal_sampler = PayloadGoalSampler(
        layout=scenario._goal_sampling_layout, max_payloads=max_payloads,
        payload_shape=shape, device=torch.device(device),
    )
    return runtime


def _common_reset_batch(swarm_start: torch.Tensor) -> MJWCommonResetBatch:
    n_reset = swarm_start.shape[0]
    return MJWCommonResetBatch(
        pool_idx=torch.zeros(n_reset, device=swarm_start.device, dtype=torch.long),
        swarm_start=swarm_start,
        initial_z_rotation=torch.zeros(n_reset, device=swarm_start.device, dtype=torch.float32),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", ["sphere", "box", "capsule"])
def test_active_payload_rows_are_centered_separated_and_leave_parking_unchanged(device: str, shape: str) -> None:
    runtime = _make_runtime(device, shape=shape)
    active = torch.tensor([
        [True, False, False, False], [False, True, False, True], [True, True, False, True],
        [True, True, True, True], [False, False, False, False],
    ], device=device)
    origins = torch.tensor([[-4., 2.], [1.5, -2.], [2., 8.], [10000., 1.], [-7., 3.]], device=device)
    common = _common_reset_batch(origins)
    positions = runtime._sample_payload_positions(
        common_reset_batch=common, active_payload_mask=active, rng=torch.Generator(device=device).manual_seed(9),
    )
    parked = runtime._sample_payload_positions(
        common_reset_batch=common, active_payload_mask=torch.zeros_like(active),
        rng=torch.Generator(device=device).manual_seed(19),
    )
    torch.testing.assert_close(positions[~active], parked[~active], rtol=0, atol=0)
    torch.testing.assert_close(positions[..., 2], torch.full_like(positions[..., 2], runtime.scenario.payload_radius))
    for row in range(4):
        active_positions = positions[row, active[row]]
        torch.testing.assert_close(active_positions[:, 0].double().mean(), origins[row, 0].double(), rtol=0, atol=.002)
        torch.testing.assert_close(
            active_positions[:, 1], (origins[row, 1] + runtime.scenario.payload_spawn_y).expand_as(active_positions[:, 1]),
        )
        sorted_x = active_positions[:, 0].sort().values
        assert ((sorted_x[1:] - sorted_x[:-1]) >= runtime._goal_sampler.layout.footprint[0]).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("max_payloads", [1, 4])
def test_payload_placement_is_repeatable_for_empty_and_partial_reset_batches(device: str, max_payloads: int) -> None:
    runtime = _make_runtime(device, max_payloads=max_payloads)
    first_rng = torch.Generator(device=device).manual_seed(101)
    second_rng = torch.Generator(device=device).manual_seed(101)
    for n_reset in (0, 1, 16, 3):
        common = _common_reset_batch(torch.zeros((n_reset, 2), device=device, dtype=torch.float32))
        first_mask = (
            runtime._sample_active_payload_mask(n_reset=n_reset, rng=first_rng)
            if n_reset else torch.empty((0, max_payloads), device=device, dtype=torch.bool)
        )
        second_mask = (
            runtime._sample_active_payload_mask(n_reset=n_reset, rng=second_rng)
            if n_reset else torch.empty((0, max_payloads), device=device, dtype=torch.bool)
        )
        torch.testing.assert_close(first_mask, second_mask, rtol=0, atol=0)
        first = runtime._sample_payload_positions(common_reset_batch=common, active_payload_mask=first_mask, rng=first_rng)
        second = runtime._sample_payload_positions(common_reset_batch=common, active_payload_mask=second_mask, rng=second_rng)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert first.shape == (n_reset, max_payloads, 3)
        if max_payloads == 1:
            torch.testing.assert_close(first[..., 0], torch.zeros_like(first[..., 0]), rtol=0, atol=0)
    assert torch.equal(first_rng.get_state(), second_rng.get_state())


@pytest.mark.parametrize("device", DEVICES)
def test_payload_slot_permutations_are_balanced_with_gaps_in_active_payload_ids(device: str) -> None:
    runtime = _make_runtime(device)
    active = torch.tensor([[True, False, True, True]], device=device).expand(4096, -1)
    common = _common_reset_batch(torch.zeros((4096, 2), device=device, dtype=torch.float32))
    positions = runtime._sample_payload_positions(
        common_reset_batch=common, active_payload_mask=active, rng=torch.Generator(device=device).manual_seed(702),
    )
    permutations, counts = positions[:, [0, 2, 3], 0].unique(dim=0, return_counts=True)
    assert permutations.shape[0] == 6
    assert ((counts - 4096 / 6).abs() < .25 * 4096 / 6).all()


@pytest.mark.parametrize("device", DEVICES)
def test_payload_placement_keeps_float32_coordinates_with_float64_default(device: str) -> None:
    runtime = _make_runtime(device)
    common = _common_reset_batch(torch.zeros((2, 2), device=device, dtype=torch.float32))
    active = torch.tensor([[True, True, False, True], [False, True, False, False]], device=device)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        positions = runtime._sample_payload_positions(
            common_reset_batch=common, active_payload_mask=active, rng=torch.Generator(device=device).manual_seed(8),
        )
    finally:
        torch.set_default_dtype(original_dtype)
    assert positions.dtype == torch.float32
    assert torch.isfinite(positions).all()
