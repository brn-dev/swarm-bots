from __future__ import annotations

import mujoco
import math
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from swarmbots import make_env, make_scenario
from swarmbots.mjw_env.mjw_swarm_bots_vector_env import _build_inactive_unit_positions
from swarmbots.mjw_env.scenarios.base_mjw_scenario import MJWCommonResetBatch
from swarmbots.mjw_env.scenarios.mjw_payload_goal_sampling import PayloadGoalSampler, make_payload_goal_layout
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig
from swarmbots.mjw_env.scenarios.mjw_multi_payload_goal_runtime import (
    MultiPayloadGoalMJWScenarioRuntime,
    _compute_multi_payload_goal_reward_kernel,
    _compute_payload_progress_np,
    _compute_payload_progress_torch,
)


@pytest.mark.parametrize("shape", ["sphere", "box", "capsule"])
@pytest.mark.parametrize("num_units", [2, 5, 12])
def test_inactive_payloads_do_not_collide_with_parked_units_or_each_other(shape: str, num_units: int) -> None:
    scenario = make_scenario(
        "SwarmBots-MultiPayloadGoal-v0", payload_shape=shape, compile_reward_kernel=False,
        unit_start_locations=MJWPreConnectedUnitLocationsConfig(num_units=num_units, pool_seeds=(42000,)),
    )
    device = torch.device("cpu")
    parked_units = _build_inactive_unit_positions(
        num_units=num_units, max_unit_extent=scenario.swarm.max_unit_extent,
        inactive_area_location=scenario.inactive_area_location, device=device,
    )
    runtime = object.__new__(MultiPayloadGoalMJWScenarioRuntime)
    runtime.scenario = scenario
    runtime.bindings = SimpleNamespace(device=device, inactive_unit_positions=parked_units)
    runtime._goal_sampler = PayloadGoalSampler(
        layout=scenario._goal_sampling_layout, max_payloads=scenario.max_payloads,
        payload_shape=shape, device=device,
    )
    active_payloads = torch.tensor([[True, False, True, False], [False, True, False, True]])
    positions = runtime._sample_payload_positions(
        common_reset_batch=MJWCommonResetBatch(
            pool_idx=torch.zeros(2, dtype=torch.long), swarm_start=torch.zeros(2, 2),
            initial_z_rotation=torch.zeros(2),
        ),
        active_payload_mask=active_payloads, rng=torch.Generator().manual_seed(9),
    )
    model = scenario.build_model()
    metadata = scenario.build_runtime_metadata(host_model=model)
    data = mujoco.MjData(model)
    inactive_unit_geoms = {
        index for index in range(model.ngeom)
        if model.body(int(model.geom_bodyid[index])).name.startswith(tuple(f"Unit{i}-" for i in range(1, num_units)))
    }
    for payload_positions, active_mask in zip(positions.numpy(), active_payloads.numpy(), strict=True):
        data.qpos[:] = model.qpos0
        for unit_index in range(1, num_units):
            body = model.body(f"Unit{unit_index}--main_body")
            address = int(model.jnt_qposadr[int(body.jntadr[0])])
            data.qpos[address:address + 3] = parked_units[unit_index].numpy()
            data.qpos[address + 3:address + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[metadata.payload_qpos_indices[:, :3]] = payload_positions
        data.qpos[metadata.payload_qpos_indices[:, 3:7]] = [1.0, 0.0, 0.0, 0.0]
        inactive_payload_geoms = {
            int(model.body_geomadr[model.body(f"Payload{index}").id]) for index in np.flatnonzero(~active_mask)
        }
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            first, second = map(int, contact.geom)
            involves_inactive_payload = first in inactive_payload_geoms or second in inactive_payload_geoms
            parked_pair = first in inactive_unit_geoms | inactive_payload_geoms and second in (
                inactive_unit_geoms | inactive_payload_geoms
            )
            if involves_inactive_payload and parked_pair:
                assert contact.dist >= -1e-6


def _sampler(*, shape="box", rect=(-2.5, 2.5, 2.5, 4.5), max_active=4, slots=8, device="cpu"):
    layout = make_payload_goal_layout(
        goal_rect=rect,
        payload_shape=shape,
        payload_radius=0.3,
        max_active_payloads=max_active,
    )
    return PayloadGoalSampler(layout=layout, max_payloads=slots, payload_shape=shape, device=torch.device(device))


def _assert_separated(goals, active, shape):
    # Compare the documented physical dimensions, independently of the sampler
    # layout, including nonadjacent active slots among padding.
    for first in range(goals.shape[1]):
        for second in range(first + 1, goals.shape[1]):
            delta = (goals[:, first] - goals[:, second]).abs()
            if shape == "sphere":
                separated = torch.linalg.vector_norm(delta, dim=-1) >= 0.6
            else:
                separated = (delta[:, 0] >= (1.2 if shape == "capsule" else 0.6)) | (delta[:, 1] >= 0.6)
            assert (separated | ~(active[:, first] & active[:, second])).all()


@pytest.mark.parametrize("shape", ["sphere", "box", "capsule"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_goals_are_bounded_repeatable_and_inactive_slots_do_not_constrain_active_goals(shape, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    sampler = _sampler(shape=shape, device=device)
    active = torch.tensor([[True, False, True, False, True, False, True, False]] * 256, device=device)
    active[::3, 4:] = False
    first_rng = torch.Generator(device=device).manual_seed(1000)
    second_rng = torch.Generator(device=device).manual_seed(1000)
    first = sampler.sample(active_payload_mask=active, rng=first_rng)
    second = sampler.sample(active_payload_mask=active, rng=second_rng)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(first_rng.get_state(), second_rng.get_state())
    assert not first[~active].any()
    visible = first[active]
    assert ((visible[:, 0] >= -2.5) & (visible[:, 0] <= 2.5)).all()
    assert ((visible[:, 1] >= 2.5) & (visible[:, 1] <= 4.5)).all()
    _assert_separated(first, active, shape)
    assert torch.unique(first[:, 0], dim=0).shape[0] > 240


@pytest.mark.parametrize("shape", ["sphere", "box", "capsule"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_bounded_sampler_falls_back_without_retrying_when_every_candidate_collides(monkeypatch, shape, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    max_x = 1.21 if shape == "capsule" else 0.61
    sampler = _sampler(shape=shape, rect=(0.0, max_x, 0.0, 0.61), slots=4, device=device)
    calls = []

    def clumped_samples(size, *, device, dtype, generator):
        calls.append(size)
        return torch.zeros(size, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "rand", clumped_samples)
    active = torch.ones((256, 4), dtype=torch.bool, device=device)
    goals = sampler.sample(active_payload_mask=active, rng=torch.Generator(device=device))
    assert len(calls) == 3
    precise = goals.double()
    assert (precise >= 0.0).all() and (precise[..., 0] <= max_x).all() and (precise[..., 1] <= 0.61).all()
    _assert_separated(goals, active, shape)


def test_single_payload_and_caller_default_dtype_do_not_change_the_float32_contract():
    sampler = _sampler(rect=(0.0, 0.01, 0.0, 0.01), max_active=1, slots=1)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        goals = sampler.sample(active_payload_mask=torch.ones((256, 1), dtype=torch.bool), rng=torch.Generator())
    finally:
        torch.set_default_dtype(original_dtype)
    assert goals.shape == (256, 1, 2) and goals.dtype == torch.float32
    assert (goals >= 0.0).all() and (goals.double() <= 0.01).all()


@pytest.mark.parametrize("rect", [
    (100000.0, 100000.1, 0.0, 1.0), (-100000.1, -100000.0, 0.0, 1.0),
    (0.0, 1.0, 100000.0, 100000.1), (0.0, 1.0, -100000.1, -100000.0),
    (1e20, 1.01e20, 1e20, 1.01e20),
])
def test_out_of_range_goal_rectangles_are_rejected_at_construction(rect: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="100000"):
        make_scenario(
            "SwarmBots-MultiPayloadGoal-v0",
            max_payloads=1,
            active_payload_count_probs={1: 1.0},
            goal_rect=rect,
            compile_reward_kernel=False,
        )


def test_goal_coordinate_limits_are_inclusive_and_keep_samples_finite() -> None:
    sampler = _sampler(rect=(-100000.0, 100000.0, -100000.0, 100000.0), slots=4)
    goals = sampler.sample(
        active_payload_mask=torch.ones((256, 4), dtype=torch.bool),
        rng=torch.Generator().manual_seed(123),
    )
    assert torch.isfinite(goals).all()
    assert (goals.abs() <= 100000.0).all()


def test_float32_unrepresentable_goal_span_is_rejected_within_coordinate_limits() -> None:
    with pytest.raises(ValueError, match="float32 span"):
        _sampler(rect=(99999.000001, 99999.000002, 0.0, 1.0), max_active=1, slots=1)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("compile_kernel", [False, True])
def test_payload_goal_distances_remain_finite_when_float32_squares_overflow(
    device: str, compile_kernel: bool,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if compile_kernel and device == "cpu":
        pytest.skip("Compiled reward regression exercises the CUDA benchmark path")
    positions = torch.tensor([[[3e20, 4e20, .3], [0., 0., .3]]], device=device)
    goals = torch.zeros((1, 2, 2), device=device)
    active = torch.tensor([[True, False]], device=device)
    progress = _compute_payload_progress_torch(
        payload_position=positions, goal_position=goals, active_payload_mask=active,
        payload_radius=.3, goal_radius=.1,
    )
    cpu_progress = _compute_payload_progress_np(
        payload_position=positions[0].cpu().numpy(), goal_position=goals[0].cpu().numpy(),
        active_payload_mask=active[0].cpu().numpy(), payload_radius=.3, goal_radius=.1,
    )
    expected_distance = math.hypot(float(positions[0, 0, 0]), float(positions[0, 0, 1]))
    assert torch.isfinite(progress).all()
    assert float(progress[0, 0]) == pytest.approx(-expected_distance, rel=1e-6)
    torch.testing.assert_close(progress[0].double().cpu(), torch.from_numpy(cpu_progress))
    kernel = _compute_multi_payload_goal_reward_kernel
    if compile_kernel:
        kernel = torch.compile(kernel, fullgraph=True)
    result = kernel(positions, goals, active, torch.ones(1, device=device, dtype=torch.bool),
                    progress, 1., 1., .99, .3, .1, 5.)
    assert all(torch.isfinite(value).all() for value in result)
    assert float(result[1][0]) == pytest.approx(expected_distance * .01, rel=1e-5)


def test_ordinary_decimal_bounds_are_rounded_inward_without_losing_finite_goals():
    sampler = _sampler(rect=(-0.61, 0.61, -0.01, 0.01), max_active=1, slots=1)
    goals = sampler.sample(
        active_payload_mask=torch.ones((4096, 1), dtype=torch.bool),
        rng=torch.Generator().manual_seed(1000),
    ).double()
    assert torch.isfinite(goals).all()
    assert ((goals[..., 0] >= -0.61) & (goals[..., 0] <= 0.61)).all()
    assert ((goals[..., 1] >= -0.01) & (goals[..., 1] <= 0.01)).all()


def test_feasibility_uses_only_positive_probability_active_counts_and_allows_narrow_center_bounds():
    scenario = make_scenario(
        "SwarmBots-MultiPayloadGoal-v0",
        max_payloads=8,
        active_payload_count_probs={2: 1.0, 8: 0.0},
        goal_rect=(0.0, 0.61, 0.0, 0.01),
        compile_reward_kernel=False,
    )
    assert scenario.get_settings()["goal_sampling"]["fallback_max_active_payloads"] == 2
    with pytest.raises(ValueError, match="guaranteed collision-free goal grid"):
        make_scenario(
            "SwarmBots-MultiPayloadGoal-v0",
            max_payloads=8,
            active_payload_count_probs={2: 1.0, 8: 1.0},
            goal_rect=(0.0, 0.61, 0.0, 0.01),
            compile_reward_kernel=False,
        )


def test_impossible_goal_bounds_are_rejected_during_scenario_construction():
    with pytest.raises(ValueError, match="goal_rect cannot fit"):
        make_scenario(
            "SwarmBots-MultiPayloadGoal-v0",
            max_payloads=2,
            active_payload_count_probs={2: 1.0},
            goal_radius=0.1,
            goal_rect=(0.0, 0.2, 0.0, 0.2),
            compile_reward_kernel=False,
        )


@pytest.mark.parametrize("shape", ["sphere", "box", "capsule"])
def test_all_payloads_can_physically_occupy_their_exact_goal_centers(shape):
    cfg = MJWPreConnectedUnitLocationsConfig(num_units=1, pool_seeds=[42000])
    scenario = make_scenario(
        "SwarmBots-MultiPayloadGoal-v0",
        payload_shape=shape,
        goal_radius=0.1,
        active_payload_count_probs={4: 1.0},
        unit_start_locations=cfg,
        compile_reward_kernel=False,
    )
    sampler = _sampler(shape=shape, slots=4)
    goals = sampler.sample(
        active_payload_mask=torch.ones((32, 4), dtype=torch.bool), rng=torch.Generator().manual_seed(1000)
    )
    model = scenario.build_model()
    metadata = scenario.build_runtime_metadata(host_model=model)
    data = mujoco.MjData(model)
    payload_geoms = {int(model.body_geomadr[model.body(f"Payload{index}").id]) for index in range(4)}
    for positions in goals.numpy():
        data.qpos[:] = model.qpos0
        data.qpos[metadata.payload_qpos_indices[:, :2]] = positions
        data.qpos[metadata.payload_qpos_indices[:, 2]] = scenario.payload_radius
        data.qpos[metadata.payload_qpos_indices[:, 3:7]] = np.array([1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            if int(contact.geom[0]) in payload_geoms and int(contact.geom[1]) in payload_geoms:
                assert contact.dist >= -1e-6


@pytest.mark.integration
def test_seeded_resets_and_partial_resets_use_the_separated_goal_sampler():
    cfg = MJWPreConnectedUnitLocationsConfig(num_units=1, pool_seeds=[42000])
    env = make_env(
        "SwarmBots-MultiPayloadGoal-v0",
        num_envs=2,
        device="cpu",
        seed=1000,
        compile_tensor_operations=False,
        scenario_kwargs=dict(
            goal_radius=0.1, unit_start_locations=cfg, compile_reward_kernel=False, reset_settle_time=0
        ),
    )
    try:
        first, _ = env.reset(seed=1000)
        first = {key: value.clone() for key, value in first.items()}
        second, _ = env.reset(seed=1000)
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        partial, _ = env.reset(seed=1001, options={"reset_mask": torch.tensor([True, False])})
        torch.testing.assert_close(partial["global_obs"][1], first["global_obs"][1], rtol=0, atol=0)
        assert not torch.equal(partial["global_obs"][0], first["global_obs"][0])
    finally:
        env.close()
