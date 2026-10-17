from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from swarmbots import ALL_BENCHMARK_IDS, make_env, make_scenario
from swarmbots.benchmark.metadata import serialize_settings
from swarmbots.mjw_env.mjw_swarm_bots_vector_env import MJWSwarmBotsVectorEnv
from swarmbots.mjw_env.random_utils import random_quat_shoemake
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWHomogeneousSwarm, MJWPreConnectedUnitLocationsConfig
from swarmbots.mjw_env.swarm.unit import get_unit_rest_min_z
from swarmbots.mjw_env.swarm.unit_config import (
    UNIT_CONFIG_CUBE_ZX,
    UNIT_CONFIG_TETRAHEDRON_XY,
    UNIT_CONFIG_TETRAHEDRON_XYZ,
    UNIT_CONFIG_TETRAHEDRON_ZX,
    UnitConfig,
)


def _lowest_geom_z(model: mujoco.MjModel, data: mujoco.MjData, geom_indices: np.ndarray) -> float:
    lowest_z = data.geom_xpos[geom_indices, 2] - model.geom_size[geom_indices, 0]
    capsule_mask = model.geom_type[geom_indices] == mujoco.mjtGeom.mjGEOM_CAPSULE
    capsule_indices = geom_indices[capsule_mask]
    lowest_z[capsule_mask] -= (
        np.abs(data.geom_xmat[capsule_indices, 8]) * model.geom_size[capsule_indices, 1]
    )
    return float(lowest_z.min())


def _active_geom_indices(model: mujoco.MjModel, active_mask: np.ndarray) -> np.ndarray:
    prefixes = tuple(f"Unit{index}-" for index in np.flatnonzero(active_mask))
    return np.array([
        index for index in range(model.ngeom)
        if model.body(int(model.geom_bodyid[index])).name.startswith(prefixes)
    ])


@pytest.mark.parametrize("unit_config", [
    UNIT_CONFIG_TETRAHEDRON_XY, UNIT_CONFIG_TETRAHEDRON_ZX, UNIT_CONFIG_TETRAHEDRON_XYZ, UNIT_CONFIG_CUBE_ZX,
])
@pytest.mark.parametrize("dimensions", [(0.1, 0.2, 0.025), (0.8, 0.01, 0.02), (0.05, 0.4, 0.08)])
@pytest.mark.parametrize("segment_1_ratio", [0.1, 0.65])
def test_analytical_unit_bottom_matches_mujoco_geometry(
    unit_config: UnitConfig, dimensions: tuple[float, float, float], segment_1_ratio: float,
) -> None:
    body_radius, leg_length, leg_radius = dimensions
    swarm = MJWHomogeneousSwarm(
        MJWPreConnectedUnitLocationsConfig(num_units=1, pool_seeds=(42000,)),
        unit_config=unit_config, body_radius=body_radius, leg_length=leg_length, leg_radius=leg_radius,
        segment_1_ratio=segment_1_ratio,
        hinge_range=(np.pi / 3,) * (3 if unit_config is UNIT_CONFIG_TETRAHEDRON_XYZ else 2),
        hinge_armature=0.0, hinge_damping=0.0, hinge_frictionloss=0.0,
    )
    model = swarm.create_swarm_spec().compile()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(42)
    quats = np.array([[1.0, 0.0, 0.0, 0.0], *[random_quat_shoemake(rng) for _ in range(12)]])
    rotation_matrices = np.empty((len(quats), 3, 3))
    actual_bottoms = []
    for index, quat in enumerate(quats):
        mujoco.mju_quat2Mat(rotation_matrices[index].reshape(9), quat)
        data.qpos[:3] = 0.0
        data.qpos[3:7] = quat
        mujoco.mj_forward(model, data)
        actual_bottoms.append(_lowest_geom_z(model, data, np.arange(model.ngeom)))
    analytical_bottoms = get_unit_rest_min_z(
        rotation_matrices, unit_config=unit_config, body_radius=body_radius,
        leg_length=leg_length, leg_radius=leg_radius,
    )
    np.testing.assert_allclose(analytical_bottoms, actual_bottoms, atol=1e-12, rtol=0)


@pytest.mark.parametrize("joints", ["xy", "zx", "xyz"])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("active_count", [1, 4, 5])
def test_pool_places_every_active_surface_above_ground(joints: str, center: bool, active_count: int) -> None:
    config = MJWPreConnectedUnitLocationsConfig(
        num_units=5, num_unit_probs=None if active_count == 5 else {active_count: 1.0, 5: 0.0}, center=center,
        pool_seeds=range(42000, 42010), unconnected_prob=0.5, ground_clearance=0.007,
    )
    swarm = make_scenario("SwarmBots-MoveToGoal-v0", joints=joints, unit_start_locations=config).swarm
    pool = swarm.build_pool(device=torch.device("cpu"))
    model = swarm.create_swarm_spec().compile()
    data = mujoco.MjData(model)
    root_addresses = [int(model.jnt_qposadr[model.body(f"Unit{index}--main_body").jntadr[0]]) for index in range(5)]
    for positions, quats, active_mask in zip(pool.positions, pool.quats, pool.active_mask, strict=True):
        np.testing.assert_array_equal(positions[~active_mask].numpy(), 0.0)
        if center:
            np.testing.assert_allclose(positions[active_mask, :2].mean(dim=0).numpy(), 0.0, atol=1e-7)
        else:
            np.testing.assert_array_equal(positions[0, :2].numpy(), 0.0)
        for index, address in enumerate(root_addresses):
            data.qpos[address:address + 3] = positions[index].numpy()
            data.qpos[address + 3:address + 7] = quats[index].numpy()
        mujoco.mj_forward(model, data)
        geom_indices = _active_geom_indices(model, active_mask.numpy())
        lowest_z = _lowest_geom_z(model, data, geom_indices)
        assert lowest_z > 0.0
        assert lowest_z == pytest.approx(config.ground_clearance, abs=2e-7)


@pytest.mark.parametrize("benchmark_id", ALL_BENCHMARK_IDS)
def test_presets_default_to_automatic_ground_placement(benchmark_id: str) -> None:
    scenario = make_scenario(benchmark_id, compile_reward_kernel=False)
    settings = serialize_settings(scenario.get_settings())["swarm"]["unit_start_locations"]
    assert settings["z_pos"] is None
    assert settings["ground_clearance"] == 0.001


@pytest.mark.parametrize("clearance", [-0.001, float("nan"), float("inf"), -float("inf")])
def test_invalid_ground_clearance_is_rejected(clearance: float) -> None:
    with pytest.raises(ValueError, match="ground_clearance"):
        MJWPreConnectedUnitLocationsConfig(num_units=1, pool_seeds=(42000,), ground_clearance=clearance)


def _assert_reset_height(env: MJWSwarmBotsVectorEnv, *, fixed_height: float | None) -> None:
    data = mujoco.MjData(env._host_model)
    for qpos, active_mask in zip(env._qpos.cpu().numpy(), env.units_active_mask.cpu().numpy(), strict=True):
        if fixed_height is None:
            data.qpos[:] = qpos
            mujoco.mj_forward(env._host_model, data)
            lowest_z = _lowest_geom_z(env._host_model, data, _active_geom_indices(env._host_model, active_mask))
            assert lowest_z > 0.0
            assert lowest_z == pytest.approx(0.001, abs=2e-7)
        else:
            root_z = qpos[env._metadata.unit_qpos_adr[active_mask] + 2]
            assert root_z.mean() == pytest.approx(fixed_height + env.scenario.swarm.max_unit_extent * 1.1)


@pytest.mark.integration
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("benchmark_id", ["SwarmBots-MoveToGoal-v0", "SwarmBots-Bridge-v0"])
@pytest.mark.parametrize("randomize_yaw", [False, True])
@pytest.mark.parametrize("fixed_height", [None, 3.0])
def test_raw_and_cpu_snapshot_resets_use_the_same_spawn_height(
    device: str, benchmark_id: str, randomize_yaw: bool, fixed_height: float | None,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    env = make_env(
        benchmark_id, num_envs=3, device=device, episode_length=1, compile_tensor_operations=False,
        scenario_kwargs=dict(
            unit_start_locations=MJWPreConnectedUnitLocationsConfig(
                num_units=3, num_unit_probs={1: 1.0, 3: 1.0}, pool_seeds=range(42000, 42005), z_pos=fixed_height,
            ),
            reset_settle_time=0.0, compile_reward_kernel=False,
            randomize_initial_swarm_z_rotation=randomize_yaw,
        ),
    )
    try:
        env.reset(seed=7)
        _assert_reset_height(env, fixed_height=fixed_height)
        previous_qpos = env._qpos.clone()
        env.reset(seed=8, options={"reset_mask": torch.tensor([True, False, True], device=env.device)})
        torch.testing.assert_close(env._qpos[1], previous_qpos[1])
        _assert_reset_height(env, fixed_height=fixed_height)

        runtime = env._scenario_runtime
        batch = runtime.sample_reset_batch(n_reset=3, rng=env._rng)
        specs = runtime.build_cpu_reset_specs(reset_batch=batch)
        snapshots = runtime.settle_cpu_reset_specs(specs=specs)
        world_indices = torch.arange(3, device=env.device)
        runtime.apply_reset_batch(world_idx=world_indices, reset_batch=batch)
        _assert_reset_height(env, fixed_height=fixed_height)
        np.testing.assert_allclose(
            env._qpos.cpu().numpy(), np.stack([snapshot.common.qpos for snapshot in snapshots]), atol=3e-7, rtol=1e-6,
        )
        runtime.apply_settled_reset_batch(world_idx=world_indices, snapshots=snapshots)
        _assert_reset_height(env, fixed_height=fixed_height)

        actions = {key: torch.zeros(space.shape, device=env.device) for key, space in env.action_space.items()}
        _, _, _, truncated, _ = env.step(actions)
        assert truncated.all()
        _assert_reset_height(env, fixed_height=fixed_height)
    finally:
        env.close()
