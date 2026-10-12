from __future__ import annotations

import mujoco
import numpy as np
import pytest

from swarmbots import ALL_BENCHMARK_IDS, make_scenario
from swarmbots.benchmark.metadata import serialize_settings
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig


def _scenario(*, joints: str, legacy: bool, num_units: int = 2):
    return make_scenario(
        "SwarmBots-MoveToGoal-v0",
        seed=42000,
        joints=joints,
        unit_start_locations=MJWPreConnectedUnitLocationsConfig(
            num_units, pool_seeds=[42000], z_pos=3.0,
        ),
        legacy_connector_positioning=legacy,
        compile_reward_kernel=False,
    )


@pytest.mark.parametrize("joints", ["xy", "zx", "xyz"])
@pytest.mark.parametrize("legacy", [False, True])
def test_connected_capsules_have_no_preload_unless_legacy_is_requested(joints, legacy):
    scenario = _scenario(joints=joints, legacy=legacy)
    swarm = scenario.swarm
    model = scenario.build_model()
    model.opt.gravity[:] = 0
    data = mujoco.MjData(model)
    positions, quats, partners, connectors, twists = swarm._generate_preconnected_sample(
        rng=np.random.default_rng(42000),
    )
    for unit_idx in range(swarm.num_units):
        root_body = model.body(f"Unit{unit_idx}--main_body")
        qpos_adr = int(model.jnt_qposadr[int(root_body.jntadr[0])])
        data.qpos[qpos_adr:qpos_adr + 3] = positions[unit_idx]
        data.qpos[qpos_adr + 3:qpos_adr + 7] = quats[unit_idx]
    connector = int(np.flatnonzero(partners[0] >= 0)[0])
    partner_connector = int(connectors[0, connector])
    name = swarm.config.get_eq_variant_name(
        0, connector, 1, partner_connector, int(twists[0, connector]),
    )
    data.eq_active[model.equality(name).id] = True
    mujoco.mj_forward(model, data)

    np.testing.assert_allclose(
        data.xpos[model.body(swarm.config.get_connector_name(0, connector)).id],
        data.xpos[model.body(swarm.config.get_connector_name(1, partner_connector)).id],
        atol=1e-12,
    )
    equality_rows = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    np.testing.assert_allclose(data.efc_pos[equality_rows], 0, atol=1e-12)
    distances = [float(data.contact[i].dist) for i in range(data.ncon)]
    if legacy:
        assert min(distances) == pytest.approx(-2 * swarm.leg_radius)
        assert np.linalg.norm(data.qacc) > 1
    else:
        assert all(distance >= -1e-12 for distance in distances)
        np.testing.assert_allclose(data.qacc, 0, atol=1e-8)


@pytest.mark.parametrize("joints", ["xy", "zx", "xyz"])
@pytest.mark.parametrize("articulated", [False, True])
def test_moving_connector_frames_preserves_robot_geometry_and_mass_matrix(joints, articulated):
    states = []
    for legacy in [False, True]:
        model = _scenario(joints=joints, legacy=legacy, num_units=1).build_model()
        data = mujoco.MjData(model)
        data.qpos[:3] = [0.0, 0.0, 3.0]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        if articulated:
            hinge_indices = model.jnt_qposadr[model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE]
            data.qpos[hinge_indices] = np.linspace(-0.25, 0.25, len(hinge_indices))
        mujoco.mj_forward(model, data)
        states.append((model, data))
    (new_model, new_data), (old_model, old_data) = states
    np.testing.assert_allclose(new_data.geom_xpos, old_data.geom_xpos, atol=1e-12)
    np.testing.assert_allclose(new_data.geom_xmat, old_data.geom_xmat, atol=1e-12)
    np.testing.assert_allclose(new_model.body_mass, old_model.body_mass, atol=1e-12)
    np.testing.assert_allclose(new_data.qM, old_data.qM, atol=1e-12)


@pytest.mark.parametrize("benchmark_id", ALL_BENCHMARK_IDS)
def test_connector_positioning_option_is_available_in_every_preset_and_metadata(benchmark_id):
    for legacy in [False, True]:
        scenario = make_scenario(
            benchmark_id, legacy_connector_positioning=legacy, compile_reward_kernel=False,
        )
        settings = serialize_settings(scenario.get_settings())["swarm"]
        assert settings["legacy_connector_positioning"] is legacy
        expected_distance = 0.3 if legacy else 0.325
        assert settings["connector_distance"] == pytest.approx(expected_distance)


def test_connector_positioning_override_rejects_conflicting_custom_swarm():
    swarm = _scenario(joints="zx", legacy=True).swarm
    assert make_scenario("SwarmBots-MoveToGoal-v0", swarm=swarm).swarm is swarm
    with pytest.raises(ValueError, match="conflicts"):
        make_scenario("SwarmBots-MoveToGoal-v0", swarm=swarm, legacy_connector_positioning=False)


@pytest.mark.parametrize("value", [None, 1, "false"])
def test_connector_positioning_override_requires_bool(value):
    with pytest.raises(TypeError, match="must be a bool"):
        make_scenario("SwarmBots-MoveToGoal-v0", legacy_connector_positioning=value)
