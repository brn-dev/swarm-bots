from __future__ import annotations

import mujoco
import numpy as np
import pytest

from swarmbots import ALL_BENCHMARK_IDS, make_scenario
from swarmbots.benchmark.metadata import serialize_settings
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig
from swarmbots.mjw_env.scenarios.mjw_move_to_goal_scenario import MJWMoveToGoalScenario


def _scenario(*, joints: str, num_units: int = 2) -> MJWMoveToGoalScenario:
    return make_scenario(
        "SwarmBots-MoveToGoal-v0",
        seed=42000,
        joints=joints,
        unit_start_locations=MJWPreConnectedUnitLocationsConfig(
            num_units, pool_seeds=[42000], z_pos=3.0,
        ),
        compile_reward_kernel=False,
    )


@pytest.mark.parametrize("joints", ["xy", "zx", "xyz"])
def test_connected_capsules_have_no_contact_preload(joints: str) -> None:
    scenario = _scenario(joints=joints)
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
    assert all(distance >= -1e-12 for distance in distances)
    np.testing.assert_allclose(data.qacc, 0, atol=1e-8)


@pytest.mark.parametrize("benchmark_id", ALL_BENCHMARK_IDS)
def test_every_preset_uses_surface_connector_spacing(benchmark_id: str) -> None:
    scenario = make_scenario(benchmark_id, compile_reward_kernel=False)
    settings = serialize_settings(scenario.get_settings())["swarm"]
    assert settings["connector_distance"] == pytest.approx(0.325)


def test_custom_swarm_is_preserved() -> None:
    swarm = _scenario(joints="zx").swarm
    assert make_scenario("SwarmBots-MoveToGoal-v0", swarm=swarm).swarm is swarm
