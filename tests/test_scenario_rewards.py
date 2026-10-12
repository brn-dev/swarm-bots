from __future__ import annotations

import pytest
import torch

from swarmbots import ALL_BENCHMARK_IDS, make_env
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig


@pytest.mark.integration
@pytest.mark.parametrize("benchmark_id", ALL_BENCHMARK_IDS)
def test_connection_topology_has_no_direct_effect_on_task_rewards(benchmark_id: str) -> None:
    env = make_env(
        benchmark_id,
        num_envs=1,
        device="cpu",
        seed=7,
        compile_tensor_operations=False,
        scenario_kwargs={
            "unit_start_locations": MJWPreConnectedUnitLocationsConfig(
                num_units=2, unconnected_prob=1.0, pool_seeds=(42000,), z_pos=0.5,
            ),
            "reset_settle_time": 0.0,
            "compile_reward_kernel": False,
        },
    )
    try:
        env.reset(seed=7)
        assert (env.partner_unit < 0).all()
        stable_mask = torch.ones(1, dtype=torch.bool)
        isolated_result = env._scenario_runtime.compute_step_rewards(stable_mask=stable_mask)
        isolated_reward = isolated_result.reward.clone()
        isolated_terms = {key: value.clone() for key, value in isolated_result.info["reward_terms"].items()}

        env.reset(seed=7)
        env.partner_unit[0, 0, 0] = 1
        env.partner_unit[0, 1, 0] = 0
        connected_result = env._scenario_runtime.compute_step_rewards(stable_mask=stable_mask)

        torch.testing.assert_close(connected_result.reward, isolated_reward)
        assert connected_result.info["reward_terms"].keys() == isolated_terms.keys()
        for key, isolated_term in isolated_terms.items():
            torch.testing.assert_close(connected_result.info["reward_terms"][key], isolated_term)
    finally:
        env.close()
