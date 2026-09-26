from __future__ import annotations

import pytest
import torch

from swarmbots import make_env


@pytest.mark.integration
def test_cpu_environment_reset_step_and_same_step_autoreset() -> None:
    env = make_env(
        "SwarmBots-MoveTo-v0",
        num_envs=2,
        device="cpu",
        seed=7,
        episode_length=1,
        scenario_kwargs={"compile_reward_kernel": False, "reset_settle_time": 0.0},
        compile_tensor_operations=False,
    )
    try:
        observations, _ = env.reset(seed=7)
        actions = {
            "actuators": torch.zeros(env.action_space["actuators"].shape),
            "connectors": torch.zeros(env.action_space["connectors"].shape),
        }

        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        assert observations["local_obs"].device.type == "cpu"
        assert next_observations["local_obs"].shape == observations["local_obs"].shape
        assert rewards.shape == (2,)
        assert not terminations.any()
        assert truncations.all()
        assert infos["_final_obs"].all()
        assert set(infos["final_obs"]) == set(observations)
    finally:
        env.close()
