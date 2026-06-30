import unittest

import numpy as np
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.testing_env import TestingSwarmBotsEnv
from swarmbots.learn.torch_running_mean_std import TorchRunningMeanStd


class _OffsetLocalObsTestingEnv(TestingSwarmBotsEnv):
    def __init__(self, *args, local_obs_offset: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._local_obs_offset = float(local_obs_offset)

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs = super()._get_obs()
        obs["local_obs"] = obs["local_obs"] + self._local_obs_offset
        return obs


def _make_env(
    *,
    n_envs: int = 2,
    n_agents: int = 3,
    n_local_obs: int = 7,
    n_global_obs: int = 1,
    max_steps: int = 100,
    local_obs_offset: float = 0.0,
) -> SwarmBotsLearnEnvWrapper:
    env_cls = TestingSwarmBotsEnv if local_obs_offset == 0.0 else _OffsetLocalObsTestingEnv
    vector_env = SyncVectorEnv(
        [
            lambda: env_cls(
                n_agents,
                n_local_obs,
                n_global_obs,
                actuators_dim=1,
                connectors_dim=1,
                max_steps=max_steps,
                **({} if local_obs_offset == 0.0 else {"local_obs_offset": local_obs_offset}),
            )
            for _ in range(n_envs)
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return SwarmBotsLearnEnvWrapper(vector_env)


def _obs(
    *,
    local_obs: torch.Tensor,
    agent_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    observations = {
        "local_obs": local_obs,
        "global_obs": torch.zeros((local_obs.shape[0], 1), dtype=local_obs.dtype),
        "hidden_local_vars": torch.zeros((local_obs.shape[0], local_obs.shape[1], 0), dtype=local_obs.dtype),
        "hidden_global_vars": torch.zeros((local_obs.shape[0], 0), dtype=local_obs.dtype),
    }
    if agent_mask is not None:
        observations["agent_mask"] = agent_mask
    return observations


class TorchFeatureWiseObsNormWrapperTests(unittest.TestCase):
    def test_agent_mask_controls_running_stats_and_observations_are_copy_on_write(self) -> None:
        env = _make_env()
        try:
            wrapper = TorchFeatureWiseObsNormWrapper(
                env,
                obs_key="local_obs",
                scalar_feature_indices=[0, 2],
                quaternion_indices=[3],
                eps=0.0,
            )
            wrapper.obs_rms = TorchRunningMeanStd(shape=(2,), initial_count=0.0)
            local_obs = torch.tensor(
                [
                    [
                        [1.0, 0.0, 10.0, -1.0, 2.0, 3.0, 4.0],
                        [100.0, 0.0, 1000.0, 1.0, 2.0, 3.0, 4.0],
                        [3.0, 0.0, 30.0, -0.5, -1.0, -2.0, -3.0],
                    ],
                    [
                        [200.0, 0.0, 2000.0, 2.0, 0.0, 0.0, 0.0],
                        [5.0, 0.0, 50.0, -2.0, 1.0, 0.0, 0.0],
                        [300.0, 0.0, 3000.0, -3.0, 0.0, 1.0, 0.0],
                    ],
                ],
                dtype=torch.float32,
            )
            original_local_obs = local_obs.clone()
            agent_mask = torch.tensor([[True, False, True], [False, True, False]])
            observations = _obs(local_obs=local_obs, agent_mask=agent_mask)

            normalized = wrapper.observations(observations)

            assert wrapper.obs_rms is not None
            torch.testing.assert_close(wrapper.obs_rms.mean, torch.tensor([3.0, 30.0], dtype=torch.float64))
            torch.testing.assert_close(
                wrapper.obs_rms.var,
                torch.tensor([8.0 / 3.0, 800.0 / 3.0], dtype=torch.float64),
            )
            self.assertEqual(float(wrapper.obs_rms.count.item()), 3.0)
            torch.testing.assert_close(observations["local_obs"], original_local_obs)
            self.assertIsNot(normalized["local_obs"], observations["local_obs"])
            self.assertIs(normalized["global_obs"], observations["global_obs"])

            expected = original_local_obs.clone()
            scalar_mean = torch.tensor([3.0, 30.0], dtype=torch.float32)
            scalar_std = torch.sqrt(torch.tensor([8.0 / 3.0, 800.0 / 3.0], dtype=torch.float32))
            expected[..., [0, 2]] = (expected[..., [0, 2]] - scalar_mean) / scalar_std
            expected[..., 3:7] = torch.where(
                expected[..., 3:4] < 0.0,
                -expected[..., 3:7],
                expected[..., 3:7],
            )
            torch.testing.assert_close(normalized["local_obs"], expected)
        finally:
            env.close()

    def test_all_inactive_agent_mask_skips_running_stats_update_but_normalizes_observations(self) -> None:
        env = _make_env(n_envs=1, n_agents=2, n_local_obs=3)
        try:
            wrapper = TorchFeatureWiseObsNormWrapper(
                env,
                obs_key="local_obs",
                scalar_feature_indices=[0, 2],
                quaternion_indices=[],
                eps=0.0,
            )
            wrapper.obs_rms = TorchRunningMeanStd(shape=(2,), initial_count=7.0)
            wrapper.obs_rms.mean = torch.tensor([10.0, 100.0], dtype=torch.float64)
            wrapper.obs_rms.var = torch.tensor([4.0, 25.0], dtype=torch.float64)
            local_obs = torch.tensor(
                [
                    [
                        [12.0, 1.0, 110.0],
                        [8.0, 2.0, 95.0],
                    ]
                ],
                dtype=torch.float32,
            )
            observations = _obs(
                local_obs=local_obs,
                agent_mask=torch.tensor([[False, False]], dtype=torch.bool),
            )

            normalized = wrapper.observations(observations)

            assert wrapper.obs_rms is not None
            torch.testing.assert_close(wrapper.obs_rms.count, torch.tensor(7.0, dtype=torch.float64))
            torch.testing.assert_close(wrapper.obs_rms.mean, torch.tensor([10.0, 100.0], dtype=torch.float64))
            torch.testing.assert_close(wrapper.obs_rms.var, torch.tensor([4.0, 25.0], dtype=torch.float64))
            expected = local_obs.clone()
            expected[..., [0, 2]] = torch.tensor([[[1.0, 2.0], [-1.0, -1.0]]])
            torch.testing.assert_close(normalized["local_obs"], expected)
        finally:
            env.close()

    def test_same_step_final_obs_is_normalized_without_updating_running_stats(self) -> None:
        env = _make_env(n_envs=1, n_agents=2, n_local_obs=2, max_steps=1, local_obs_offset=2.0)
        try:
            wrapper = TorchFeatureWiseObsNormWrapper(
                env,
                obs_key="local_obs",
                scalar_feature_indices=[0],
                quaternion_indices=[],
                eps=1.0,
            )
            wrapper.obs_rms = TorchRunningMeanStd(shape=(1,), initial_count=0.0)

            reset_obs, _info = wrapper.reset()
            assert wrapper.obs_rms is not None
            torch.testing.assert_close(wrapper.obs_rms.count, torch.tensor(2.0, dtype=torch.float64))

            actions = torch.zeros(
                (1, wrapper.n_agents, wrapper.action_space.total_agent_action_dim),
                dtype=torch.float32,
            )
            same_step_reset_obs, _rewards, _terminations, truncations, infos = wrapper.step(actions)

            self.assertTrue(bool(truncations[0].item()))
            expected_reset_obs = torch.tensor([[[0.0, 2.0], [0.0, 2.0]]], dtype=torch.float32)
            torch.testing.assert_close(reset_obs["local_obs"], expected_reset_obs)
            torch.testing.assert_close(same_step_reset_obs["local_obs"], expected_reset_obs)
            torch.testing.assert_close(wrapper.obs_rms.count, torch.tensor(4.0, dtype=torch.float64))
            self.assertTrue(wrapper.update_running_mean)
            transformed_final_obs = infos["final_obs"]
            self.assertIsInstance(transformed_final_obs[0]["local_obs"], torch.Tensor)
            expected = torch.tensor(
                [[1.0, 3.0], [1.0, 3.0]],
                dtype=torch.float32,
            )
            torch.testing.assert_close(transformed_final_obs[0]["local_obs"], expected)
        finally:
            env.close()

    def test_empty_scalar_and_quaternion_indices_return_original_observations(self) -> None:
        env = _make_env()
        try:
            wrapper = TorchFeatureWiseObsNormWrapper(
                env,
                obs_key="local_obs",
                scalar_feature_indices=[],
                quaternion_indices=[],
            )
            observations = _obs(local_obs=torch.zeros((2, 3, 7), dtype=torch.float32))

            normalized = wrapper.observations(observations)

            self.assertIs(normalized, observations)
            self.assertIsNone(wrapper.obs_rms)
        finally:
            env.close()

    def test_rejects_out_of_bounds_and_overlapping_indices(self) -> None:
        env = _make_env(n_local_obs=4)
        try:
            with self.assertRaisesRegex(ValueError, "scalar_feature_indices out of bounds"):
                TorchFeatureWiseObsNormWrapper(
                    env,
                    obs_key="local_obs",
                    scalar_feature_indices=[4],
                    quaternion_indices=[],
                )
            with self.assertRaisesRegex(ValueError, "quaternion_indices out of bounds"):
                TorchFeatureWiseObsNormWrapper(
                    env,
                    obs_key="local_obs",
                    scalar_feature_indices=[],
                    quaternion_indices=[1],
                )
            with self.assertRaisesRegex(ValueError, "overlap"):
                TorchFeatureWiseObsNormWrapper(
                    env,
                    obs_key="local_obs",
                    scalar_feature_indices=[2],
                    quaternion_indices=[0],
                )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
