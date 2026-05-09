import unittest

import numpy as np
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.testing_env import TestingSwarmBotsEnv
from swarmbots.learn.torch_running_mean_std import TorchRunningMeanStd


def _make_env(
    *,
    n_envs: int = 2,
    n_agents: int = 3,
    n_local_obs: int = 7,
    n_global_obs: int = 1,
) -> SwarmBotsLearnEnvWrapper:
    vector_env = SyncVectorEnv(
        [
            lambda: TestingSwarmBotsEnv(
                n_agents,
                n_local_obs,
                n_global_obs,
                actuators_dim=1,
                connectors_dim=1,
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


def _single_env_final_obs(local_obs: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "local_obs": local_obs.astype(np.float32),
        "global_obs": np.zeros((1,), dtype=np.float32),
        "hidden_local_vars": np.zeros((local_obs.shape[0], 0), dtype=np.float32),
        "hidden_global_vars": np.zeros((0,), dtype=np.float32),
    }


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

    def test_final_obs_is_normalized_without_updating_running_stats(self) -> None:
        env = _make_env(n_agents=2, n_local_obs=2)
        try:
            wrapper = TorchFeatureWiseObsNormWrapper(
                env,
                obs_key="local_obs",
                scalar_feature_indices=[0],
                quaternion_indices=[],
                eps=0.0,
            )
            wrapper.obs_rms = TorchRunningMeanStd(shape=(1,), initial_count=0.0)
            live_obs = _obs(
                local_obs=torch.tensor(
                    [
                        [[0.0, 0.0], [2.0, 0.0]],
                        [[4.0, 0.0], [6.0, 0.0]],
                    ],
                    dtype=torch.float32,
                )
            )
            wrapper.observations(live_obs)
            assert wrapper.obs_rms is not None
            count_before = wrapper.obs_rms.count.clone()

            final_obs = np.empty((2,), dtype=object)
            final_obs[0] = _single_env_final_obs(np.array([[13.0, 0.0], [-7.0, 0.0]], dtype=np.float32))
            final_obs[1] = _single_env_final_obs(np.array([[1000.0, 0.0], [1001.0, 0.0]], dtype=np.float32))
            transformed = wrapper._transform_infos(
                {
                    "final_obs": final_obs,
                    "_final_obs": np.array([True, False]),
                }
            )

            torch.testing.assert_close(wrapper.obs_rms.count, count_before)
            self.assertTrue(wrapper.update_running_mean)
            transformed_final_obs = transformed["final_obs"]
            self.assertIsInstance(transformed_final_obs[0]["local_obs"], torch.Tensor)
            self.assertIsInstance(transformed_final_obs[1]["local_obs"], np.ndarray)
            expected = torch.tensor(
                [[10.0 / np.sqrt(5.0), 0.0], [-10.0 / np.sqrt(5.0), 0.0]],
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
