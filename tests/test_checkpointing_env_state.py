import unittest
from typing import Any
from unittest.mock import patch

import torch
from gymnasium import spaces

from swarmbots.learn.checkpointing import apply_env_state, capture_env_state, freeze_env_normalization, move_env_to_device
from swarmbots.learn.algos.base_algorithm import BaseAlgorithm
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.env_wrappers.torch_normalize_reward_wrapper import TorchNormalizeRewardWrapper
from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.torch_running_mean_std import TorchRunningMeanStd


class _DummyTorchEnv:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.obs_dtype = torch.float32
        self.reward_dtype = torch.float32
        self._n_envs = 2
        self.num_envs = 2
        self.n_agents = 3
        self.local_obs_dim = 4
        self.global_obs_dim = 2
        self.hidden_local_vars_dim = 1
        self.hidden_global_vars_dim = 1
        self.has_agent_mask = True
        self.observation_space = spaces.Dict({
            "local_obs": spaces.Box(-float("inf"), float("inf"), shape=(2, 3, 4), dtype=float),
            "global_obs": spaces.Box(-float("inf"), float("inf"), shape=(2, 2), dtype=float),
            "hidden_local_vars": spaces.Box(-float("inf"), float("inf"), shape=(2, 3, 1), dtype=float),
            "hidden_global_vars": spaces.Box(-float("inf"), float("inf"), shape=(2, 1), dtype=float),
            "agent_mask": spaces.MultiBinary((2, 3)),
        })
        self.action_space = VectorHybridActionSpace({
            "move": spaces.Box(-1.0, 1.0, shape=(2, 3, 2), dtype=float),
        })

    def set_device(self, device: torch.device | str) -> None:
        self.device = torch.device(device)


class _CheckpointLoadAlgorithm(BaseAlgorithm):
    def get_hyper_parameters(self) -> dict[str, Any]:
        return {}

    def _get_optimizer_state_dict(self) -> dict[str, Any]:
        return {}

    def _apply_optimizer_state_dict(
        self,
        state_dict: dict[str, Any],
        missing_keys: list[str],
        unexpected_keys: list[str],
    ) -> None:
        _ = state_dict, missing_keys, unexpected_keys

    def _apply_learning_rate(self, lr: Any) -> None:
        _ = lr

    def perform_iteration(self, *args: Any, **kwargs: Any) -> tuple[dict[str, Any], int]:
        _ = args, kwargs
        return {}, 0


def _set_rms(rms: TorchRunningMeanStd, *, offset: float) -> None:
    rms.mean = torch.arange(rms.mean.numel(), dtype=rms.mean.dtype, device=rms.mean.device).reshape_as(rms.mean) + offset
    rms.var = torch.arange(rms.var.numel(), dtype=rms.var.dtype, device=rms.var.device).reshape_as(rms.var) + offset + 10.0
    rms.count = torch.tensor(offset + 100.0, dtype=rms.count.dtype, device=rms.count.device)


class CheckpointEnvStateTests(unittest.TestCase):
    def test_load_aligns_compiled_checkpoint_keys_to_uncompiled_policy(self) -> None:
        policy = torch.nn.Linear(3, 2)
        expected_state = {
            key: value.detach().clone()
            for key, value in policy.state_dict().items()
        }
        compiled_state = {
            f"_orig_mod.{key}": value
            for key, value in expected_state.items()
        }
        with torch.no_grad():
            policy.weight.zero_()
            policy.bias.zero_()

        algorithm = _CheckpointLoadAlgorithm(policy=policy, env=object(), learning_rate=1e-3)
        checkpoint = {
            "policy_state_dict": compiled_state,
            "n_total_iterations": 7,
            "n_total_updates": 11,
            "n_total_timesteps": 100_000_768,
        }

        with patch(
            "swarmbots.learn.algos.base_algorithm.load_checkpoint",
            return_value=checkpoint,
        ):
            algorithm.load("unused.pt")

        for key, value in policy.state_dict().items():
            self.assertTrue(torch.equal(value, expected_state[key]))
        self.assertEqual(algorithm.n_total_iterations, 7)
        self.assertEqual(algorithm.n_total_updates, 11)
        self.assertEqual(algorithm.n_total_timesteps, 100_000_768)

    def test_torch_obs_norm_restores_legacy_featurewise_wrapper_state_by_obs_key(self) -> None:
        wrapper = TorchFeatureWiseObsNormWrapper(
            _DummyTorchEnv(),
            obs_key="local_obs",
            scalar_feature_indices=[0, 2],
            quaternion_indices=[],
        )
        assert wrapper.obs_rms is not None
        _set_rms(wrapper.obs_rms, offset=3.0)

        restored = TorchFeatureWiseObsNormWrapper(
            _DummyTorchEnv(),
            obs_key="local_obs",
            scalar_feature_indices=[0, 2],
            quaternion_indices=[],
        )
        assert restored.obs_rms is not None
        env_state = capture_env_state(wrapper)
        env_state[0]["wrapper_class"] = "FeatureWiseObsNormWrapper"

        apply_env_state(restored, env_state)

        self.assertTrue(torch.equal(restored.obs_rms.mean, wrapper.obs_rms.mean))
        self.assertTrue(torch.equal(restored.obs_rms.var, wrapper.obs_rms.var))
        self.assertTrue(torch.equal(restored.obs_rms.count, wrapper.obs_rms.count))

    def test_torch_reward_norm_restores_legacy_normalize_reward_state(self) -> None:
        wrapper = TorchNormalizeRewardWrapper(_DummyTorchEnv(), gamma=0.5)
        _set_rms(wrapper.return_rms, offset=7.0)
        restored = TorchNormalizeRewardWrapper(_DummyTorchEnv(), gamma=0.5)
        env_state = capture_env_state(wrapper)
        env_state[0]["wrapper_class"] = "NormalizeReward"

        apply_env_state(restored, env_state)

        self.assertTrue(torch.equal(restored.return_rms.mean, wrapper.return_rms.mean))
        self.assertTrue(torch.equal(restored.return_rms.var, wrapper.return_rms.var))
        self.assertTrue(torch.equal(restored.return_rms.count, wrapper.return_rms.count))

    def test_capture_env_state_snapshots_running_stats(self) -> None:
        wrapper = TorchNormalizeRewardWrapper(_DummyTorchEnv(), gamma=0.5)
        restored = TorchNormalizeRewardWrapper(_DummyTorchEnv(), gamma=0.5)
        _set_rms(wrapper.return_rms, offset=7.0)

        env_state = capture_env_state(wrapper)
        captured_mean = wrapper.return_rms.mean.clone()
        captured_var = wrapper.return_rms.var.clone()
        captured_count = wrapper.return_rms.count.clone()
        _set_rms(wrapper.return_rms, offset=99.0)

        apply_env_state(restored, env_state)

        self.assertTrue(torch.equal(restored.return_rms.mean, captured_mean))
        self.assertTrue(torch.equal(restored.return_rms.var, captured_var))
        self.assertTrue(torch.equal(restored.return_rms.count, captured_count))

    def test_apply_env_state_rejects_obs_key_mismatch_and_missing_wrapper(self) -> None:
        wrapper = TorchFeatureWiseObsNormWrapper(
            _DummyTorchEnv(),
            obs_key="local_obs",
            scalar_feature_indices=[0],
            quaternion_indices=[],
        )
        assert wrapper.obs_rms is not None

        with self.assertRaisesRegex(ValueError, "Obs keys not equal"):
            apply_env_state(
                wrapper,
                [{
                    "wrapper_class": "FeatureWiseObsNormWrapper",
                    "obs_key": "global_obs",
                    "obs_rms": TorchRunningMeanStd(shape=(1,)),
                }],
            )

        with self.assertRaisesRegex(ValueError, "No saved env_state entry"):
            apply_env_state(
                wrapper,
                [{
                    "wrapper_class": "TorchNormalizeRewardWrapper",
                    "return_rms": TorchRunningMeanStd(shape=()),
                }],
            )

    def test_freeze_and_move_env_walk_full_wrapper_chain(self) -> None:
        env = _DummyTorchEnv()
        reward_wrapper = TorchNormalizeRewardWrapper(env)
        obs_wrapper = TorchFeatureWiseObsNormWrapper(
            reward_wrapper,
            obs_key="local_obs",
            scalar_feature_indices=[0],
            quaternion_indices=[],
        )

        freeze_env_normalization(obs_wrapper)
        move_env_to_device(obs_wrapper, "cpu")

        self.assertFalse(obs_wrapper.update_running_mean)
        self.assertFalse(reward_wrapper.update_running_mean)
        self.assertEqual(obs_wrapper.device, torch.device("cpu"))
        self.assertEqual(reward_wrapper.device, torch.device("cpu"))
        self.assertEqual(env.device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
