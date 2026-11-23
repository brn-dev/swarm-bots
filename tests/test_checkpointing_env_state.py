import unittest
import os
from copy import deepcopy
from functools import partial
from pathlib import Path
from stat import S_IMODE
from tempfile import TemporaryDirectory
from typing import Any
import torch
import pytest
from gymnasium import spaces

from swarmbots.learn.checkpointing import (
    apply_env_state,
    capture_env_state,
    freeze_env_normalization,
    move_env_to_device,
)
from swarmbots.learn.algos.base_algorithm import BaseAlgorithm
from swarmbots.learn.algos.ppo.ppo import PPO, StepsRolloutMode
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.env_wrappers.torch_normalize_reward_wrapper import TorchNormalizeRewardWrapper
from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.scheduling.schedulers import ScheduledHyperParameter, SchedulerManager
from swarmbots.learn.torch_running_mean_std import TorchRunningMeanStd
from tests.test_base_algorithm_lifecycle import _Policy
from tests.test_ppo_parameter_lr_multipliers import _DummyEnv, _DummyPolicy, _make_ppo


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


@pytest.mark.parametrize(("source_dtype", "target_dtype", "value", "valid"), [
    (torch.float8_e4m3fn, torch.float8_e4m3fn, 7.0, True),
    (torch.float32, torch.float8_e4m3fn, 512.0, False),
    (torch.float64, torch.float64, 1e80, True),
])
def test_checkpoint_validates_frozen_parameter_in_destination_precision(
    tmp_path: Path, source_dtype: torch.dtype, target_dtype: torch.dtype, value: float, valid: bool,
) -> None:
    source_policy, target_policy = _Policy(), _Policy()
    source_policy.frozen_weight = torch.nn.Parameter(
        torch.tensor([value], dtype=source_dtype), requires_grad=False,
    )
    target_policy.frozen_weight = torch.nn.Parameter(
        torch.tensor([1.0], dtype=target_dtype), requires_grad=False,
    )
    source = _CheckpointLoadAlgorithm(policy=source_policy, env=object(), learning_rate=1e-3)
    target = _CheckpointLoadAlgorithm(policy=target_policy, env=object(), learning_rate=1e-3)
    path = tmp_path / "frozen.pt"
    source.save(path)

    if valid:
        target.load(path, restore_optimizer_state=False)
        assert target_policy.frozen_weight.dtype == target_dtype
        assert target_policy.frozen_weight.double().item() == value
    else:
        with pytest.raises(ValueError, match="Non-finite checkpoint parameter frozen_weight"):
            target.load(path, restore_optimizer_state=False)
        assert target_policy.frozen_weight.double().item() == 1.0


def test_checkpoint_round_trips_module_extra_state_and_identifies_retry_candidate(tmp_path: Path) -> None:
    class ModuleWithMetadata(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.metadata: dict[str, Any] = {}

        def get_extra_state(self) -> dict[str, Any]:
            return self.metadata

        def set_extra_state(self, state: dict[str, Any]) -> None:
            self.metadata = state

    policy = _Policy()
    policy.custom = ModuleWithMetadata()
    trainer = _CheckpointLoadAlgorithm(policy=policy, env=object(), learning_rate=1e-3)
    policy.custom.metadata = {"mode": "sample", "levels": torch.tensor([0.25, 0.75])}
    digest = trainer._current_best_candidate()
    assert digest is not None
    path = tmp_path / "model.pt"
    trainer.save(path)
    policy.custom.metadata = {"mode": "fixed"}
    assert trainer._current_best_candidate() != digest
    trainer.load(path)
    assert policy.custom.metadata["mode"] == "sample"
    torch.testing.assert_close(policy.custom.metadata["levels"], torch.tensor([0.25, 0.75]))
    assert trainer._current_best_candidate() == digest


class _LazyEncoderPolicy(_DummyPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.shared = torch.nn.Sequential(torch.nn.LazyLinear(3), torch.nn.LazyBatchNorm1d())

    def num_parameters(self, learnable_only: bool = True) -> int:
        # PPO requests the declared model budget before the encoder receives its first three-feature input.
        encoder_parameters = 3 * 3 + 3 + 2 * 3
        return encoder_parameters + sum(
            param.numel() for name, param in self.named_parameters() if not name.startswith("shared.")
        )


def _make_lazy_ppo() -> PPO:
    policy = _LazyEncoderPolicy()
    return PPO(policy, _DummyEnv(), rollout_mode=StepsRolloutMode(2), train_device="cpu", rollout_device="cpu")


def _step_lazy_encoder(trainer: PPO) -> None:
    batch = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    trainer.policy.shared(batch).square().mean().backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)


def test_lazy_ppo_restores_parameters_buffers_and_resumes_adam(tmp_path: Path) -> None:
    source = _make_lazy_ppo()
    path = tmp_path / "lazy.pt"
    source.save(path, optimizer_state_dict=source._get_optimizer_state_dict())
    target = _make_lazy_ppo()
    weight = target.policy.shared[0].weight
    target.load(path)
    assert torch.nn.parameter.is_lazy(weight)
    assert target._current_best_candidate() == source._current_best_candidate() is not None

    _step_lazy_encoder(source)
    source.save(path, optimizer_state_dict=source._get_optimizer_state_dict())
    target.load(path)
    assert target.policy.shared[0].weight is weight and weight in target.optimizer.state
    torch.testing.assert_close(target.policy.state_dict(), source.policy.state_dict())
    torch.testing.assert_close(target.optimizer.state_dict(), source.optimizer.state_dict())
    for trainer in (source, target):
        _step_lazy_encoder(trainer)
    torch.testing.assert_close(target.policy.state_dict(), source.policy.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(target.optimizer.state_dict(), source.optimizer.state_dict(), rtol=0, atol=0)


def test_lazy_ppo_rejects_wrong_adam_shape_before_materialization(tmp_path: Path) -> None:
    source = _make_lazy_ppo()
    _step_lazy_encoder(source)
    state = source._get_optimizer_state_dict()
    slots = next(iter(state["state"].values()))
    slots["exp_avg"] = torch.ones(1)
    path = tmp_path / "bad.pt"
    source.save(path, optimizer_state_dict=state)
    target = _make_lazy_ppo()
    with pytest.raises(ValueError, match="Invalid checkpoint Adam moment exp_avg"):
        target.load(path)
    assert torch.nn.parameter.is_lazy(target.policy.shared[0].weight)
    assert torch.nn.parameter.is_lazy(target.policy.shared[1].running_mean)
    assert not target.optimizer.state


def test_rejected_extra_state_keeps_lazy_encoder_uninitialized(tmp_path: Path) -> None:
    class Versioned(torch.nn.Module):
        def get_extra_state(self) -> int:
            return 1

        def set_extra_state(self, state: int) -> None:
            if state != 1:
                raise ValueError("Unsupported version")

    source, target = _make_lazy_ppo(), _make_lazy_ppo()
    source.policy.versioned = Versioned()
    target.policy.versioned = Versioned()
    weight = target.policy.shared[0].weight
    batch_norm = target.policy.shared[1]
    _step_lazy_encoder(source)
    path = tmp_path / "versioned_lazy.pt"
    source.save(path, optimizer_state_dict=source._get_optimizer_state_dict())
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["policy_state_dict"]["versioned._extra_state"] = 2
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="Unsupported version"):
        target.load(path)
    assert target.policy.shared[0].weight is weight and torch.nn.parameter.is_lazy(weight)
    assert torch.nn.parameter.is_lazy(batch_norm.running_mean) and batch_norm.num_features == 0
    assert target.policy.shared[0].in_features == 0
    assert not target.optimizer.state
    _step_lazy_encoder(target)
    assert weight in target.optimizer.state


def test_checkpoint_publishes_near_filename_limit(tmp_path: Path) -> None:
    path = tmp_path / ("x" * 247 + ".pt")
    sidecar = path.with_name(path.name + ".json")
    path.touch()
    sidecar.touch()
    trainer = _CheckpointLoadAlgorithm(policy=_Policy(), env=object(), learning_rate=1e-3)
    trainer.save(path)
    trainer.load(path)
    assert sidecar.stat().st_size > 0


def test_rejected_module_extra_state_restores_policy_and_schedulers(tmp_path: Path) -> None:
    class VersionedModule(torch.nn.Module):
        def __init__(self, version: int) -> None:
            super().__init__()
            self.register_buffer("schema", torch.tensor(version))
            self.metadata = {"version": version}

        def get_extra_state(self) -> dict[str, int]:
            return self.metadata

        def set_extra_state(self, state: dict[str, int]) -> None:
            self.metadata.update(state)
            if self.metadata["version"] != self.schema.item():
                raise ValueError("Metadata version does not match the loaded schema")

    source, _ = _make_ppo()
    target, _ = _make_ppo()
    for trainer, version in ((source, 2), (target, 1)):
        trainer.policy.custom = VersionedModule(version)
        trainer.gamma = 0.4 * version
        trainer.n_total_timesteps = 10 * version
        trainer.scheduler_manager = SchedulerManager([ScheduledHyperParameter(
            name="gamma", scheduler=None, get_value=partial(getattr, trainer, "gamma"),
            apply=partial(setattr, trainer, "gamma"), state={"version": version}, enabled=version == 2,
        )])

    path = tmp_path / "versioned.pt"
    source.save(path, optimizer_state_dict=source._get_optimizer_state_dict())
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["policy_state_dict"]["custom._extra_state"]["version"] = 3
    bad_path = tmp_path / "rejected.pt"
    torch.save(checkpoint, bad_path)
    previous_policy = deepcopy(target.policy.state_dict())
    previous_schedule = target.scheduler_manager.state_dict()

    with pytest.raises(ValueError, match="loaded schema"):
        target.load(bad_path)
    torch.testing.assert_close(target.policy.state_dict(), previous_policy)
    assert target.scheduler_manager.state_dict() == previous_schedule
    assert target.n_total_timesteps == 10

    target.load(path)
    torch.testing.assert_close(target.policy.state_dict(), source.policy.state_dict())
    assert target.scheduler_manager.state_dict() == source.scheduler_manager.state_dict()
    assert target.n_total_timesteps == 20


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_checkpoint_publication_respects_creation_and_existing_modes(tmp_path: Path) -> None:
    trainer = _CheckpointLoadAlgorithm(policy=_Policy(), env=object(), learning_rate=1e-3)
    path = tmp_path / "model.pt"
    sidecar = path.with_suffix(".pt.json")
    control = tmp_path / "ordinary"
    control.touch()
    trainer.save(path)
    assert S_IMODE(path.stat().st_mode) == S_IMODE(sidecar.stat().st_mode) == S_IMODE(control.stat().st_mode)
    path.chmod(0o660)
    sidecar.chmod(0o640)
    trainer.save(path)
    assert S_IMODE(path.stat().st_mode) == 0o660
    assert S_IMODE(sidecar.stat().st_mode) == 0o640


class CheckpointEnvStateTests(unittest.TestCase):
    def test_load_aligns_compiled_checkpoint_keys_to_uncompiled_policy(self) -> None:
        policy = _Policy()
        policy.register_buffer("bias", torch.ones(2))
        with torch.no_grad():
            policy.weight.fill_(3.0)
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
            "policy_loss_weights": {},
            "algorithm_state_dict": {},
            "rollout_warmup_done": False,
            "pending_best_digest": None,
            "n_total_iterations": 7,
            "n_total_updates": 11,
            "n_total_timesteps": 100_000_768,
            "best_return_ema": None,
            "env_state": [],
            "episode_return_ema": algorithm._episode_return_ema.state_dict(),
            "episode_success_rate_ema": algorithm._episode_success_rate_ema.state_dict(),
        }

        with TemporaryDirectory() as directory:
            path = Path(directory) / "compiled.pt"
            torch.save(checkpoint, path)
            algorithm.load(path)

        for key, value in policy.state_dict().items():
            self.assertTrue(torch.equal(value, expected_state[key]))
        self.assertEqual(algorithm.n_total_iterations, 7)
        self.assertEqual(algorithm.n_total_updates, 11)
        self.assertEqual(algorithm.n_total_timesteps, 100_000_768)

    def test_torch_obs_norm_restores_state_by_obs_key(self) -> None:
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

        apply_env_state(restored, env_state)

        self.assertTrue(torch.equal(restored.obs_rms.mean, wrapper.obs_rms.mean))
        self.assertTrue(torch.equal(restored.obs_rms.var, wrapper.obs_rms.var))
        self.assertTrue(torch.equal(restored.obs_rms.count, wrapper.obs_rms.count))

    def test_torch_reward_norm_restores_state(self) -> None:
        wrapper = TorchNormalizeRewardWrapper(_DummyTorchEnv(), gamma=0.5)
        _set_rms(wrapper.return_rms, offset=7.0)
        restored = TorchNormalizeRewardWrapper(_DummyTorchEnv(), gamma=0.5)
        env_state = capture_env_state(wrapper)

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

    def test_rejected_inner_stats_preserve_all_wrappers(self) -> None:
        reward = TorchNormalizeRewardWrapper(_DummyTorchEnv())
        wrapper = TorchFeatureWiseObsNormWrapper(reward, obs_key="local_obs", scalar_feature_indices=[0],
                                                  quaternion_indices=[])
        states = capture_env_state(wrapper)
        original = wrapper.obs_rms.mean.clone()
        states[0]["obs_rms"].mean.fill_(9)
        states[1]["return_rms"].count = torch.ones(3)
        with self.assertRaisesRegex(ValueError, "count shape"):
            apply_env_state(wrapper, states)
        torch.testing.assert_close(wrapper.obs_rms.mean, original)

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
                    "wrapper_class": "TorchFeatureWiseObsNormWrapper",
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
