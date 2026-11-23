from __future__ import annotations

from collections.abc import Callable
from threading import Event, Lock, RLock
from types import MethodType
from typing import Any

import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from torch import nn

from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import SquashedDiagGaussianConfig
from swarmbots.learn.algos.ppo.ppo_policy import PPOActorConfig, PPOPolicy, PPOPolicyConfig
from swarmbots.learn.benchmark_policy import BenchmarkPolicy
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_tensor_ops import NormalizedObservations
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.env_wrappers.torch_transition_obs_wrapper import TorchTransitionObsWrapper
from swarmbots.learn.testing_env import TestingSwarmBotsEnv as FixtureSwarmEnv


class _MaskedEnv(FixtureSwarmEnv):
    def __init__(self, max_steps):
        super().__init__(n_agents=1, n_local_obs=1, n_global_obs=1,
                         actuators_dim=1, connectors_dim=1, max_steps=max_steps)
        self.observation_space = spaces.Dict({**self.observation_space.spaces, "agent_mask": spaces.MultiBinary(1)})

    def _get_obs(self):
        return {**super()._get_obs(), "agent_mask": np.ones(1, dtype=np.bool_)}


class _Actor(nn.Module):
    gsde_enabled = False

    def requires_previous_actions(self):
        return False

    def initial_temporal_state(self, *args, **kwargs):
        return None

    def act_with_temporal_state(self, *, local_obs, global_obs, agent_mask, **kwargs):
        assert "hidden_local_vars" not in kwargs and "hidden_global_vars" not in kwargs
        self.seen = {"local_obs": local_obs.clone(), "global_obs": global_obs.clone()}
        assert local_obs.shape[-1] == 4 and global_obs.shape[-1] == 2
        return local_obs.new_tensor([0.25, 1.]).expand(2, 1, 2).clone(), None


@pytest.mark.parametrize("new_obs_first", [False, True])
@pytest.mark.parametrize("normalize_binary", [False, True])
@pytest.mark.parametrize("outer_normalizer", [False, True])
def test_benchmark_transition_inputs_match_training_and_do_not_change_training_history(
    new_obs_first, normalize_binary, outer_normalizer,
):
    vector = SyncVectorEnv([lambda: _MaskedEnv(1), lambda: _MaskedEnv(3)], autoreset_mode=AutoresetMode.SAME_STEP)
    env = SwarmBotsLearnEnvWrapper(vector)
    for key in ("local_obs", "global_obs"):
        env = TorchFeatureWiseObsNormWrapper(env, key, [0], [], compile_tensor_operations=False)
        env.update_running_mean = False
        env.obs_rms.mean.fill_(2.)
        env.obs_rms.var.fill_(4.)
    env = TorchTransitionObsWrapper(
        env, new_obs_first=new_obs_first, normalize_prev_binary_actions=normalize_binary,
        compile_tensor_operations=False,
    )
    transition_wrapper = env
    if outer_normalizer:
        env = TorchFeatureWiseObsNormWrapper(env, "local_obs", [0, 3], [], compile_tensor_operations=False)
        env.update_running_mean = False
        env.obs_rms.mean.fill_(1.)
        env.obs_rms.var.fill_(4.)
    try:
        expected, _ = env.reset(seed=7)
        actor = _Actor()
        adapter = BenchmarkPolicy(actor, env)
        starts = torch.ones(2, dtype=torch.bool)
        for _ in range(4):
            raw = {key: torch.as_tensor(value).clone() for key, value in vector._observations.items()}
            history = transition_wrapper._prev_local.clone()
            previous_features = transition_wrapper._current_prev_actions.clone()
            adapter(raw, starts)
            for key in actor.seen:
                torch.testing.assert_close(actor.seen[key], expected[key])
            torch.testing.assert_close(transition_wrapper._prev_local, history)
            torch.testing.assert_close(transition_wrapper._current_prev_actions, previous_features)
            packed = torch.tensor([0.25, 1.]).expand(2, 1, 2).clone()
            expected, _, terminated, truncated, _ = env.step(packed)
            starts = terminated | truncated
    finally:
        env.close()


_CUDA = (pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))


@pytest.mark.parametrize("actor_device", ["cpu", pytest.param("cuda", marks=_CUDA)])
@pytest.mark.parametrize("instance_override", [False, True])
@pytest.mark.parametrize("lock_factory", [Lock, RLock], ids=["lock", "rlock"])
@torch.inference_mode()
def test_benchmark_preserves_custom_normalizer_without_touching_training_state(
    actor_device: str, instance_override: bool, lock_factory: Callable[[], Any],
) -> None:
    lock_type = type(lock_factory())

    def offset_observations(
        normalizer: TorchFeatureWiseObsNormWrapper, observations: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        assert type(normalizer.lock) is lock_type
        if lock_factory is RLock:
            assert not normalizer.lock._is_owned()
        assert normalizer.lock.acquire(blocking=False)
        try:
            normalized = TorchFeatureWiseObsNormWrapper.observations(normalizer, observations)
            normalizer.calls.append(normalizer.update_running_mean)
            return {**normalized, "local_obs": normalized["local_obs"] + normalizer.offset}
        finally:
            normalizer.lock.release()

    class OffsetNormalizer(TorchFeatureWiseObsNormWrapper):
        __slots__ = ("offset", "lock")

        observations = offset_observations

    vector = SyncVectorEnv([lambda: _MaskedEnv(3)] * 2, autoreset_mode=AutoresetMode.SAME_STEP)
    wrapper_type = TorchFeatureWiseObsNormWrapper if instance_override else OffsetNormalizer
    normalizer = wrapper_type(
        SwarmBotsLearnEnvWrapper(vector), "local_obs", [0], [], compile_tensor_operations=False,
    )
    if instance_override:
        normalizer.observations = MethodType(offset_observations, normalizer)
    with torch.inference_mode(False):
        offset_source = nn.Parameter(torch.tensor([7.0]))
        normalizer.offset = offset_source[0]
    normalizer.lock = lock_factory()
    normalizer.calls = []
    env = TorchTransitionObsWrapper(normalizer, compile_tensor_operations=False)
    try:
        wrapped, _ = env.reset()
        actor = _Actor()
        actor.register_buffer("_placement", torch.empty(0, device=actor_device))
        adapter = BenchmarkPolicy(actor, env)
        full_obs = vector._observations
        raw = {key: torch.as_tensor(full_obs[key]) for key in ("local_obs", "global_obs", "agent_mask")}
        raw["agent_mask"] = raw["agent_mask"].bool()
        starts = torch.ones(2, dtype=torch.bool)

        with normalizer.lock:
            adapter(raw, starts)
        assert actor.seen["local_obs"].device.type == actor_device
        torch.testing.assert_close(actor.seen["local_obs"].cpu(), wrapped["local_obs"])
        assert normalizer.update_running_mean
        assert normalizer.calls == [True]

        normalizer.obs_rms.mean.fill_(20.0)
        with torch.inference_mode(False), torch.no_grad():
            offset_source.fill_(40.0)
        adapter(raw, starts)
        torch.testing.assert_close(actor.seen["local_obs"].cpu(), wrapped["local_obs"])
        assert normalizer.calls == [True]
    finally:
        env.close()


@pytest.mark.parametrize(("actor_device", "simulation_device"), [
    ("cpu", "cpu"), pytest.param("cuda", "cuda", marks=_CUDA),
    pytest.param("cuda", "cpu", marks=_CUDA), pytest.param("cpu", "cuda", marks=_CUDA),
])
@pytest.mark.parametrize("normalizer_kind", ["builtin", "passive_subclass", "custom_frozen_operation"])
@torch.inference_mode()
def test_frozen_benchmark_actions_match_wrapped_rollout_on_actor_device(
    actor_device: str, simulation_device: str, normalizer_kind: str,
) -> None:
    class TaggedNormalizer(TorchFeatureWiseObsNormWrapper):
        pass

    vector = SyncVectorEnv([lambda: _MaskedEnv(1), lambda: _MaskedEnv(3)], autoreset_mode=AutoresetMode.SAME_STEP)
    action_env = SwarmBotsLearnEnvWrapper(vector)
    wrapper_type = TorchFeatureWiseObsNormWrapper if normalizer_kind == "builtin" else TaggedNormalizer
    normalizer = wrapper_type(action_env, "local_obs", [0], [], compile_tensor_operations=False)
    if normalizer_kind == "passive_subclass":
        normalizer.ready = Event()
    elif normalizer_kind == "custom_frozen_operation":
        builtin_frozen = normalizer._normalize_frozen_observations

        def offset_frozen_observations(*args: Any) -> NormalizedObservations:
            obs, mean, var, count = builtin_frozen(*args)
            return obs + 7.0, mean, var, count

        normalizer._normalize_frozen_observations = offset_frozen_observations
    normalizer.update_running_mean = False
    normalizer.obs_rms.mean.fill_(2.0)
    normalizer.obs_rms.var.fill_(4.0)
    env = TorchTransitionObsWrapper(normalizer, normalize_prev_binary_actions=True, compile_tensor_operations=False)
    try:
        policy = PPOPolicy(env, PPOPolicyConfig(
            actor_config=PPOActorConfig(latent_pi_dim_per_agent=8),
            continuous_config=SquashedDiagGaussianConfig(std=0.2, std_learnable=False),
        ))
        adapter = BenchmarkPolicy(policy, env)
        policy.to(actor_device).eval()
        wrapped, _ = env.reset(seed=7)
        starts = torch.ones(2, dtype=torch.bool)
        for step in range(4):
            public = ("local_obs", "global_obs", "agent_mask")
            expected = policy.act(**{key: wrapped[key].to(actor_device) for key in public}, deterministic=True)
            raw = {key: torch.as_tensor(vector._observations[key], device=simulation_device) for key in public}
            raw["agent_mask"] = raw["agent_mask"].bool()
            if step:
                normalizer.obs_rms.mean.fill_(1000.0)
            actual = adapter(raw, starts.to(simulation_device))
            torch.testing.assert_close(actual, action_env._actions_to_env(expected))
            assert normalizer.obs_rms.mean.device.type == "cpu"
            normalizer.obs_rms.mean.fill_(2.0)
            wrapped, _, terminated, truncated, _ = env.step(expected.cpu())
            starts = terminated | truncated
    finally:
        env.close()
