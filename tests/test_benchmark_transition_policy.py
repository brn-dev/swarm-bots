from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from torch import nn

from swarmbots.learn.benchmark_policy import BenchmarkPolicy
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
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
