from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.sac.scenario_obs_encoder import (
    ScenarioFieldEncoderConfig,
    ScenarioObservationSpec,
    TMASACScenarioEncoderConfig,
)
from swarmbots.learn.algos.off_policy.actor_heads import TMASACActorHeadConfig
from swarmbots.learn.algos.sac.tmasac_policy import TMASACPolicy, TMASACPolicyConfig
from swarmbots.learn.benchmark_policy import as_benchmark_policy
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.multi_scenario_vector_env import MultiScenarioVectorEnv
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.testing_env import TestingSwarmBotsEnv as FixtureSwarmEnv


class _MaskedScenarioEnv(FixtureSwarmEnv):
    def __init__(self, global_obs_dim: int) -> None:
        super().__init__(
            n_agents=2, n_local_obs=4, n_global_obs=global_obs_dim, actuators_dim=1, connectors_dim=1,
            n_hidden_local_vars=1, n_hidden_global_vars=1, continuous_connector_actions=True,
        )
        self.observation_space = spaces.Dict({**self.observation_space.spaces, "agent_mask": spaces.MultiBinary(2)})

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {**super()._get_obs(), "agent_mask": np.ones(2, dtype=bool)}


@pytest.fixture
def scenario_training() -> Iterator[tuple[TorchFeatureWiseObsNormWrapper, TMASACPolicy]]:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    scenarios = ("wall", "payload")
    global_dims = (0, 3)
    env = SwarmBotsLearnEnvWrapper(MultiScenarioVectorEnv({
        name: SyncVectorEnv(
            [lambda dim=dim: _MaskedScenarioEnv(dim)], autoreset_mode=AutoresetMode.SAME_STEP,
        )
        for name, dim in zip(scenarios, global_dims, strict=True)
    }))
    env = TorchFeatureWiseObsNormWrapper(env, "global_obs", [0, 1, 2], [], compile_tensor_operations=False)
    env.update_running_mean = False
    env.obs_rms.mean.fill_(0.5)
    env.obs_rms.var.fill_(2.0)
    encoder = MATEncoderConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
    policy = TMASACPolicy(env, TMASACPolicyConfig(
        actor_encoder_config=encoder,
        critic_encoder_config=encoder,
        actor_head_config=TMASACActorHeadConfig(hidden_dims=[8]),
        scenario_encoder_config=TMASACScenarioEncoderConfig(
            scenarios=tuple(
                ScenarioObservationSpec(index, name, dim, 1, 1)
                for index, (name, dim) in enumerate(zip(scenarios, global_dims, strict=True))
            ),
            global_obs=ScenarioFieldEncoderConfig(output_dim=4),
            hidden_local_vars=ScenarioFieldEncoderConfig(output_dim=3),
            hidden_global_vars=ScenarioFieldEncoderConfig(output_dim=3),
            scenario_embedding_dim=2,
        ),
    ))
    try:
        yield env, policy
    finally:
        env.close()
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("scenario_id", [0, 1])
def test_benchmark_adapter_selects_scenario_for_canonical_observations(
    scenario_training: tuple[TorchFeatureWiseObsNormWrapper, TMASACPolicy], scenario_id: int,
) -> None:
    env, policy = scenario_training
    raw_obs, _ = env.unwrapped.reset()
    raw_obs = env._obs_to_torch(raw_obs)
    global_dim = (0, 3)[scenario_id]
    canonical_obs = {key: value[scenario_id:scenario_id + 1].clone() for key, value in raw_obs.items()}
    del canonical_obs["scenario_id"]
    canonical_obs["global_obs"] = canonical_obs["global_obs"][..., :global_dim]
    canonical_obs["hidden_local_vars"].fill_(1234)
    canonical_obs["hidden_global_vars"].fill_(1234)
    expected_obs = env.observations({key: value[scenario_id:scenario_id + 1] for key, value in raw_obs.items()})
    with torch.no_grad():
        expected_actions = env._actions_to_env(policy.act(
            local_obs=expected_obs["local_obs"], global_obs=expected_obs["global_obs"],
            agent_mask=expected_obs["agent_mask"], scenario_ids=torch.tensor([scenario_id]), deterministic=True,
        ))
    adapter = as_benchmark_policy(SimpleNamespace(policy=policy, env=env), scenario_id=scenario_id)
    with patch.object(policy, "act_with_temporal_state", wraps=policy.act_with_temporal_state) as act:
        actual_actions = adapter(canonical_obs, torch.ones(1, dtype=torch.bool))
    torch.testing.assert_close(actual_actions, expected_actions)
    assert "hidden_local_vars" not in act.call_args.kwargs
    assert "hidden_global_vars" not in act.call_args.kwargs
    torch.testing.assert_close(act.call_args.kwargs["scenario_ids"], torch.tensor([scenario_id]))
    torch.testing.assert_close(act.call_args.kwargs["global_obs"], expected_obs["global_obs"])


def test_benchmark_adapter_forwards_observation_scenarios_before_selected_default(
    scenario_training: tuple[TorchFeatureWiseObsNormWrapper, TMASACPolicy],
) -> None:
    env, policy = scenario_training
    raw_obs, _ = env.unwrapped.reset()
    raw_obs = env._obs_to_torch(raw_obs)
    expected_obs = env.observations(raw_obs)
    with torch.no_grad():
        expected_actions = env._actions_to_env(policy.act(
            local_obs=expected_obs["local_obs"], global_obs=expected_obs["global_obs"],
            agent_mask=expected_obs["agent_mask"], scenario_ids=raw_obs["scenario_id"], deterministic=True,
        ))
    for selected_default in (None, 1):
        adapter = as_benchmark_policy(SimpleNamespace(policy=policy, env=env), scenario_id=selected_default)
        with patch.object(policy, "act_with_temporal_state", wraps=policy.act_with_temporal_state) as act:
            actual_actions = adapter(raw_obs, torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(actual_actions, expected_actions)
        torch.testing.assert_close(act.call_args.kwargs["scenario_ids"], raw_obs["scenario_id"])
        assert "hidden_local_vars" not in act.call_args.kwargs
        assert "hidden_global_vars" not in act.call_args.kwargs
