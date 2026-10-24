from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import SquashedDiagGaussianConfig
from swarmbots.learn.algos.ppo.ppo import PPO, StepsRolloutMode
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat.mat_ind_policy import MATIndPolicy, MATIndPolicyConfig
from swarmbots.learn.algos.ppo.ppo_policy import PPOActorConfig, PPOPolicy, PPOPolicyConfig
from swarmbots.learn.algos.ppo.ppo_rollout import collect_step_rollout_batch, collect_steps
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPORolloutBuffer
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamplerConfig
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoderConfig
from swarmbots.learn.algos.r_mat.r_mat_ind_policy import RMATIndPolicy, RMATIndPolicyConfig
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.testing_env import TestingSwarmBotsEnv as SwarmTestEnv
from tests.test_same_step_pipeline import _ConstantValuePolicy, _ValueOnlyBootstrapPolicy, _make_scripted_env


DEVICES = ["cpu", pytest.param("cuda", marks=[
    pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
])]


@pytest.mark.parametrize("device", DEVICES)
def test_compiled_collector_reuses_graphs_when_truncated_lanes_change(
    device: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    graphs: list[torch.fx.GraphModule] = []
    original_compile = torch.compile

    def backend(graph: torch.fx.GraphModule, _inputs: list[torch.Tensor]) -> Any:
        graphs.append(graph)
        return graph.forward

    def compile_module(function: Any, **_kwargs: Any) -> Any:
        return original_compile(function, backend=backend, fullgraph=True, dynamic=False)

    monkeypatch.setattr(torch, "compile", compile_module)
    env = _make_scripted_env((0, (2,), "truncate"), (1, (3,), "truncate"), (2, (), "none"))
    env.set_device(device)
    policy = MATIndPolicy(env, MATIndPolicyConfig(
        encoder_config=MATEncoderConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16),
        continuous_config=SquashedDiagGaussianConfig(std=0.5, std_learnable=False), compile_modules=True,
    ))
    buffer = PPORolloutBuffer(
        max_episode_length=8, observation_space=env.observation_space, action_space=env.action_space,
        gamma=1.0, gae_lambda=1.0, rollout_device=device, train_device="cpu",
    )
    state = None
    try:
        for _ in range(2):
            _, _, _, state = collect_step_rollout_batch(env, policy, buffer, 12, state)
        graph_count = len(graphs)
        assert graph_count > 0
        for _ in range(4):
            _, _, _, state = collect_step_rollout_batch(env, policy, buffer, 12, state)
        assert len(graphs) == graph_count
    finally:
        env.close()
        torch._dynamo.reset()


@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_collectors_preserve_returns_across_mixed_episode_boundaries(segmented: bool, device: str) -> None:
    env = _make_scripted_env((0, (2,), "truncate"), (1, (3,), "terminate"), (2, (), "none"))
    try:
        env.set_device(device)
        policy = _ConstantValuePolicy(action_dim=env.action_space.total_agent_action_dim)
        buffer = PPORolloutBuffer(
            max_episode_length=8, observation_space=env.observation_space, action_space=env.action_space,
            gamma=1.0, gae_lambda=1.0, rollout_device=device, train_device="cpu",
        )
        collector = collect_steps if segmented else collect_step_rollout_batch
        with patch.object(policy, "predict_values", wraps=policy.predict_values) as predict:
            result, _, _, _ = collector(env, policy, buffer, 12)
        assert all(call.kwargs["local_obs"].shape[0] == 3 for call in predict.call_args_list)
        if segmented:
            actual = torch.empty(3, 4)
            for episode in result:
                start = episode.rollout_start_step
                actual[episode.rollout_env_idx, start:start + len(episode.rewards)] = episode.returns
        else:
            actual = result.returns.cpu()
        torch.testing.assert_close(actual, torch.tensor([
            [105., 104., 205., 204.],
            [36., 25., 13., 1212.],
            [2194., 2173., 2151., 2128.],
        ]))
    finally:
        env.close()


@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("done_mode,expected_calls", [("none", 1), ("terminate", 0), ("truncate", 2)])
def test_collectors_only_bootstrap_truncations_and_unfinished_rollout_boundary(
    segmented: bool, device: str, done_mode: str, expected_calls: int,
) -> None:
    env = _make_scripted_env((0, (2,), done_mode))
    try:
        env.set_device(device)
        policy = _ValueOnlyBootstrapPolicy(action_dim=env.action_space.total_agent_action_dim)
        buffer = PPORolloutBuffer(
            max_episode_length=8, observation_space=env.observation_space, action_space=env.action_space,
            gamma=1.0, gae_lambda=1.0, rollout_device=device, train_device="cpu",
        )
        collector = collect_steps if segmented else collect_step_rollout_batch
        result, _, _, _ = collector(env, policy, buffer, 4)
        assert policy.predict_value_calls == expected_calls
        assert policy.forward_calls == 4
        if segmented:
            assert len(result) == (1 if done_mode == "none" else 2)
            if done_mode == "terminate":
                assert all(episode.final_value == 0 for episode in result)
            elif done_mode == "truncate":
                torch.testing.assert_close(result[0].final_value, torch.tensor(102.5))
                torch.testing.assert_close(result[1].final_value, torch.tensor(202.5))
        elif done_mode == "terminate":
            assert torch.count_nonzero(result.bootstrap_values) == 0
    finally:
        env.close()


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_ppo_load_restarts_collector_with_restored_normalization(
    recurrent: bool, device: str, tmp_path: Path,
) -> None:
    env = TorchFeatureWiseObsNormWrapper(
        SwarmBotsLearnEnvWrapper(SyncVectorEnv(
            [lambda: SwarmTestEnv(2, 3, 1, 1, 1, max_steps=8)], autoreset_mode=AutoresetMode.SAME_STEP,
        )), obs_key="local_obs", scalar_feature_indices=[0], quaternion_indices=[], compile_tensor_operations=False,
    )
    env.set_device(device)
    continuous = SquashedDiagGaussianConfig(std=0.5, std_learnable=False)
    policy = RMATIndPolicy(env, RMATIndPolicyConfig(
        encoder_config=RMATEncoderConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16),
        continuous_config=continuous,
    )) if recurrent else PPOPolicy(env, PPOPolicyConfig(
        actor_config=PPOActorConfig(hidden_dims=[8], latent_pi_dim_per_agent=8), continuous_config=continuous,
    ))
    trainer = PPO(
        policy, env, rollout_mode=StepsRolloutMode(2), sampler_config=PPOSamplerConfig(batch_size=1),
        train_device=device, rollout_device=device,
    )
    collector = collect_steps if recurrent else collect_step_rollout_batch
    try:
        env.reset()
        checkpoint = tmp_path / "initial.pt"
        trainer.save(checkpoint, optimizer_state_dict=trainer._get_optimizer_state_dict())
        _, _, _, trainer._rollout_state = collector(env, policy, trainer.rollout_buffer, 2)
        assert trainer._rollout_state.obs["local_obs"][..., 0].abs().sum() > 0
        trainer.load(checkpoint)
        assert trainer._rollout_state is None
        assert trainer.rollout_buffer.episodes == []
        assert torch.count_nonzero(trainer.rollout_buffer.accumulator.step) == 0
        env.update_running_mean = False
        with patch.object(policy, "forward_with_temporal_state", wraps=policy.forward_with_temporal_state) as forward:
            result, _, _, _ = collector(env, policy, trainer.rollout_buffer, 1, trainer._rollout_state)
        first_obs = result[0].local_obs[0] if recurrent else result.local_obs[0, 0]
        assert torch.count_nonzero(first_obs[..., 0]) == 0
        assert forward.call_args.kwargs["episode_start_mask"].all()
        state = forward.call_args.kwargs["temporal_state"]
        if recurrent:
            assert all(torch.count_nonzero(tensor) == 0 for side in state.values() for layer in side for tensor in layer)
    finally:
        env.close()
