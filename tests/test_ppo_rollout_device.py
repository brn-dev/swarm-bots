from collections.abc import Iterator
from typing import Any, Callable

import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.action_dists.gsde_action_dist import GSDEConfig
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import SquashedDiagGaussianConfig
from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo import PPO, StepsRolloutMode
from swarmbots.learn.algos.ppo.ppo_policy import PPOActorConfig, PPOPolicy, PPOPolicyConfig
from swarmbots.learn.algos.ppo.ppo_rollout import collect_step_rollout_batch, collect_steps, warmup_rollout_steps
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPORolloutBuffer
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoderConfig
from swarmbots.learn.algos.r_mat.r_mat_ind_policy import RMATIndPolicy, RMATIndPolicyConfig
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.gsde_reset import GSDEEpisodeResetMode
from swarmbots.learn.temporal_state import clone_detach_temporal_state, move_temporal_state
from swarmbots.learn.testing_env import TestingSwarmBotsEnv as SwarmTestEnv

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


@pytest.fixture(scope="module", autouse=True)
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def env() -> Iterator[SwarmBotsLearnEnvWrapper]:
    wrapper = SwarmBotsLearnEnvWrapper(SyncVectorEnv(
        [lambda: SwarmTestEnv(2, 3, 1, 1, 1, max_steps=8)], autoreset_mode=AutoresetMode.SAME_STEP,
    ))
    try:
        yield wrapper
    finally:
        wrapper.close()


def make_policy(env: SwarmBotsLearnEnvWrapper, kind: str) -> BasePPOPolicy[Any, Any]:
    continuous = GSDEConfig(base_std=0.5) if kind == "gsde" else SquashedDiagGaussianConfig(std=0.5, std_learnable=False)
    if kind == "lstm":
        return RMATIndPolicy(env, RMATIndPolicyConfig(
            encoder_config=RMATEncoderConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16),
            continuous_config=continuous,
        ))
    return PPOPolicy(env, PPOPolicyConfig(
        actor_config=PPOActorConfig(latent_pi_dim_per_agent=8), continuous_config=continuous,
    ))


def make_buffer(env: SwarmBotsLearnEnvWrapper, device: str) -> PPORolloutBuffer:
    return PPORolloutBuffer(
        max_episode_length=8, observation_space=env.observation_space, action_space=env.action_space,
        gamma=0.99, gae_lambda=0.95, rollout_device=device, train_device="cpu",
    )


@pytest.mark.parametrize("collector", [collect_step_rollout_batch, collect_steps])
@pytest.mark.parametrize("kind", ["mlp", "gsde"])
@pytest.mark.parametrize("warmup", [False, True])
def test_rollout_device_moves_environment_and_continues_existing_episodes(
    env: SwarmBotsLearnEnvWrapper, collector: Callable[..., Any], kind: str, warmup: bool,
) -> None:
    policy = make_policy(env, kind)
    reset_mode = GSDEEpisodeResetMode() if policy.gsde_enabled else None
    state = warmup_rollout_steps(env, policy, 1, gsde_reset_mode=reset_mode) if warmup else None
    _, _, _, state = collector(env, policy, make_buffer(env, "cuda"), 2, state, reset_mode)
    assert env.device.type == "cuda"
    assert state.obs["local_obs"].device.type == "cuda"
    expected_noise = (
        move_temporal_state(clone_detach_temporal_state(policy.action_dist.get_temporal_correlation_state()), device="cpu")
        if policy.gsde_enabled else None
    )
    steps_before = state.rollout_step_idx
    result, _, _, state = collector(env, policy, make_buffer(env, "cpu"), 1, state, reset_mode)
    assert env.device.type == "cpu"
    assert state.obs["local_obs"].device.type == "cpu"
    assert state.rollout_step_idx == steps_before + 1
    first_observation = result[0].local_obs[0, 0, 0] if collector is collect_steps else result.local_obs[0, 0, 0, 0]
    assert first_observation.item() == steps_before % 8
    assert state.obs["local_obs"][0, 0, 0].item() == state.rollout_step_idx % 8
    if policy.gsde_enabled:
        torch.testing.assert_close(policy.action_dist.get_temporal_correlation_state(), expected_noise)


@pytest.mark.parametrize("collector", [collect_step_rollout_batch, collect_steps])
@pytest.mark.parametrize("change_std", ["set_std", "scale_std"])
def test_cpu_rollout_moves_noise_recreated_during_cuda_training(
    env: SwarmBotsLearnEnvWrapper, collector: Callable[..., Any], change_std: str,
) -> None:
    policy = make_policy(env, "gsde")
    reset_mode = GSDEEpisodeResetMode()
    state = warmup_rollout_steps(env, policy, 1, gsde_reset_mode=reset_mode)
    policy.to("cuda")
    getattr(policy.action_dist, change_std)(0.4)
    expected_noise = move_temporal_state(
        clone_detach_temporal_state(policy.action_dist.get_temporal_correlation_state()), device="cpu",
    )
    _, _, _, state = collector(env, policy, make_buffer(env, "cpu"), 1, state, reset_mode)
    torch.testing.assert_close(policy.action_dist.get_temporal_correlation_state(), expected_noise)
    assert state.rollout_step_idx == 2
    assert state.obs["local_obs"][0, 0, 0].item() == 2


def test_recurrent_rollout_preserves_actor_and_critic_histories_across_devices(
    env: SwarmBotsLearnEnvWrapper,
) -> None:
    policy = make_policy(env, "lstm")
    state = warmup_rollout_steps(env, policy, 1)
    for device in ("cuda", "cpu"):
        expected_state = move_temporal_state(clone_detach_temporal_state(state.temporal_state), device="cpu")
        episodes, _, _, state = collect_steps(env, policy, make_buffer(env, device), 1, state)
        torch.testing.assert_close(episodes[0].initial_temporal_state, expected_state)
        assert state.temporal_state["actor"][0][0].device.type == device
        assert state.temporal_state["critic"][0][0].device.type == device
    assert state.rollout_step_idx == 3
    assert state.obs["local_obs"][0, 0, 0].item() == 3


def test_ppo_warmup_uses_the_configured_rollout_device(env: SwarmBotsLearnEnvWrapper) -> None:
    policy = make_policy(env, "mlp")
    algorithm = PPO(
        policy, env, train_device="cpu", rollout_device="cuda", rollout_mode=StepsRolloutMode(1),
        rollout_warmup_steps_per_env=1,
    )
    algorithm._before_learn_loop()
    assert env.device.type == "cuda"
    assert algorithm._rollout_state.obs["local_obs"].device.type == "cuda"
    assert next(policy.parameters()).device.type == "cuda"
    assert algorithm.n_total_timesteps == 0
