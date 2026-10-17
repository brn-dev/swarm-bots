from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.action_dists.gsde_action_dist import GSDEConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.ppo.ppo import PPO, StepsRolloutMode
from swarmbots.learn.algos.ppo.ppo_policy import PPOActorConfig, PPOPolicy, PPOPolicyConfig
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamplerConfig
from swarmbots.learn.algos.sac import SAC, SACNOPConfig, TMASACPolicy, TMASACPolicyConfig
from swarmbots.learn.benchmark_policy import BenchmarkPolicy, as_benchmark_policy
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.gsde_reset import (
    GSDEResetMode, GSDEEpisodeResetMode, GSDEIntervalResetMode, GSDEProbabilityResetMode,
)
from swarmbots.learn.testing_env import TestingSwarmBotsEnv as FixtureEnv

GSDEActorFixture = tuple[SwarmBotsLearnEnvWrapper, PPOPolicy, dict[str, torch.Tensor]]


@pytest.fixture
def gsde_actor() -> Iterator[GSDEActorFixture]:
    env = SwarmBotsLearnEnvWrapper(SyncVectorEnv([
        lambda: FixtureEnv(
            n_agents=2, n_local_obs=3, n_global_obs=1, actuators_dim=1, connectors_dim=1,
            continuous_connector_actions=True,
        )
        for _ in range(2)
    ], autoreset_mode=AutoresetMode.SAME_STEP))
    policy = PPOPolicy(env, PPOPolicyConfig(
        actor_config=PPOActorConfig(latent_pi_dim_per_agent=8), continuous_config=GSDEConfig(base_std=.3),
    ))
    observations, _ = env.reset(seed=7)
    observations["agent_mask"] = torch.ones(2, 2, dtype=torch.bool)
    try:
        yield env, policy, observations
    finally:
        env.close()


def test_benchmark_gsde_preserves_noise_until_each_episode_ends(gsde_actor: GSDEActorFixture) -> None:
    env, policy, observations = gsde_actor
    policy.action_dist.reset_temporal_correlations_on_step(batch_shape=(2, 2))
    training_noise = tuple(state.clone() for state in policy.action_dist.get_temporal_correlation_state())
    adapter = BenchmarkPolicy(policy, env, deterministic=False)
    first_actions = adapter(observations, torch.ones(2, dtype=torch.bool))
    initial_noise = tuple(state.clone() for state in adapter.distribution_state)
    assert all(state.shape[:2] == (2, 2) for state in initial_noise)
    assert not torch.equal(initial_noise[0][:, 0], initial_noise[0][:, 1])
    for _ in range(3):
        torch.testing.assert_close(adapter(observations, torch.zeros(2, dtype=torch.bool)), first_actions)
        torch.testing.assert_close(adapter.distribution_state, initial_noise)
    adapter(observations, torch.tensor([True, False]))
    for previous, current in zip(initial_noise, adapter.distribution_state, strict=True):
        assert not torch.equal(previous[0], current[0])
        torch.testing.assert_close(previous[1], current[1])
    torch.testing.assert_close(policy.action_dist.get_temporal_correlation_state(), training_noise)


def test_benchmark_gsde_inherits_interval_and_allows_episode_override(gsde_actor: GSDEActorFixture) -> None:
    env, policy, observations = gsde_actor
    trainer = SimpleNamespace(policy=policy, env=env, gsde_reset_mode=GSDEIntervalResetMode(3))
    for override in (None, GSDEEpisodeResetMode()):
        adapter = as_benchmark_policy(trainer, deterministic=False, gsde_reset_mode=override)
        previous_noise = None
        for step in range(7):
            adapter(observations, torch.full((2,), step == 0, dtype=torch.bool))
            if previous_noise is not None:
                should_reset = override is None and step % 3 == 0
                for previous, current in zip(previous_noise, adapter.distribution_state, strict=True):
                    assert torch.equal(previous, current) is not should_reset
            previous_noise = tuple(state.clone() for state in adapter.distribution_state)


def test_benchmark_gsde_probability_does_not_reset_every_agent_every_step(gsde_actor: GSDEActorFixture) -> None:
    env, policy, observations = gsde_actor
    torch.manual_seed(42)
    adapter = BenchmarkPolicy(policy, env, deterministic=False, gsde_reset_mode=GSDEProbabilityResetMode(.25))
    adapter(observations, torch.ones(2, dtype=torch.bool))
    counts = torch.zeros(2, 2, dtype=torch.long)
    for _ in range(20):
        previous = adapter.distribution_state[0].clone()
        adapter(observations, torch.zeros(2, dtype=torch.bool))
        counts += (previous != adapter.distribution_state[0]).any(dim=(-2, -1))
    assert ((counts > 0) & (counts < 20)).all()


@pytest.mark.parametrize("saved_mode", [
    GSDEEpisodeResetMode(), GSDEIntervalResetMode(3), GSDEProbabilityResetMode(.25),
])
@pytest.mark.parametrize("variant", ["ppo", "tmasac"])
@pytest.mark.parametrize("override", [False, True])
def test_checkpoint_recording_restores_gsde_schedule(
    gsde_actor: GSDEActorFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    saved_mode: GSDEResetMode, variant: Literal["ppo", "tmasac"], override: bool,
) -> None:
    import swarmbots.learn.checkpoint_recording as recording

    env, policy, _ = gsde_actor
    reset_mode = saved_mode
    trainer: PPO | SAC
    if variant == "ppo":
        trainer = PPO(
            policy, env, rollout_mode=StepsRolloutMode(4), sampler_config=PPOSamplerConfig(batch_size=2),
            gsde_reset_mode=reset_mode, train_device="cpu", rollout_device="cpu",
        )
    else:
        encoder_config = MATEncoderConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
        sac_policy = TMASACPolicy(env, TMASACPolicyConfig(
            actor_encoder_config=encoder_config, critic_encoder_config=encoder_config,
            continuous_config=GSDEConfig(base_std=.3), nop_config=SACNOPConfig(enabled=False),
        ))
        trainer = SAC(
            sac_policy, env, buffer_capacity_per_env=8, batch_size=2, learning_starts=0,
            gsde_reset_mode=reset_mode, train_device="cpu", rollout_device="cpu", replay_storage_device="cpu",
        )
    checkpoint_path = tmp_path / "gsde.pt"
    trainer.save(checkpoint_path)
    monkeypatch.setattr(recording, "_make_policy_env", lambda *args, **kwargs: (env, trainer.policy))
    captured = []
    monkeypatch.setattr(recording, "_record_episodes", lambda env, adapter, **kwargs: captured.append(adapter))
    selected = GSDEIntervalResetMode(7) if override else None
    recording.record_checkpoint(
        checkpoint_path, "SwarmBots-WallEasy-v0", variant, video_folder=tmp_path,
        deterministic=False, gsde_reset_mode=selected,
    )
    assert captured[0].gsde_reset_mode == (selected if override else reset_mode)
