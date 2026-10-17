from collections.abc import Callable
from typing import Any

import pytest
import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.gsde_action_dist import GSDEConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat.mat_ind_policy import MATIndPolicy, MATIndPolicyConfig
from swarmbots.learn.algos.ppo.ppo import PPO, StepsRolloutMode
from swarmbots.learn.algos.ppo.ppo_rollout import collect_step_rollout_batch, collect_steps
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamplerConfig
from swarmbots.learn.gsde_reset import GSDEEpisodeResetMode, GSDEIntervalResetMode, resolve_gsde_reset_mode
from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace


class _EpisodeEnv:
    device = torch.device("cpu")
    n_agents = 2
    local_obs_dim = 3
    global_obs_dim = 1
    hidden_local_vars_dim = 0
    hidden_global_vars_dim = 0

    def __init__(self) -> None:
        self.step_counts = torch.zeros(2, dtype=torch.long)
        self.episode_lengths = torch.tensor([3, 100])
        self.action_space = VectorHybridActionSpace({"cont": spaces.Box(-1.0, 1.0, (2, 2, 1))})
        self.observation_space = spaces.Dict({
            "local_obs": spaces.Box(-1000.0, 1000.0, (2, 2, 3)),
            "global_obs": spaces.Box(-1000.0, 1000.0, (2, 1)),
            "hidden_local_vars": spaces.Box(-1.0, 1.0, (2, 2, 0)),
            "hidden_global_vars": spaces.Box(-1.0, 1.0, (2, 0)),
        })

    def set_device(self, device: torch.device | str) -> None:
        assert torch.device(device) == self.device

    def _observations(self) -> dict[str, torch.Tensor]:
        return {
            "local_obs": self.step_counts.float().view(2, 1, 1).expand(2, 2, 3).clone(),
            "global_obs": self.step_counts.float().unsqueeze(-1),
            "hidden_local_vars": torch.empty(2, 2, 0),
            "hidden_global_vars": torch.empty(2, 0),
        }

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        self.step_counts.zero_()
        return self._observations(), {}

    def _obs_to_torch(self, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return observations

    def step(
            self,
            actions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        assert torch.isfinite(actions).all()
        self.step_counts += 1
        done = self.step_counts >= self.episode_lengths
        final_observations = self._observations()
        self.step_counts[done] = 0
        return (
            self._observations(), torch.ones(2), done, torch.zeros(2, dtype=torch.bool),
            {"final_obs": final_observations, "_final_obs": done},
        )


@pytest.mark.parametrize("collector", [collect_steps, collect_step_rollout_batch])
@pytest.mark.parametrize("disable_after_rollout", [False, True])
def test_disabling_gsde_step_resets_preserves_noise_until_each_episode_ends(
        collector: Callable[..., Any],
        disable_after_rollout: bool,
) -> None:
    env = _EpisodeEnv()
    policy = MATIndPolicy(env, MATIndPolicyConfig(
        encoder_config=MATEncoderConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16),
        continuous_config=GSDEConfig(base_std=0.5),
    ))
    algorithm = PPO(
        policy, env, rollout_mode=StepsRolloutMode(4), sampler_config=PPOSamplerConfig(batch_size=2),
        gsde_reset_mode=GSDEIntervalResetMode(interval=1), train_device="cpu", rollout_device="cpu",
    )
    if not disable_after_rollout:
        assert algorithm._execute_command("disable_gsde_reset", "", None)
    distribution = policy.action_dist.distributions[0]
    _, _, _, rollout_state = collector(
        env, policy, algorithm.rollout_buffer, n_steps=2, gsde_reset_mode=algorithm.gsde_reset_mode,
    )
    initial_noise = distribution.get_temporal_correlation_state().clone()
    if disable_after_rollout:
        assert algorithm._execute_command("disable_gsde_reset", "", None)
    assert algorithm._serialize_gsde_reset_mode(algorithm.gsde_reset_mode) == {"mode": "episode"}
    _, _, _, rollout_state = collector(
        env, policy, algorithm.rollout_buffer, n_steps=2, rollout_state=rollout_state,
        gsde_reset_mode=algorithm.gsde_reset_mode,
    )
    torch.testing.assert_close(distribution.get_temporal_correlation_state(), initial_noise)

    collector(
        env, policy, algorithm.rollout_buffer, n_steps=2, rollout_state=rollout_state,
        gsde_reset_mode=algorithm.gsde_reset_mode,
    )
    reset_noise = distribution.get_temporal_correlation_state()
    assert tuple(reset_noise.shape[:2]) == (2, 2)
    assert not torch.equal(reset_noise[0], initial_noise[0])
    torch.testing.assert_close(reset_noise[1], initial_noise[1])


def test_episode_only_gsde_reset_mode_is_explicit() -> None:
    mode = GSDEEpisodeResetMode()
    assert resolve_gsde_reset_mode(gsde_enabled=True, reset_mode=mode) is mode
    assert resolve_gsde_reset_mode(gsde_enabled=False, reset_mode=mode) is None
    with pytest.raises(ValueError, match="gsde_reset_mode is required"):
        resolve_gsde_reset_mode(gsde_enabled=True, reset_mode=None)
