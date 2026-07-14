import unittest

import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo import PPO, StepsRolloutMode, WholeEpisodesRolloutMode
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSampler, PPOSamplerConfig
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.testing_env import TestingSwarmBotsEnv


class _DummyActionDist:
    has_gsde = False
    distributions: tuple[object, ...] = ()

    def reset_temporal_correlations_on_ep_start(self, mask: torch.Tensor) -> None:
        _ = mask

    def reset_temporal_correlations_on_step(
            self,
            mask: torch.Tensor | None = None,
            batch_shape: tuple[int, ...] | None = None,
    ) -> None:
        _ = mask
        _ = batch_shape


class _DummyPolicy(BasePPOPolicy[PPOSampler, PPOSamplerConfig]):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self._action_dim = action_dim
        self._dummy = torch.nn.Parameter(torch.zeros(()))
        self.action_dist = _DummyActionDist()

    def forward(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _ = global_obs
        _ = hidden_local_vars
        _ = hidden_global_vars
        _ = agent_mask
        _ = previous_actions
        _ = deterministic
        batch_size, n_agents, _ = local_obs.shape
        actions = torch.zeros(
            (batch_size, n_agents, self._action_dim),
            device=local_obs.device,
            dtype=local_obs.dtype,
        )
        log_probs = torch.zeros((batch_size, n_agents), device=local_obs.device, dtype=local_obs.dtype)
        values = torch.zeros((batch_size,), device=local_obs.device, dtype=local_obs.dtype)
        return actions + (self._dummy * 0.0), log_probs, values

    def predict_values(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = global_obs
        _ = hidden_local_vars
        _ = hidden_global_vars
        _ = agent_mask
        _ = previous_actions
        return torch.zeros((local_obs.shape[0],), device=local_obs.device, dtype=local_obs.dtype) + (self._dummy * 0.0)

    def _evaluate_actions(self, batch, action_splitter=None):
        raise NotImplementedError

    def make_sampler(
            self,
            episodes: list[PPOEpisodeSegment],
            config: PPOSamplerConfig,
    ) -> PPOSampler:
        raise NotImplementedError

    def get_hyper_parameters(self) -> dict[str, float]:
        return {}

    def get_grad_norms(self) -> dict[str, float]:
        return {}

    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> torch.Tensor:
        actions, _, _ = self.forward(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
        )
        return actions

    def requires_previous_actions(self) -> bool:
        return False


def _make_env(n_envs: int, max_steps: int = 32) -> SwarmBotsLearnEnvWrapper:
    vector_env = SyncVectorEnv(
        [lambda: TestingSwarmBotsEnv(2, 3, 2, 1, 1, max_steps=max_steps) for _ in range(n_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return SwarmBotsLearnEnvWrapper(vector_env)


class PPORolloutBufferSizingTests(unittest.TestCase):
    def test_steps_rollout_sizes_buffer_to_rollout_segment_length(self) -> None:
        env = _make_env(n_envs=512)
        try:
            policy = _DummyPolicy(action_dim=env.action_space.total_agent_action_dim)
            algo = PPO(
                policy=policy,
                env=env,
                rollout_mode=StepsRolloutMode(4048),
                max_episode_length=512,
            )

            self.assertEqual(algo.max_episode_length, 512)
            self.assertEqual(algo.rollout_buffer_max_episode_length, 8)
            self.assertEqual(algo.rollout_buffer.max_episode_length, 8)
            self.assertEqual(algo.get_hyper_parameters()["rollout_buffer_max_episode_length"], 8)
        finally:
            env.close()

    def test_whole_episode_rollout_keeps_full_episode_capacity(self) -> None:
        env = _make_env(n_envs=16)
        try:
            policy = _DummyPolicy(action_dim=env.action_space.total_agent_action_dim)
            algo = PPO(
                policy=policy,
                env=env,
                rollout_mode=WholeEpisodesRolloutMode(2),
                max_episode_length=512,
            )

            self.assertEqual(algo.rollout_buffer_max_episode_length, 512)
            self.assertEqual(algo.rollout_buffer.max_episode_length, 512)
        finally:
            env.close()

    def test_rollout_warmup_advances_state_without_training_counters_or_buffer(self) -> None:
        env = _make_env(n_envs=4)
        try:
            policy = _DummyPolicy(action_dim=env.action_space.total_agent_action_dim)
            algo = PPO(
                policy=policy,
                env=env,
                rollout_mode=StepsRolloutMode(8),
                max_episode_length=32,
                rollout_warmup_steps_per_env=3,
            )

            algo._before_learn_loop()

            self.assertEqual(algo.n_total_timesteps, 0)
            self.assertEqual(algo.n_total_iterations, 0)
            self.assertEqual(len(algo.rollout_buffer.episodes), 0)
            self.assertEqual(int(algo.rollout_buffer.accumulator.step.sum().item()), 0)
            self.assertIsNotNone(algo._rollout_state)
            assert algo._rollout_state is not None
            self.assertEqual(algo._rollout_state.rollout_step_idx, 3)
        finally:
            env.close()

    def test_rollout_warmup_continues_from_same_step_reset_observation(self) -> None:
        env = _make_env(n_envs=2, max_steps=2)
        try:
            policy = _DummyPolicy(action_dim=env.action_space.total_agent_action_dim)
            algo = PPO(
                policy=policy,
                env=env,
                rollout_mode=StepsRolloutMode(4),
                max_episode_length=2,
                rollout_warmup_steps_per_env=3,
            )

            algo._before_learn_loop()

            assert algo._rollout_state is not None
            torch.testing.assert_close(
                algo._rollout_state.obs["local_obs"],
                torch.ones_like(algo._rollout_state.obs["local_obs"]),
            )
            self.assertFalse(algo._rollout_state.episode_start_mask.any().item())
            self.assertEqual(algo._rollout_state.rollout_step_idx, 3)
            self.assertEqual(algo.n_total_timesteps, 0)
            self.assertEqual(len(algo.rollout_buffer.episodes), 0)
        finally:
            env.close()

    def test_rollout_warmup_is_skipped_for_started_training_state(self) -> None:
        env = _make_env(n_envs=4)
        try:
            policy = _DummyPolicy(action_dim=env.action_space.total_agent_action_dim)
            algo = PPO(
                policy=policy,
                env=env,
                rollout_mode=StepsRolloutMode(8),
                max_episode_length=32,
                rollout_warmup_steps_per_env=3,
            )
            algo.n_total_timesteps = 8

            algo._before_learn_loop()

            self.assertIsNone(algo._rollout_state)
            self.assertTrue(algo._rollout_warmup_done)
            self.assertEqual(len(algo.rollout_buffer.episodes), 0)
            self.assertEqual(int(algo.rollout_buffer.accumulator.step.sum().item()), 0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
