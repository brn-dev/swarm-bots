import unittest
from typing import Any
from unittest.mock import patch

import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.algos.off_policy import OffPolicyReplayBuffer, OffPolicySamplerConfig, collect_steps
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper


class _ScriptedEnv(gymnasium.Env):
    metadata = {}

    def __init__(self, *, env_id: int, done_steps: tuple[int, ...], done_mode: str) -> None:
        super().__init__()
        self.env_id = env_id
        self.done_steps = done_steps
        self.done_mode = done_mode
        self.episode_id = 0
        self.step_count = 0
        self.episode_return = 0.0
        self.observation_space = spaces.Dict({
            "local_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(2, 3), dtype=np.float32),
            "global_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "hidden_local_vars": spaces.Box(low=-np.inf, high=np.inf, shape=(2, 1), dtype=np.float32),
            "hidden_global_vars": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        })
        self.action_space = spaces.Dict({
            "actuators": spaces.Box(low=-1.0, high=1.0, shape=(2, 1), dtype=np.float32),
            "connectors": spaces.MultiBinary((2, 1)),
        })

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        _ = options
        super().reset(seed=seed)
        self.episode_id += 1
        self.step_count = 0
        self.episode_return = 0.0
        return self._obs(), {}

    def step(
            self,
            action: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        _ = action
        self.step_count += 1
        reward = float(self.env_id * 10 + self.step_count)
        self.episode_return += reward
        terminated = self.done_mode == "terminate" and self.step_count in self.done_steps
        truncated = self.done_mode == "truncate" and self.step_count in self.done_steps
        info: dict[str, Any] = {}
        if terminated or truncated:
            info["episode"] = {"r": np.float64(self.episode_return), "success": np.bool_(terminated)}
        return self._obs(), reward, terminated, truncated, info

    def _obs(self) -> dict[str, np.ndarray]:
        base = float(self.env_id * 1000 + self.episode_id * 100 + self.step_count)
        return {
            "local_obs": np.full((2, 3), base, dtype=np.float32),
            "global_obs": np.array([base, base + 0.5], dtype=np.float32),
            "hidden_local_vars": np.full((2, 1), base + 1.0, dtype=np.float32),
            "hidden_global_vars": np.array([base + 2.0], dtype=np.float32),
        }


class _PreviousActionPolicy(BasePolicy):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim

    @property
    def gsde_enabled(self) -> bool:
        return False

    def get_hyper_parameters(self) -> dict[str, Any]:
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
        _ = local_obs
        _ = global_obs
        _ = hidden_local_vars
        _ = hidden_global_vars
        _ = agent_mask
        _ = deterministic
        if previous_actions is None:
            raise AssertionError("previous_actions should be tracked for off-policy rollouts")
        return previous_actions[..., :1].add(1.0).expand(*previous_actions.shape[:-1], self.action_dim).clone()

    def update_loss_weights(self, **weights: float) -> None:
        if weights:
            raise ValueError(f"Unknown loss weights: {sorted(weights)}")

    def requires_previous_actions(self) -> bool:
        return True


class _NoPreviousActionPolicy(BasePolicy):
    @property
    def gsde_enabled(self) -> bool:
        return False

    def get_hyper_parameters(self) -> dict[str, Any]:
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
        _ = global_obs
        _ = hidden_local_vars
        _ = hidden_global_vars
        _ = agent_mask
        _ = deterministic
        if previous_actions is not None:
            raise AssertionError("previous_actions should not be passed to policies that do not require them")
        return torch.zeros((*local_obs.shape[:2], 2), dtype=local_obs.dtype, device=local_obs.device)

    def update_loss_weights(self, **weights: float) -> None:
        if weights:
            raise ValueError(f"Unknown loss weights: {sorted(weights)}")

    def requires_previous_actions(self) -> bool:
        return False


def _make_env(*configs: tuple[int, tuple[int, ...], str]) -> SwarmBotsLearnEnvWrapper:
    vector_env = SyncVectorEnv(
        [
            (lambda env_id=env_id, done_steps=done_steps, done_mode=done_mode: _ScriptedEnv(
                env_id=env_id,
                done_steps=done_steps,
                done_mode=done_mode,
            ))
            for env_id, done_steps, done_mode in configs
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return SwarmBotsLearnEnvWrapper(vector_env)


def _make_buffer(
        env: SwarmBotsLearnEnvWrapper,
        *,
        capacity: int = 16,
        max_segment_length: int = 8,
) -> OffPolicyReplayBuffer:
    return OffPolicyReplayBuffer(
        capacity_transitions=capacity,
        max_segment_length=max_segment_length,
        observation_space=env.observation_space,
        action_space=env.action_space,
        rollout_device="cpu",
        train_device="cpu",
    )


class OffPolicyRolloutBufferTests(unittest.TestCase):
    def test_collect_steps_stores_final_obs_and_previous_actions(self) -> None:
        env = _make_env((1, (2,), "truncate"))
        try:
            policy = _PreviousActionPolicy(action_dim=env.action_space.total_agent_action_dim)
            replay_buffer = _make_buffer(env)

            segments, episode_infos, _metrics, rollout_state = collect_steps(
                env=env,
                policy=policy,
                replay_buffer=replay_buffer,
                n_steps=2,
            )

            self.assertEqual(len(segments), 1)
            self.assertEqual(replay_buffer.n_transitions, 2)
            segment = segments[0]
            torch.testing.assert_close(segment.local_obs[:, 0, 0].cpu(), torch.tensor([1100.0, 1101.0]))
            torch.testing.assert_close(segment.next_local_obs[:, 0, 0].cpu(), torch.tensor([1101.0, 1102.0]))
            torch.testing.assert_close(segment.previous_actions[:, 0, 0].cpu(), torch.tensor([0.0, 1.0]))
            torch.testing.assert_close(segment.actions[:, 0, 0].cpu(), torch.tensor([1.0, 2.0]))
            torch.testing.assert_close(segment.next_previous_actions[:, 0, 0].cpu(), torch.tensor([1.0, 2.0]))
            self.assertTrue(bool(segment.truncations[-1].item()))
            self.assertFalse(bool(segment.terminations[-1].item()))
            self.assertEqual(episode_infos[0]["r"], 23.0)
            self.assertEqual(_first_obs_value(rollout_state.obs["local_obs"][0]), 1200.0)
            torch.testing.assert_close(rollout_state.previous_actions[0, :, 0].cpu(), torch.zeros(2))
        finally:
            env.close()

    def test_partial_segments_flush_and_resume_mid_episode(self) -> None:
        env = _make_env((1, (4,), "truncate"))
        try:
            policy = _PreviousActionPolicy(action_dim=env.action_space.total_agent_action_dim)
            replay_buffer = _make_buffer(env)

            first_segments, _episode_infos, _metrics, rollout_state = collect_steps(
                env=env,
                policy=policy,
                replay_buffer=replay_buffer,
                n_steps=2,
            )
            second_segments, _episode_infos, _metrics, _rollout_state = collect_steps(
                env=env,
                policy=policy,
                replay_buffer=replay_buffer,
                n_steps=1,
                rollout_state=rollout_state,
            )

            self.assertEqual(len(first_segments), 1)
            self.assertTrue(first_segments[0].is_true_episode_start)
            self.assertEqual(first_segments[0].n_steps, 2)
            self.assertFalse(bool(first_segments[0].dones.any().item()))
            self.assertEqual(len(second_segments), 1)
            self.assertFalse(second_segments[0].is_true_episode_start)
            self.assertEqual(second_segments[0].rollout_start_step, 2)
            torch.testing.assert_close(second_segments[0].local_obs[:, 0, 0].cpu(), torch.tensor([1102.0]))
            self.assertEqual(replay_buffer.n_transitions, 3)
        finally:
            env.close()

    def test_long_episodes_are_split_at_max_segment_length(self) -> None:
        env = _make_env((1, (10,), "truncate"))
        try:
            policy = _PreviousActionPolicy(action_dim=env.action_space.total_agent_action_dim)
            replay_buffer = _make_buffer(env, max_segment_length=2)

            segments, _episode_infos, _metrics, _rollout_state = collect_steps(
                env=env,
                policy=policy,
                replay_buffer=replay_buffer,
                n_steps=5,
            )

            self.assertEqual([segment.n_steps for segment in segments], [2, 2, 1])
            self.assertTrue(segments[0].is_true_episode_start)
            self.assertFalse(segments[1].is_true_episode_start)
            self.assertFalse(segments[2].is_true_episode_start)
            self.assertFalse(bool(segments[0].dones.any().item()))
            self.assertFalse(bool(segments[1].dones.any().item()))
            torch.testing.assert_close(segments[1].local_obs[:, 0, 0].cpu(), torch.tensor([1102.0, 1103.0]))
            torch.testing.assert_close(segments[2].local_obs[:, 0, 0].cpu(), torch.tensor([1104.0]))
            self.assertEqual(replay_buffer.n_transitions, 5)
        finally:
            env.close()

    def test_policy_without_previous_actions_does_not_receive_action_history(self) -> None:
        env = _make_env((1, (3,), "truncate"))
        try:
            policy = _NoPreviousActionPolicy()
            replay_buffer = _make_buffer(env)

            segments, _episode_infos, _metrics, rollout_state = collect_steps(
                env=env,
                policy=policy,
                replay_buffer=replay_buffer,
                n_steps=2,
            )

            self.assertEqual(len(segments), 1)
            torch.testing.assert_close(segments[0].previous_actions[:, 0, 0].cpu(), torch.zeros(2))
            torch.testing.assert_close(segments[0].next_previous_actions[:, 0, 0].cpu(), torch.zeros(2))
            self.assertIsNotNone(rollout_state.previous_actions)
        finally:
            env.close()

    def test_sample_transition_batch_keeps_fields_aligned(self) -> None:
        env = _make_env((1, (4,), "truncate"))
        try:
            policy = _PreviousActionPolicy(action_dim=env.action_space.total_agent_action_dim)
            replay_buffer = _make_buffer(env)
            collect_steps(env=env, policy=policy, replay_buffer=replay_buffer, n_steps=3)

            with patch("torch.randint", return_value=torch.tensor([2, 0])):
                batch = replay_buffer.sample_transition_batch(batch_size=2)

            torch.testing.assert_close(batch.local_obs[:, 0, 0].cpu(), torch.tensor([1102.0, 1100.0]))
            torch.testing.assert_close(batch.next_local_obs[:, 0, 0].cpu(), torch.tensor([1103.0, 1101.0]))
            torch.testing.assert_close(batch.previous_actions[:, 0, 0].cpu(), torch.tensor([2.0, 0.0]))
            torch.testing.assert_close(batch.actions[:, 0, 0].cpu(), torch.tensor([3.0, 1.0]))
            torch.testing.assert_close(batch.next_previous_actions[:, 0, 0].cpu(), torch.tensor([3.0, 1.0]))
            self.assertFalse(bool(batch.dones.any().item()))
        finally:
            env.close()

    def test_without_replacement_sampling_rejects_oversized_batches(self) -> None:
        env = _make_env((1, (4,), "truncate"))
        try:
            policy = _PreviousActionPolicy(action_dim=env.action_space.total_agent_action_dim)
            replay_buffer = _make_buffer(env)
            collect_steps(env=env, policy=policy, replay_buffer=replay_buffer, n_steps=2)

            sampler = replay_buffer.make_sampler(OffPolicySamplerConfig(batch_size=3, replacement=False))
            with self.assertRaisesRegex(ValueError, "without replacement"):
                sampler.sample_batch()
        finally:
            env.close()

    def test_replay_buffer_discards_oldest_transitions_by_transition_capacity(self) -> None:
        env = _make_env((1, (4,), "truncate"))
        try:
            policy = _PreviousActionPolicy(action_dim=env.action_space.total_agent_action_dim)
            replay_buffer = _make_buffer(env, capacity=2)
            _segments, _episode_infos, _metrics, rollout_state = collect_steps(
                env=env,
                policy=policy,
                replay_buffer=replay_buffer,
                n_steps=2,
            )
            collect_steps(
                env=env,
                policy=policy,
                replay_buffer=replay_buffer,
                n_steps=1,
                rollout_state=rollout_state,
            )

            self.assertEqual(replay_buffer.n_transitions, 2)
            with patch("torch.randint", return_value=torch.tensor([0, 1])):
                batch = replay_buffer.sample_transition_batch(batch_size=2)
            torch.testing.assert_close(batch.local_obs[:, 0, 0].cpu(), torch.tensor([1101.0, 1102.0]))
            torch.testing.assert_close(batch.next_local_obs[:, 0, 0].cpu(), torch.tensor([1102.0, 1103.0]))
        finally:
            env.close()

    def test_sequence_batch_follows_env_time_not_global_insert_order(self) -> None:
        env = _make_env((1, (5,), "truncate"), (2, (5,), "truncate"))
        try:
            policy = _PreviousActionPolicy(action_dim=env.action_space.total_agent_action_dim)
            replay_buffer = _make_buffer(env, capacity=16)
            collect_steps(env=env, policy=policy, replay_buffer=replay_buffer, n_steps=6)

            with patch("torch.randint", return_value=torch.tensor([0])):
                batch = replay_buffer.sample_sequence_batch(batch_size=1, horizon=3, require_full=True)

            self.assertEqual(tuple(batch.local_obs.shape[:2]), (1, 3))
            torch.testing.assert_close(batch.local_obs[0, :, 0, 0].cpu(), torch.tensor([1100.0, 1101.0, 1102.0]))
            torch.testing.assert_close(batch.next_local_obs[0, :, 0, 0].cpu(), torch.tensor([1101.0, 1102.0, 1103.0]))
            torch.testing.assert_close(batch.time_mask.cpu(), torch.tensor([[True, True, True]]))
        finally:
            env.close()

    def test_n_step_batch_accumulates_rewards_and_discount(self) -> None:
        env = _make_env((1, (5,), "truncate"))
        try:
            policy = _PreviousActionPolicy(action_dim=env.action_space.total_agent_action_dim)
            replay_buffer = _make_buffer(env, capacity=16)
            collect_steps(env=env, policy=policy, replay_buffer=replay_buffer, n_steps=3)

            with patch("torch.randint", return_value=torch.tensor([0])):
                batch = replay_buffer.sample_n_step_transition_batch(batch_size=1, n_steps=2, gamma=0.5)

            torch.testing.assert_close(batch.rewards.cpu(), torch.tensor([17.0]))
            torch.testing.assert_close(batch.discounts.cpu(), torch.tensor([0.25]))
            torch.testing.assert_close(batch.next_local_obs[:, 0, 0].cpu(), torch.tensor([1102.0]))
            torch.testing.assert_close(batch.steps.cpu(), torch.tensor([2]))
        finally:
            env.close()

    def test_n_step_batch_truncates_at_replay_frontier(self) -> None:
        env = _make_env((1, (5,), "truncate"))
        try:
            policy = _PreviousActionPolicy(action_dim=env.action_space.total_agent_action_dim)
            replay_buffer = _make_buffer(env, capacity=16)
            collect_steps(env=env, policy=policy, replay_buffer=replay_buffer, n_steps=1)

            with patch("torch.randint", return_value=torch.tensor([0])):
                batch = replay_buffer.sample_n_step_transition_batch(batch_size=1, n_steps=2, gamma=0.5)

            torch.testing.assert_close(batch.rewards.cpu(), torch.tensor([11.0]))
            torch.testing.assert_close(batch.discounts.cpu(), torch.tensor([0.5]))
            torch.testing.assert_close(batch.next_local_obs[:, 0, 0].cpu(), torch.tensor([1101.0]))
            torch.testing.assert_close(batch.steps.cpu(), torch.tensor([1]))
            self.assertFalse(bool(batch.terminations.any().item()))
            self.assertFalse(bool(batch.truncations.any().item()))
        finally:
            env.close()


def _first_obs_value(obs: torch.Tensor) -> float:
    return float(obs.detach().cpu()[0, 0].item())


if __name__ == "__main__":
    unittest.main()
