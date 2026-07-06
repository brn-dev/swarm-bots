import unittest
from typing import Any
from unittest.mock import patch

import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv

import swarmbots.learn.algos.off_policy.rollout as off_policy_rollout
from swarmbots.learn.algos.off_policy import OffPolicyReplayBuffer, collect_off_policy_steps
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper


class _ScriptedOffPolicyEnv(gymnasium.Env):
    metadata = {}

    def __init__(
            self,
            *,
            done_steps: tuple[int, ...] = (),
            terminate_steps: tuple[int, ...] = (),
            env_id: int = 0,
            include_agent_mask: bool = False,
    ) -> None:
        super().__init__()
        self.done_steps = done_steps
        self.terminate_steps = terminate_steps
        self.env_id = env_id
        self.include_agent_mask = include_agent_mask
        self.episode_id = 0
        self.step_count = 0
        self.episode_return = 0.0
        observation_spaces: dict[str, spaces.Space] = {
            "local_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(2, 3), dtype=np.float32),
            "global_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "hidden_local_vars": spaces.Box(low=-np.inf, high=np.inf, shape=(2, 1), dtype=np.float32),
            "hidden_global_vars": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        }
        if include_agent_mask:
            observation_spaces["agent_mask"] = spaces.MultiBinary(2)
        self.observation_space = spaces.Dict(observation_spaces)
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
        reward = float(self.step_count)
        self.episode_return += reward
        terminated = self.step_count in self.terminate_steps
        truncated = self.step_count in self.done_steps
        info: dict[str, Any] = {}
        if terminated or truncated:
            info["episode"] = {"r": np.float64(self.episode_return), "success": np.bool_(terminated)}
        return self._obs(), reward, terminated, truncated, info

    def _obs(self) -> dict[str, np.ndarray]:
        value = float(self.env_id * 1000 + self.episode_id * 100 + self.step_count)
        obs = {
            "local_obs": np.full((2, 3), value, dtype=np.float32),
            "global_obs": np.full((2,), value + 0.5, dtype=np.float32),
            "hidden_local_vars": np.full((2, 1), value + 1.0, dtype=np.float32),
            "hidden_global_vars": np.full((1,), value + 2.0, dtype=np.float32),
        }
        if self.include_agent_mask:
            obs["agent_mask"] = np.asarray(
                [True, (self.episode_id + self.step_count) % 2 == 0],
                dtype=np.bool_,
            )
        return obs


def _make_env(
        *,
        done_steps: tuple[int, ...] = (),
        terminate_steps: tuple[int, ...] = (),
) -> SwarmBotsLearnEnvWrapper:
    vector_env = SyncVectorEnv(
        [lambda: _ScriptedOffPolicyEnv(done_steps=done_steps, terminate_steps=terminate_steps)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return SwarmBotsLearnEnvWrapper(vector_env)


def _make_agent_mask_env(*, done_steps: tuple[int, ...] = ()) -> SwarmBotsLearnEnvWrapper:
    vector_env = SyncVectorEnv(
        [lambda: _ScriptedOffPolicyEnv(done_steps=done_steps, include_agent_mask=True)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return SwarmBotsLearnEnvWrapper(vector_env)


def _make_multi_env(*done_steps_per_env: tuple[int, ...]) -> SwarmBotsLearnEnvWrapper:
    vector_env = SyncVectorEnv(
        [
            (lambda env_id=env_id, done_steps=done_steps: _ScriptedOffPolicyEnv(
                done_steps=done_steps,
                env_id=env_id,
            ))
            for env_id, done_steps in enumerate(done_steps_per_env)
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return SwarmBotsLearnEnvWrapper(vector_env)


def _make_buffer(
        env: SwarmBotsLearnEnvWrapper,
        *,
        capacity_per_env: int,
        store_previous_actions: bool = False,
) -> OffPolicyReplayBuffer:
    return OffPolicyReplayBuffer(
        capacity_per_env=capacity_per_env,
        observation_space=env.observation_space,
        action_space=env.action_space,
        store_previous_actions=store_previous_actions,
        storage_device="cpu",
        train_device="cpu",
    )


def _obs(value: float) -> dict[str, torch.Tensor]:
    return {
        "local_obs": torch.full((1, 2, 3), value),
        "global_obs": torch.full((1, 2), value + 0.5),
        "hidden_local_vars": torch.full((1, 2, 1), value + 1.0),
        "hidden_global_vars": torch.full((1, 1), value + 2.0),
    }


def _actions(value: float) -> torch.Tensor:
    return torch.full((1, 2, 2), value)


class _PreviousActionPolicy(BasePolicy):
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
            raise AssertionError("previous_actions must be passed to policies that require them")
        return previous_actions + 1.0

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
            raise AssertionError("previous_actions must not be passed to policies that do not require them")
        return torch.zeros((*local_obs.shape[:2], 2), dtype=local_obs.dtype, device=local_obs.device)

    def update_loss_weights(self, **weights: float) -> None:
        if weights:
            raise ValueError(f"Unknown loss weights: {sorted(weights)}")

    def requires_previous_actions(self) -> bool:
        return False


class OffPolicyReplayTests(unittest.TestCase):
    def test_ring_observation_slots_survive_wraparound(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=3)
            for step in range(5):
                buffer.add(
                    obs=_obs(float(step)),
                    actions=_actions(float(step)),
                    rewards=torch.tensor([float(step)]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([False]),
                    next_obs=_obs(float(step + 1)),
                )

            batch = buffer.get_all()
            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [2.0, 3.0, 4.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [3.0, 4.0, 5.0])
        finally:
            env.close()

    def test_sample_defaults_to_replacement_and_allows_oversized_batches(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=2)
            for step in range(2):
                buffer.add(
                    obs=_obs(float(step)),
                    actions=_actions(float(step)),
                    rewards=torch.tensor([float(step)]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([False]),
                    next_obs=_obs(float(step + 1)),
                )

            batch = buffer.sample(5)
            self.assertEqual(batch.actions.shape[0], 5)
            self.assertEqual(batch.local_obs.shape[0], 5)
        finally:
            env.close()

    def test_sample_without_replacement_rejects_oversized_batches(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=2)
            buffer.add(
                obs=_obs(0.0),
                actions=_actions(0.0),
                rewards=torch.tensor([0.0]),
                terminations=torch.tensor([False]),
                truncations=torch.tensor([False]),
                next_obs=_obs(1.0),
            )

            with self.assertRaisesRegex(ValueError, "without replacement"):
                buffer.sample(2, replacement=False)
        finally:
            env.close()

    def test_replacement_sample_keeps_wrapped_ring_fields_aligned(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=3)
            for step in range(5):
                buffer.add(
                    obs=_obs(float(step)),
                    actions=_actions(float(step + 10)),
                    rewards=torch.tensor([float(step + 20)]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([False]),
                    next_obs=_obs(float(step + 1)),
                )

            with patch("torch.randint", return_value=torch.tensor([0, 2])):
                batch = buffer.sample(2, replacement=True)

            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [2.0, 4.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [3.0, 5.0])
            self.assertEqual(batch.actions[:, 0, 0].tolist(), [12.0, 14.0])
            self.assertEqual(batch.rewards.tolist(), [22.0, 24.0])
        finally:
            env.close()

    def test_done_heavy_observation_stream_survives_wraparound(self) -> None:
        env = _make_env(done_steps=(1,))
        try:
            buffer = _make_buffer(env, capacity_per_env=2)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=5,
                random_actions=True,
            )

            batch = buffer.get_all()
            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [400.0, 500.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [401.0, 501.0])
            self.assertEqual(batch.truncations.tolist(), [True, True])
        finally:
            env.close()

    def test_direct_done_add_uses_streamed_obs_slots_after_first_insert(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=2)
            for step in range(3):
                buffer.add(
                    obs=_obs(float(step * 100)),
                    actions=_actions(float(step)),
                    rewards=torch.tensor([float(step)]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([True]),
                    next_obs=_obs(float((step + 1) * 100)),
                    terminal_obs=_obs(float(step * 100 + 1)),
                )

            batch = buffer.get_all()
            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [100.0, 200.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [101.0, 201.0])
            self.assertEqual(batch.truncations.tolist(), [True, True])
        finally:
            env.close()

    def test_copy_current_obs_true_after_stream_start_is_rejected(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=2)
            buffer.add(
                obs=_obs(0.0),
                actions=_actions(0.0),
                rewards=torch.tensor([0.0]),
                terminations=torch.tensor([False]),
                truncations=torch.tensor([False]),
                next_obs=_obs(1.0),
            )

            with self.assertRaisesRegex(ValueError, "copy_current_obs=True"):
                buffer.add(
                    obs=_obs(1.0),
                    actions=_actions(1.0),
                    rewards=torch.tensor([1.0]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([False]),
                    next_obs=_obs(2.0),
                    copy_current_obs=True,
                )
        finally:
            env.close()

    def test_collect_uses_final_obs_for_done_transition_and_reset_obs_afterward(self) -> None:
        env = _make_env(done_steps=(2,))
        try:
            buffer = _make_buffer(env, capacity_per_env=4, store_previous_actions=True)
            with patch(
                    "swarmbots.learn.algos.off_policy.rollout.sample_random_actions",
                    side_effect=[_actions(1.0), _actions(2.0), _actions(3.0)],
            ):
                episode_infos, metrics, rollout_state = collect_off_policy_steps(
                    env=env,
                    replay_buffer=buffer,
                    n_steps=3,
                    random_actions=True,
                )

            batch = buffer.get_all()
            self.assertEqual(len(buffer), 3)
            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [100.0, 101.0, 200.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [101.0, 102.0, 201.0])
            self.assertEqual(batch.truncations.tolist(), [False, True, False])
            self.assertEqual(batch.previous_actions[:, 0, 0].tolist(), [0.0, 1.0, 0.0])
            self.assertEqual(batch.actions[:, 0, 0].tolist(), [1.0, 2.0, 3.0])
            self.assertEqual(batch.next_previous_actions[:, 0, 0].tolist(), [1.0, 2.0, 3.0])
            self.assertEqual(len(episode_infos), 1)
            self.assertEqual(metrics["transitions_collected"], 3)
            self.assertTrue(torch.equal(rollout_state.episode_start_mask.cpu(), torch.tensor([False])))
            self.assertEqual(rollout_state.previous_actions[:, 0, 0].tolist(), [3.0])
        finally:
            env.close()

    def test_collect_stores_termination_flag_and_terminal_next_obs(self) -> None:
        env = _make_env(terminate_steps=(2,))
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=3,
                random_actions=True,
            )

            batch = buffer.get_all()
            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [100.0, 101.0, 200.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [101.0, 102.0, 201.0])
            self.assertEqual(batch.terminations.tolist(), [False, True, False])
            self.assertEqual(batch.truncations.tolist(), [False, False, False])
        finally:
            env.close()

    def test_collect_resumes_with_rollout_state_in_same_buffer(self) -> None:
        env = _make_env(done_steps=(3,))
        try:
            buffer = _make_buffer(env, capacity_per_env=8)
            _episode_infos, _metrics, rollout_state = collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=2,
                random_actions=True,
            )
            episode_infos, metrics, rollout_state = collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=2,
                rollout_state=rollout_state,
                random_actions=True,
            )

            batch = buffer.get_all()
            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [100.0, 101.0, 102.0, 200.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [101.0, 102.0, 103.0, 201.0])
            self.assertEqual(batch.truncations.tolist(), [False, False, True, False])
            self.assertEqual(len(episode_infos), 1)
            self.assertEqual(metrics["transitions_collected"], 2)
            self.assertEqual(rollout_state.rollout_step_idx, 4)
            self.assertEqual(float(rollout_state.obs["local_obs"][0, 0, 0].item()), 201.0)
        finally:
            env.close()

    def test_collect_only_snapshots_current_obs_when_buffer_needs_it(self) -> None:
        env = _make_env(done_steps=(4,))
        try:
            buffer = _make_buffer(env, capacity_per_env=8)
            with patch(
                    "swarmbots.learn.algos.off_policy.rollout.snapshot_obs",
                    wraps=off_policy_rollout.snapshot_obs,
            ) as snapshot_mock:
                _episode_infos, _metrics, rollout_state = collect_off_policy_steps(
                    env=env,
                    replay_buffer=buffer,
                    n_steps=3,
                    random_actions=True,
                )

            self.assertEqual(snapshot_mock.call_count, 1)

            with patch(
                    "swarmbots.learn.algos.off_policy.rollout.snapshot_obs",
                    wraps=off_policy_rollout.snapshot_obs,
            ) as snapshot_mock:
                collect_off_policy_steps(
                    env=env,
                    replay_buffer=buffer,
                    n_steps=2,
                    rollout_state=rollout_state,
                    random_actions=True,
                )

            self.assertEqual(snapshot_mock.call_count, 0)
        finally:
            env.close()

    def test_collect_applies_explicit_rollout_device_to_env(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            with patch.object(env, "set_device", wraps=env.set_device) as set_device_mock:
                _episode_infos, _metrics, rollout_state = collect_off_policy_steps(
                    env=env,
                    replay_buffer=buffer,
                    n_steps=1,
                    random_actions=True,
                    rollout_device="cpu",
                )

            set_device_mock.assert_called_once_with(torch.device("cpu"))
            self.assertEqual(rollout_state.obs["local_obs"].device, torch.device("cpu"))
        finally:
            env.close()

    def test_collect_preserves_agent_mask_for_terminal_and_reset_obs(self) -> None:
        env = _make_agent_mask_env(done_steps=(2,))
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=3,
                random_actions=True,
            )

            batch = buffer.get_all()
            self.assertEqual(
                batch.agent_mask.tolist(),
                [[True, False], [True, True], [True, True]],
            )
            self.assertEqual(
                batch.next_agent_mask.tolist(),
                [[True, True], [True, False], [True, False]],
            )
        finally:
            env.close()

    def test_policy_rollout_stores_previous_action_history(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4, store_previous_actions=True)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=3,
                policy=_PreviousActionPolicy(),
            )

            batch = buffer.get_all()
            self.assertEqual(batch.previous_actions[:, 0, 0].tolist(), [0.0, 1.0, 2.0])
            self.assertEqual(batch.actions[:, 0, 0].tolist(), [1.0, 2.0, 3.0])
            self.assertEqual(batch.next_previous_actions[:, 0, 0].tolist(), [1.0, 2.0, 3.0])
        finally:
            env.close()

    def test_policy_requiring_previous_actions_requires_replay_storage(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            with self.assertRaisesRegex(ValueError, "store_previous_actions=True"):
                collect_off_policy_steps(
                    env=env,
                    replay_buffer=buffer,
                    n_steps=2,
                    policy=_PreviousActionPolicy(),
                )
        finally:
            env.close()

    def test_policy_without_previous_actions_does_not_receive_tracked_history(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4, store_previous_actions=True)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=2,
                policy=_NoPreviousActionPolicy(),
            )

            batch = buffer.get_all()
            self.assertEqual(batch.previous_actions[:, 0, 0].tolist(), [0.0, 0.0])
            self.assertEqual(batch.next_previous_actions[:, 0, 0].tolist(), [0.0, 0.0])
        finally:
            env.close()

    def test_previous_action_storage_is_optional(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=2,
                random_actions=True,
            )

            batch = buffer.get_all()
            self.assertIsNone(batch.previous_actions)
            self.assertIsNone(batch.next_previous_actions)
        finally:
            env.close()

    def test_collect_keeps_terminal_obs_per_done_lane(self) -> None:
        env = _make_multi_env((), (2,))
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=6,
                random_actions=True,
            )

            batch = buffer.get_all()
            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [100.0, 101.0, 102.0, 1100.0, 1101.0, 1200.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [101.0, 102.0, 103.0, 1101.0, 1102.0, 1201.0])
            self.assertEqual(batch.truncations.tolist(), [False, False, False, False, True, False])
        finally:
            env.close()

    def test_multi_env_asymmetric_done_lanes_survive_wraparound(self) -> None:
        env = _make_multi_env((), (1,))
        try:
            buffer = _make_buffer(env, capacity_per_env=3)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=10,
                random_actions=True,
            )

            batch = buffer.get_all()
            self.assertEqual(
                batch.local_obs[:, 0, 0].tolist(),
                [102.0, 103.0, 104.0, 1300.0, 1400.0, 1500.0],
            )
            self.assertEqual(
                batch.next_local_obs[:, 0, 0].tolist(),
                [103.0, 104.0, 105.0, 1301.0, 1401.0, 1501.0],
            )
            self.assertEqual(batch.truncations.tolist(), [False, False, False, True, True, True])
        finally:
            env.close()

    def test_collect_rejects_non_multiple_transition_count(self) -> None:
        env = _make_multi_env((), ())
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            with self.assertRaisesRegex(ValueError, "multiple"):
                collect_off_policy_steps(
                    env=env,
                    replay_buffer=buffer,
                    n_steps=3,
                    random_actions=True,
                )
        finally:
            env.close()

    def test_collect_requires_rollout_state_for_non_empty_buffer(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=1,
                random_actions=True,
            )

            with self.assertRaisesRegex(ValueError, "rollout_state"):
                collect_off_policy_steps(
                    env=env,
                    replay_buffer=buffer,
                    n_steps=1,
                    random_actions=True,
                )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
