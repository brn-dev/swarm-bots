import unittest
from typing import Any

import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo_rollout import collect_steps, collect_whole_episodes
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPORolloutBuffer, PPOEpisodeSegment
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSampler, PPOSamplerConfig
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.env_wrappers.torch_normalize_reward_wrapper import TorchNormalizeRewardWrapper
from swarmbots.learn.env_wrappers.torch_progress_guidance_ep_stats_wrapper import TorchProgressGuidanceEpisodeStatsWrapper
from swarmbots.learn.env_wrappers.torch_record_episode_statistics_wrapper import TorchRecordEpisodeStatisticsWrapper
from swarmbots.learn.env_wrappers.torch_transition_obs_wrapper import TorchTransitionObsWrapper
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.gsde_reset import GSDEIntervalResetMode, GSDEProbabilityResetMode
from swarmbots.learn.testing_env import TestingSwarmBotsEnv


class _DummyActionDist:
    has_gsde = False

    def reset_temporal_correlations_on_ep_start(self, mask: torch.Tensor) -> None:
        _ = mask

    def reset_temporal_correlations_on_step(
            self,
            mask: torch.Tensor | None = None,
            batch_shape: tuple[int, ...] | None = None,
    ) -> None:
        _ = mask
        _ = batch_shape


class _SpyGSDEActionDist:
    has_gsde = True

    def __init__(self) -> None:
        self.episode_resets: list[torch.Tensor] = []
        self.step_resets: list[tuple[torch.Tensor | None, tuple[int, ...] | None]] = []

    def reset_temporal_correlations_on_ep_start(self, mask: torch.Tensor) -> None:
        self.episode_resets.append(mask.clone())

    def reset_temporal_correlations_on_step(
            self,
            mask: torch.Tensor | None = None,
            batch_shape: tuple[int, ...] | None = None,
    ) -> None:
        self.step_resets.append((None if mask is None else mask.clone(), batch_shape))


class _BaseTestPolicy(BasePPOPolicy[PPOSampler, PPOSamplerConfig]):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self._action_dim = action_dim
        self.action_dist = _DummyActionDist()

    def _evaluate_actions(self, batch, action_splitter=None):
        raise NotImplementedError

    def predict_values(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, _, values = self.forward(
            local_obs,
            global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
        )
        return values

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
            local_obs,
            global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
        )
        return actions


class _ConstantValuePolicy(_BaseTestPolicy):
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
        actions = torch.zeros((batch_size, n_agents, self._action_dim), device=local_obs.device, dtype=local_obs.dtype)
        log_probs = torch.zeros((batch_size, n_agents), device=local_obs.device, dtype=local_obs.dtype)
        values = local_obs[..., 0].mean(dim=1)
        return actions, log_probs, values

    def requires_previous_actions(self) -> bool:
        return False


class _PreviousActionPolicy(_BaseTestPolicy):
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
        _ = deterministic
        batch_size, n_agents, _ = local_obs.shape
        if previous_actions is None:
            raise AssertionError("previous_actions should be provided")
        action_value = previous_actions[..., :1] + 1.0
        actions = action_value.expand(batch_size, n_agents, self._action_dim).clone()
        log_probs = torch.zeros((batch_size, n_agents), device=local_obs.device, dtype=local_obs.dtype)
        values = local_obs[..., 0].mean(dim=1)
        return actions, log_probs, values

    def requires_previous_actions(self) -> bool:
        return True


class _ValueOnlyBootstrapPolicy(_BaseTestPolicy):
    def __init__(self, action_dim: int) -> None:
        super().__init__(action_dim=action_dim)
        self.forward_calls = 0
        self.predict_value_calls = 0
        self.temporal_state = 0

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
        self.forward_calls += 1
        batch_size, n_agents, _ = local_obs.shape
        actions = torch.zeros((batch_size, n_agents, self._action_dim), device=local_obs.device, dtype=local_obs.dtype)
        log_probs = torch.zeros((batch_size, n_agents), device=local_obs.device, dtype=local_obs.dtype)
        values = torch.full((batch_size,), -1234.0, device=local_obs.device, dtype=local_obs.dtype)
        return actions, log_probs, values

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
        self.predict_value_calls += 1
        self.temporal_state += 1
        return local_obs[..., 0].mean(dim=1) + 0.5

    def get_temporal_state_snapshot(self) -> int:
        return self.temporal_state

    def restore_temporal_state_snapshot(self, snapshot: int) -> None:
        self.temporal_state = snapshot

    def requires_previous_actions(self) -> bool:
        return False


class _ScriptedRolloutEnv(gymnasium.Env):
    metadata = {}

    def __init__(
            self,
            *,
            env_id: int,
            done_steps: tuple[int, ...],
            done_mode: str,
    ) -> None:
        super().__init__()
        self.env_id = env_id
        self.done_steps = tuple(done_steps)
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
            info["episode"] = {
                "r": np.float64(self.episode_return),
                "env_marker": np.float64(self.env_id * 100 + self.step_count),
            }
        return self._obs(), reward, terminated, truncated, info

    def _obs(self) -> dict[str, np.ndarray]:
        base = float(self.env_id * 1000 + self.episode_id * 100 + self.step_count)
        local_obs = np.full((2, 3), base, dtype=np.float32)
        global_obs = np.array([base, base + 0.5], dtype=np.float32)
        hidden_local_vars = np.full((2, 1), base + 1.0, dtype=np.float32)
        hidden_global_vars = np.array([base + 2.0], dtype=np.float32)
        return {
            "local_obs": local_obs,
            "global_obs": global_obs,
            "hidden_local_vars": hidden_local_vars,
            "hidden_global_vars": hidden_global_vars,
        }


class _AgentMaskRolloutEnv(_ScriptedRolloutEnv):
    def __init__(
            self,
            *,
            env_id: int,
            done_steps: tuple[int, ...],
            done_mode: str,
    ) -> None:
        super().__init__(env_id=env_id, done_steps=done_steps, done_mode=done_mode)
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            "agent_mask": spaces.MultiBinary(2),
        })

    def _obs(self) -> dict[str, np.ndarray]:
        obs = super()._obs()
        mask = np.array(
            [1, 1 if ((self.episode_id + self.step_count) % 2 == 0) else 0],
            dtype=np.int8,
        )
        obs["agent_mask"] = mask
        return obs


class _RewardInfoRolloutEnv(_ScriptedRolloutEnv):
    def step(
            self,
            action: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        info.pop("episode", None)
        info["progress_reward"] = np.float64(self.env_id + self.step_count)
        info["guidance_reward"] = np.float64(10 * self.env_id + self.step_count)
        return obs, reward, terminated, truncated, info


class _EnvProxy:
    def __init__(self, env: SwarmBotsLearnEnvWrapper, mutate_step_infos) -> None:
        self.env = env
        self._mutate_step_infos = mutate_step_infos

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return obs, rewards, terminations, truncations, self._mutate_step_infos(infos)

    def _obs_to_torch(self, obs):
        return self.env._obs_to_torch(obs)

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, item: str):
        return getattr(self.env, item)


class _ObsReuseProxy:
    def __init__(self, env: SwarmBotsLearnEnvWrapper) -> None:
        self.env = env
        self._shared_obs: dict[str, torch.Tensor] | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._shared_obs = obs
        return obs, info

    def step(self, actions):
        next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
        if self._shared_obs is None:
            raise AssertionError("reset() must be called before step()")

        for key, value in next_obs.items():
            self._shared_obs[key].copy_(value)
        return self._shared_obs, rewards, terminations, truncations, infos

    def close(self) -> None:
        self.env.close()

    def _obs_to_torch(self, obs):
        return self.env._obs_to_torch(obs)

    def __getattr__(self, item: str):
        return getattr(self.env, item)


def _to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu()


def _first_obs_value(obs: torch.Tensor) -> float:
    return float(_to_cpu(obs)[0, 0].item())


def _make_single_env(max_steps: int = 2) -> SwarmBotsLearnEnvWrapper:
    vector_env = SyncVectorEnv(
        [lambda: TestingSwarmBotsEnv(2, 3, 2, 1, 1, max_steps=max_steps)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return SwarmBotsLearnEnvWrapper(vector_env)


def _make_transition_wrapped_env(max_steps: int = 2) -> SwarmBotsLearnEnvWrapper:
    env = _make_single_env(max_steps=max_steps)
    env = TorchFeatureWiseObsNormWrapper(env, obs_key="local_obs", scalar_feature_indices=[], quaternion_indices=[])
    env = TorchFeatureWiseObsNormWrapper(env, obs_key="global_obs", scalar_feature_indices=[], quaternion_indices=[])
    env = TorchFeatureWiseObsNormWrapper(
        env,
        obs_key="hidden_local_vars",
        scalar_feature_indices=[],
        quaternion_indices=[],
    )
    env = TorchFeatureWiseObsNormWrapper(
        env,
        obs_key="hidden_global_vars",
        scalar_feature_indices=[],
        quaternion_indices=[],
    )
    env = TorchTransitionObsWrapper(env)
    return env


def _make_scripted_env(*configs: tuple[int, tuple[int, ...], str]) -> SwarmBotsLearnEnvWrapper:
    vector_env = SyncVectorEnv(
        [
            (lambda env_id=env_id, done_steps=done_steps, done_mode=done_mode: _ScriptedRolloutEnv(
                env_id=env_id,
                done_steps=done_steps,
                done_mode=done_mode,
            ))
            for env_id, done_steps, done_mode in configs
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return SwarmBotsLearnEnvWrapper(vector_env)


def _make_agent_mask_env(*configs: tuple[int, tuple[int, ...], str]) -> SwarmBotsLearnEnvWrapper:
    vector_env = SyncVectorEnv(
        [
            (lambda env_id=env_id, done_steps=done_steps, done_mode=done_mode: _AgentMaskRolloutEnv(
                env_id=env_id,
                done_steps=done_steps,
                done_mode=done_mode,
            ))
            for env_id, done_steps, done_mode in configs
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return SwarmBotsLearnEnvWrapper(vector_env)


def _make_reward_info_env(*configs: tuple[int, tuple[int, ...], str]) -> SwarmBotsLearnEnvWrapper:
    vector_env = SyncVectorEnv(
        [
            (lambda env_id=env_id, done_steps=done_steps, done_mode=done_mode: _RewardInfoRolloutEnv(
                env_id=env_id,
                done_steps=done_steps,
                done_mode=done_mode,
            ))
            for env_id, done_steps, done_mode in configs
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return SwarmBotsLearnEnvWrapper(vector_env)


def _make_buffer(env: SwarmBotsLearnEnvWrapper) -> PPORolloutBuffer:
    return PPORolloutBuffer(
        max_episode_length=8,
        observation_space=env.observation_space,
        action_space=env.action_space,
        gamma=1.0,
        gae_lambda=1.0,
        rollout_device="cpu",
        train_device="cpu",
    )


def _transition_actions_from_local(obs: dict[str, torch.Tensor], *, scale: float = 1.0) -> torch.Tensor:
    return torch.cat(
        (
            obs["local_obs"][..., :1] * scale,
            torch.zeros((*obs["local_obs"].shape[:2], 1), dtype=obs["local_obs"].dtype, device=obs["local_obs"].device),
        ),
        dim=-1,
    )


def _assert_transition_obs_parts(
        test_case: unittest.TestCase,
        obs: dict[str, torch.Tensor],
        *,
        current_local: torch.Tensor,
        prev_actions: torch.Tensor,
        prev_local: torch.Tensor,
        current_global: torch.Tensor,
        prev_global: torch.Tensor,
        new_obs_first: bool = True,
) -> None:
    if new_obs_first:
        actual_current_local = obs["local_obs"][..., :3]
        actual_prev_actions = obs["local_obs"][..., 3:5]
        actual_prev_local = obs["local_obs"][..., 5:8]
        actual_current_global = obs["global_obs"][..., :2]
        actual_prev_global = obs["global_obs"][..., 2:4]
    else:
        actual_prev_local = obs["local_obs"][..., :3]
        actual_prev_actions = obs["local_obs"][..., 3:5]
        actual_current_local = obs["local_obs"][..., 5:8]
        actual_prev_global = obs["global_obs"][..., :2]
        actual_current_global = obs["global_obs"][..., 2:4]

    torch.testing.assert_close(actual_current_local, current_local)
    torch.testing.assert_close(actual_prev_actions, prev_actions)
    torch.testing.assert_close(actual_prev_local, prev_local)
    torch.testing.assert_close(actual_current_global, current_global)
    torch.testing.assert_close(actual_prev_global, prev_global)


class SameStepPipelineTests(unittest.TestCase):
    def test_learn_env_wrapper_requires_same_step(self) -> None:
        vector_env = SyncVectorEnv(
            [lambda: TestingSwarmBotsEnv(2, 3, 2, 1, 1, max_steps=2)],
            autoreset_mode=AutoresetMode.NEXT_STEP,
        )
        try:
            with self.assertRaisesRegex(ValueError, "autoreset_mode=SAME_STEP"):
                SwarmBotsLearnEnvWrapper(vector_env)
        finally:
            vector_env.close()

    def test_collect_steps_uses_final_obs_for_truncation_bootstrap(self) -> None:
        env = _make_single_env(max_steps=2)
        try:
            policy = _ConstantValuePolicy(action_dim=env.action_space.total_agent_action_dim)
            buffer = _make_buffer(env)

            episodes, _episode_infos, _metrics, rollout_state = collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=2,
            )

            self.assertEqual(len(episodes), 1)
            episode = episodes[0]
            self.assertTrue(torch.equal(_to_cpu(episode.local_obs[:, 0, 0]), torch.tensor([0.0, 1.0])))
            self.assertTrue(torch.all(_to_cpu(episode.final_local_obs[:, 0]) == 2.0))
            self.assertTrue(torch.allclose(_to_cpu(episode.final_value), torch.tensor(2.0)))
            self.assertTrue(bool(rollout_state.episode_start_mask[0].item()))
            self.assertTrue(torch.all(_to_cpu(rollout_state.obs["local_obs"][0, :, 0]) == 0.0))
        finally:
            env.close()

    def test_collect_steps_uses_value_only_path_for_bootstrap_values(self) -> None:
        env = _make_single_env(max_steps=2)
        try:
            policy = _ValueOnlyBootstrapPolicy(action_dim=env.action_space.total_agent_action_dim)
            buffer = _make_buffer(env)

            episodes, _episode_infos, _metrics, _rollout_state = collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=2,
            )

            self.assertEqual(len(episodes), 1)
            self.assertEqual(float(_to_cpu(episodes[0].final_value).item()), 2.5)
            self.assertEqual(policy.forward_calls, 2)
            self.assertGreaterEqual(policy.predict_value_calls, 2)
            self.assertEqual(policy.temporal_state, 0)
        finally:
            env.close()

    def test_collect_steps_resumes_from_reset_obs_without_dummy_final_step(self) -> None:
        env = _make_single_env(max_steps=2)
        try:
            policy = _ConstantValuePolicy(action_dim=env.action_space.total_agent_action_dim)
            buffer = _make_buffer(env)

            _episodes, _episode_infos, _metrics, rollout_state = collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=2,
            )
            resumed_episodes, _episode_infos, _metrics, _next_state = collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=1,
                rollout_state=rollout_state,
            )

            self.assertEqual(len(resumed_episodes), 1)
            resumed_episode = resumed_episodes[0]
            self.assertEqual(resumed_episode.local_obs.shape[0], 1)
            self.assertTrue(torch.all(_to_cpu(resumed_episode.local_obs[0, :, 0]) == 0.0))
            self.assertTrue(resumed_episode.is_true_episode_start)
        finally:
            env.close()

    def test_collect_steps_zeroes_bootstrap_value_for_terminations(self) -> None:
        env = _make_scripted_env((1, (2,), "terminate"))
        try:
            policy = _ConstantValuePolicy(action_dim=env.action_space.total_agent_action_dim)
            buffer = _make_buffer(env)

            episodes, episode_infos, _metrics, rollout_state = collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=2,
            )

            self.assertEqual(len(episodes), 1)
            self.assertEqual(len(episode_infos), 1)
            episode = episodes[0]
            self.assertEqual(_first_obs_value(episode.final_local_obs), 1102.0)
            self.assertEqual(float(_to_cpu(episode.final_value).item()), 0.0)
            self.assertEqual(episode_infos[0]["env_marker"], np.float64(102.0))
            self.assertEqual(_first_obs_value(rollout_state.obs["local_obs"][0]), 1200.0)
        finally:
            env.close()

    def test_collect_steps_handles_mixed_done_and_partial_envs(self) -> None:
        env = _make_scripted_env(
            (1, (2,), "truncate"),
            (2, (4,), "truncate"),
        )
        try:
            policy = _ConstantValuePolicy(action_dim=env.action_space.total_agent_action_dim)
            buffer = _make_buffer(env)

            episodes, episode_infos, _metrics, rollout_state = collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=4,
            )

            self.assertEqual(len(episodes), 2)
            completed_episode, partial_episode = episodes
            self.assertEqual(completed_episode.local_obs.shape[0], 2)
            self.assertEqual(_first_obs_value(completed_episode.final_local_obs), 1102.0)
            self.assertEqual(float(_to_cpu(completed_episode.final_value).item()), 1102.0)
            self.assertEqual(len(episode_infos), 1)
            self.assertEqual(episode_infos[0]["env_marker"], np.float64(102.0))

            self.assertEqual(partial_episode.local_obs.shape[0], 2)
            self.assertTrue(partial_episode.is_true_episode_start)
            self.assertEqual(_first_obs_value(partial_episode.local_obs[0]), 2100.0)
            self.assertEqual(_first_obs_value(partial_episode.final_local_obs), 2102.0)
            self.assertEqual(float(_to_cpu(partial_episode.final_value).item()), 2102.0)

            self.assertTrue(torch.equal(_to_cpu(rollout_state.episode_start_mask), torch.tensor([True, False])))
            self.assertEqual(_first_obs_value(rollout_state.obs["local_obs"][0]), 1200.0)
            self.assertEqual(_first_obs_value(rollout_state.obs["local_obs"][1]), 2102.0)
        finally:
            env.close()

    def test_collect_whole_episodes_collects_done_env_infos_across_envs(self) -> None:
        env = _make_scripted_env(
            (1, (2,), "truncate"),
            (2, (3,), "truncate"),
        )
        try:
            policy = _ConstantValuePolicy(action_dim=env.action_space.total_agent_action_dim)
            buffer = _make_buffer(env)

            episodes, episode_infos, _metrics = collect_whole_episodes(
                env=env,
                policy=policy,
                buffer=buffer,
                n_episodes=2,
            )

            self.assertEqual(len(episodes), 2)
            self.assertEqual([episode.local_obs.shape[0] for episode in episodes], [2, 3])
            self.assertEqual([_first_obs_value(episode.final_local_obs) for episode in episodes], [1102.0, 2103.0])
            self.assertEqual([info["env_marker"] for info in episode_infos], [np.float64(102.0), np.float64(203.0)])
        finally:
            env.close()

    def test_partial_rollout_resume_marks_mid_episode_as_not_true_start(self) -> None:
        env = _make_scripted_env((1, (4,), "truncate"))
        try:
            policy = _PreviousActionPolicy(action_dim=env.action_space.total_agent_action_dim)
            buffer = _make_buffer(env)

            first_episodes, _episode_infos, _metrics, rollout_state = collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=2,
            )
            self.assertEqual(len(first_episodes), 1)
            self.assertTrue(first_episodes[0].is_true_episode_start)
            self.assertTrue(torch.allclose(_to_cpu(first_episodes[0].initial_previous_actions), torch.zeros((2, 2))))

            resumed_episodes, _episode_infos, _metrics, _next_state = collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=1,
                rollout_state=rollout_state,
            )

            self.assertEqual(len(resumed_episodes), 1)
            resumed_episode = resumed_episodes[0]
            self.assertFalse(resumed_episode.is_true_episode_start)
            self.assertTrue(torch.allclose(_to_cpu(resumed_episode.initial_previous_actions), torch.full((2, 2), 2.0)))
            self.assertEqual(resumed_episode.local_obs.shape[0], 1)
            self.assertEqual(_first_obs_value(resumed_episode.local_obs[0]), 1102.0)
        finally:
            env.close()

    def test_collect_steps_handles_simultaneous_dones(self) -> None:
        env = _make_scripted_env(
            (1, (1,), "truncate"),
            (2, (1,), "truncate"),
        )
        try:
            policy = _ConstantValuePolicy(action_dim=env.action_space.total_agent_action_dim)
            buffer = _make_buffer(env)

            episodes, episode_infos, _metrics, rollout_state = collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=2,
            )

            self.assertEqual(len(episodes), 2)
            self.assertEqual([episode.local_obs.shape[0] for episode in episodes], [1, 1])
            self.assertEqual([_first_obs_value(episode.final_local_obs) for episode in episodes], [1101.0, 2101.0])
            self.assertEqual([info["env_marker"] for info in episode_infos], [np.float64(101.0), np.float64(201.0)])
            self.assertTrue(torch.equal(_to_cpu(rollout_state.episode_start_mask), torch.tensor([True, True])))
            self.assertEqual(_first_obs_value(rollout_state.obs["local_obs"][0]), 1200.0)
            self.assertEqual(_first_obs_value(rollout_state.obs["local_obs"][1]), 2200.0)
        finally:
            env.close()

    def test_previous_actions_are_reset_after_done_before_next_episode(self) -> None:
        env = _make_scripted_env((1, (2,), "truncate"))
        try:
            policy = _PreviousActionPolicy(action_dim=env.action_space.total_agent_action_dim)
            buffer = _make_buffer(env)

            completed_episodes, _episode_infos, _metrics, rollout_state = collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=2,
            )
            resumed_episodes, _episode_infos, _metrics, _next_state = collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=1,
                rollout_state=rollout_state,
            )

            self.assertEqual(len(completed_episodes), 1)
            self.assertTrue(torch.allclose(_to_cpu(rollout_state.previous_actions), torch.zeros((1, 2, 2))))
            self.assertEqual(len(resumed_episodes), 1)
            resumed_episode = resumed_episodes[0]
            self.assertTrue(resumed_episode.is_true_episode_start)
            self.assertTrue(torch.allclose(_to_cpu(resumed_episode.initial_previous_actions), torch.zeros((2, 2))))
        finally:
            env.close()

    def test_transition_obs_can_normalize_previous_binary_actions(self) -> None:
        env = _make_single_env(max_steps=3)
        env = TorchTransitionObsWrapper(env, normalize_prev_binary_actions=True)
        try:
            obs, _ = env.reset()
            torch.testing.assert_close(obs["local_obs"][0, :, 3:5], torch.zeros((2, 2)))

            actions = torch.tensor([[[0.25, 1.0], [-0.5, 0.0]]], dtype=torch.float32)
            obs, _, _, _, _ = env.step(actions)

            expected_prev_actions = torch.tensor([[0.25, 1.0], [-0.5, -1.0]], dtype=torch.float32)
            torch.testing.assert_close(obs["local_obs"][0, :, 3:5], expected_prev_actions)
        finally:
            env.close()

    def test_transition_obs_non_shuffle_reset_starts_with_zero_history(self) -> None:
        env = TorchTransitionObsWrapper(_make_scripted_env((1, (99,), "truncate")))
        try:
            obs, _info = env.reset()

            _assert_transition_obs_parts(
                self,
                obs,
                current_local=torch.full((1, 2, 3), 1100.0),
                prev_actions=torch.zeros((1, 2, 2)),
                prev_local=torch.zeros((1, 2, 3)),
                current_global=torch.tensor([[1100.0, 1100.5]]),
                prev_global=torch.zeros((1, 2)),
            )
        finally:
            env.close()

    def test_transition_obs_non_shuffle_non_done_step_carries_previous_obs_and_actions(self) -> None:
        env = TorchTransitionObsWrapper(_make_scripted_env((1, (99,), "truncate")))
        try:
            obs, _info = env.reset()
            actions = torch.tensor([[[0.25, 1.0], [-0.5, 0.0]]], dtype=torch.float32)

            next_obs, _rewards, terminations, truncations, infos = env.step(actions)

            self.assertFalse(bool(terminations[0].item()))
            self.assertFalse(bool(truncations[0].item()))
            if "_final_obs" in infos:
                self.assertFalse(bool(torch.as_tensor(infos["_final_obs"])[0].item()))
            _assert_transition_obs_parts(
                self,
                next_obs,
                current_local=torch.full((1, 2, 3), 1101.0),
                prev_actions=actions,
                prev_local=obs["local_obs"][..., :3],
                current_global=torch.tensor([[1101.0, 1101.5]]),
                prev_global=obs["global_obs"][..., :2],
            )
        finally:
            env.close()

    def test_transition_obs_non_shuffle_same_step_done_zeroes_reset_obs_but_not_final_obs(self) -> None:
        env = TorchTransitionObsWrapper(_make_scripted_env((1, (1,), "truncate")))
        try:
            obs, _info = env.reset()
            actions = torch.tensor([[[0.25, 1.0], [-0.5, 0.0]]], dtype=torch.float32)

            reset_obs, _rewards, _terminations, truncations, infos = env.step(actions)

            self.assertTrue(bool(truncations[0].item()))
            _assert_transition_obs_parts(
                self,
                reset_obs,
                current_local=torch.full((1, 2, 3), 1200.0),
                prev_actions=torch.zeros((1, 2, 2)),
                prev_local=torch.zeros((1, 2, 3)),
                current_global=torch.tensor([[1200.0, 1200.5]]),
                prev_global=torch.zeros((1, 2)),
            )
            _assert_transition_obs_parts(
                self,
                {
                    "local_obs": infos["final_obs"][0]["local_obs"].unsqueeze(0),
                    "global_obs": infos["final_obs"][0]["global_obs"].unsqueeze(0),
                },
                current_local=torch.full((1, 2, 3), 1101.0),
                prev_actions=actions,
                prev_local=obs["local_obs"][..., :3],
                current_global=torch.tensor([[1101.0, 1101.5]]),
                prev_global=obs["global_obs"][..., :2],
            )
        finally:
            env.close()

    def test_transition_obs_non_shuffle_false_new_obs_first_preserves_done_layout(self) -> None:
        env = TorchTransitionObsWrapper(_make_scripted_env((1, (1,), "truncate")), new_obs_first=False)
        try:
            obs, _info = env.reset()
            actions = torch.tensor([[[0.25, 1.0], [-0.5, 0.0]]], dtype=torch.float32)

            reset_obs, _rewards, _terminations, truncations, infos = env.step(actions)

            self.assertTrue(bool(truncations[0].item()))
            _assert_transition_obs_parts(
                self,
                reset_obs,
                current_local=torch.full((1, 2, 3), 1200.0),
                prev_actions=torch.zeros((1, 2, 2)),
                prev_local=torch.zeros((1, 2, 3)),
                current_global=torch.tensor([[1200.0, 1200.5]]),
                prev_global=torch.zeros((1, 2)),
                new_obs_first=False,
            )
            _assert_transition_obs_parts(
                self,
                {
                    "local_obs": infos["final_obs"][0]["local_obs"].unsqueeze(0),
                    "global_obs": infos["final_obs"][0]["global_obs"].unsqueeze(0),
                },
                current_local=torch.full((1, 2, 3), 1101.0),
                prev_actions=actions,
                prev_local=obs["local_obs"][..., 5:8],
                current_global=torch.tensor([[1101.0, 1101.5]]),
                prev_global=obs["global_obs"][..., 2:4],
                new_obs_first=False,
            )
        finally:
            env.close()

    def test_transition_obs_non_shuffle_explicit_partial_reset_preserves_unreset_history(self) -> None:
        env = TorchTransitionObsWrapper(_make_scripted_env(
            (1, (99,), "truncate"),
            (2, (99,), "truncate"),
        ))
        try:
            obs, _info = env.reset()
            first_actions = _transition_actions_from_local(obs, scale=0.01)
            first_next_obs, _rewards, _terminations, truncations, _infos = env.step(first_actions)
            self.assertTrue(torch.equal(truncations.cpu(), torch.tensor([False, False])))

            second_actions = _transition_actions_from_local(first_next_obs, scale=0.01)
            second_next_obs, _rewards, _terminations, truncations, _infos = env.step(second_actions)
            self.assertTrue(torch.equal(truncations.cpu(), torch.tensor([False, False])))

            partial_reset_obs, _info = env.reset(options={"reset_mask": np.array([True, False])})

            _assert_transition_obs_parts(
                self,
                partial_reset_obs,
                current_local=torch.stack((
                    torch.full((2, 3), 1200.0),
                    torch.full((2, 3), 2102.0),
                )).to(torch.float32),
                prev_actions=torch.stack((
                    torch.zeros((2, 2)),
                    second_actions[1],
                )).to(torch.float32),
                prev_local=torch.stack((
                    torch.zeros((2, 3)),
                    first_next_obs["local_obs"][1, :, :3],
                )).to(torch.float32),
                current_global=torch.tensor([[1200.0, 1200.5], [2102.0, 2102.5]]),
                prev_global=torch.stack((
                    torch.zeros((2,)),
                    first_next_obs["global_obs"][1, :2],
                )).to(torch.float32),
            )

            third_actions = _transition_actions_from_local(partial_reset_obs, scale=0.01)
            third_next_obs, _rewards, _terminations, truncations, _infos = env.step(third_actions)

            self.assertTrue(torch.equal(truncations.cpu(), torch.tensor([False, False])))
            _assert_transition_obs_parts(
                self,
                third_next_obs,
                current_local=torch.stack((
                    torch.full((2, 3), 1201.0),
                    torch.full((2, 3), 2103.0),
                )).to(torch.float32),
                prev_actions=third_actions,
                prev_local=partial_reset_obs["local_obs"][..., :3],
                current_global=torch.tensor([[1201.0, 1201.5], [2103.0, 2103.5]]),
                prev_global=partial_reset_obs["global_obs"][..., :2],
            )

        finally:
            env.close()

    def test_full_non_shuffle_wrapper_chain_preserves_same_step_final_obs_and_stats(self) -> None:
        env: SwarmBotsLearnEnvWrapper = _make_reward_info_env(
            (1, (2,), "truncate"),
            (2, (3,), "truncate"),
        )
        env = TorchRecordEpisodeStatisticsWrapper(env)
        env = TorchProgressGuidanceEpisodeStatsWrapper(env)
        env = TorchFeatureWiseObsNormWrapper(env, obs_key="local_obs", scalar_feature_indices=[], quaternion_indices=[])
        env = TorchFeatureWiseObsNormWrapper(env, obs_key="global_obs", scalar_feature_indices=[], quaternion_indices=[])
        env = TorchTransitionObsWrapper(env)
        env = TorchNormalizeRewardWrapper(env, gamma=1.0)
        env.update_running_mean = False
        try:
            obs, _info = env.reset()
            first_actions = _transition_actions_from_local(obs, scale=0.01)
            first_next_obs, rewards, _terminations, truncations, _infos = env.step(first_actions)
            self.assertTrue(torch.isfinite(rewards).all())
            self.assertTrue(torch.equal(truncations.cpu(), torch.tensor([False, False])))

            second_actions = _transition_actions_from_local(first_next_obs, scale=0.01)
            reset_obs, rewards, _terminations, truncations, infos = env.step(second_actions)

            self.assertTrue(torch.isfinite(rewards).all())
            self.assertTrue(torch.equal(truncations.cpu(), torch.tensor([True, False])))
            _assert_transition_obs_parts(
                self,
                reset_obs,
                current_local=torch.stack((
                    torch.full((2, 3), 1200.0),
                    torch.full((2, 3), 2102.0),
                )).to(torch.float32),
                prev_actions=torch.stack((
                    torch.zeros((2, 2)),
                    second_actions[1],
                )).to(torch.float32),
                prev_local=torch.stack((
                    torch.zeros((2, 3)),
                    first_next_obs["local_obs"][1, :, :3],
                )).to(torch.float32),
                current_global=torch.tensor([[1200.0, 1200.5], [2102.0, 2102.5]]),
                prev_global=torch.stack((
                    torch.zeros((2,)),
                    first_next_obs["global_obs"][1, :2],
                )).to(torch.float32),
            )
            _assert_transition_obs_parts(
                self,
                {
                    "local_obs": infos["final_obs"][0]["local_obs"].unsqueeze(0),
                    "global_obs": infos["final_obs"][0]["global_obs"].unsqueeze(0),
                },
                current_local=torch.full((1, 2, 3), 1102.0),
                prev_actions=second_actions[0:1],
                prev_local=first_next_obs["local_obs"][0:1, :, :3],
                current_global=torch.tensor([[1102.0, 1102.5]]),
                prev_global=first_next_obs["global_obs"][0:1, :2],
            )
            self.assertTrue(torch.equal(torch.as_tensor(infos["_episode"]).cpu(), torch.tensor([True, False])))
            self.assertEqual(float(infos["episode"]["r"][0].item()), 23.0)
            self.assertEqual(int(infos["episode"]["l"][0].item()), 2)
            self.assertEqual(float(infos["episode"]["progress_reward"][0].item()), 5.0)
            self.assertEqual(float(infos["episode"]["guidance_reward"][0].item()), 23.0)
        finally:
            env.close()

    def test_collect_steps_preserves_agent_mask_from_final_obs(self) -> None:
        env = _make_agent_mask_env((1, (2,), "truncate"))
        try:
            policy = _ConstantValuePolicy(action_dim=env.action_space.total_agent_action_dim)
            buffer = _make_buffer(env)

            episodes, _episode_infos, _metrics, rollout_state = collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=2,
            )

            self.assertEqual(len(episodes), 1)
            episode = episodes[0]
            self.assertIsNotNone(episode.agent_mask)
            self.assertIsNotNone(episode.final_agent_mask)
            self.assertTrue(torch.equal(_to_cpu(episode.agent_mask[:, 1]), torch.tensor([0, 1], dtype=torch.bool)))
            self.assertTrue(torch.equal(_to_cpu(episode.final_agent_mask), torch.tensor([True, False])))
            self.assertTrue(torch.equal(_to_cpu(rollout_state.obs["agent_mask"][0]), torch.tensor([True, True])))
        finally:
            env.close()

    def test_collect_steps_raises_when_final_obs_missing_for_done(self) -> None:
        env = _make_scripted_env((1, (2,), "truncate"))
        proxy = _EnvProxy(
            env,
            mutate_step_infos=lambda infos: (
                {
                    key: value
                    for key, value in infos.items()
                    if key not in {"final_obs", "_final_obs"}
                }
            ),
        )
        try:
            policy = _ConstantValuePolicy(action_dim=proxy.action_space.total_agent_action_dim)
            buffer = _make_buffer(proxy)
            with self.assertRaisesRegex(ValueError, "final_obs"):
                collect_steps(
                    env=proxy,
                    policy=policy,
                    buffer=buffer,
                    n_steps=2,
                )
        finally:
            proxy.close()

    def test_collect_steps_raises_when_agent_mask_missing_from_final_obs(self) -> None:
        env = _make_agent_mask_env((1, (2,), "truncate"))

        def _drop_agent_mask(infos: dict[str, Any]) -> dict[str, Any]:
            mutated_infos = dict(infos)
            if "final_obs" not in mutated_infos or "_final_obs" not in mutated_infos:
                return mutated_infos
            final_obs = np.asarray(mutated_infos["final_obs"], dtype=object).copy()
            for idx, has_final_obs in enumerate(np.asarray(mutated_infos["_final_obs"], dtype=bool).reshape(-1)):
                if not has_final_obs:
                    continue
                obs = dict(final_obs[idx])
                obs.pop("agent_mask", None)
                final_obs[idx] = obs
            mutated_infos["final_obs"] = final_obs
            return mutated_infos

        proxy = _EnvProxy(env, mutate_step_infos=_drop_agent_mask)
        try:
            policy = _ConstantValuePolicy(action_dim=proxy.action_space.total_agent_action_dim)
            buffer = _make_buffer(proxy)
            with self.assertRaisesRegex(ValueError, "agent_mask"):
                collect_steps(
                    env=proxy,
                    policy=policy,
                    buffer=buffer,
                    n_steps=2,
                )
        finally:
            proxy.close()

    def test_collect_steps_handles_final_obs_through_torch_wrapper_chain(self) -> None:
        env = _make_transition_wrapped_env(max_steps=2)
        try:
            policy = _ConstantValuePolicy(action_dim=env.action_space.total_agent_action_dim)
            buffer = _make_buffer(env)
            episodes, _episode_infos, _metrics, rollout_state = collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=2,
            )

            self.assertEqual(len(episodes), 1)
            episode = episodes[0]
            self.assertEqual(episode.final_local_obs.shape[-1], env.local_obs_dim)
            self.assertEqual(episode.final_global_obs.shape[-1], env.global_obs_dim)
            self.assertEqual(rollout_state.obs["local_obs"].shape[-1], env.local_obs_dim)
            self.assertEqual(_first_obs_value(episode.final_local_obs), 2.0)
        finally:
            env.close()

    def test_collect_steps_raises_when_final_obs_mask_mismatches_dones(self) -> None:
        env = _make_scripted_env((1, (2,), "truncate"))

        def _flip_final_obs_mask(infos: dict[str, Any]) -> dict[str, Any]:
            mutated_infos = dict(infos)
            if "_final_obs" not in mutated_infos:
                return mutated_infos
            mutated_infos["_final_obs"] = np.zeros_like(mutated_infos["_final_obs"], dtype=np.bool_)
            return mutated_infos

        proxy = _EnvProxy(env, mutate_step_infos=_flip_final_obs_mask)
        try:
            policy = _ConstantValuePolicy(action_dim=proxy.action_space.total_agent_action_dim)
            buffer = _make_buffer(proxy)
            with self.assertRaisesRegex(ValueError, "_final_obs"):
                collect_steps(
                    env=proxy,
                    policy=policy,
                    buffer=buffer,
                    n_steps=2,
                )
        finally:
            proxy.close()

    def test_collect_steps_raises_when_episode_mask_mismatches_dones(self) -> None:
        env = _make_scripted_env((1, (2,), "truncate"))

        def _flip_episode_mask(infos: dict[str, Any]) -> dict[str, Any]:
            mutated_infos = dict(infos)
            if "final_info" not in mutated_infos:
                return mutated_infos
            final_info = dict(mutated_infos["final_info"])
            if "_episode" in final_info:
                final_info["_episode"] = np.zeros_like(final_info["_episode"], dtype=np.bool_)
            mutated_infos["final_info"] = final_info
            return mutated_infos

        proxy = _EnvProxy(env, mutate_step_infos=_flip_episode_mask)
        try:
            policy = _ConstantValuePolicy(action_dim=proxy.action_space.total_agent_action_dim)
            buffer = _make_buffer(proxy)
            with self.assertRaisesRegex(ValueError, "_episode"):
                collect_steps(
                    env=proxy,
                    policy=policy,
                    buffer=buffer,
                    n_steps=2,
                )
        finally:
            proxy.close()

    def test_collect_steps_accepts_top_level_episode_info_without_final_info(self) -> None:
        env = _make_scripted_env((1, (2,), "truncate"))

        def _hoist_episode_info(infos: dict[str, Any]) -> dict[str, Any]:
            mutated_infos = dict(infos)
            if "final_info" not in mutated_infos:
                return mutated_infos
            final_info = dict(mutated_infos.pop("final_info"))
            mutated_infos.pop("_final_info", None)
            mutated_infos["episode"] = final_info["episode"]
            mutated_infos["_episode"] = final_info["_episode"]
            return mutated_infos

        proxy = _EnvProxy(env, mutate_step_infos=_hoist_episode_info)
        try:
            policy = _ConstantValuePolicy(action_dim=proxy.action_space.total_agent_action_dim)
            buffer = _make_buffer(proxy)
            episodes, episode_infos, _metrics, _rollout_state = collect_steps(
                env=proxy,
                policy=policy,
                buffer=buffer,
                n_steps=2,
            )

            self.assertEqual(len(episodes), 1)
            self.assertEqual(len(episode_infos), 1)
            self.assertEqual(episode_infos[0]["env_marker"], np.float64(102.0))
        finally:
            proxy.close()

    def test_collect_steps_rejects_non_positive_n_steps(self) -> None:
        env = _make_scripted_env((1, (4,), "truncate"))
        try:
            policy = _ConstantValuePolicy(action_dim=env.action_space.total_agent_action_dim)
            buffer = _make_buffer(env)
            with self.assertRaisesRegex(ValueError, "n_steps must be > 0"):
                collect_steps(
                    env=env,
                    policy=policy,
                    buffer=buffer,
                    n_steps=0,
                )
        finally:
            env.close()

    def test_collect_steps_partial_rollout_without_done_has_no_episode_infos(self) -> None:
        env = _make_scripted_env((1, (4,), "truncate"))
        try:
            policy = _ConstantValuePolicy(action_dim=env.action_space.total_agent_action_dim)
            buffer = _make_buffer(env)
            episodes, episode_infos, _metrics, rollout_state = collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=2,
            )

            self.assertEqual(len(episodes), 1)
            self.assertEqual(episode_infos, [])
            self.assertEqual(episodes[0].local_obs.shape[0], 2)
            self.assertTrue(episodes[0].is_true_episode_start)
            self.assertFalse(bool(rollout_state.episode_start_mask[0].item()))
        finally:
            env.close()

    def test_gsde_interval_reset_does_not_reset_on_non_interval_steps(self) -> None:
        env = _make_single_env(max_steps=10)
        try:
            policy = _ConstantValuePolicy(action_dim=env.action_space.total_agent_action_dim)
            action_dist = _SpyGSDEActionDist()
            policy.action_dist = action_dist
            buffer = _make_buffer(env)

            collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=4,
                gsde_reset_mode=GSDEIntervalResetMode(interval=3),
            )

            self.assertEqual(len(action_dist.step_resets), 2)
            for reset_mask, batch_shape in action_dist.step_resets:
                self.assertIsNone(batch_shape)
                self.assertIsNotNone(reset_mask)
                self.assertEqual(tuple(reset_mask.shape), (1, 2))
                self.assertTrue(bool(reset_mask.all().item()))
        finally:
            env.close()

    def test_gsde_probability_reset_passes_agent_masks_without_batch_shape(self) -> None:
        env = _make_scripted_env(
            (1, (3,), "truncate"),
            (2, (4,), "truncate"),
        )
        try:
            torch.manual_seed(1234)
            policy = _ConstantValuePolicy(action_dim=env.action_space.total_agent_action_dim)
            action_dist = _SpyGSDEActionDist()
            policy.action_dist = action_dist
            buffer = _make_buffer(env)

            collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=6,
                gsde_reset_mode=GSDEProbabilityResetMode(probability=0.5),
            )

            self.assertEqual(len(action_dist.step_resets), 3)
            for reset_mask, batch_shape in action_dist.step_resets:
                self.assertIsNone(batch_shape)
                self.assertIsNotNone(reset_mask)
                self.assertEqual(reset_mask.dtype, torch.bool)
                self.assertEqual(tuple(reset_mask.shape), (2, 2))
            self.assertTrue(all(tuple(mask.shape) == (2,) for mask in action_dist.episode_resets))
            self.assertTrue(any(bool(mask.any().item()) for mask in action_dist.episode_resets[1:]))
        finally:
            env.close()

    def test_gsde_rollout_state_resume_keeps_interval_cadence(self) -> None:
        env = _make_single_env(max_steps=10)
        try:
            policy = _ConstantValuePolicy(action_dim=env.action_space.total_agent_action_dim)
            action_dist = _SpyGSDEActionDist()
            policy.action_dist = action_dist
            buffer = _make_buffer(env)

            _episodes, _episode_infos, _metrics, rollout_state = collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=3,
                gsde_reset_mode=GSDEIntervalResetMode(interval=2),
            )
            collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=2,
                rollout_state=rollout_state,
                gsde_reset_mode=GSDEIntervalResetMode(interval=2),
            )

            self.assertEqual(len(action_dist.step_resets), 3)
            for reset_mask, batch_shape in action_dist.step_resets:
                self.assertIsNone(batch_shape)
                self.assertIsNotNone(reset_mask)
                self.assertEqual(tuple(reset_mask.shape), (1, 2))
        finally:
            env.close()

    def test_collect_steps_snapshots_obs_before_step_when_env_reuses_buffers(self) -> None:
        base_env = _make_scripted_env((1, (2,), "truncate"))
        env = _ObsReuseProxy(base_env)
        try:
            policy = _ConstantValuePolicy(action_dim=env.action_space.total_agent_action_dim)
            buffer = _make_buffer(env)
            episodes, _episode_infos, _metrics, _rollout_state = collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=2,
            )

            self.assertEqual(len(episodes), 1)
            episode = episodes[0]
            self.assertTrue(torch.equal(_to_cpu(episode.local_obs[:, 0, 0]), torch.tensor([1100.0, 1101.0])))
            self.assertEqual(_first_obs_value(episode.final_local_obs), 1102.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
