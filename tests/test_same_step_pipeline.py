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
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
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


class _BaseTestPolicy(BasePPOPolicy[PPOSampler, PPOSamplerConfig]):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self._action_dim = action_dim
        self.action_dist = _DummyActionDist()

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
            resumed_episodes, _episode_infos, _metrics, _next_state = collect_steps(
                env=env,
                policy=policy,
                buffer=buffer,
                n_steps=1,
                rollout_state=rollout_state,
            )

            self.assertEqual(len(first_episodes), 1)
            self.assertTrue(first_episodes[0].is_true_episode_start)
            self.assertTrue(torch.allclose(_to_cpu(first_episodes[0].initial_previous_actions), torch.zeros((2, 2))))
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


if __name__ == "__main__":
    unittest.main()
