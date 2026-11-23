import unittest
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo import PPO, StepsRolloutMode
from swarmbots.learn.algos.ppo.ppo_rollout import collect_step_rollout_batch, collect_steps, warmup_rollout_steps
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment, PPORolloutBuffer
from swarmbots.learn.algos.r_mat.r_ppo_wm_sampler import RPPOWMSampler, RPPOWMSamplerConfig
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMBatchSampler, PPOWMSamplerConfig
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.env_wrappers.torch_transition_obs_wrapper import TorchTransitionObsWrapper
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper, TorchObs
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


def test_dense_ppo_rollout_keeps_later_precision_and_previous_actions() -> None:
    env = _make_env(n_envs=2)
    policy = _DummyPolicy(env.action_space.total_agent_action_dim)
    buffer = PPORolloutBuffer(
        max_episode_length=8, observation_space=env.observation_space, action_space=env.action_space,
        gamma=0.99, gae_lambda=0.95, rollout_device="cpu", train_device="cpu",
    )
    draws = iter([
        (0.135792468, torch.float64, torch.float32),
        (0.2345678, torch.float16, torch.float16),
        (0.3456789, torch.float32, torch.float32),
        (0.4567891, torch.float16, torch.bfloat16),
    ])
    emitted_actions: list[torch.Tensor] = []
    emitted_log_probs: list[torch.Tensor] = []
    conditioned_actions: list[torch.Tensor] = []
    original_forward = policy.forward

    def forward(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, log_probs, values = original_forward(**kwargs)
        value, action_dtype, log_prob_dtype = next(draws)
        actions = torch.full_like(actions, value, dtype=action_dtype)
        log_probs = torch.full_like(log_probs, -value, dtype=log_prob_dtype)
        emitted_actions.append(actions)
        emitted_log_probs.append(log_probs)
        conditioned_actions.append(kwargs["previous_actions"])
        return actions, log_probs, values

    try:
        with (
            patch.object(policy, "requires_previous_actions", return_value=True),
            patch.object(policy, "forward", side_effect=forward),
        ):
            _, _, _, state = collect_step_rollout_batch(env, policy, buffer, n_steps=2)
            batch, _, _, _ = collect_step_rollout_batch(env, policy, buffer, n_steps=6, rollout_state=state)
        torch.testing.assert_close(batch.actions, torch.stack(emitted_actions[1:], dim=1), rtol=0, atol=0)
        torch.testing.assert_close(batch.log_probs, torch.stack(emitted_log_probs[1:], dim=1), rtol=0, atol=0)
        torch.testing.assert_close(batch.previous_actions, torch.stack(conditioned_actions[1:], dim=1), rtol=0, atol=0)
    finally:
        env.close()


class _StationaryObservationEnv(TestingSwarmBotsEnv):
    def _get_obs(self) -> dict:
        obs = super()._get_obs()
        obs["local_obs"][..., :2] = 3.0
        obs["global_obs"][..., :2] = 3.0
        return obs


class _WorldModelTestPolicy(_DummyPolicy):
    uses_world_model_targets = True

    def initial_temporal_state(
        self, batch_size: int, n_agents: int, *, device: torch.device | str, dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros((batch_size, 1), device=device, dtype=dtype)


class _ConstantLaneEnv(TestingSwarmBotsEnv):
    def __init__(self, values: tuple[float, float], max_steps: int) -> None:
        super().__init__(2, 3, 2, 1, 1, max_steps=max_steps)
        self.values = values

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs = super()._get_obs()
        for key in ("local_obs", "global_obs"):
            obs[key][..., :2] = self.values
        return obs


class _JumpInGlobalObservationEnv(TestingSwarmBotsEnv):
    def __init__(self, value: float) -> None:
        super().__init__(2, 3, 2, 1, 1, max_steps=32)
        self.value = value

    def _get_obs(self) -> dict[str, np.ndarray]:
        obs = super()._get_obs()
        obs["global_obs"][0] = self.value if self._step_count else 0.
        return obs


class _TrackingScalarNorm(TorchFeatureWiseObsNormWrapper):
    def __init__(self, env: BaseLearnEnvWrapper, key: str, indices: list[int]) -> None:
        super().__init__(env, key, indices, [])
        self.frames = []
        self.obs_rms.mean.fill_(2)
        self.obs_rms.count.fill_(4)

    def observations(self, observations: TorchObs, *, sample_lanes: torch.Tensor | None = None) -> TorchObs:
        result = super().observations(observations, sample_lanes=sample_lanes)
        if self.update_running_mean:
            self.frames.append((self.obs_rms.mean.clone(), (self.obs_rms.var + self.epsilon).sqrt()))
        return result


@pytest.mark.parametrize("recurrent", (False, True))
def test_nop_rebases_half_precision_observations_into_float32_targets(recurrent: bool) -> None:
    vector_env = SyncVectorEnv(
        [lambda: _JumpInGlobalObservationEnv(0.), lambda: _JumpInGlobalObservationEnv(2000.)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    env = TorchFeatureWiseObsNormWrapper(
        SwarmBotsLearnEnvWrapper(vector_env, obs_dtype=torch.float16), "global_obs", [0], [],
    )
    env.obs_rms.count.fill_(100.)
    env.obs_rms.var.zero_()
    try:
        policy = _WorldModelTestPolicy(env.action_space.total_agent_action_dim)
        buffer = PPORolloutBuffer(1, env.observation_space, env.action_space, 0.99, 0.95, train_device="cpu")
        if recurrent:
            episodes, *_ = collect_steps(env, policy, buffer, n_steps=2)
            sampler = RPPOWMSampler(episodes, RPPOWMSamplerConfig(batch_size=2, num_next_steps=1, sequence_length=1))
        else:
            rollout, *_ = collect_step_rollout_batch(env, policy, buffer, n_steps=2)
            sampler = PPOWMBatchSampler(rollout, PPOWMSamplerConfig(batch_size=2, num_next_steps=1))
        rebased = sampler.next_global_obs[..., 0].reshape(-1)
        expected_scale = torch.tensor(env.epsilon, dtype=torch.float16).sqrt().item()
        assert rebased.dtype == torch.float32
        # Half-precision quantization of the successor can leave a small stationary residual.
        assert rebased[0].item() == pytest.approx(0., abs=2.)
        assert rebased[1].item() == pytest.approx(2000. / expected_scale, rel=0.002)
    finally:
        env.close()


@pytest.mark.parametrize("recurrent", (False, True))
@pytest.mark.parametrize(("stack_depth", "enclosing_norms"), ((2, False), (2, True), (4, True)))
def test_nested_nop_history_uses_reset_padding_and_real_signed_values(
    recurrent: bool, stack_depth: int, enclosing_norms: bool,
) -> None:
    lane_values = ((0.0, -3.0), (4.0, 3.0))
    episode_lengths = (3, 5)
    vector_env = SyncVectorEnv(
        [lambda v=v, n=n: _ConstantLaneEnv(v, n) for v, n in zip(lane_values, episode_lengths, strict=True)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    env = SwarmBotsLearnEnvWrapper(vector_env)
    keys = ("local_obs", "global_obs")
    inner, middle, outer = {}, {}, {}
    for key in keys:
        env = inner[key] = _TrackingScalarNorm(env, key, [0, 1])
    env = TorchTransitionObsWrapper(env, new_obs_first=True)
    history_offsets = {
        key: env.observation_space[key].shape[-1] - env.env.observation_space[key].shape[-1] for key in keys
    }
    if enclosing_norms:
        for key in keys:
            env = middle[key] = _TrackingScalarNorm(env, key, [history_offsets[key] + 1])
    for _ in range(stack_depth - 1):
        env = TorchTransitionObsWrapper(env, new_obs_first=False)
    if enclosing_norms:
        for key in keys:
            env = outer[key] = _TrackingScalarNorm(env, key, [history_offsets[key] + 1])
    try:
        policy = _WorldModelTestPolicy(env.action_space.total_agent_action_dim)
        buffer = PPORolloutBuffer(4, env.observation_space, env.action_space, 0.99, 0.95, train_device="cpu")
        collect = collect_steps if recurrent else collect_step_rollout_batch
        warmup = 6
        state = warmup_rollout_steps(env, policy, n_steps=warmup * env.num_envs)
        for rollout in range(2):
            result, _, _, state = collect(env, policy, buffer, n_steps=8, rollout_state=state)
            if recurrent:
                sampler = RPPOWMSampler(result, RPPOWMSamplerConfig(batch_size=2, num_next_steps=3, sequence_length=4))
            else:
                sampler = PPOWMBatchSampler(result, PPOWMSamplerConfig(batch_size=8, num_next_steps=3))
            sample = next(sampler.sample())
            valid = sample.wm_target_time_mask.reshape(2, 4, 3)
            for key in keys:
                offset = history_offsets[key]
                targets = getattr(sample, f"next_{key}").reshape(2, 4, 3, -1)[..., offset:offset + 2]
                for lane, step, horizon_idx in valid.nonzero().tolist():
                    origin = warmup + 4 * rollout + step
                    successor_age = origin % episode_lengths[lane] + horizon_idx + 1
                    mean, scale = inner[key].frames[origin - stack_depth]
                    expected = torch.zeros_like(mean)
                    if successor_age >= stack_depth:
                        expected = (expected.new_tensor(lane_values[lane]) - mean) / scale
                    if enclosing_norms:
                        if successor_age >= stack_depth - 1:
                            middle_mean, middle_scale = middle[key].frames[origin - (stack_depth - 1)]
                            expected[1:] = (expected[1:] - middle_mean) / middle_scale
                        outer_mean, outer_scale = outer[key].frames[origin]
                        expected[1:] = (expected[1:] - outer_mean) / outer_scale
                    torch.testing.assert_close(targets[lane, step, horizon_idx], expected.to(targets.dtype))
    finally:
        env.close()


@pytest.mark.parametrize("recurrent", (False, True))
@pytest.mark.parametrize("local_width", (1, 2))
def test_nop_history_survives_independent_normalizer_freeze_and_thaw(recurrent: bool, local_width: int) -> None:
    vector_env = SyncVectorEnv(
        [lambda v=v: _ConstantLaneEnv(v, 32) for v in ((1.0, -3.0), (5.0, 4.0))],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    env = SwarmBotsLearnEnvWrapper(vector_env)
    widths = {"local_obs": local_width, "global_obs": 1}
    normalizers = {}
    offsets = {key: [0] for key in widths}
    for key, width in widths.items():
        env = normalizers[key] = TorchFeatureWiseObsNormWrapper(env, key, list(range(width)), [])
        env.obs_rms.mean.fill_(-6 if key == "local_obs" else 8)
        env.obs_rms.count.fill_(4)
        env.update_running_mean = False
    for _ in range(2):
        env = TorchTransitionObsWrapper(env)
        for key, positions in offsets.items():
            stride = env.observation_space[key].shape[-1] - env.env.observation_space[key].shape[-1]
            offsets[key] = positions + [stride + p for p in positions]
    try:
        policy = _WorldModelTestPolicy(env.action_space.total_agent_action_dim)
        buffer = PPORolloutBuffer(1, env.observation_space, env.action_space, 0.99, 0.95, train_device="cpu")
        collect = collect_steps if recurrent else collect_step_rollout_batch
        state = warmup_rollout_steps(env, policy, n_steps=4 * env.num_envs)
        settings = ((True, False), (False, True), (True, True), (False, False), (False, False), (True, True))
        for local_updates, global_updates in settings:
            normalizers["local_obs"].update_running_mean = local_updates
            normalizers["global_obs"].update_running_mean = global_updates
            result, _, _, state = collect(env, policy, buffer, n_steps=env.num_envs, rollout_state=state)
            if recurrent:
                sampler = RPPOWMSampler(result, RPPOWMSamplerConfig(batch_size=2, num_next_steps=1, sequence_length=1))
            else:
                sampler = PPOWMBatchSampler(result, PPOWMSamplerConfig(batch_size=2, num_next_steps=1))
            sample = next(sampler.sample())
            assert sample.wm_target_time_mask.all()
            for key, width in widths.items():
                # Each raw observation is constant within its lane. Every stacked successor,
                # expressed in the origin's frame, must therefore equal that origin.
                columns = [p + i for p in offsets[key] for i in range(width)]
                features = env.observation_space[key].shape[-1]
                target = getattr(sample, f"next_{key}").reshape(-1, features)
                origin = getattr(sample, key).reshape(-1, features)
                torch.testing.assert_close(target[:, columns], origin[:, columns])
    finally:
        env.close()


@pytest.mark.parametrize("recurrent", (False, True))
@pytest.mark.parametrize("new_obs_first", (None, False, True))
def test_nop_targets_keep_the_origin_frame_across_steps_resets_and_rollouts(
    recurrent: bool, new_obs_first: bool | None,
) -> None:
    vector_env = SyncVectorEnv(
        [
            lambda: _StationaryObservationEnv(2, 3, 2, 1, 1, max_steps=3),
            lambda: TestingSwarmBotsEnv(2, 3, 2, 1, 1, max_steps=3),
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    env = SwarmBotsLearnEnvWrapper(vector_env)
    obs_widths = {key: env.observation_space[key].shape[-1] for key in ("local_obs", "global_obs")}
    if new_obs_first is not None:
        for key in obs_widths:
            env = TorchFeatureWiseObsNormWrapper(env, key, [0], [])
        env = TorchTransitionObsWrapper(env, new_obs_first=new_obs_first)
    strides = {key: env.observation_space[key].shape[-1] - width for key, width in obs_widths.items()}
    current_offsets = {key: stride if new_obs_first is False else 0 for key, stride in strides.items()}
    for key, offset in current_offsets.items():
        env = TorchFeatureWiseObsNormWrapper(env, key, [offset + 1, offset], [])
        if new_obs_first is not None:
            env = TorchFeatureWiseObsNormWrapper(env, key, [offset + 1], [])
            env.update_running_mean = False
            env.obs_rms.mean.fill_(1.5)
            env.obs_rms.var.fill_(4.0)
    try:
        policy = _WorldModelTestPolicy(env.action_space.total_agent_action_dim)
        buffer = PPORolloutBuffer(4, env.observation_space, env.action_space, 0.99, 0.95, train_device="cpu")
        collect = collect_steps if recurrent else collect_step_rollout_batch
        warmup_steps = 2
        state = warmup_rollout_steps(env, policy, n_steps=warmup_steps * env.num_envs)
        for rollout in range(2):
            result, _, _, state = collect(env, policy, buffer, n_steps=8, rollout_state=state)
            if recurrent:
                config = RPPOWMSamplerConfig(batch_size=2, num_next_steps=3, sequence_length=4)
                sampler = RPPOWMSampler(result, config)
            else:
                assert buffer._accumulator is None
                sampler = PPOWMBatchSampler(result, PPOWMSamplerConfig(batch_size=8, num_next_steps=3))
            sample = next(sampler.sample())
            valid = sample.wm_target_time_mask.reshape(2, 4, 3)
            for key, offset in current_offsets.items():
                origin = getattr(sample, key).reshape(2, 4, -1)[..., offset:offset + 2]
                target = getattr(sample, f"next_{key}").reshape(2, 4, 3, -1)[..., offset:offset + 2]
                stationary = origin[0]
                assert (stationary.amax(0) - stationary.amin(0) > 0.001).all()
                torch.testing.assert_close(target[0][valid[0]], stationary[:, None].expand_as(target[0])[valid[0]])
                # The raw lane gap (3 - episode step) gives the inverse scale of each composed frame.
                raw_origin = (torch.arange(4) + rollout * 4 + warmup_steps) % 3
                inv_scale = (stationary - origin[1]) / (3 - raw_origin[:, None])
                expected = origin[1, :, None] + inv_scale[:, None] * torch.arange(1, 4)[:, None]
                torch.testing.assert_close(target[1][valid[1]], expected[valid[1]])
                if new_obs_first is not None:
                    history_offset = strides[key] - offset
                    history = getattr(sample, key).reshape(2, 4, -1)[0, :, history_offset]
                    next_history = getattr(sample, f"next_{key}").reshape(2, 4, 3, -1)[0, :, :, history_offset]
                    # After the reset sentinel leaves the history, its constant raw scalar has zero delta.
                    valid_history = valid[0] & (raw_origin != 0)[:, None]
                    torch.testing.assert_close(next_history[valid_history], history[:, None].expand_as(next_history)[valid_history])
    finally:
        env.close()


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

    def test_default_rollout_uses_fixed_step_collection(self) -> None:
        env = _make_env(n_envs=16)
        try:
            policy = _DummyPolicy(action_dim=env.action_space.total_agent_action_dim)
            algo = PPO(
                policy=policy,
                env=env,
                max_episode_length=512,
            )

            self.assertEqual(algo.rollout_mode.n_steps_per_rollout, 2048)
            self.assertEqual(algo.rollout_buffer_max_episode_length, 128)
            self.assertEqual(algo.rollout_buffer.max_episode_length, 128)
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
