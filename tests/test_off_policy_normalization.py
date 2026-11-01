import gc
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn import as_benchmark_policy
from swarmbots.learn.algos.off_policy import collect_off_policy_steps
from swarmbots.learn.algos.off_policy.base_algorithm import OffPolicyAlgorithm
from swarmbots.learn.algos.off_policy.normalization import ReplayNormalization, _normalization_operation
from swarmbots.learn.algos.sac import SAC, RecurrentSAC
from swarmbots.learn.algos.td3 import DDPG, TD3, RecurrentTD3
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_tensor_ops import build_obs_norm_tensor_operation
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.env_wrappers.torch_normalize_reward_wrapper import TorchNormalizeRewardWrapper
from swarmbots.learn.env_wrappers.torch_transition_obs_wrapper import TorchTransitionObsWrapper
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage
from swarmbots.learn.evaluation import FrozenEvaluationRunner
from swarmbots.learn.obs_indices import ObsIndices
from tests.test_deterministic_baselines import make_policy
from tests.test_off_policy_replay import _ScriptedOffPolicyEnv
from tests.test_sac_algorithm import _make_env as make_quaternion_env


@pytest.fixture(scope="module", autouse=True)
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@contextmanager
def _isolated_normalization_compiler_caches() -> Iterator[None]:
    cached_operations = (_normalization_operation, build_obs_norm_tensor_operation)
    for operation in cached_operations:
        operation.cache_clear()
    try:
        yield
    finally:
        for operation in cached_operations:
            operation.cache_clear()


class _PhysicalStateEnv(_ScriptedOffPolicyEnv):
    def __init__(self) -> None:
        super().__init__(done_steps=(3,), include_agent_mask=True)
        self.action_space.spaces["connectors"] = spaces.Box(-1.0, 1.0, (2, 1), dtype=np.float32)
        self.value = 10.0
        self.close_calls = 0

    def _obs(self) -> dict[str, np.ndarray]:
        observations = super()._obs()
        for key, value in observations.items():
            if key != "agent_mask":
                value.fill(self.value)
        return observations

    def step(self, action: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        observations, _, terminated, truncated, infos = super().step(action)
        return observations, 1.0, terminated, truncated, infos

    def close(self) -> None:
        self.close_calls += 1
        super().close()


def _make_normalization_env(
    *, device: str = "cpu", transition_observations: bool = False,
    compile_normalization: bool | None = False,
) -> TorchNormalizeRewardWrapper:
    env = SwarmBotsLearnEnvWrapper(SyncVectorEnv(
        [_PhysicalStateEnv], autoreset_mode=AutoresetMode.SAME_STEP,
    ), device=device)
    for key in ("local_obs", "global_obs", "hidden_local_vars", "hidden_global_vars"):
        env = TorchFeatureWiseObsNormWrapper(
            env, obs_key=key, scalar_feature_indices=[0], quaternion_indices=[],
            compile_tensor_operations=compile_normalization,
        )
    if transition_observations:
        env = TorchTransitionObsWrapper(env, compile_tensor_operations=False)
    return TorchNormalizeRewardWrapper(env)


def _make_algorithm(
    variant: str = "tmasac", *, device: str = "cpu", transition_observations: bool = False,
    compile_normalization: bool | None = False, use_nop: bool = True, independent_nop_sampling: bool = False,
) -> OffPolicyAlgorithm:
    env = _make_normalization_env(
        device=device, transition_observations=transition_observations,
        compile_normalization=compile_normalization,
    )
    recurrent_td3 = variant in {"tmatd3", "tmatd3_dec"}
    recurrent = recurrent_td3 or variant in {"tmasac__lstm", "tmasac__slstm"}
    policy_options = {"td3_recurrent_actor": True} if recurrent_td3 else {}
    policy = make_policy(
        env, variant, model_scale=None, use_nop=use_nop, world_model_num_next_steps=3,
        obs_indices=ObsIndices(**{
            field.name: [0] if field.name == "local_scalar_indices" else [] for field in fields(ObsIndices)
        }),
        transition_model_d_model=8, transition_model_nhead=2, **policy_options,
    )
    algorithm_class = (
        RecurrentTD3 if recurrent_td3 else RecurrentSAC if recurrent
        else DDPG if variant.startswith("maddpg") else TD3 if variant.startswith("matd3") else SAC
    )
    return algorithm_class(
        policy, env, buffer_capacity_per_env=64, learning_starts=32, batch_size=2,
        rollout_steps_per_iteration=1, gradient_steps=1,
        replay_storage_device=device, train_device=device, rollout_device=device,
        replay_compile_tensor_operations=False, sac_compile_tensor_operations=False,
        sac_compile_optimizer_steps=False,
        **({"burn_in_steps": 1, "learning_steps": 3, "temporal_state_store_interval": 1} if recurrent else {}),
        **({"independent_nop_sampling": True} if independent_nop_sampling else {}),
    )


def _collect(algorithm: OffPolicyAlgorithm, steps: int) -> None:
    for _ in range(steps):
        algorithm.perform_iteration(ExponentialMovingAverage(0.1), ExponentialMovingAverage(0.1), False)


def _observation_wrappers(algorithm: OffPolicyAlgorithm) -> list[TorchFeatureWiseObsNormWrapper]:
    return [layout.wrapper for layout in algorithm._normalization.observation_layouts]


def _set_statistics(algorithm: OffPolicyAlgorithm, mean: float, variance: float) -> None:
    for wrapper in _observation_wrappers(algorithm):
        wrapper.obs_rms.mean.fill_(mean)
        wrapper.obs_rms.var.fill_(variance)


@pytest.mark.parametrize("transition_observations", [False, True])
@pytest.mark.parametrize("obs_key,duplicate_indices", [
    ("local_obs", [0]), ("local_obs", [1]), ("local_obs", []),
    ("global_obs", [0]), ("global_obs", [1]), ("global_obs", []),
    ("hidden_local_vars", [0]), ("hidden_global_vars", [0]),
])
def test_duplicate_normalization_keys_are_rejected_without_changing_wrapper_behavior(
    transition_observations: bool, obs_key: str, duplicate_indices: list[int],
) -> None:
    env = _make_normalization_env(transition_observations=transition_observations)
    duplicate = TorchFeatureWiseObsNormWrapper(
        env, obs_key, duplicate_indices, [], compile_tensor_operations=False,
    )
    try:
        with pytest.raises(ValueError, match=f"duplicate observation normalization wrappers for key {obs_key!r}"):
            ReplayNormalization(duplicate)
        current = duplicate
        while hasattr(current, "env"):
            if isinstance(current, TorchFeatureWiseObsNormWrapper):
                assert current.normalize_output
            if isinstance(current, TorchNormalizeRewardWrapper):
                assert current.normalize_rewards
            current = current.env
    finally:
        duplicate.close()


@pytest.mark.parametrize("variant", [
    "tmasac", "tmasac_dec", "masac__deepset_critic", "matd3__deepset_critic", "maddpg__deepset_critic",
    "tmasac__lstm", "tmasac__slstm", "tmatd3", "tmatd3_dec",
])
def test_raw_replay_keeps_physical_states_and_statistics_adapt_to_later_states(variant: str) -> None:
    algorithm = _make_algorithm(variant)
    try:
        algorithm._before_learn_loop()
        assert algorithm._rollout_state is None
        assert algorithm.env.unwrapped.envs[0].episode_id == 0
        _collect(algorithm, 4)
        assert algorithm.n_total_timesteps == len(algorithm.replay_buffer) == 4
        assert algorithm.n_total_updates == 0
        batch = algorithm.replay_buffer.get_all()
        for name in ("local_obs", "global_obs", "hidden_local_vars", "hidden_global_vars"):
            torch.testing.assert_close(getattr(batch, name), torch.full_like(getattr(batch, name), 10.0))
            torch.testing.assert_close(getattr(batch, f"next_{name}"), getattr(batch, name), rtol=0, atol=0)
        torch.testing.assert_close(batch.rewards, torch.ones_like(batch.rewards), rtol=0, atol=0)
        means = [wrapper.obs_rms.mean.clone() for wrapper in _observation_wrappers(algorithm)]
        algorithm.env.unwrapped.envs[0].value = 100.0
        _collect(algorithm, 4)
        for wrapper, previous_mean in zip(_observation_wrappers(algorithm), means, strict=True):
            assert wrapper.update_running_mean
            assert torch.all(wrapper.obs_rms.mean > previous_mean)
        raw = algorithm.replay_buffer.get_all()
        normalized = algorithm._normalization.snapshot(device=torch.device("cpu")).normalize_batch(raw)
        wrapper = _observation_wrappers(algorithm)[0]
        expected = (raw.local_obs[..., 0] - wrapper.obs_rms.mean.float()) / torch.sqrt(
            wrapper.obs_rms.var.float() + wrapper._eps
        )
        torch.testing.assert_close(normalized.local_obs[..., 0], expected)
        torch.testing.assert_close(algorithm.replay_buffer.get_all().local_obs, raw.local_obs, rtol=0, atol=0)
    finally:
        algorithm.env.close()


@pytest.mark.parametrize("variant", ["tmasac", "tmasac__lstm", "matd3__deepset_critic", "tmatd3"])
@pytest.mark.parametrize("use_nop,independent", [(False, False), (True, False), (True, True)])
def test_learner_burn_in_and_nop_share_one_current_normalization_snapshot(
    variant: str, use_nop: bool, independent: bool,
) -> None:
    if independent and variant in {"tmasac__lstm", "tmatd3"}:
        pytest.skip("Recurrent learners use sequence NOP sampling rather than independent NOP batches.")
    algorithm = _make_algorithm(variant, use_nop=use_nop, independent_nop_sampling=independent)
    try:
        _collect(algorithm, 8)
        _set_statistics(algorithm, mean=2.0, variance=4.0)
        algorithm.env.return_rms.var.fill_(4.0)
        batches = []
        nop_batches = []

        def capture_batch(batch: Any, **kwargs: Any) -> tuple[dict[str, torch.Tensor], float, float]:
            batches.append(batch)
            if kwargs["nop_batch"] is not None:
                nop_batches.append(kwargs["nop_batch"])
            _set_statistics(algorithm, mean=100.0, variance=100.0)
            algorithm.env.return_rms.var.fill_(100.0)
            return {"critic_loss": batch.rewards.mean()}, 0.0, 0.0

        with patch.object(algorithm, "_train_step", side_effect=capture_batch):
            algorithm.train(gradient_steps=2)
        expected = (10.0 - 2.0) / (4.0 + 1e-6) ** 0.5
        for batch in batches:
            torch.testing.assert_close(batch.local_obs[..., 0], torch.full_like(batch.local_obs[..., 0], expected))
            torch.testing.assert_close(batch.next_local_obs[..., 0], batch.local_obs[..., 0], rtol=0, atol=0)
            torch.testing.assert_close(batch.rewards, torch.full_like(batch.rewards, 0.5))
        for batch in nop_batches:
            valid = batch.train_mask
            torch.testing.assert_close(batch.local_obs[..., 0][valid], torch.full_like(
                batch.local_obs[..., 0][valid], expected,
            ))
            assert torch.count_nonzero((batch.next_local_obs - batch.local_obs)[valid]) == 0
        raw = algorithm.replay_buffer.get_all()
        torch.testing.assert_close(raw.local_obs, torch.full_like(raw.local_obs, 10.0))
        torch.testing.assert_close(raw.rewards, torch.ones_like(raw.rewards))
    finally:
        algorithm.env.close()


def test_checkpoint_restores_statistics_and_continues_updating_them(tmp_path: Path) -> None:
    original = _make_algorithm("tmasac__lstm")
    restored = _make_algorithm("tmasac__lstm")
    try:
        _collect(original, 4)
        checkpoint = tmp_path / "normalized.pt"
        original.save(checkpoint, optimizer_state_dict=original._get_optimizer_state_dict())
        restored.load(checkpoint)
        assert restored._rollout_state is None
        assert len(restored.replay_buffer) == 0
        for previous, current in zip(_observation_wrappers(original), _observation_wrappers(restored), strict=True):
            assert current.update_running_mean
            torch.testing.assert_close(current.obs_rms.mean, previous.obs_rms.mean)
            torch.testing.assert_close(current.obs_rms.var, previous.obs_rms.var)
        counts = [wrapper.obs_rms.count.clone() for wrapper in _observation_wrappers(restored)]
        _collect(restored, 4)
        for wrapper, previous_count in zip(_observation_wrappers(restored), counts, strict=True):
            assert wrapper.obs_rms.count > previous_count
        torch.testing.assert_close(restored.replay_buffer.get_all().rewards, torch.ones(4))
    finally:
        original.env.close()
        restored.env.close()


def test_public_collection_stores_raw_observations_and_advances_recurrent_history() -> None:
    algorithm = _make_algorithm("tmasac__lstm")
    try:
        _, _, state = collect_off_policy_steps(
            algorithm.env, algorithm.replay_buffer, 8, policy=algorithm.policy, random_actions=True,
        )
        batch = algorithm.replay_buffer.get_all()
        torch.testing.assert_close(batch.local_obs, torch.full_like(batch.local_obs, 10.0))
        assert state.temporal_state is not None
        algorithm.train(gradient_steps=1)
        algorithm._rollout_state = state
        _collect(algorithm, 1)
        assert len(algorithm.replay_buffer) == 9
    finally:
        algorithm.env.close()


def test_transition_history_is_raw_and_both_observation_halves_use_current_statistics() -> None:
    algorithm = _make_algorithm(transition_observations=True)
    try:
        _collect(algorithm, 4)
        wrapper = algorithm.env.env
        assert isinstance(wrapper, TorchTransitionObsWrapper)
        raw = algorithm.replay_buffer.get_all()
        assert torch.count_nonzero(raw.local_obs[0, ..., wrapper._n_local_obs:]) == 0
        _set_statistics(algorithm, mean=2.0, variance=4.0)
        counts = [normalizer.obs_rms.count.clone() for normalizer in _observation_wrappers(algorithm)]
        normalized = algorithm._normalization.snapshot(device=torch.device("cpu")).normalize_batch(raw)
        expected = (10.0 - 2.0) / (4.0 + 1e-6) ** 0.5
        torch.testing.assert_close(normalized.local_obs[1, ..., 0], torch.full((2,), expected))
        torch.testing.assert_close(normalized.local_obs[1, ..., -wrapper._n_local_obs], torch.full((2,), expected))
        absent_history = (0.0 - 2.0) / (4.0 + 1e-6) ** 0.5
        torch.testing.assert_close(normalized.local_obs[3, ..., -wrapper._n_local_obs], torch.full((2,), absent_history))
        # Done successors are terminal observations with real history, rather than the autoreset's empty context.
        torch.testing.assert_close(
            normalized.next_local_obs[..., -wrapper._n_local_obs], torch.full((4, 2), expected),
        )
        action_slice = slice(wrapper._n_local_obs, wrapper._n_local_obs + wrapper._n_action_features)
        torch.testing.assert_close(normalized.local_obs[..., action_slice], raw.local_obs[..., action_slice])
        for normalizer, count in zip(_observation_wrappers(algorithm), counts, strict=True):
            torch.testing.assert_close(normalizer.obs_rms.count, count, rtol=0, atol=0)
    finally:
        algorithm.env.close()


def test_quaternion_canonicalization_does_not_modify_raw_inputs() -> None:
    source = make_quaternion_env()
    wrapper = TorchFeatureWiseObsNormWrapper(
        source, obs_key="local_obs", scalar_feature_indices=[], quaternion_indices=[0],
        compile_tensor_operations=False,
    )
    normalization = ReplayNormalization(wrapper)
    try:
        raw = {"local_obs": torch.tensor([[[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])}
        normalized = normalization.snapshot(device=torch.device("cpu")).normalize_observations(raw)
        torch.testing.assert_close(normalized["local_obs"][..., 0], torch.ones((1, 2)))
        assert raw["local_obs"][0, 0, 0] == -1
    finally:
        wrapper.close()


def test_benchmark_adapter_normalizes_raw_training_environment_observations_without_updating_statistics() -> None:
    algorithm = _make_algorithm()
    try:
        _collect(algorithm, 4)
        observations, _ = algorithm.env.reset()
        _set_statistics(algorithm, mean=2.0, variance=4.0)
        observed = []
        original_act = algorithm.policy.act_with_temporal_state

        def capture(**kwargs: Any) -> Any:
            observed.append(kwargs["local_obs"].clone())
            return original_act(**kwargs)

        adapter = as_benchmark_policy(algorithm)
        with patch.object(algorithm.policy, "act_with_temporal_state", side_effect=capture):
            adapter(observations, torch.ones(1, dtype=torch.bool))
        expected = (10.0 - 2.0) / (4.0 + 1e-6) ** 0.5
        torch.testing.assert_close(observed[0][..., 0], torch.full((1, 2), expected))
        assert all(not wrapper.normalize_output and wrapper.update_running_mean for wrapper in _observation_wrappers(algorithm))
    finally:
        algorithm.env.close()


@pytest.mark.parametrize("variant", ["tmasac", "matd3__deepset_critic"])
@pytest.mark.parametrize("compile_policy", [False, True])
def test_rollout_replay_and_both_evaluators_agree_on_missing_and_populated_transition_history(
    variant: str, compile_policy: bool,
) -> None:
    algorithm = _make_algorithm(variant, transition_observations=True)
    original_policy = algorithm.policy
    if compile_policy:
        algorithm.policy = torch.compile(original_policy, backend="eager")
    runner = FrozenEvaluationRunner(
        make_env=lambda: _make_normalization_env(transition_observations=True),
        training_env=algorithm.env, policy=algorithm.policy,
        episodes_per_env=2,
    )
    try:
        _set_statistics(algorithm, mean=2.0, variance=4.0)
        for wrapper in _observation_wrappers(algorithm):
            wrapper.update_running_mean = False
        observed = []
        original_act = original_policy.act_with_temporal_state

        def capture(**kwargs: Any) -> Any:
            observed.append((
                kwargs["local_obs"].clone(), kwargs["global_obs"].clone(), kwargs["episode_start_mask"].clone(),
            ))
            return original_act(**kwargs)

        adapter = as_benchmark_policy(algorithm)
        assert adapter.policy is algorithm.policy
        raw, _ = algorithm.env.reset()
        expected = algorithm._normalization.snapshot(device=torch.device("cpu")).normalize_observations(raw)
        base = algorithm.env.unwrapped.envs[0]
        public = {key: torch.as_tensor(value).unsqueeze(0) for key, value in base._obs().items()}
        with patch.object(original_policy, "act_with_temporal_state", side_effect=capture):
            adapter(public, torch.ones(1, dtype=torch.bool))
            runner.evaluate(timesteps=0, milestone_percentage=0)
        for local, global_obs, _ in (observed[0], observed[1]):
            torch.testing.assert_close(local, expected["local_obs"])
            torch.testing.assert_close(global_obs, expected["global_obs"])
        assert expected["local_obs"][0, 0, -3] < 0
        for local, global_obs, starts in observed:
            expected_history = torch.where(starts, -2.0, 8.0) / (4.0 + 1e-6) ** 0.5
            torch.testing.assert_close(local[..., -3], expected_history[:, None].expand(1, 2))
            torch.testing.assert_close(global_obs[..., -2], expected_history)
    finally:
        algorithm.env.close()
        runner.close()


def test_whole_policy_compilation_preserves_on_policy_normalization_behavior() -> None:
    env = _make_normalization_env(transition_observations=True)
    try:
        policy = make_policy(env, "mat_ind", model_scale=None, use_nop=False)
        compiled_policy = torch.compile(policy, backend="eager")
        assert ReplayNormalization.for_policy(env, compiled_policy) is None
        assert env.normalize_rewards
        current = env
        while hasattr(current, "env"):
            if isinstance(current, TorchFeatureWiseObsNormWrapper):
                assert current.normalize_output
            current = current.env
    finally:
        env.close()


def test_closing_frozen_evaluation_releases_its_normalization_wrappers() -> None:
    algorithm = _make_algorithm()
    references = []

    def make_environment() -> TorchFeatureWiseObsNormWrapper:
        env = TorchFeatureWiseObsNormWrapper(
            make_quaternion_env(), "local_obs", [0], [], compile_tensor_operations=False,
        )
        references.append(weakref.ref(env))
        return env

    runner = FrozenEvaluationRunner(make_env=make_environment, training_env=object(), policy=algorithm.policy)
    try:
        runner.env
        assert references[0]() is not None
        runner.close()
        gc.collect()
        assert references[0]() is None
    finally:
        runner.close()
        algorithm.env.close()


@pytest.mark.parametrize("duplicate_indices", [[], [0]])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=[pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]),
])
def test_failed_frozen_initialization_closes_invalid_environments_and_revalidates_retries(
    duplicate_indices: list[int], device: str,
) -> None:
    algorithm = _make_algorithm(device=device, transition_observations=True)
    _set_statistics(algorithm, mean=2.0, variance=4.0)
    environments = []

    def make_environment() -> TorchNormalizeRewardWrapper | TorchFeatureWiseObsNormWrapper:
        torch.manual_seed(987)
        env = _make_normalization_env(device=device, transition_observations=True)
        if len(environments) < 2:
            env = TorchFeatureWiseObsNormWrapper(
                env, "local_obs", duplicate_indices, [], compile_tensor_operations=False,
            )
        environments.append(env)
        return env

    runner = FrozenEvaluationRunner(make_env=make_environment, training_env=algorithm.env, policy=algorithm.policy)
    observed = []
    original_act = algorithm.policy.act_with_temporal_state

    def capture(**kwargs: Any) -> Any:
        observed.append(kwargs["local_obs"].clone())
        return original_act(**kwargs)

    rng_state = torch.random.get_rng_state().clone()
    try:
        for attempt in range(2):
            with pytest.raises(ValueError, match="duplicate observation normalization wrappers for key 'local_obs'"):
                runner.evaluate(timesteps=0, milestone_percentage=0)
            assert len(environments) == attempt + 1
            assert environments[-1].unwrapped.envs[0].close_calls == 1
            assert torch.equal(torch.random.get_rng_state(), rng_state)
        with patch.object(algorithm.policy, "act_with_temporal_state", side_effect=capture):
            metrics = runner.evaluate(timesteps=0, milestone_percentage=0)
        assert metrics["eval_episodes"] == 2
        assert len(environments) == 3
        assert runner.env is environments[-1]
        assert environments[-1].unwrapped.envs[0].close_calls == 0
        expected_reset_history = -2.0 / (4.0 + 1e-6) ** 0.5
        torch.testing.assert_close(observed[0][..., -3], torch.full_like(observed[0][..., -3], expected_reset_history))
        runner.close()
        assert all(env.unwrapped.envs[0].close_calls == 1 for env in environments)
    finally:
        runner.close()
        for env in environments:
            env.close()
        algorithm.env.close()


@pytest.mark.parametrize("variant", ["tmasac", "tmasac__lstm", "matd3__deepset_critic", "tmatd3"])
def test_adaptively_normalized_replay_supports_actual_nop_and_policy_updates(variant: str) -> None:
    algorithm = _make_algorithm(variant)
    try:
        _collect(algorithm, 8)
        metrics = algorithm.train(gradient_steps=2)
        assert metrics["updates"] == 2
        assert all(torch.isfinite(parameter).all() for parameter in algorithm.policy.parameters())
        assert all(wrapper.update_running_mean for wrapper in _observation_wrappers(algorithm))
    finally:
        algorithm.env.close()


def test_absent_observation_normalization_does_not_require_a_cuda_compiler() -> None:
    with _isolated_normalization_compiler_caches(), patch(
        "torch.compile", side_effect=AssertionError("Unexpected compilation"),
    ):
        ReplayNormalization(object()).snapshot(device=torch.device("cuda"))


def test_compiler_mocks_do_not_leak_into_subsequent_normalization_compilation() -> None:
    real_compile = torch.compile
    with _isolated_normalization_compiler_caches(), patch(
        "torch.compile", side_effect=lambda function, **kwargs: function,
    ):
        build_obs_norm_tensor_operation(
            compile_operation=True, compile_mode="reduce-overhead", normalize_output=False,
        )
        _normalization_operation(True)
    with patch("torch.compile", wraps=real_compile) as compiler:
        build_obs_norm_tensor_operation(
            compile_operation=True, compile_mode="reduce-overhead", normalize_output=False,
        )
        _normalization_operation(True)
    assert {call.args[0].__name__ for call in compiler.call_args_list} == {
        "_update_observation_statistics", "_normalize_tensors",
    }


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compile_normalization", [None, False, True])
def test_cuda_normalization_respects_compilation_selection_across_training_and_evaluation(
    compile_normalization: bool | None,
) -> None:
    algorithm = None
    runner = None
    try:
        with _isolated_normalization_compiler_caches(), patch(
            "torch.compile", side_effect=lambda function, **kwargs: function,
        ) as compiler:
            algorithm = _make_algorithm(
                "tmasac__lstm", device="cuda", transition_observations=True,
                compile_normalization=compile_normalization,
            )
            collect_off_policy_steps(
                algorithm.env, algorithm.replay_buffer, 8, policy=algorithm.policy, random_actions=False,
            )
            metrics = algorithm.train(gradient_steps=2)
            assert metrics["updates"] == 2
            public = {
                key: torch.as_tensor(value, device="cuda").unsqueeze(0)
                for key, value in algorithm.env.unwrapped.envs[0]._obs().items()
            }
            as_benchmark_policy(algorithm)(public, torch.ones(1, dtype=torch.bool, device="cuda"))
            runner = FrozenEvaluationRunner(
                make_env=lambda: _make_normalization_env(
                    device="cuda", transition_observations=True,
                    compile_normalization=compile_normalization,
                ),
                training_env=algorithm.env, policy=algorithm.policy,
            )
            runner.evaluate(timesteps=0, milestone_percentage=0)
            if compile_normalization is False:
                compiler.assert_not_called()
            else:
                assert any(call.args[0].__name__ == "_normalize_tensors" for call in compiler.call_args_list)
            assert all(torch.isfinite(parameter).all() for parameter in algorithm.policy.parameters())
    finally:
        if runner is not None:
            runner.close()
        if algorithm is not None:
            algorithm.env.close()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_normalization_compilation_depends_on_participating_observation_fields() -> None:
    local = TorchFeatureWiseObsNormWrapper(make_quaternion_env(), "local_obs", [0], [])
    env = TorchFeatureWiseObsNormWrapper(
        local, "hidden_local_vars", [0], [], compile_tensor_operations=False,
    )
    normalization = ReplayNormalization(env)
    observations = {
        "local_obs": torch.full((1, 2, 4), 5.0, device="cuda"),
        "hidden_local_vars": torch.full((1, 2, 1), 7.0, device="cuda"),
    }
    local.obs_rms.mean.fill_(2.0)
    env.obs_rms.mean.fill_(3.0)
    local.obs_rms.var.fill_(4.0)
    env.obs_rms.var.fill_(4.0)
    try:
        with _isolated_normalization_compiler_caches(), patch("torch.compile") as compiler:
            public = normalization.snapshot(device=torch.device("cuda"), public_only=True)
            compiler.assert_called_once()
            compiled = compiler.return_value
            compiled.side_effect = compiler.call_args.args[0]
            public_observations = public.normalize_observations(observations)
            full_observations = normalization.snapshot(device=torch.device("cuda")).normalize_observations(observations)
            compiled.assert_called_once()
            torch.testing.assert_close(public_observations["local_obs"], full_observations["local_obs"])
            torch.testing.assert_close(public_observations["hidden_local_vars"], observations["hidden_local_vars"])
            torch.testing.assert_close(
                full_observations["hidden_local_vars"],
                torch.full_like(observations["hidden_local_vars"], 4.0 / (4.0 + env._eps) ** 0.5),
            )
    finally:
        env.close()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("transition_observations", [False, True])
def test_compiled_cuda_adaptive_normalization_preserves_raw_replay_and_retained_batches(
    transition_observations: bool,
) -> None:
    torch.compiler.reset()
    algorithm = _make_algorithm(
        "tmasac__lstm", device="cuda", compile_normalization=True,
        transition_observations=transition_observations,
    )
    try:
        _collect(algorithm, 8)
        raw = algorithm.replay_buffer.get_all()
        torch.testing.assert_close(raw.local_obs[..., :3], torch.full_like(raw.local_obs[..., :3], 10.0), rtol=0, atol=0)
        snapshot = algorithm._normalization.snapshot(device=torch.device("cuda:0"))
        first = snapshot.normalize_batch(raw)
        retained = first.local_obs.clone()
        _set_statistics(algorithm, mean=100.0, variance=100.0)
        algorithm._normalization.snapshot(device=torch.device("cuda:0")).normalize_batch(raw)
        torch.testing.assert_close(first.local_obs, retained, rtol=0, atol=0)
        algorithm.train(gradient_steps=2)
        assert all(torch.isfinite(parameter).all() for parameter in algorithm.policy.parameters())
        reset_observations, _ = algorithm.env.reset()
        expected = algorithm._normalization.snapshot(device=torch.device("cuda:0")).normalize_observations(
            reset_observations,
        )
        public = {
            key: torch.as_tensor(value, device="cuda").unsqueeze(0)
            for key, value in algorithm.env.unwrapped.envs[0]._obs().items()
        }
        seen = []
        original_act = algorithm.policy.act_with_temporal_state

        def capture(**kwargs: Any) -> Any:
            seen.append(kwargs["local_obs"].clone())
            return original_act(**kwargs)

        with patch.object(algorithm.policy, "act_with_temporal_state", side_effect=capture):
            as_benchmark_policy(algorithm)(public, torch.ones(1, dtype=torch.bool, device="cuda"))
        torch.testing.assert_close(seen[0], expected["local_obs"], rtol=1e-5, atol=1e-5)
    finally:
        algorithm.env.close()
        torch.compiler.reset()
