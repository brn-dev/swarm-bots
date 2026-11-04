from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.algos.off_policy.base_algorithm import OffPolicyAlgorithm
from swarmbots.learn.algos.off_policy.replay_buffer import OffPolicyReplayEpisodeSegmentBatch
from swarmbots.learn.algos.off_policy.replay_exposure import resolve_replay_updates
from swarmbots.learn.algos.sac import RecurrentSAC, SAC
from swarmbots.learn.algos.td3 import RecurrentTD3, TD3
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage
from swarmbots.learn.testing_env import TestingSwarmBotsEnv as SwarmTestEnv
from tests.test_deterministic_baselines import make_policy
from tests.test_sac_algorithm import _make_env, _make_policy


@pytest.fixture(scope="module", autouse=True)
def single_torch_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _make_multi_env() -> SwarmBotsLearnEnvWrapper:
    return SwarmBotsLearnEnvWrapper(SyncVectorEnv(
        [
            lambda: SwarmTestEnv(
                n_agents=2, n_local_obs=4, n_global_obs=2, actuators_dim=1, connectors_dim=1,
                n_hidden_local_vars=1, n_hidden_global_vars=1, max_steps=7,
                continuous_connector_actions=True,
            )
            for _ in range(3)
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    ))


def _make_algorithm(
    env: SwarmBotsLearnEnvWrapper,
    family: Literal["sac", "td3"],
    *,
    recurrent: bool,
    batch_size: int,
    rollout_steps: int | None,
    exposure: float,
    learning_steps: int = 4,
    burn_in_steps: int = 2,
    temporal_state_store_interval: int = 3,
) -> OffPolicyAlgorithm:
    variant = "tmasac__lstm" if recurrent else "tmasac"
    if family == "td3":
        variant = "tmatd3"
    policy = make_policy(
        env, variant, model_scale=None, rmat_experimental_compile_lstm=False,
        **({"td3_recurrent_actor": True, "td3_recurrent_critic": True} if recurrent and family == "td3" else {}),
    )
    algorithm_class = (RecurrentSAC if recurrent else SAC) if family == "sac" else (
        RecurrentTD3 if recurrent else TD3
    )
    return algorithm_class(
        policy, env, batch_size=batch_size, buffer_capacity_per_env=16, learning_starts=0,
        rollout_steps_per_iteration=rollout_steps, replay_exposure=exposure,
        learning_rate_warmup_updates=0, train_device="cpu", rollout_device="cpu", replay_storage_device="cpu",
        **({
            "learning_steps": learning_steps, "burn_in_steps": burn_in_steps,
            "temporal_state_store_interval": temporal_state_store_interval,
        } if recurrent else {}),
    )


def _iterate(algorithm: OffPolicyAlgorithm) -> dict[str, Any]:
    metrics, collected = algorithm.perform_iteration(
        ExponentialMovingAverage(0.1), ExponentialMovingAverage(0.1), update_ema=False,
    )
    assert collected == metrics["transitions_collected"]
    return metrics


def _iterate_with_scored_batches(algorithm: OffPolicyAlgorithm) -> tuple[dict[str, Any], list[int]]:
    with patch.object(algorithm, "_train_step", wraps=algorithm._train_step) as train_step:
        metrics = _iterate(algorithm)
    scored_positions = []
    for call in train_step.call_args_list:
        batch = call.args[0]
        scored_positions.append(
            int(batch.train_mask.sum())
            if isinstance(batch, OffPolicyReplayEpisodeSegmentBatch) else batch.actions.shape[0]
        )
    return metrics, scored_positions


@pytest.mark.parametrize("rollout_steps,exposure,expected_updates", [
    (8, 1.0, 1), (16, 2.5, 5), (80, 0.1, 1), (800, 0.07, 7),
])
def test_fractional_exposure_accepts_integral_update_counts(
    rollout_steps: int, exposure: float, expected_updates: int,
) -> None:
    assert resolve_replay_updates(
        replay_exposure=exposure, rollout_steps=rollout_steps, learning_positions_per_batch=8,
    ) == expected_updates


@pytest.mark.parametrize("family", ["sac", "td3"])
@pytest.mark.parametrize("recurrent", [False, True])
def test_fractional_exposure_scores_each_multi_environment_collection(
    family: Literal["sac", "td3"], recurrent: bool,
) -> None:
    env = _make_multi_env()
    try:
        algorithm = _make_algorithm(
            env, family, recurrent=recurrent, batch_size=3 if recurrent else 12, rollout_steps=48, exposure=0.5,
        )
        for iteration in range(1, 3):
            parameter_groups = (list(algorithm.policy.actor_parameters()), list(algorithm.policy.critic_parameters()))
            parameters_before = [[parameter.detach().clone() for parameter in group] for group in parameter_groups]
            metrics, scored_positions = _iterate_with_scored_batches(algorithm)
            assert metrics["transitions_collected"] == 48
            assert metrics["updates"] == 2
            assert scored_positions == [12, 12]
            assert sum(scored_positions) == 0.5 * metrics["transitions_collected"]
            assert algorithm.n_total_timesteps == 48 * iteration
            assert algorithm.n_total_updates == 2 * iteration
            for before, after in zip(parameters_before, parameter_groups, strict=True):
                assert any(not torch.equal(previous, current) for previous, current in zip(before, after, strict=True))
    finally:
        env.close()


@pytest.mark.parametrize("family", ["sac", "td3"])
@pytest.mark.parametrize("recurrent", [False, True])
def test_omitted_collection_keeps_one_vector_step(
    family: Literal["sac", "td3"], recurrent: bool,
) -> None:
    env = _make_multi_env()
    try:
        expected_positions = 20 if recurrent else 5
        algorithm = _make_algorithm(
            env, family, recurrent=recurrent, batch_size=5,
            rollout_steps=None, exposure=expected_positions / 3,
        )
        assert algorithm.rollout_steps_per_iteration == 3
        assert algorithm._resolved_gradient_steps(3) == 1
        assert _iterate(algorithm)["transitions_collected"] == 3
    finally:
        env.close()


@pytest.mark.parametrize("family", ["sac", "td3"])
@pytest.mark.parametrize("recurrent", [False, True])
def test_omitted_collection_rejects_fractional_updates_without_extending_rollout(
    family: Literal["sac", "td3"], recurrent: bool,
) -> None:
    env = _make_multi_env()
    try:
        with pytest.raises(ValueError, match="whole number of updates"):
            _make_algorithm(
                env, family, recurrent=recurrent, batch_size=5, rollout_steps=None, exposure=1 / 3,
            )
    finally:
        env.close()


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("save_optimizer,restore_optimizer", [(False, True), (True, False), (True, True)])
def test_whole_update_schedule_continues_through_checkpoint_and_replay_refill(
    tmp_path: Path, recurrent: bool, save_optimizer: bool, restore_optimizer: bool,
) -> None:
    env = _make_env()
    try:
        def make_algorithm() -> SAC:
            policy = (
                make_policy(env, "tmasac__lstm", model_scale=None, rmat_experimental_compile_lstm=False)
                if recurrent else _make_policy(env)
            )
            return (RecurrentSAC if recurrent else SAC)(
                policy, env, batch_size=2, buffer_capacity_per_env=16, learning_starts=0,
                rollout_steps_per_iteration=6 if recurrent else 2, replay_exposure=1.0,
                train_device="cpu", rollout_device="cpu", replay_storage_device="cpu",
                **({"learning_steps": 3, "burn_in_steps": 1, "temporal_state_store_interval": 1} if recurrent else {}),
            )

        source = make_algorithm()
        assert _iterate(source)["updates"] == 1
        assert source.actor_optimizer.state
        checkpoint = tmp_path / "whole-updates.pt"
        source.save(checkpoint, optimizer_state_dict=source._get_optimizer_state_dict() if save_optimizer else None)
        restored = make_algorithm()
        restored.load(checkpoint, restore_optimizer_state=restore_optimizer)
        assert len(restored.replay_buffer) == 0
        assert bool(restored.actor_optimizer.state) == (save_optimizer and restore_optimizer)
        assert _iterate(restored)["updates"] == 1
        assert restored.n_total_updates == 2
    finally:
        env.close()


@pytest.mark.parametrize("family", ["sac", "td3"])
def test_resume_uses_changed_recurrent_schedule_after_fresh_replay_refill(
    tmp_path: Path, family: Literal["sac", "td3"],
) -> None:
    env = _make_multi_env()
    try:
        source = _make_algorithm(
            env, family, recurrent=True, batch_size=5, rollout_steps=60, exposure=2 / 3,
        )
        assert _iterate(source)["updates"] == 2
        checkpoint = tmp_path / "changed-schedule.pt"
        source.save(checkpoint, optimizer_state_dict=source._get_optimizer_state_dict())
        restored = _make_algorithm(
            env, family, recurrent=True, batch_size=2, rollout_steps=15, exposure=4 / 3,
            learning_steps=5, burn_in_steps=4, temporal_state_store_interval=2,
        )
        restored.load(checkpoint)
        assert len(restored.replay_buffer) == 0
        assert (restored.batch_size, restored.learning_steps, restored.burn_in_steps) == (2, 5, 4)
        assert restored.temporal_state_store_interval == 2
        assert restored.replay_exposure == 4 / 3
        critic_before = [parameter.detach().clone() for parameter in restored.policy.critic_parameters()]
        skipped, scored_positions = _iterate_with_scored_batches(restored)
        assert skipped["transitions_collected"] == 15
        assert skipped["updates"] == 0
        assert skipped["training_skipped"]
        assert scored_positions == []
        assert restored.n_total_updates == 2
        metrics, scored_positions = _iterate_with_scored_batches(restored)
        assert metrics["transitions_collected"] == 15
        assert metrics["updates"] == 2
        assert scored_positions == [10, 10]
        assert sum(scored_positions) == pytest.approx((4 / 3) * metrics["transitions_collected"])
        assert restored.n_total_updates == 4
        assert restored.n_total_timesteps == 90
        assert any(
            not torch.equal(before, after)
            for before, after in zip(critic_before, restored.policy.critic_parameters(), strict=True)
        )
        assert restored.train(gradient_steps=3)["updates"] == 3
        assert restored.n_total_updates == 7
        assert restored.n_total_timesteps == 90
    finally:
        env.close()


@pytest.mark.parametrize("rollout_steps,exposure", [(1, 1.0), (2, 0.25), (16, 0.1), (16, 0.75)])
def test_fractional_update_counts_are_rejected(rollout_steps: int, exposure: float) -> None:
    env = _make_env()
    try:
        with pytest.raises(ValueError, match="whole number of updates"):
            SAC(
                _make_policy(env), env, batch_size=8, buffer_capacity_per_env=16, learning_starts=0,
                rollout_steps_per_iteration=rollout_steps, replay_exposure=exposure,
                train_device="cpu", rollout_device="cpu", replay_storage_device="cpu",
            )
    finally:
        env.close()


@pytest.mark.parametrize("exposure", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_replay_exposure_is_rejected(exposure: float) -> None:
    env = _make_env()
    try:
        with pytest.raises(ValueError, match="replay_exposure"):
            SAC(policy=_make_policy(env), env=env, replay_exposure=exposure)
    finally:
        env.close()
