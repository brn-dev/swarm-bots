from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.algos.off_policy import collect_off_policy_steps
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


@pytest.mark.parametrize("recurrent", [False, True])
def test_interrupted_collection_restarts_from_aligned_replay(recurrent: bool) -> None:
    env = _make_env(max_steps=20)
    try:
        trainer = _make_algorithm(
            env, "sac", recurrent=recurrent, batch_size=2, rollout_steps=6,
            exposure=4 / 3 if recurrent else 1 / 3,
        )
        completed_iterations = int(recurrent)
        if completed_iterations:
            assert _iterate(trainer)["updates"] == 1

        step = env.step
        steps = 0

        def interrupt_after_step(actions: torch.Tensor) -> Any:
            nonlocal steps
            result = step(actions)
            steps += 1
            if steps == 2:
                raise KeyboardInterrupt("collection interrupted")
            return result

        with patch.object(env, "step", side_effect=interrupt_after_step):
            with pytest.raises(KeyboardInterrupt, match="collection interrupted"):
                _iterate(trainer)

        assert len(trainer.replay_buffer) == 0
        assert trainer.n_total_timesteps == 6 * completed_iterations
        assert trainer.n_total_iterations == completed_iterations
        assert trainer.n_total_updates == completed_iterations

        assert _iterate(trainer)["updates"] == 1
        replay = trainer.replay_buffer.get_all()
        torch.testing.assert_close(replay.local_obs[:, 0, 0], torch.arange(6.0))
        torch.testing.assert_close(replay.next_local_obs[:, 0, 0], torch.arange(1.0, 7.0))
        assert trainer.n_total_updates == completed_iterations + 1
        assert max(int(state["step"]) for state in trainer.critic_optimizer.state.values()) == completed_iterations + 1
        if recurrent:
            segment = trainer.replay_buffer.sample_episode_segments(1, segment_length=4, burn_in_steps=2)
            torch.testing.assert_close(
                segment.initial_temporal_state,
                trainer.policy.initial_temporal_state(
                    batch_size=1, n_agents=env.n_agents, device=segment.local_obs.device, dtype=segment.local_obs.dtype,
                ),
            )
    finally:
        env.close()


def test_replay_reset_failure_preserves_collection_error_and_blocks_stale_sampling() -> None:
    env = _make_env()
    try:
        trainer = _make_algorithm(env, "sac", recurrent=False, batch_size=2, rollout_steps=2, exposure=1)
        assert _iterate(trainer)["updates"] == 1
        with (
            patch.object(env, "step", side_effect=RuntimeError("collection error")),
            patch.object(trainer.replay_buffer, "reset", side_effect=RuntimeError("reset error")),
        ):
            with pytest.raises(RuntimeError, match="collection error") as interrupted:
                _iterate(trainer)
            assert any("reset error" in note for note in interrupted.value.__notes__)
            with pytest.raises(RuntimeError, match="reset error"):
                trainer.train(gradient_steps=1)

        assert _iterate(trainer)["updates"] == 1
        torch.testing.assert_close(trainer.replay_buffer.get_all().local_obs[:, 0, 0], torch.arange(2.0))
    finally:
        env.close()


def test_rollout_diagnostics_failure_keeps_collected_replay_and_counters() -> None:
    env = _make_env()
    try:
        trainer = _make_algorithm(env, "sac", recurrent=False, batch_size=2, rollout_steps=2, exposure=1)
        with patch.object(trainer, "_compute_action_metrics", side_effect=RuntimeError("diagnostics failed")):
            with pytest.raises(RuntimeError, match="diagnostics failed"):
                _iterate(trainer)

        assert len(trainer.replay_buffer) == 2
        assert trainer.n_total_timesteps == 2
        assert trainer.n_total_iterations == 1
        assert trainer.n_total_updates == 0
        assert _iterate(trainer)["updates"] == 1
        torch.testing.assert_close(trainer.replay_buffer.get_all().local_obs[:, 0, 0], torch.arange(4.0))
    finally:
        env.close()


@pytest.mark.parametrize("family", ["sac", "td3"])
@pytest.mark.parametrize("recurrent", [False, True])
def test_reporting_failure_preserves_actor_and_target_update_cadence(
    family: Literal["sac", "td3"], recurrent: bool,
) -> None:
    env = _make_env()
    try:
        trainer = _make_algorithm(env, family, recurrent=recurrent, batch_size=2, rollout_steps=8, exposure=1)
        trainer.policy_delay = 2
        trainer.target_update_interval = 2
        collect_off_policy_steps(env, trainer.replay_buffer, n_steps=8, policy=trainer.policy, random_actions=True)
        for count in (1, 2, 3):
            target = trainer.policy.critic_target
            previous_target = [parameter.detach().clone() for parameter in target.parameters()]
            if count < 3:
                with patch.object(trainer, "_preserve_train_step_result", side_effect=RuntimeError("reporting failed")):
                    with pytest.raises(RuntimeError, match="reporting failed"):
                        trainer.train(gradient_steps=1)
            else:
                assert trainer.train(gradient_steps=1)["updates"] == 1

            assert trainer.n_total_updates == count
            assert max(int(state["step"]) for state in trainer.critic_optimizer.state.values()) == count
            actor_steps = [int(state["step"]) for state in trainer.actor_optimizer.state.values()]
            assert max(actor_steps, default=0) == count // 2
            target_changed = any(
                not torch.equal(previous, current)
                for previous, current in zip(previous_target, target.parameters(), strict=True)
            )
            assert target_changed == (count % 2 == (1 if family == "sac" else 0))
    finally:
        env.close()


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
            SAC(
                policy=_make_policy(env), env=env, replay_exposure=exposure,
                train_device="cpu", rollout_device="cpu", replay_storage_device="cpu",
            )
    finally:
        env.close()
