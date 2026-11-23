import csv
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from textwrap import dedent
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any, BinaryIO
from unittest.mock import Mock, patch
from zipfile import ZipFile

import torch
import pytest

from swarmbots.learn.algos.base_algorithm import BaseAlgorithm, LearningRate
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.evaluation import ScheduledEvaluationHook
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage, HybridEMA
from swarmbots.learn.metrics_logger import MetricsLogger, summed
from swarmbots.learn.torch_running_mean_std import TorchRunningMeanStd


class _Policy(BasePolicy):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.weight

    @property
    def gsde_enabled(self) -> bool:
        return False

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
        _ = (
            local_obs,
            global_obs,
            hidden_local_vars,
            hidden_global_vars,
            agent_mask,
            previous_actions,
            deterministic,
        )
        return torch.zeros(1, 1, 1)

    def update_loss_weights(self, **weights: float) -> None:
        _ = weights

    def requires_previous_actions(self) -> bool:
        return False


class _Env:
    unwrapped = None


class _Algorithm(BaseAlgorithm):
    def __init__(self, *, timesteps_per_iteration: int = 4) -> None:
        super().__init__(policy=_Policy(), env=_Env(), learning_rate=1e-3)
        self.timesteps_per_iteration = timesteps_per_iteration
        self.saved_checkpoints: list[tuple[Path, dict[str, Any] | None, float | None]] = []
        self.loaded_optimizer_state: dict[str, Any] | None = None

    def get_hyper_parameters(self) -> dict[str, int | float]:
        return {"timesteps_per_iteration": self.timesteps_per_iteration, "learning_rate": self.learning_rate}

    def _get_optimizer_state_dict(self) -> dict[str, Any]:
        return {"optimizer_step": self.n_total_updates}

    def _apply_optimizer_state_dict(
            self,
            state_dict: dict[str, Any],
            missing_keys: list[str],
            unexpected_keys: list[str],
    ) -> None:
        assert not missing_keys
        assert not unexpected_keys
        self.loaded_optimizer_state = state_dict

    def _apply_learning_rate(self, lr: LearningRate) -> None:
        _ = lr

    def perform_iteration(
            self,
            episode_return_ema: ExponentialMovingAverage,
            episode_success_rate_ema: ExponentialMovingAverage,
            update_ema: bool,
    ) -> tuple[dict[str, Any], int]:
        _ = episode_return_ema, episode_success_rate_ema, update_ema
        self.n_total_iterations += 1
        self.n_total_updates += 1
        self.n_total_timesteps += self.timesteps_per_iteration
        return {}, self.timesteps_per_iteration

    def save(
            self,
            path: str | Path,
            optimizer_state_dict: dict[str, Any] | None = None,
            return_ema: float | None = None,
    ) -> None:
        self.saved_checkpoints.append((Path(path), optimizer_state_dict, return_ema))


def test_load_calls_custom_optimizer_checkpoint_validator(tmp_path: Path) -> None:
    class CustomAlgorithm(_Algorithm):
        def _validate_optimizer_checkpoint(self, checkpoint: dict[str, Any]) -> None:
            super()._validate_optimizer_checkpoint(checkpoint)
            self.validated_steps = checkpoint["n_total_timesteps"]

    source = CustomAlgorithm()
    source.n_total_timesteps = 12
    path = tmp_path / "custom.pt"
    BaseAlgorithm.save(source, path, optimizer_state_dict=source._get_optimizer_state_dict())
    target = CustomAlgorithm()
    target.load(path)
    assert target.validated_steps == target.n_total_timesteps == 12
    assert target.loaded_optimizer_state == source._get_optimizer_state_dict()


def test_learn_without_periodic_saves_still_writes_final_checkpoint(tmp_path: Path) -> None:
    algorithm = _Algorithm()

    result = algorithm.learn(
        max_total_timesteps=8,
        run_dir=tmp_path,
        log_interval=None,
        save_interval=None,
        save_optimizer=True,
        enable_command_prompt=False,
    )

    assert result is algorithm
    assert algorithm.n_total_timesteps == 8
    assert algorithm.n_total_iterations == 2
    assert algorithm.saved_checkpoints == [
        (
            tmp_path / "models" / "model_8_steps_final.pt",
            {"optimizer_step": 2},
            None,
        )
    ]
    assert algorithm._active_run_dir is None
    assert algorithm._active_max_total_timesteps is None


@pytest.mark.parametrize("save_optimizer", [False, True])
def test_inspection_queue_preserves_hooks_and_scheduled_saving(tmp_path: Path, save_optimizer: bool) -> None:
    algorithm = _Algorithm()
    algorithm._command_queue.put("show_hps; set_lr:0.2; pause:1h; stop; save:unexpected.pt")
    hook_steps: list[int] = []

    def inspect_iteration(trainer: BaseAlgorithm, metrics: dict[str, Any], rollout_steps: int) -> None:
        hook_steps.append(trainer.n_total_timesteps)
        assert trainer.execute_command("show_id")

    algorithm.learn(
        max_total_timesteps=8, run_dir=tmp_path, log_interval=None, save_interval=1,
        save_optimizer=save_optimizer, enable_command_prompt=False,
        post_iteration_hooks=[inspect_iteration],
    )

    assert hook_steps == [4, 8]
    assert algorithm.learning_rate == 1e-3
    assert [path.name for path, _, _ in algorithm.saved_checkpoints] == [
        "model_4_steps.pt", "model_8_steps.pt", "model_8_steps_final.pt",
    ]
    assert [optimizer for _, optimizer, _ in algorithm.saved_checkpoints] == (
        [{"optimizer_step": 1}, {"optimizer_step": 2}, {"optimizer_step": 2}]
        if save_optimizer else [None, None, None]
    )
    assert len(list(tmp_path.glob("run_metadata*.json"))) == 1
    assert not (tmp_path / "command_log.jsonl").exists()
    assert algorithm._active_run_dir is None
    assert algorithm._active_max_total_timesteps is None


@pytest.mark.parametrize("budgets", [
    {}, {"max_total_timesteps": 8, "additional_timesteps": 4}, {"max_total_timesteps": 0},
    {"additional_timesteps": True}, {"additional_timesteps": 1.5}, {"max_total_timesteps": float("inf")},
])
def test_invalid_timestep_budget_fails_before_training(budgets: dict[str, Any], tmp_path: Path) -> None:
    trainer = _Algorithm()
    with patch.object(trainer, "_before_learn_loop", side_effect=AssertionError("Unexpected training")):
        with pytest.raises(ValueError, match="exactly one|positive integer"):
            trainer.learn(**budgets, run_dir=tmp_path / "run", enable_command_prompt=False)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("retry_target", [3, 8])
def test_retry_evaluates_interrupted_milestone_before_further_training(retry_target: int) -> None:
    trainer = _Algorithm(timesteps_per_iteration=4)
    runner = Mock()
    runner.evaluate.side_effect = [RuntimeError("evaluation interrupted"), {"eval_episodes": 4}]
    logger = Mock()
    hook = ScheduledEvaluationHook(algorithm=trainer, total_timesteps=3, milestones=[100], runner=runner,
                                  metrics_logger=logger)
    iterations: list[int] = []

    def record_iteration(algorithm: BaseAlgorithm, metrics: dict[str, Any], rollout_steps: int) -> None:
        iterations.append(algorithm.n_total_timesteps)

    options = dict(post_iteration_hooks=[record_iteration, hook], log_interval=None, enable_command_prompt=False)
    with pytest.raises(RuntimeError, match="evaluation interrupted"):
        trainer.learn(max_total_timesteps=3, **options)
    trainer.learn(max_total_timesteps=retry_target, **options)

    assert [call.kwargs["timesteps"] for call in runner.evaluate.call_args_list] == [4, 4]
    logger.log.assert_called_once_with({"eval_episodes": 4})
    assert iterations == ([4, 8] if retry_target == 8 else [4])


def test_failed_training_iteration_does_not_queue_an_evaluation_retry() -> None:
    trainer = _Algorithm(timesteps_per_iteration=4)
    runner = Mock()
    hook = ScheduledEvaluationHook(algorithm=trainer, total_timesteps=4, milestones=[100], runner=runner,
                                  metrics_logger=Mock())
    options = dict(post_iteration_hooks=[hook], log_interval=None, enable_command_prompt=False)

    def fail_after_collection(*args: Any, **kwargs: Any) -> None:
        trainer.n_total_timesteps += 4
        raise RuntimeError("optimizer failed")

    with patch.object(trainer, "perform_iteration", side_effect=fail_after_collection):
        with pytest.raises(RuntimeError, match="optimizer failed"):
            trainer.learn(max_total_timesteps=4, **options)
    trainer.learn(max_total_timesteps=4, **options)
    runner.evaluate.assert_not_called()
    trainer.learn(max_total_timesteps=8, **options)
    runner.evaluate.assert_called_once_with(timesteps=8, milestone_percentage=100.0, gsde_reset_mode=None)


@pytest.mark.parametrize("retry_target", [4, 8])
def test_retry_finishes_remaining_hooks_for_the_original_rollout(retry_target: int) -> None:
    trainer = _Algorithm(timesteps_per_iteration=4)
    runners = [Mock(), Mock()]
    for runner in runners:
        runner.evaluate.side_effect = [RuntimeError("interrupted"), {"episodes": 2}]
    evaluations = [
        ScheduledEvaluationHook(algorithm=trainer, total_timesteps=4, milestones=[100], runner=runner,
                                metrics_logger=Mock())
        for runner in runners
    ]
    processed: list[tuple[int, int, int]] = []

    def stamp(algorithm: BaseAlgorithm, metrics: dict[str, Any], rollout_steps: int) -> None:
        metrics["original_step"] = algorithm.n_total_timesteps

    def record(algorithm: BaseAlgorithm, metrics: dict[str, Any], rollout_steps: int) -> None:
        processed.append((algorithm.n_total_timesteps, metrics["original_step"], rollout_steps))

    options = dict(post_iteration_hooks=[stamp, *evaluations, record], log_interval=None, enable_command_prompt=False)
    for budget in (4, retry_target):
        with pytest.raises(RuntimeError, match="interrupted"):
            trainer.learn(max_total_timesteps=budget, **options)
    trainer.learn(max_total_timesteps=retry_target, **options)
    assert processed == ([(4, 4, 4)] if retry_target == 4 else [(4, 4, 4), (8, 8, 4)])
    assert [[c.kwargs["timesteps"] for c in r.evaluate.call_args_list] for r in runners] == [[4, 4], [4, 4]]


def test_checkpoint_round_trip_restores_policy_counters_and_optimizer(tmp_path: Path) -> None:
    source = _Algorithm()
    source.policy.weight.data.fill_(3.0)
    source.n_total_iterations = 7
    source.n_total_updates = 11
    source.n_total_timesteps = 13
    source._best_return_ema = 4.5
    checkpoint_path = tmp_path / "checkpoint.pt"

    BaseAlgorithm.save(
        source,
        checkpoint_path,
        optimizer_state_dict={"optimizer_step": 11},
        return_ema=4.0,
    )

    restored = _Algorithm()
    restored.load(checkpoint_path)

    torch.testing.assert_close(restored.policy.weight, torch.tensor(3.0))
    assert restored.n_total_iterations == 7
    assert restored.n_total_updates == 11
    assert restored.n_total_timesteps == 13
    assert restored._best_return_ema == 4.5
    assert restored.loaded_optimizer_state == {"optimizer_step": 11}


@pytest.mark.parametrize("field", [
    "policy_state_dict", "algorithm_state_dict", "hyper_parameters", "n_total_iterations", "n_total_updates",
    "n_total_timesteps", "best_return_ema", "env_state", "episode_return_ema", "episode_success_rate_ema",
    "rollout_warmup_done", "pending_best_digest",
])
def test_checkpoint_requires_current_fields(tmp_path: Path, field: str) -> None:
    source = _Algorithm()
    path = tmp_path / "checkpoint.pt"
    BaseAlgorithm.save(source, path, optimizer_state_dict=source._get_optimizer_state_dict())
    checkpoint = torch.load(path, weights_only=False)
    del checkpoint[field]
    torch.save(checkpoint, path)

    with pytest.raises(KeyError, match=field):
        _Algorithm().load(path)


def test_segmented_and_resumed_learn_match_uninterrupted_metrics(tmp_path: Path) -> None:
    class EpisodeAlgorithm(_Algorithm):
        def perform_iteration(self, episode_return_ema: ExponentialMovingAverage,
                              episode_success_rate_ema: ExponentialMovingAverage,
                              update_ema: bool) -> tuple[dict[str, Any], int]:
            if update_ema:
                episode_return_ema.update(float(self.n_total_iterations))
                episode_success_rate_ema.update(float(self.n_total_iterations % 2))
            return super().perform_iteration(episode_return_ema, episode_success_rate_ema, update_ema)

    options = {"log_interval": None, "enable_command_prompt": False}
    uninterrupted = EpisodeAlgorithm(timesteps_per_iteration=1)
    uninterrupted.learn(max_total_timesteps=125, episode_return_ema_alpha=0.2, **options)
    segmented = EpisodeAlgorithm(timesteps_per_iteration=1)
    segmented.learn(max_total_timesteps=63, episode_return_ema_alpha=0.2, **options)
    path = tmp_path / "continuation.pt"
    BaseAlgorithm.save(segmented, path)
    resumed = EpisodeAlgorithm(timesteps_per_iteration=1)
    resumed.load(path)
    for trainer in (resumed, segmented):
        trainer.learn(max_total_timesteps=125, **options)
        assert trainer._episode_return_ema.state_dict() == uninterrupted._episode_return_ema.state_dict()
        assert trainer._episode_success_rate_ema.state_dict() == uninterrupted._episode_success_rate_ema.state_dict()
        assert trainer._best_return_ema == uninterrupted._best_return_ema == uninterrupted._final_return_ema
        assert trainer._best_return_ema is not None


@pytest.mark.parametrize("options", [{"save_interval": 0}, {"episode_return_ema_alpha": float("nan")}])
def test_invalid_learn_settings_fail_before_creating_logger_or_run(tmp_path: Path, options: dict) -> None:
    run = tmp_path / "invalid"
    with pytest.raises(ValueError):
        _Algorithm().learn(additional_timesteps=4, run_dir=run, **options)
    assert not run.exists()


def test_interrupted_checkpoint_write_preserves_previous_model_and_metadata(tmp_path: Path, monkeypatch) -> None:
    algorithm = _Algorithm()
    path = tmp_path / "checkpoint.pt"
    BaseAlgorithm.save(algorithm, path)
    metadata = path.with_suffix(".pt.json").read_bytes()
    algorithm.n_total_timesteps = 9

    def interrupted_save(state: dict, staged: Path) -> None:
        staged.write_bytes(b"truncated")
        raise OSError("interrupted")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="interrupted"):
        BaseAlgorithm.save(algorithm, path)
    assert torch.load(path, weights_only=False)["n_total_timesteps"] == 0
    assert path.with_suffix(".pt.json").read_bytes() == metadata
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize("locked_name", ["checkpoint.pt", "checkpoint.pt.json"])
def test_publication_failure_preserves_previous_checkpoint_pair(tmp_path: Path, monkeypatch, locked_name: str) -> None:
    algorithm = _Algorithm()
    path = tmp_path / "checkpoint.pt"
    BaseAlgorithm.save(algorithm, path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    original_replace = Path.replace

    def reject_locked(source: Path, destination: Path) -> Path:
        if destination.name == locked_name:
            raise PermissionError(locked_name)
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", reject_locked)
    algorithm.n_total_timesteps = 17
    with pytest.raises(PermissionError):
        BaseAlgorithm.save(algorithm, path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("case_alias", (False, True))
def test_competing_checkpoint_writers_publish_matching_scores_and_counters(tmp_path: Path, case_alias: bool) -> None:
    path = tmp_path / "shared.pt"
    second_path = path.with_name(path.name.upper()) if case_alias else path
    if case_alias:
        path.touch()
        if not second_path.exists():
            pytest.skip("requires case-insensitive filenames")
    first_publishing, second_serialized, second_done = Event(), Event(), Event()
    replace = Path.replace
    save = torch.save

    def publish(source: Path, destination: Path) -> Path:
        if destination == path and not first_publishing.is_set():
            first_publishing.set()
            assert second_serialized.wait(timeout=10)
            second_done.wait(timeout=0.2)
        return replace(source, destination)

    def serialize(payload: dict, staged: Path) -> None:
        save(payload, staged)
        if payload["n_total_timesteps"] == 2:
            second_serialized.set()

    def save_at(step: int) -> None:
        trainer = _Algorithm()
        trainer.n_total_timesteps = step
        BaseAlgorithm.save(trainer, second_path if step == 2 else path, return_ema=float(step))
        if step == 2:
            second_done.set()

    with patch.object(Path, "replace", publish), patch.object(torch, "save", serialize), ThreadPoolExecutor(2) as pool:
        first = pool.submit(save_at, 1)
        assert first_publishing.wait(timeout=10)
        second = pool.submit(save_at, 2)
        first.result(timeout=10)
        second.result(timeout=10)

    metadata = json.loads(path.with_suffix(".pt.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(path, weights_only=False)
    assert metadata == {key: checkpoint[key] for key in metadata}


class _ImprovingAlgorithm(_Algorithm):
    save = BaseAlgorithm.save

    def perform_iteration(
        self, episode_return_ema: ExponentialMovingAverage, episode_success_rate_ema: ExponentialMovingAverage,
        update_ema: bool,
    ) -> tuple[dict[str, Any], int]:
        with torch.no_grad():
            self.policy.weight.add_(1.0)
        if update_ema:
            for _ in range(100):
                episode_return_ema.update(self.policy.weight.item())
        return super().perform_iteration(episode_return_ema, episode_success_rate_ema, update_ema)


def test_interrupted_evaluation_preserves_training_log_and_checkpoints(tmp_path: Path) -> None:
    trainer = _ImprovingAlgorithm(timesteps_per_iteration=1)
    trainer.n_total_iterations = 99
    runner = Mock()
    runner.evaluate.side_effect = [RuntimeError("interrupted"), {"eval_episodes": 2}]
    hook = ScheduledEvaluationHook(algorithm=trainer, total_timesteps=1, milestones=[100], runner=runner,
                                  metrics_logger=Mock())
    options = dict(max_total_timesteps=1, run_dir=tmp_path, log_interval=1, save_interval=1,
                   post_iteration_hooks=[hook], enable_command_prompt=False)
    with pytest.raises(RuntimeError, match="interrupted"):
        trainer.learn(**options)
    trainer.learn(**options)

    with (tmp_path / "log.csv").open(newline="", encoding="utf-8") as log:
        rows = list(csv.DictReader(log, delimiter=";"))
    assert [int(row["timesteps"]) for row in rows] == [1]
    assert trainer._best_return_ema == 1.0
    for model in ("model_1_steps.pt", "model_1_steps_final.pt"):
        assert (tmp_path / "models" / model).is_file()
    assert len(list((tmp_path / "models" / "best").rglob("*.pt"))) == 1


@pytest.mark.parametrize("retry_budget", [1, 2])
def test_learn_retries_best_policy_before_another_update_or_final_save(tmp_path: Path, retry_budget: int) -> None:
    trainer = _ImprovingAlgorithm(timesteps_per_iteration=1)
    trainer.n_total_iterations = 99
    options = dict(run_dir=tmp_path, best_rotation_n=2, enable_command_prompt=False, log_interval=None)
    with patch.object(Path, "replace", autospec=True, side_effect=PermissionError("locked")):
        with pytest.raises(PermissionError, match="locked"):
            trainer.learn(max_total_timesteps=1, **options)
    assert trainer.n_total_timesteps == 1
    assert trainer._best_return_ema is None

    trainer.learn(max_total_timesteps=retry_budget, **options)
    paths = list((tmp_path / "models" / "best").rglob("*.pt"))
    checkpoints = [torch.load(path, weights_only=False) for path in paths]
    assert {checkpoint["policy_state_dict"]["weight"].item() for checkpoint in checkpoints} == set(
        range(1, retry_budget + 1)
    )
    first = next(checkpoint for checkpoint in checkpoints if checkpoint["n_total_timesteps"] == 1)
    assert first["return_ema"] == first["best_return_ema"] == 1.0
    final = torch.load(tmp_path / "models" / f"model_{retry_budget}_steps_final.pt", weights_only=False)
    assert final["best_return_ema"] == trainer._best_return_ema


def test_failed_iteration_does_not_become_a_best_checkpoint_on_retry(tmp_path: Path) -> None:
    trainer = _ImprovingAlgorithm(timesteps_per_iteration=1)
    trainer.n_total_iterations = 99
    options = dict(run_dir=tmp_path, enable_command_prompt=False, log_interval=None)
    perform_iteration = trainer.perform_iteration

    def interrupted_iteration(*args: Any, **kwargs: Any) -> None:
        perform_iteration(*args, **kwargs)
        raise RuntimeError("optimizer update interrupted")

    with patch.object(trainer, "perform_iteration", side_effect=interrupted_iteration):
        with pytest.raises(RuntimeError, match="optimizer update interrupted"):
            trainer.learn(max_total_timesteps=1, **options)
    trainer.learn(max_total_timesteps=1, **options)
    assert trainer._best_return_ema is None
    assert not list((tmp_path / "models" / "best").rglob("*.pt"))
    trainer.learn(max_total_timesteps=2, **options)
    best_path, = (tmp_path / "models" / "best").rglob("*.pt")
    assert torch.load(best_path, weights_only=False)["n_total_timesteps"] == 2


@pytest.mark.parametrize("intervening_action", [
    "manual_update", "load_other", "reload_pending", "ignore_saved_best", "partial_policy", "skip_env_stats",
])
def test_only_matching_pending_policy_is_retried(tmp_path: Path, intervening_action: str) -> None:
    trainer = _ImprovingAlgorithm(timesteps_per_iteration=1)
    trainer.n_total_iterations = 99
    options = dict(run_dir=tmp_path, enable_command_prompt=False, log_interval=None)
    with patch.object(Path, "replace", autospec=True, side_effect=PermissionError("locked")):
        with pytest.raises(PermissionError, match="locked"):
            trainer.learn(max_total_timesteps=1, **options)

    if intervening_action == "manual_update":
        trainer.perform_iteration(trainer._episode_return_ema, trainer._episode_success_rate_ema, update_ema=True)
    elif intervening_action == "load_other":
        other = _ImprovingAlgorithm(timesteps_per_iteration=1)
        other.n_total_iterations = 99
        other.perform_iteration(other._episode_return_ema, other._episode_success_rate_ema, update_ema=True)
        other.save(tmp_path / "other.pt")
        trainer.load(tmp_path / "other.pt")
    else:
        trainer.save(tmp_path / "recovery.pt")
        trainer = _ImprovingAlgorithm(timesteps_per_iteration=1)
        if intervening_action == "partial_policy":
            trainer.policy.extra = torch.nn.Linear(1, 1)
        trainer.load(
            tmp_path / "recovery.pt", recover_best_return_ema=intervening_action != "ignore_saved_best",
            strict_load_state_dict=intervening_action != "partial_policy",
            restore_env_state=intervening_action != "skip_env_stats",
        )

    trainer.learn(max_total_timesteps=trainer.n_total_timesteps, **options)
    best_paths = list((tmp_path / "models" / "best").rglob("*.pt"))
    assert len(best_paths) == int(intervening_action == "reload_pending")
    if best_paths:
        best = torch.load(best_paths[0], weights_only=False)
        assert best["best_return_ema"] == best["return_ema"] == 1.0
        assert best["pending_best_digest"] is None


@pytest.mark.parametrize("change", ["load_weights", "assign_weights", "update_stats", "replace_stats", "omit_stats_on_load"])
def test_failed_best_save_does_not_promote_changed_weights_or_normalization(tmp_path: Path, change: str) -> None:
    trainer = _ImprovingAlgorithm(timesteps_per_iteration=1)
    trainer.n_total_iterations = 99
    rms = TorchRunningMeanStd(shape=(1,), device="cpu")
    trainer.env = SimpleNamespace(env=trainer.env, unwrapped=None, obs_rms=rms)
    options = dict(run_dir=tmp_path, enable_command_prompt=False, log_interval=None)
    with patch.object(Path, "replace", autospec=True, side_effect=PermissionError("locked")):
        with pytest.raises(PermissionError, match="locked"):
            trainer.learn(max_total_timesteps=1, **options)

    if change in {"load_weights", "assign_weights"}:
        trainer.policy.load_state_dict({"weight": torch.tensor(-5.0)}, assign=change == "assign_weights")
    elif change == "replace_stats":
        rms.mean = torch.ones_like(rms.mean)
    elif change == "update_stats":
        rms.mean.add_(1)
    recovery_path = tmp_path / "recovery.pt"
    trainer.save(recovery_path)
    pending = torch.load(recovery_path, weights_only=False)["pending_best_digest"]
    if change == "omit_stats_on_load":
        assert pending is not None
        trainer = _ImprovingAlgorithm(timesteps_per_iteration=1)
        trainer.load(recovery_path)
    else:
        assert pending is None
    trainer.learn(max_total_timesteps=1, **options)
    assert not list((tmp_path / "models" / "best").rglob("*.pt"))


def test_pending_best_retries_with_an_inference_tensor(tmp_path: Path) -> None:
    trainer = _ImprovingAlgorithm(timesteps_per_iteration=1)
    trainer.n_total_iterations = 99
    with torch.inference_mode():
        trainer.policy.register_buffer("offset", torch.zeros(0))
    options = dict(run_dir=tmp_path, max_total_timesteps=1, enable_command_prompt=False, log_interval=None)
    with patch.object(Path, "replace", autospec=True, side_effect=PermissionError("locked")):
        with pytest.raises(PermissionError, match="locked"):
            trainer.learn(**options)
    trainer.learn(**options)
    assert len(list((tmp_path / "models" / "best").rglob("*.pt"))) == 1


class _UnorderedExtraState(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.labels = {f"label-{index}" for index in range(16)}

    def get_extra_state(self) -> dict[str, set[str]]:
        return {"labels": self.labels}

    def set_extra_state(self, state: dict[str, set[str]]) -> None:
        self.labels = state["labels"]


def _extra_state_best_trainer() -> _ImprovingAlgorithm:
    trainer = _ImprovingAlgorithm(timesteps_per_iteration=1)
    trainer.n_total_iterations = 99
    trainer.policy.metadata = _UnorderedExtraState()
    return trainer


def test_pending_best_with_unordered_extra_state_retries_in_a_new_process(tmp_path: Path) -> None:
    script = dedent("""\
        import json
        import sys
        from pathlib import Path
        from unittest.mock import patch
        from tests.test_base_algorithm_lifecycle import _extra_state_best_trainer

        run_dir = Path(sys.argv[1])
        checkpoint = run_dir / "recovery.pt"
        trainer = _extra_state_best_trainer()
        options = dict(run_dir=run_dir, max_total_timesteps=1, enable_command_prompt=False, log_interval=None)
        if sys.argv[2] == "save":
            with patch.object(Path, "replace", autospec=True, side_effect=PermissionError("locked")):
                try:
                    trainer.learn(**options)
                except PermissionError:
                    pass
                else:
                    raise AssertionError("The best checkpoint write did not fail")
            trainer.save(checkpoint)
        else:
            trainer.load(checkpoint)
            trainer.learn(**options)
        print(json.dumps(list(trainer.policy.metadata.labels)))
    """)
    orders = []
    for action, seed in (("save", "11"), ("load", "17")):
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), action],
            env={**os.environ, "PYTHONHASHSEED": seed}, check=True, capture_output=True, text=True,
        )
        orders.append(json.loads(result.stdout.splitlines()[-1]))
    assert set(orders[0]) == set(orders[1]) and orders[0] != orders[1]
    best, = (tmp_path / "models" / "best").rglob("*.pt")
    saved = torch.load(best, weights_only=False)
    assert saved["policy_state_dict"]["metadata._extra_state"]["labels"] == set(orders[0])
    assert saved["n_total_timesteps"] == saved["best_return_ema"] == 1


def test_pending_best_retries_from_standalone_checkpoint_with_identity_hashed_members(tmp_path: Path) -> None:
    class TensorMetadata(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.tensor(4)
            self.labels = {torch.tensor(index) for index in range(16) if index != 4} | {self.anchor}

        def get_extra_state(self) -> dict[str, Any]:
            return {"anchor": self.anchor, "labels": self.labels}

        def set_extra_state(self, state: dict[str, Any]) -> None:
            self.anchor = state["anchor"]
            self.labels = state["labels"]

    source = _ImprovingAlgorithm(timesteps_per_iteration=1)
    source.n_total_iterations = 99
    source.policy.metadata = TensorMetadata()
    options = dict(run_dir=tmp_path, max_total_timesteps=1, enable_command_prompt=False, log_interval=None)
    with patch.object(Path, "replace", autospec=True, side_effect=PermissionError("locked")):
        with pytest.raises(PermissionError, match="locked"):
            source.learn(**options)
    recovery = tmp_path / "recovery.pt"
    source.save(recovery)
    moved = recovery.rename(tmp_path / "standalone.pt")
    assert not moved.with_name(f"{moved.name}.json").exists()
    with ZipFile(moved) as archive:
        assert archive.comment.isascii()
        assert archive.testzip() is None

    restored = _ImprovingAlgorithm(timesteps_per_iteration=1)
    restored.policy.metadata = TensorMetadata()
    restored.load(moved)
    assert {label.item() for label in restored.policy.metadata.labels} == set(range(16))
    assert any(label is restored.policy.metadata.anchor for label in restored.policy.metadata.labels)
    restored.learn(**options)
    best, = (tmp_path / "models" / "best").rglob("*.pt")
    saved = torch.load(best, weights_only=False)
    assert saved["n_total_timesteps"] == saved["best_return_ema"] == 1


@pytest.mark.parametrize("change_after_load", [False, True])
def test_pending_best_rebinds_extra_state_reconstructed_by_the_module(
    tmp_path: Path, change_after_load: bool,
) -> None:
    class ReorderedExtraState(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.labels = {0, 8, 16, 24}

        def get_extra_state(self) -> dict[str, set[int]]:
            return {"labels": self.labels}

        def set_extra_state(self, state: dict[str, set[int]]) -> None:
            self.labels = set(sorted(state["labels"], reverse=True))

    source = _ImprovingAlgorithm(timesteps_per_iteration=1)
    source.n_total_iterations = 99
    source.policy.metadata = ReorderedExtraState()
    options = dict(run_dir=tmp_path, max_total_timesteps=1, enable_command_prompt=False, log_interval=None)
    with patch.object(Path, "replace", autospec=True, side_effect=PermissionError("locked")):
        with pytest.raises(PermissionError, match="locked"):
            source.learn(**options)
    path = tmp_path / "recovery.pt"
    source.save(path)
    restored = _ImprovingAlgorithm(timesteps_per_iteration=1)
    restored.policy.metadata = ReorderedExtraState()
    restored.load(path)
    assert restored.policy.metadata.labels == source.policy.metadata.labels
    assert list(restored.policy.metadata.labels) != list(source.policy.metadata.labels)
    if change_after_load:
        restored.policy.metadata.labels.add(32)
    restored.learn(**options)
    assert len(list((tmp_path / "models" / "best").rglob("*.pt"))) == int(not change_after_load)


@pytest.mark.parametrize("sink", ["csv", "partial_csv", "wandb"])
def test_final_evaluation_retries_the_unfinished_metric_sink(tmp_path: Path, sink: str) -> None:
    trainer = _Algorithm(timesteps_per_iteration=1)
    runner = Mock()
    runner.evaluate.return_value = {"timesteps": 1, "eval_success_rate": 75.0}
    log_path = tmp_path / "evaluation.csv"
    wandb_run = Mock() if sink == "wandb" else None
    metrics_logger = MetricsLogger(tmp_path, filename=log_path.name, console_keys=[], wandb_run=wandb_run)
    hook = ScheduledEvaluationHook(
        algorithm=trainer, total_timesteps=1, milestones=[100], runner=runner, metrics_logger=metrics_logger,
    )
    options = dict(max_total_timesteps=1, post_iteration_hooks=[hook], enable_command_prompt=False, log_interval=None)
    if sink == "csv":
        log_path.mkdir()
        try:
            with pytest.raises(OSError):
                trainer.learn(**options)
        finally:
            log_path.rmdir()
    elif sink == "partial_csv":
        metrics_logger.log({"timesteps": 0})
        metrics_logger.flush()
        previous_rows = log_path.read_bytes()
        write_row = csv.DictWriter.writerow

        def fail_after_writing_row(writer: csv.DictWriter, row: dict[str, Any]) -> None:
            write_row(writer, row)
            if "eval_success_rate" in row:
                raise OSError("CSV write interrupted")

        with patch.object(csv.DictWriter, "writerow", fail_after_writing_row):
            with pytest.raises(OSError, match="CSV write interrupted"):
                trainer.learn(**options)
        assert log_path.read_bytes() == previous_rows
    else:
        with patch.object(metrics_logger, "_log_to_wandb", side_effect=[TimeoutError("sink unavailable"), None]) as wandb:
            with pytest.raises(TimeoutError, match="sink unavailable"):
                trainer.learn(**options)
            trainer.learn(**options)
        assert wandb.call_count == 2
    try:
        trainer.learn(**options)
        with log_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file, delimiter=";"))
        expected_timesteps = ["0", "1"] if sink == "partial_csv" else ["1"]
        assert [row["timesteps"] for row in rows] == expected_timesteps
        assert float(rows[-1]["eval_success_rate"]) == 75.0
        runner.evaluate.assert_called_once()
    finally:
        hook.close()


def test_scheduled_evaluation_retains_custom_logger_overrides(tmp_path: Path) -> None:
    class CustomLogger(MetricsLogger):
        should_interrupt = True

        def log(self, metrics: dict[str, Any]) -> None:
            super().log({**metrics, "custom": "recorded"})

        def flush(self) -> None:
            super().flush()
            if self.should_interrupt:
                self.should_interrupt = False
                raise OSError("custom logger interrupted")

    trainer = _Algorithm(timesteps_per_iteration=1)
    runner = Mock()
    runner.evaluate.return_value = {"timesteps": 1}
    metrics_logger = CustomLogger(tmp_path, filename="evaluation.csv", console_keys=[])
    hook = ScheduledEvaluationHook(
        algorithm=trainer, total_timesteps=1, milestones=[100], runner=runner, metrics_logger=metrics_logger,
    )
    options = dict(max_total_timesteps=1, post_iteration_hooks=[hook], enable_command_prompt=False, log_interval=None)
    try:
        with pytest.raises(OSError, match="custom logger interrupted"):
            trainer.learn(**options)
        trainer.learn(**options)
        with (tmp_path / "evaluation.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file, delimiter=";"))
        assert [(row["timesteps"], row["custom"]) for row in rows] == [("1", "recorded")]
        runner.evaluate.assert_called_once()
    finally:
        hook.close()


def test_scheduled_evaluation_appends_while_csv_reader_is_open(tmp_path: Path) -> None:
    trainer = _Algorithm(timesteps_per_iteration=1)
    runner = Mock()
    runner.evaluate.side_effect = [{"timesteps": 1}, {"timesteps": 2}]
    metrics_logger = MetricsLogger(tmp_path, filename="evaluation.csv", console_keys=[])
    hook = ScheduledEvaluationHook(
        algorithm=trainer, total_timesteps=2, milestones=[50, 100], runner=runner, metrics_logger=metrics_logger,
    )
    try:
        trainer.n_total_timesteps = 1
        hook(trainer, {}, 1)
        with (tmp_path / "evaluation.csv").open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter=";")
            assert [row["timesteps"] for row in reader] == ["1"]
            trainer.n_total_timesteps = 2
            hook(trainer, {}, 1)
            assert [row["timesteps"] for row in reader] == ["2"]
    finally:
        hook.close()


@pytest.mark.parametrize("has_previous_row", [False, True])
def test_scheduled_evaluation_appends_to_csv_without_final_newline(tmp_path: Path, has_previous_row: bool) -> None:
    trainer = _Algorithm(timesteps_per_iteration=1)
    runner = Mock()
    runner.evaluate.return_value = {"timesteps": 1, "diagnostic": "second"}
    log_path = tmp_path / "evaluation.csv"
    log_path.write_bytes(b"timesteps;diagnostic;timestamp" + (b"\r\n0;first;initial" if has_previous_row else b""))
    metrics_logger = MetricsLogger(tmp_path, filename=log_path.name, console_keys=[])
    hook = ScheduledEvaluationHook(
        algorithm=trainer, total_timesteps=1, milestones=[100], runner=runner, metrics_logger=metrics_logger,
    )
    try:
        trainer.n_total_timesteps = 1
        hook(trainer, {}, 1)
        with log_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file, delimiter=";"))
        expected_rows = [("0", "first")] if has_previous_row else []
        assert [(row["timesteps"], row["diagnostic"]) for row in rows] == [*expected_rows, ("1", "second")]
    finally:
        hook.close()


def test_scheduled_evaluation_preserves_large_csv_fields(tmp_path: Path) -> None:
    trainer = _Algorithm(timesteps_per_iteration=1)
    runner = Mock()
    diagnostic = '"quoted";' + "é" * 140_000 + "\ncontinued"
    runner.evaluate.side_effect = [
        {"timesteps": 1, "diagnostic": diagnostic},
        {"timesteps": 2, "diagnostic": "next"},
        {"timesteps": 3, "diagnostic": "expanded", "new_metric": 4},
    ]
    metrics_logger = MetricsLogger(tmp_path, filename="evaluation.csv", console_keys=[])
    hook = ScheduledEvaluationHook(
        algorithm=trainer, total_timesteps=3, milestones=[30, 60, 100], runner=runner, metrics_logger=metrics_logger,
    )
    original_limit = csv.field_size_limit(128 * 1024)
    try:
        for timestep in (1, 2, 3):
            trainer.n_total_timesteps = timestep
            hook(trainer, {}, 1)
        assert csv.field_size_limit() == 128 * 1024
        csv.field_size_limit(len(diagnostic))
        with (tmp_path / "evaluation.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file, delimiter=";"))
        assert [(row["timesteps"], row["diagnostic"], row["new_metric"]) for row in rows] == [
            ("1", diagnostic, ""), ("2", "next", ""), ("3", "expanded", "4"),
        ]
    finally:
        csv.field_size_limit(original_limit)
        hook.close()


@pytest.mark.parametrize("buffer_size", [2, 3])
def test_scheduled_evaluation_aggregates_pending_training_metrics(tmp_path: Path, buffer_size: int) -> None:
    trainer = _Algorithm(timesteps_per_iteration=1)
    runner = Mock()
    runner.evaluate.return_value = {"timesteps": 1, "combined": summed(10)}
    metrics_logger = MetricsLogger(
        tmp_path, filename="evaluation.csv", console_keys=[], buffer_size=buffer_size,
    )
    hook = ScheduledEvaluationHook(
        algorithm=trainer, total_timesteps=1, milestones=[100], runner=runner, metrics_logger=metrics_logger,
    )
    try:
        metrics_logger.log({"timesteps": 0, "combined": summed(2)})
        trainer.n_total_timesteps = 1
        hook(trainer, {}, 1)
        metrics_logger.log({"timesteps": 2, "combined": summed(3)})
        metrics_logger.flush()
        with (tmp_path / "evaluation.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file, delimiter=";"))
        assert [(row["timesteps"], row["combined"]) for row in rows] == [("1", "12"), ("2", "3")]
    finally:
        hook.close()


@pytest.mark.parametrize("failure", ["short_write", "write_error", "rollback_error"])
def test_scheduled_evaluation_retries_interrupted_same_schema_csv_append(tmp_path: Path, failure: str) -> None:
    trainer = _Algorithm(timesteps_per_iteration=1)
    runner = Mock()
    runner.evaluate.return_value = {"timesteps": 1}
    log_path = tmp_path / "evaluation.csv"
    metrics_logger = MetricsLogger(tmp_path, filename=log_path.name, console_keys=[])
    hook = ScheduledEvaluationHook(
        algorithm=trainer, total_timesteps=1, milestones=[100], runner=runner, metrics_logger=metrics_logger,
    )
    original_open = Path.open

    @contextmanager
    def open_interrupted_append(path: Path, **kwargs: Any) -> Iterator[BinaryIO]:
        with original_open(path, "r+b", **kwargs) as file:
            original_write = file.write
            original_truncate = file.truncate

            def interrupted_write(row: bytes) -> int:
                written = original_write(row[:len(row) // 2])
                if failure == "short_write":
                    return written
                raise OSError("write interrupted")

            def truncate(offset: int) -> int:
                if failure == "rollback_error":
                    raise OSError("rollback interrupted")
                return original_truncate(offset)

            with patch.object(file, "write", interrupted_write), patch.object(file, "truncate", truncate):
                yield file

    def open_file(path: Path, mode: str = "r", **kwargs: Any) -> Any:
        if path == log_path and mode == "r+b":
            return open_interrupted_append(path, **kwargs)
        return original_open(path, mode, **kwargs)

    try:
        metrics_logger.log({"timesteps": 0})
        metrics_logger.flush()
        previous = log_path.read_bytes()
        trainer.n_total_timesteps = 1
        with patch.object(Path, "open", open_file), pytest.raises(OSError):
            hook(trainer, {}, 1)
        if failure != "rollback_error":
            assert log_path.read_bytes() == previous
        hook(trainer, {}, 1)
        with log_path.open(newline="", encoding="utf-8") as file:
            previous_row, recovered_row = csv.DictReader(file, delimiter=";")
        assert [previous_row["timesteps"], recovered_row["timesteps"]] == ["0", "1"]
        assert set(recovered_row) == {"timesteps", "timestamp"}
        datetime.fromisoformat(recovered_row["timestamp"])
        runner.evaluate.assert_called_once()
    finally:
        hook.close()


@pytest.mark.parametrize("field", ["extra_state", "weight", "missing_seal", "missing_extra_state_metadata"])
def test_pending_best_with_archived_extra_state_rejects_changed_payload(tmp_path: Path, field: str) -> None:
    trainer = _extra_state_best_trainer()
    options = dict(run_dir=tmp_path, max_total_timesteps=1, enable_command_prompt=False, log_interval=None)
    with patch.object(Path, "replace", autospec=True, side_effect=PermissionError("locked")):
        with pytest.raises(PermissionError, match="locked"):
            trainer.learn(**options)
    path = tmp_path / "recovery.pt"
    trainer.save(path)
    recovered = torch.load(path, weights_only=False)
    assert recovered["pending_best_digest"] is not None
    if field == "extra_state":
        recovered["policy_state_dict"]["metadata._extra_state"]["labels"].add("different")
    elif field == "weight":
        recovered["policy_state_dict"]["weight"].fill_(-4.0)
    else:
        del recovered["pending_best_archive_checksum"]
        if field == "missing_extra_state_metadata":
            del recovered["pending_best_extra_state"]
    torch.save(recovered, path)
    trainer = _extra_state_best_trainer()
    trainer.load(path)
    trainer.learn(**options)
    assert not list((tmp_path / "models" / "best").rglob("*.pt"))


def test_failed_candidate_inspection_preserves_original_save_error(tmp_path: Path) -> None:
    trainer = _ImprovingAlgorithm(timesteps_per_iteration=1)
    trainer.n_total_iterations = 99
    options = dict(run_dir=tmp_path, max_total_timesteps=1, enable_command_prompt=False, log_interval=None)
    with patch("swarmbots.learn.algos.base_algorithm.best_candidate_digest", side_effect=RuntimeError("unavailable")) as scan:
        with patch.object(Path, "replace", autospec=True, side_effect=PermissionError("locked")):
            with pytest.raises(PermissionError, match="locked"):
                trainer.learn(**options)
        trainer.learn(**options)
    assert scan.call_count == 1
    assert not list((tmp_path / "models" / "best").rglob("*.pt"))


def test_ordinary_saves_do_not_inspect_candidate_weights(tmp_path: Path) -> None:
    trainer = _ImprovingAlgorithm(timesteps_per_iteration=1)
    trainer.n_total_iterations = 99
    with patch("swarmbots.learn.algos.base_algorithm.best_candidate_digest") as scan:
        trainer.learn(max_total_timesteps=1, run_dir=tmp_path, enable_command_prompt=False, log_interval=None)
        trainer.save(tmp_path / "manual.pt")
    scan.assert_not_called()


def test_hybrid_ema_rejects_a_nonzero_sum_with_zero_samples() -> None:
    state = HybridEMA(0.1).state_dict()
    state["pre_exponential_sum"] = 1000.0
    with pytest.raises(ValueError, match="sum"):
        HybridEMA.from_state_dict(state)


@pytest.mark.parametrize("previous_samples", [0, 1, 2])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_ema_sample_preserves_all_history(previous_samples: int, value: float) -> None:
    average = HybridEMA(alpha=0.1, n_pre_exponential_samples=2)
    for _ in range(previous_samples):
        average.update(2.0)
    before = average.state_dict()
    with pytest.raises(ValueError, match="finite"):
        average.update(value)
    assert average.state_dict() == before


def test_overflowing_ema_sample_sum_preserves_history() -> None:
    average = HybridEMA(alpha=0.1)
    average.update(sys.float_info.max)
    before = average.state_dict()
    with pytest.raises(ValueError, match="finite"):
        average.update(sys.float_info.max)
    assert average.state_dict() == before


def test_nonfinite_episode_cannot_replace_a_loadable_best(tmp_path: Path) -> None:
    trainer = _ImprovingAlgorithm(timesteps_per_iteration=1)
    trainer.n_total_iterations = 99
    perform_iteration = trainer.perform_iteration

    def collect_episode(episode_return_ema: ExponentialMovingAverage, episode_success_rate_ema: ExponentialMovingAverage,
                        update_ema: bool) -> tuple[dict[str, Any], int]:
        if trainer.n_total_updates:
            episode_return_ema.update(float("nan"))
        return perform_iteration(episode_return_ema, episode_success_rate_ema, update_ema)

    with patch.object(trainer, "perform_iteration", side_effect=collect_episode):
        with pytest.raises(ValueError, match="finite"):
            trainer.learn(max_total_timesteps=2, run_dir=tmp_path, enable_command_prompt=False, log_interval=None)
    best_path, = (tmp_path / "models" / "best").rglob("*.pt")
    restored = _ImprovingAlgorithm()
    restored.load(best_path)
    assert restored._best_return_ema == restored._episode_return_ema.get() == 1.0
    with pytest.raises(ValueError, match="finite"):
        restored.save(best_path, return_ema=float("inf"))
    assert torch.load(best_path, weights_only=False)["return_ema"] == 1.0


def test_best_rotation_keeps_both_sessions_started_in_the_same_second(tmp_path: Path) -> None:
    trainer = _ImprovingAlgorithm(timesteps_per_iteration=1)
    trainer.n_total_iterations = 99
    with patch("swarmbots.learn.algos.base_algorithm.datetime") as clock:
        clock.now.return_value = datetime(2026, 1, 1, 12)
        for _ in range(2):
            trainer.learn(additional_timesteps=1, run_dir=tmp_path, best_rotation_n=2,
                          enable_command_prompt=False, log_interval=None)
    paths = (tmp_path / "models" / "best").rglob("*.pt")
    assert sorted(torch.load(path, weights_only=False)["n_total_timesteps"] for path in paths) == [1, 2]


def test_base_checkpoint_preserves_custom_learning_rate_mapping(tmp_path: Path) -> None:
    source = _Algorithm()
    source.learning_rate = {"actor": 0.001, "critic": 0.01}
    path = tmp_path / "custom.pt"
    BaseAlgorithm.save(source, path, optimizer_state_dict=source._get_optimizer_state_dict())
    target = _Algorithm()
    target.load(path)
    assert target.learning_rate == source.learning_rate


def test_checkpoint_can_omit_optimizer_state(tmp_path: Path) -> None:
    source = _Algorithm()
    source.policy.weight.data.fill_(2.0)
    path = tmp_path / "weights-and-metadata.pt"
    BaseAlgorithm.save(source, path)
    restored = _Algorithm()

    restored.load(path)

    torch.testing.assert_close(restored.policy.weight, source.policy.weight)
    assert restored.loaded_optimizer_state is None


class _OptimizerAlgorithm(_Algorithm):
    def __init__(self, *, compiled: bool) -> None:
        super().__init__()
        if compiled:
            self.policy = torch.compile(self.policy, backend="eager")
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def _get_optimizer_state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def _apply_optimizer_state_dict(
        self, state_dict: dict[str, Any], missing_keys: list[str], unexpected_keys: list[str],
    ) -> None:
        self.optimizer.load_state_dict(state_dict)


@pytest.mark.parametrize("compiled_source", [False, True])
@pytest.mark.parametrize("compiled_target", [False, True])
def test_checkpoint_continuation_preserves_adam_across_compile_modes(
    tmp_path: Path, compiled_source: bool, compiled_target: bool,
) -> None:
    def update(algorithm: _OptimizerAlgorithm) -> None:
        algorithm.optimizer.zero_grad()
        (algorithm.policy(torch.ones(2)) - 1.0).square().mean().backward()
        algorithm.optimizer.step()

    source = _OptimizerAlgorithm(compiled=compiled_source)
    update(source)
    path = tmp_path / "checkpoint.pt"
    BaseAlgorithm.save(source, path, optimizer_state_dict=source._get_optimizer_state_dict())
    restored = _OptimizerAlgorithm(compiled=compiled_target)
    restored.load(path)

    update(source)
    update(restored)

    torch.testing.assert_close(restored.policy.weight, source.policy.weight, rtol=0, atol=0)
    for source_state, restored_state in zip(source.optimizer.state.values(), restored.optimizer.state.values(), strict=True):
        for key, value in source_state.items():
            torch.testing.assert_close(restored_state[key], value, rtol=0, atol=0)
