import abc
import json
import math
import os
import pathlib
import queue
import threading
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from numbers import Integral
from pathlib import Path
from typing import Any, Optional, Collection, Callable, Self
from uuid import uuid4

from loguru import logger

try:
    logger.level("SAVE")
except ValueError:
    logger.level("SAVE", no=21, color="<magenta>")

from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.checkpointing import (
    align_torch_compile_state_dict_keys,
    best_candidate_digest,
    prepare_env_state,
    install_env_state,
    restore_best_candidate_digest,
    snapshot_best_candidate_extra_state,
    snapshot_policy_for_extra_state,
    validate_policy_state,
    capture_env_state,
    load_training_checkpoint,
    write_checkpoint,
)
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage, HybridEMA
from swarmbots.learn.loss_weights import prepare_checkpoint_loss_weights
from swarmbots.learn.logging_levels import ConsoleSelection, LoggingLevel
from swarmbots.learn.metrics_logger import (
    MetricsLogger,
    mean_std,
    rate,
    summed,
)
from swarmbots.learn.performance_timer import PerformanceTimer
from swarmbots.learn.run_metadata import collect_source_revision
from swarmbots.learn.serialization_utils import serialize_value
from swarmbots.utils.machine_specs import collect_machine_specs




MIN_ITERATIONS_FOR_EMA = 10
MIN_ITERATIONS_FOR_BEST = 100

LearningRate = float | list[float] | dict[str, float]
LearnIterationHook = Callable[["BaseAlgorithm", dict[str, Any], int], None]


@dataclass
class _PendingIterationHooks:
    timesteps: int
    hooks: tuple[LearnIterationHook, ...]
    metrics: dict[str, Any]
    rollout_steps: int


def validate_timestep_budget(value: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("The training timestep budget must be a positive integer")
    return int(value)


def validate_scalar_learning_rate(learning_rate: LearningRate) -> float:
    if not isinstance(learning_rate, float):
        raise TypeError("learning_rate must be a float")
    if not math.isfinite(learning_rate) or learning_rate < 0.0:
        raise ValueError("learning_rate must be finite and nonnegative")
    return learning_rate

MEAN_STD_LOG_METRIC_KEYS = frozenset({
    "rollout_time",
    "env_reset_time",
    "to_rollout_device_time",
    "total_reset_noise_time",
    "total_policy_forward_time",
    "total_env_step_time",
    "total_buffer_add_time",
    "rollout_finalize_time",
    "rollout_batch_builder_init_time",
    "rollout_batch_finalize_time",
    "to_train_device_time",
    "sampler_init_time",
    "total_sampling_time",
    "total_update_time",
    "metrics_time",
    "train_time",
    "total_compute_grad_norms_time",
    "expl_var",
    "grad_clip_frac",
})
SUM_LOG_METRIC_KEYS = frozenset({"updates", "transitions_collected"})


def _apply_log_metric_reductions(metrics: dict[str, Any]) -> dict[str, Any]:
    logging_metrics = metrics.copy()
    for key in MEAN_STD_LOG_METRIC_KEYS:
        if key in logging_metrics and logging_metrics[key] is not None:
            logging_metrics[key] = mean_std(logging_metrics[key])
    for key, value in metrics.items():
        if value is None:
            continue
        if key in SUM_LOG_METRIC_KEYS or (key.startswith("scenario/") and key.endswith("/episodes")):
            logging_metrics[key] = summed(value)
        elif key == "ep_success_rate__n" or key.endswith("/ep_success_rate__n"):
            rate_key = key.removesuffix("__n")
            if metrics.get(rate_key) is not None:
                logging_metrics[rate_key] = rate(metrics[rate_key] * value, value)
            logging_metrics[key] = summed(value)
    return logging_metrics


class BaseAlgorithm(abc.ABC):

    def __init__(
            self,
            policy: BasePolicy,
            env: BaseLearnEnvWrapper,
            learning_rate: LearningRate,
            *,
            rollout_warmup_steps_per_env: int = 0,
    ) -> None:
        self.policy = policy
        self.env = env

        self.learning_rate = learning_rate
        self.training_metadata: dict[str, Any] = {}

        self.n_total_iterations = 0
        self.n_total_updates = 0
        self.n_total_timesteps = 0
        self.rollout_warmup_steps_per_env = int(rollout_warmup_steps_per_env)
        if self.rollout_warmup_steps_per_env < 0:
            raise ValueError(f"rollout_warmup_steps_per_env must be >= 0, got {self.rollout_warmup_steps_per_env}")
        self._rollout_warmup_done = False
        self._command_queue: queue.Queue[str] = queue.Queue()
        self._command_prompt_started: bool = False
        self._active_run_dir: Path | None = None
        self._last_return_ema: float | None = None
        self._final_return_ema: float | None = None
        self._episode_return_ema = HybridEMA(alpha=0.01)
        self._episode_success_rate_ema = HybridEMA(alpha=0.01)
        self._best_return_ema: float | None = None
        self._pending_best_save: bytes | None = None
        self._pending_iteration_hooks: _PendingIterationHooks | None = None
        self._active_max_total_timesteps: int | None = None
        self._active_learn_started_monotonic: float | None = None
        self._active_learn_started_timesteps: int | None = None

    @abc.abstractmethod
    def get_hyper_parameters(self) -> dict[str, Any]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_optimizer_state_dict(self) -> dict[str, Any]:
        raise NotImplementedError()

    def _validate_optimizer_checkpoint(
            self,
            checkpoint: dict[str, Any],
    ) -> None:
        """Reject invalid optimizer settings before loading any trainer state."""

    def _restore_checkpoint_schedulers(self, optimizer_state: dict[str, Any]) -> None:
        """Apply checked scheduler settings before copying any checkpoint tensors."""

    def _get_algorithm_state_dict(self) -> dict[str, Any]:
        return {}

    def _validate_algorithm_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass

    def _apply_algorithm_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass

    @abc.abstractmethod
    def _apply_optimizer_state_dict(
            self,
            state_dict: dict[str, Any],
            missing_keys: list[str],
            unexpected_keys: list[str],
    ) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def _apply_learning_rate(self, lr: LearningRate) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def perform_iteration(
            self,
            episode_return_ema: ExponentialMovingAverage,
            episode_success_rate_ema: ExponentialMovingAverage,
            update_ema: bool,
    ) -> tuple[dict[str, Any], int]:
        """
        :return: metrics, rollout_steps
        """
        raise NotImplementedError()

    def _before_learn_loop(self) -> None:
        pass

    def _run_iteration_hooks(
            self, hooks: tuple[LearnIterationHook, ...], metrics: dict[str, Any], rollout_steps: int,
    ) -> None:
        for index, hook in enumerate(hooks):
            try:
                hook(self, metrics, rollout_steps)
            except BaseException:
                self._pending_iteration_hooks = _PendingIterationHooks(
                    self.n_total_timesteps, hooks[index:], metrics, rollout_steps,
                )
                raise

    def learn(
            self,
            max_total_timesteps: int | None = None,
            additional_timesteps: int | None = None,
            run_dir: Optional[str | pathlib.Path] = None,
            log_interval: int = 1,
            save_interval: Optional[int] = None,
            save_optimizer: bool = True,
            extra_run_metadata: dict[str, Any] | None = None,
            episode_return_ema_alpha: float | None = None,
            best_rotation_n: int = 1,
            wandb_project: str | None = None,
            wandb_entity: str | None = None,
            wandb_run_name: str | None = None,
            wandb_group: str | None = None,
            wandb_tags: list[str] | None = None,
            wandb_mode: str | None = None,
            wandb_kwargs: dict[str, Any] | None = None,
            logging_ignore_keys_for_persistence: list[str] | None = None,
            logging_console_keys: ConsoleSelection = "default",
            logging_console_level: LoggingLevel = "minimal",
            logging_persistence_level: LoggingLevel = "full",
            logging_buffer_size: int = 1,
            compress_metrics_log_on_exit: bool = False,
            enable_command_prompt: bool = True,
            post_iteration_hooks: Collection[LearnIterationHook] | None = None,
    ) -> Self:
        if (max_total_timesteps is None) == (additional_timesteps is None):
            raise ValueError("Supply exactly one of max_total_timesteps and additional_timesteps")
        if additional_timesteps is None:
            max_total_timesteps = validate_timestep_budget(max_total_timesteps)
        else:
            max_total_timesteps = self.n_total_timesteps + validate_timestep_budget(additional_timesteps)
        if type(best_rotation_n) is not int or best_rotation_n < 1:
            raise ValueError("best_rotation_n must be a positive integer")
        if log_interval is not None and log_interval < 1:
            raise ValueError(f"log_interval must be >= 1 or None, got {log_interval}")
        if logging_buffer_size < 1:
            raise ValueError(f"logging_buffer_size must be >= 1, got {logging_buffer_size}")

        if save_interval is not None and (type(save_interval) is not int or save_interval < 1):
            raise ValueError("save_interval must be a positive integer or None")
        if episode_return_ema_alpha is not None:
            if not 0 <= episode_return_ema_alpha <= 1:
                raise ValueError("episode_return_ema_alpha must be between 0 and 1")
            self._episode_return_ema.alpha = self._episode_success_rate_ema.alpha = episode_return_ema_alpha

        if run_dir is not None:
            run_dir = pathlib.Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_run_metadata(run_dir, extra_run_metadata)

        best_models_dir: pathlib.Path | None = None
        learn_started_at = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if run_dir is not None:
            best_models_dir = run_dir / "models" / "best" / f"{learn_started_at}_{uuid4().hex}"

        wandb_config: dict[str, Any] | None = None
        if wandb_project is not None:
            wandb_config = {
                "hyper_parameters": self.get_hyper_parameters(),
                "machine_specs": collect_machine_specs(),
            }
            if extra_run_metadata:
                wandb_config["extra_run_metadata"] = json.loads(json.dumps(extra_run_metadata, default=str))
            if run_dir is not None:
                wandb_config["run_dir"] = str(run_dir)

            if wandb_run_name is None and run_dir is not None:
                wandb_run_name = run_dir.name

        metric_logger = MetricsLogger(
            log_dir=run_dir,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_run_name=wandb_run_name,
            wandb_group=wandb_group,
            wandb_tags=wandb_tags,
            wandb_config=wandb_config,
            wandb_mode=wandb_mode,
            wandb_kwargs=wandb_kwargs,
            wandb_step_key="timesteps",
            ignore_keys_for_persistence=logging_ignore_keys_for_persistence,
            console_keys=logging_console_keys,
            console_level=logging_console_level,
            persistence_level=logging_persistence_level,
            buffer_size=logging_buffer_size,
        )
        episode_return_ema = self._episode_return_ema
        episode_success_rate_ema = self._episode_success_rate_ema

        self._active_run_dir = run_dir
        self._active_max_total_timesteps = max_total_timesteps
        self._active_learn_started_monotonic = time.monotonic()
        self._active_learn_started_timesteps = self.n_total_timesteps
        self._last_return_ema = episode_return_ema.get()
        self._final_return_ema = None
        should_compress_metrics_log = False

        try:
            if enable_command_prompt:
                self._maybe_start_command_prompt()

            retry_best = self._can_retry_best_save()
            self._pending_best_save = None
            best_save_counter = 0
            if retry_best:
                best_save_counter = self._maybe_save_best_ema_model(best_models_dir, best_rotation_n, save_optimizer, 0)
            hooks = tuple(post_iteration_hooks or ())
            pending = self._pending_iteration_hooks
            self._pending_iteration_hooks = None
            if pending is not None and pending.timesteps == self.n_total_timesteps:
                remaining = tuple(hook for hook in pending.hooks if hook in hooks)
                self._run_iteration_hooks(remaining, pending.metrics, pending.rollout_steps)
            self._before_learn_loop()

            while self.n_total_timesteps < max_total_timesteps:
                self._poll_and_execute_commands()

                iter_timer = PerformanceTimer().start()
                metrics, rollout_steps = self.perform_iteration(
                    episode_return_ema,
                    episode_success_rate_ema,
                    update_ema=self.n_total_iterations >= MIN_ITERATIONS_FOR_EMA
                )
                iter_duration = iter_timer.stop().get_duration()

                try:
                    self._run_iteration_hooks(hooks, metrics, rollout_steps)
                finally:
                    # A hook can fail after the optimizer has already completed this iteration.
                    current_return_ema = episode_return_ema.get()
                    current_success_rate_ema = episode_success_rate_ema.get()
                    current_success_rate_ema_percent = (
                        None if current_success_rate_ema is None else 100.0 * current_success_rate_ema
                    )
                    self._last_return_ema = current_return_ema
                    best_save_counter = self._maybe_save_best_ema_model(
                        best_models_dir, best_rotation_n, save_optimizer, best_save_counter,
                    )

                    if log_interval is not None and self.n_total_iterations % log_interval == 0:
                        metric_logger.log({
                            'learn_start': learn_started_at,
                            'iteration': self.n_total_iterations,
                            'timesteps': self.n_total_timesteps,
                            'learning_rate': self.learning_rate,
                            **_apply_log_metric_reductions(metrics),
                            'ep_rew_ema': current_return_ema,
                            'ep_success_rate_ema': current_success_rate_ema_percent,
                            'best_ep_rew_ema': self._best_return_ema,
                            'fps': rate(rollout_steps, iter_duration),
                        })

                    if save_interval is not None and run_dir is not None and self.n_total_iterations % save_interval == 0:
                        save_path = run_dir / f"models/model_{self.n_total_timesteps}_steps.pt"
                        self.save(
                            save_path,
                            optimizer_state_dict=self._get_optimizer_state_dict() if save_optimizer else None,
                            return_ema=current_return_ema
                        )
                        logger.log("SAVE", f"Saved model to {save_path.as_posix()}")

            if run_dir is not None:
                save_path = run_dir / f"models/model_{self.n_total_timesteps}_steps_final.pt"
                self.save(
                    save_path,
                    optimizer_state_dict=self._get_optimizer_state_dict() if save_optimizer else None,
                    return_ema=episode_return_ema.get()
                )
                logger.log("SAVE", f"Saved final model to {save_path.as_posix()}")

            should_compress_metrics_log = compress_metrics_log_on_exit

        finally:
            self._final_return_ema = self._last_return_ema
            try:
                try:
                    metric_logger.close()
                finally:
                    if should_compress_metrics_log:
                        metric_logger.compress_persisted_log()
            finally:
                self._active_run_dir = None
                self._active_max_total_timesteps = None
                self._active_learn_started_monotonic = None
                self._active_learn_started_timesteps = None
                self._last_return_ema = None

        return self

    def set_learning_rate(self, lr: LearningRate) -> None:
        self._apply_learning_rate(lr)
        self.learning_rate = lr

    def _write_run_metadata(self, run_dir: Path, extra_run_metadata: dict[str, Any] | None) -> None:
        metadata: dict[str, Any] = {
            "algorithm": type(self).__name__,
            "hyper_parameters": self.get_hyper_parameters(),
            "policy_hyper_parameters": self.policy.get_hyper_parameters(),
            "model_scale": getattr(self.policy, "model_scale", None),
            "policy_repr": str(self.policy),
            "env_repr": str(self.env),
        }
        if extra_run_metadata:
            metadata.update(extra_run_metadata)
        metadata.update(self.training_metadata)
        get_settings = getattr(self.env.unwrapped, "get_settings", None)
        if callable(get_settings):
            metadata["environment_settings"] = serialize_value(get_settings())
        metadata["source_revision"] = collect_source_revision()
        metadata["machine_specs"] = collect_machine_specs()
        metadata["iterations"] = self.n_total_iterations
        metadata["updates"] = self.n_total_updates
        metadata["timesteps"] = self.n_total_timesteps
        metadata_json: dict[str, Any] = json.loads(json.dumps(serialize_value(metadata), default=str))

        existing_metadata_files = list(run_dir.glob("run_metadata*.json"))
        if existing_metadata_files:
            def _sort_key(path: Path) -> tuple[int, float, str]:
                step = _parse_step_from_metadata_filename(path)
                step_key = step if step is not None else -1
                return step_key, path.stat().st_mtime, path.name

            latest_path = max(existing_metadata_files, key=_sort_key)
            latest_metadata = _read_metadata_json(latest_path)
            if _normalize_run_metadata_for_comparison(latest_metadata) == _normalize_run_metadata_for_comparison(metadata_json):
                return

        step = int(self.n_total_timesteps)
        base_path = run_dir / f"run_metadata_{step}.json"
        if base_path.exists():
            existing = _read_metadata_json(base_path)
            if _normalize_run_metadata_for_comparison(existing) == _normalize_run_metadata_for_comparison(metadata_json):
                return

            suffix_idx = 1
            while (run_dir / f"run_metadata_{step}_{suffix_idx}.json").exists():
                suffix_idx += 1
            metadata_path = run_dir / f"run_metadata_{step}_{suffix_idx}.json"
        else:
            metadata_path = base_path

        metadata_path.write_text(json.dumps(metadata_json, indent=2, sort_keys=True), encoding="utf-8")

    def _current_best_candidate(self) -> bytes | None:
        try:
            counters = self.n_total_iterations, self.n_total_updates, self.n_total_timesteps
            return best_candidate_digest(
                self.policy.state_dict(), capture_env_state(self.env), counters, self._episode_return_ema.get(),
            )
        except Exception:
            # Failed storage or an unavailable device may also prevent identifying the candidate.
            # Preserve the original save error and leave an unverifiable candidate ineligible.
            return None

    def _can_retry_best_save(self) -> bool:
        return self._pending_best_save is not None and self._pending_best_save == self._current_best_candidate()

    def _maybe_save_best_ema_model(
            self,
            best_models_dir: pathlib.Path | None,
            best_rotation_n: int,
            save_optimizer: bool,
            best_save_counter: int,
    ) -> int:
        score = self._episode_return_ema.get()
        if score is None or self.n_total_iterations < MIN_ITERATIONS_FOR_BEST:
            return best_save_counter
        if self._best_return_ema is not None and score <= self._best_return_ema:
            return best_save_counter

        previous_best = self._best_return_ema
        self._best_return_ema = score
        if best_models_dir is None:
            return best_save_counter

        if best_rotation_n == 1:
            best_save_path = best_models_dir / "model_best.pt"
        else:
            best_save_idx = best_save_counter % best_rotation_n
            best_save_path = best_models_dir / f"model_best_{best_save_idx}.pt"

        # The candidate belongs in its own checkpoint. On failure it must remain eligible for retry.
        try:
            self.save(
                best_save_path,
                optimizer_state_dict=self._get_optimizer_state_dict() if save_optimizer else None,
                return_ema=score,
            )
        except BaseException:
            self._best_return_ema = previous_best
            self._pending_best_save = self._current_best_candidate()
            raise
        logger.log(
            "SAVE",
            f"Saved best-EMA model to {best_save_path.as_posix()} (ep_rew_ema={score:.4f})",
        )
        return best_save_counter + 1

    def save(
            self,
            path: str | Path,
            optimizer_state_dict: Optional[dict[str, Any]] = None,
            return_ema: Optional[float] = None,
    ) -> None:
        if return_ema is None:
            return_ema = self._episode_return_ema.get()
        if return_ema is not None and not math.isfinite(return_ema):
            raise ValueError("Checkpoint return EMA must be finite")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            'n_total_iterations': self.n_total_iterations,
            'n_total_updates': self.n_total_updates,
            'n_total_timesteps': self.n_total_timesteps,
            'return_ema': return_ema,
            'best_return_ema': self._best_return_ema,
        }
        save_dict = {
            **metadata,
            'policy_state_dict': self.policy.state_dict(),
            'policy_loss_weights': self.policy.get_loss_weights(),
            'algorithm_state_dict': self._get_algorithm_state_dict(),
            'policy_hyper_parameters': self.policy.get_hyper_parameters(),
            'hyper_parameters': self.get_hyper_parameters(),
            'model_scale': getattr(self.policy, 'model_scale', None),
            'env_state': capture_env_state(self.env),
            'episode_return_ema': self._episode_return_ema.state_dict(),
            'episode_success_rate_ema': self._episode_success_rate_ema.state_dict(),
            'rollout_warmup_done': self._rollout_warmup_done,
            'pending_best_digest': None,
        }
        if self._pending_best_save is not None:
            self._attach_pending_best_candidate(save_dict)

        if optimizer_state_dict is not None:
            save_dict['optimizer_state_dict'] = optimizer_state_dict

        write_checkpoint(path, save_dict, metadata)

    def _attach_pending_best_candidate(self, payload: dict[str, Any]) -> None:
        try:
            policy_state = payload["policy_state_dict"]
            extra_state = snapshot_best_candidate_extra_state(policy_state)
            counters = payload["n_total_iterations"], payload["n_total_updates"], payload["n_total_timesteps"]
            digest = best_candidate_digest(
                policy_state, payload["env_state"], counters, payload["episode_return_ema"]["ema"], extra_state=extra_state,
            )
        except Exception:
            return
        if digest == self._pending_best_save:
            payload["pending_best_digest"] = digest
            if extra_state is not None:
                payload["pending_best_extra_state"] = extra_state

    def load(
            self,
            path: str | Path,
            *,
            map_location: Any | None = "cpu",
            recover_best_return_ema: bool = True,
            strict_load_state_dict: bool = True,
            restore_optimizer_state: bool = True,
            restore_env_state: bool = True,
    ) -> None:
        checkpoint, verified_best_archive = load_training_checkpoint(path, map_location=map_location)
        algorithm_state_dict = checkpoint["algorithm_state_dict"]
        self._validate_algorithm_state_dict(algorithm_state_dict)
        optimizer_state_dict = (
            checkpoint.get("optimizer_state_dict")
            if restore_optimizer_state
            else None
        )
        episode_return_ema = HybridEMA.from_state_dict(checkpoint["episode_return_ema"])
        episode_success_rate_ema = HybridEMA.from_state_dict(checkpoint["episode_success_rate_ema"])
        counters = tuple(checkpoint[key] for key in ("n_total_iterations", "n_total_updates", "n_total_timesteps"))
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("Checkpoint counters must be nonnegative integers")
        warmup_done = checkpoint["rollout_warmup_done"]
        if type(warmup_done) is not bool:
            raise ValueError("Checkpoint rollout_warmup_done must be a boolean")
        pending_best_digest = checkpoint["pending_best_digest"]
        if pending_best_digest is not None and not isinstance(pending_best_digest, bytes):
            raise ValueError("Checkpoint pending_best_digest must contain bytes or None")
        pending_best_extra_state = checkpoint.get("pending_best_extra_state")
        if pending_best_extra_state is not None and not isinstance(pending_best_extra_state, bytes):
            raise ValueError("Checkpoint pending_best_extra_state must contain bytes or None")
        best_return = checkpoint["best_return_ema"] if recover_best_return_ema else self._best_return_ema
        if best_return is not None and not math.isfinite(best_return):
            raise ValueError("Checkpoint best return must be finite")
        env_updates = prepare_env_state(self.env, checkpoint["env_state"]) if restore_env_state else []
        target_policy = self.policy.state_dict()
        target_policy_keys = target_policy.keys()
        policy_state_dict = align_torch_compile_state_dict_keys(
            checkpoint["policy_state_dict"],
            target_keys=target_policy_keys,
        )
        matching_policy = policy_state_dict.keys() == target_policy_keys
        restore_optimizer = optimizer_state_dict is not None and (strict_load_state_dict or matching_policy)
        if restore_optimizer:
            saved_learning_rate = checkpoint["hyper_parameters"]["learning_rate"]
            self._validate_optimizer_checkpoint(checkpoint)
        validate_policy_state(
            policy_state_dict, target_policy, strict=strict_load_state_dict,
            parameter_names=dict(self.policy.named_parameters(remove_duplicate=False)).keys(),
        )
        loss_weights = (
            prepare_checkpoint_loss_weights(checkpoint["policy_loss_weights"], self.policy.get_loss_weights())
            if matching_policy else {}
        )
        restore_policy = snapshot_policy_for_extra_state(self.policy, policy_state_dict, target_policy)
        previous_optimizer_state = (
            self._get_optimizer_state_dict() if restore_policy is not None and restore_optimizer else None
        )
        if restore_optimizer:
            self._restore_checkpoint_schedulers(optimizer_state_dict)
        try:
            missing_keys, unexpected_keys = self.policy.load_state_dict(policy_state_dict, strict=strict_load_state_dict)
        except BaseException:
            if restore_policy is not None:
                restore_policy()
            if previous_optimizer_state is not None:
                self._restore_checkpoint_schedulers(previous_optimizer_state)
            raise
        if missing_keys or unexpected_keys:
            logger.warning(f'{missing_keys = },  {unexpected_keys = }')
        elif loss_weights:
            self.policy.update_loss_weights(**loss_weights)

        self._apply_algorithm_state_dict(algorithm_state_dict)

        if restore_optimizer and not (missing_keys or unexpected_keys):
            # Warmup schedules recompute optimizer rates from the base rate on the next update.
            self.learning_rate = saved_learning_rate
            self._apply_optimizer_state_dict(optimizer_state_dict, missing_keys=[], unexpected_keys=[])
        else:
            if optimizer_state_dict is not None:
                logger.warning("Skipping optimizer state restore because policy keys differ")
            self._apply_learning_rate(self.learning_rate)

        self.n_total_iterations, self.n_total_updates, self.n_total_timesteps = counters
        self._rollout_warmup_done = warmup_done
        self._best_return_ema = best_return
        self._episode_return_ema = episode_return_ema
        self._episode_success_rate_ema = episode_success_rate_ema
        retry_best = recover_best_return_ema and restore_env_state and not (missing_keys or unexpected_keys)
        self._final_return_ema = episode_return_ema.get()
        install_env_state(env_updates)
        self._pending_best_save = pending_best_digest if retry_best else None
        if self._pending_best_save is not None:
            try:
                self._pending_best_save = restore_best_candidate_digest(
                    self._pending_best_save, pending_best_extra_state, self.policy.state_dict(),
                    capture_env_state(self.env), counters, episode_return_ema.get(), verified_archive=verified_best_archive,
                )
            except Exception:
                self._pending_best_save = None
        self._pending_iteration_hooks = None

    def execute_command(self, cmd: str) -> bool:
        """Run a read-only inspection command and report whether it succeeded."""
        handlers = {
            "show_hps": self._cmd_show_hyper_parameters,
            "show_run_path": self._cmd_show_run_path,
            "show_id": self._cmd_show_run_path,
            "pid": self._cmd_show_pid,
            "show_pid": self._cmd_show_pid,
            "show_reward_weights": self._cmd_show_reward_weights,
            "show_rw": self._cmd_show_reward_weights,
            "show_eta": self._cmd_show_eta,
            "eta": self._cmd_show_eta,
            "record_status": self._cmd_record_status,
            "record_progress": self._cmd_record_status,
            "recording_status": self._cmd_record_status,
            "recording_progress": self._cmd_record_status,
        }
        handler = handlers.get(cmd.strip().lower())
        if handler is None:
            logger.error(f'Unknown inspection command "{cmd}"')
            return False
        try:
            handler()
        except Exception:
            logger.exception('Executing inspection command failed')
            return False
        return True

    def _maybe_start_command_prompt(self) -> None:
        if not self._command_prompt_started:
            self._command_prompt_started = _start_command_prompt(self._command_queue)

    def _poll_and_execute_commands(self) -> None:
        while True:
            try:
                raw = self._command_queue.get_nowait()
            except queue.Empty:
                return
            self._execute_command_line(raw)

    def _execute_command_line(self, raw: str) -> None:
        for command in raw.split(";"):
            if command.strip():
                self.execute_command(command)

    def _cmd_show_hyper_parameters(self) -> None:
        logger.info(self.get_hyper_parameters())

    def _cmd_show_run_path(self) -> None:
        if self._active_run_dir is None:
            logger.info({"run_path": None, "run_id": None})
            return

        run_dir = self._active_run_dir.resolve()
        logger.info({"run_path": run_dir.as_posix(), "run_id": run_dir.name})

    def _cmd_show_pid(self) -> None:
        logger.info({"pid": os.getpid()})

    def _cmd_record_status(self) -> None:
        get_status_fn = getattr(self.env.unwrapped, "get_video_recording_status", None)
        if callable(get_status_fn):
            logger.info({"recording": get_status_fn(), "recording_mode": "live_env"})
            return

        logger.info(
            {
                "recording": {
                    "active": False,
                    "message": "Recording is not configured for this run.",
                },
                "recording_mode": "unavailable",
            }
        )

    def _cmd_show_reward_weights(self) -> None:
        logger.info({"reward_weights": self.env.unwrapped.get_reward_weights()})

    def _cmd_show_eta(self) -> None:
        eta = self._build_run_eta_estimate()
        if eta is None:
            logger.info({
                "estimated_finish_at": None,
                "remaining_duration": None,
                "remaining_timesteps": None,
                "observed_throughput_tps": None,
                "reason": "Need at least one completed iteration in the current learn() call.",
            })
            return

        logger.info(eta)

    def _build_run_eta_estimate(
            self,
            current_time: datetime | None = None,
            elapsed_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        max_total_timesteps = self._active_max_total_timesteps
        if max_total_timesteps is None:
            return None

        learn_started_timesteps = self._active_learn_started_timesteps
        learn_started_monotonic = self._active_learn_started_monotonic
        if learn_started_timesteps is None or learn_started_monotonic is None:
            return None

        completed_timesteps = self.n_total_timesteps - learn_started_timesteps
        if completed_timesteps <= 0:
            return None

        elapsed_seconds = time.monotonic() - learn_started_monotonic if elapsed_seconds is None else elapsed_seconds
        if elapsed_seconds <= 0.0:
            return None

        current_time = datetime.now().astimezone() if current_time is None else current_time
        remaining_timesteps = max(0, max_total_timesteps - self.n_total_timesteps)
        observed_throughput_tps = completed_timesteps / elapsed_seconds
        remaining_duration_seconds = remaining_timesteps / observed_throughput_tps
        estimated_finish_at = current_time + timedelta(seconds=remaining_duration_seconds)
        return {
            "estimated_finish_at": estimated_finish_at.isoformat(timespec="seconds"),
            "remaining_duration": _format_duration_seconds(remaining_duration_seconds),
            "remaining_timesteps": remaining_timesteps,
            "observed_throughput_tps": round(observed_throughput_tps, 3),
        }


def _parse_step_from_metadata_filename(path: Path) -> int | None:
    if path.name == "run_metadata.json":
        return None
    if not (path.name.startswith("run_metadata_") and path.name.endswith(".json")):
        return None

    suffix = path.name.removeprefix("run_metadata_").removesuffix(".json")
    step_str = suffix.split("_", 1)[0]
    if not step_str.isdigit():
        return None
    return int(step_str)


def _read_metadata_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


_RUN_METADATA_VOLATILE_KEYS: frozenset[str] = frozenset({
    "iterations",
    "updates",
    "timesteps",
    "load_path",
    "script",
})


def _normalize_run_metadata_for_comparison(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    return {k: v for k, v in metadata.items() if k not in _RUN_METADATA_VOLATILE_KEYS}


def _format_duration_seconds(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return "".join(parts)


def _start_command_prompt(cmd_q: queue.Queue[str]) -> bool:
    use_prompt_toolkit = True
    try:
        from prompt_toolkit import prompt as pt_prompt
        from prompt_toolkit.patch_stdout import patch_stdout as pt_patch_stdout
    except ImportError:
        logger.warning(
            "Command prompt disabled (prompt_toolkit not installed). "
            "Install it with `pip install prompt_toolkit`."
        )
        use_prompt_toolkit = False

    if not sys.stdin or not sys.stdin.isatty():
        use_prompt_toolkit = False
        logger.warning("Command prompt may not work (stdin is not a TTY).")

    def _run() -> None:
        while True:
            try:
                if use_prompt_toolkit:
                    with pt_patch_stdout():
                        cmd = pt_prompt("> ")
                else:
                    cmd = input("> ")
            except (EOFError, KeyboardInterrupt):
                logger.debug("EOF: Exiting command prompt")
                return
            cmd_q.put(cmd.strip())

    threading.Thread(target=_run, daemon=True).start()
    return True
