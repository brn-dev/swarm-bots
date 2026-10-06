import csv
import gzip
import json
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from typing import Any
from unittest import mock

import torch
from loguru import logger

from swarmbots.learn.algos.base_algorithm import BaseAlgorithm
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage
from swarmbots.learn.logging_levels import LOGGING_LEVELS
from swarmbots.learn.metrics_logger import MetricsLogger, mean_std, rate, summed
from swarmbots.learn.summary_statistics import compute_summary_statistics


@contextmanager
def capture_log_messages() -> Iterator[list[tuple[str, str, str]]]:
    messages: list[tuple[str, str, str]] = []
    sink_id = logger.add(
        lambda message: messages.append(
            (message.record["level"].name, message.record["function"], message.record["message"])
        ),
        format="{message}",
    )
    try:
        yield messages
    finally:
        logger.remove(sink_id)


@contextmanager
def enable_mock_wandb() -> Iterator[None]:
    with mock.patch.multiple(
        "swarmbots.learn.metrics_logger", wandb_available=True, wandb=mock.Mock(), create=True
    ):
        yield


class _DummyPolicy(BasePolicy):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

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
        _ = local_obs
        _ = global_obs
        _ = hidden_local_vars
        _ = hidden_global_vars
        _ = agent_mask
        _ = previous_actions
        _ = deterministic
        return torch.zeros((1, 1, 1))

    def update_loss_weights(self, **weights: float) -> None:
        _ = weights

    def requires_previous_actions(self) -> bool:
        return False


class _DummyEnv:
    unwrapped = None

    def __str__(self) -> str:
        return "_DummyEnv()"


class _DummyAlgorithm(BaseAlgorithm):
    def get_hyper_parameters(self) -> dict[str, float]:
        return {"dummy": 1.0}

    def _get_optimizer_state_dict(self) -> dict[str, float]:
        return {}

    def _apply_optimizer_state_dict(
            self,
            state_dict: dict[str, float],
            missing_keys: list[str],
            unexpected_keys: list[str],
    ) -> None:
        _ = state_dict
        _ = missing_keys
        _ = unexpected_keys

    def _apply_learning_rate(self, lr: float) -> None:
        _ = lr

    def perform_iteration(
            self,
            episode_return_ema: ExponentialMovingAverage,
            episode_success_rate_ema: ExponentialMovingAverage,
            update_ema: bool,
    ) -> tuple[dict[str, Any], int]:
        _ = episode_return_ema
        _ = episode_success_rate_ema
        _ = update_ema
        self.n_total_iterations += 1
        self.n_total_updates += 1
        self.n_total_timesteps += 1
        return {"metric": 1.0, "total_updates": self.n_total_updates}, 1


class _ReturnEmaAlgorithm(_DummyAlgorithm):
    def perform_iteration(
            self,
            episode_return_ema: ExponentialMovingAverage,
            episode_success_rate_ema: ExponentialMovingAverage,
            update_ema: bool,
    ) -> tuple[dict[str, Any], int]:
        _ = episode_success_rate_ema
        if update_ema:
            for _ in range(100):
                episode_return_ema.update(15.0)
        self.n_total_iterations += 1
        self.n_total_updates += 1
        self.n_total_timesteps += 1
        return {"metric": 1.0, "total_updates": self.n_total_updates}, 1


class _SuccessRateEmaAlgorithm(_DummyAlgorithm):
    def perform_iteration(
            self,
            episode_return_ema: ExponentialMovingAverage,
            episode_success_rate_ema: ExponentialMovingAverage,
            update_ema: bool,
    ) -> tuple[dict[str, Any], int]:
        if update_ema:
            for success in (1.0, 0.0) * 50:
                episode_return_ema.update(0.0)
                episode_success_rate_ema.update(success)
        self.n_total_iterations += 1
        self.n_total_updates += 1
        self.n_total_timesteps += 1
        return {"metric": 1.0, "total_updates": self.n_total_updates}, 1


class _DiagnosticAlgorithm(_DummyAlgorithm):
    def perform_iteration(
        self,
        episode_return_ema: ExponentialMovingAverage,
        episode_success_rate_ema: ExponentialMovingAverage,
        update_ema: bool,
    ) -> tuple[dict[str, Any], int]:
        metrics, steps = super().perform_iteration(episode_return_ema, episode_success_rate_ema, update_ema)
        metrics.update(
            critic_loss=compute_summary_statistics([0.08, 0.09]),
            actor_loss=compute_summary_statistics([-0.12, -0.11]),
            critic_nop_loss_scaled=compute_summary_statistics([0.06, 0.062]),
            critic_nop_scalar_loss=compute_summary_statistics([0.13, 0.14]),
            sampling_time=0.01,
            replay_act=compute_summary_statistics([0.1, 0.2]),
        )
        return metrics, steps


class MetricsLoggerTests(unittest.TestCase):
    def test_default_training_console_is_compact_and_csv_keeps_all_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, capture_log_messages() as messages:
            algo = _DiagnosticAlgorithm(policy=_DummyPolicy(), env=_DummyEnv(), learning_rate=3e-4)
            algo.learn(
                max_total_timesteps=1,
                run_dir=tmp_dir,
                save_optimizer=False,
                enable_command_prompt=False,
            )
            output = next(message for _level, function, message in messages if function == "_log_to_console")
            self.assertIn("it: 1 | steps: 1 | upd: 1", output)
            self.assertIn("ret: n/a | succ%: n/a", output)
            self.assertIn("q_loss: 0.085 | pi_loss: -0.115 | c_nop: 0.061", output)
            self.assertIn("lr: 3.00e-04", output)
            self.assertNotIn("±", output)
            for diagnostic in ("learn_start", "timestamp", "critic_nop_scalar_loss", "sampling_time", "replay_act"):
                self.assertNotIn(diagnostic, output)
            self.assertFalse(any(level == "WARNING" for level, _function, _message in messages))
            with (Path(tmp_dir) / "log.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle, delimiter=";"))
            self.assertAlmostEqual(float(row["critic_loss__std"]), 0.005)
            self.assertAlmostEqual(float(row["critic_nop_scalar_loss__mean"]), 0.135)
            self.assertIn("sampling_time", row)
            self.assertIn("timestamp", row)
            self.assertNotIn("q_loss", row)

    def test_custom_console_keys_preserve_order_aliases_and_formats(self) -> None:
        with capture_log_messages() as messages:
            algo = _DummyAlgorithm(policy=_DummyPolicy(), env=_DummyEnv(), learning_rate=1e-3)
            algo.learn(
                max_total_timesteps=1,
                logging_console_keys=[("metric", ".1f", "m"), "iteration", ("learning_rate", ".2e", "lr")],
                save_optimizer=False,
                enable_command_prompt=False,
            )
            output = next(message for _level, function, message in messages if function == "_log_to_console")
            displayed = [tuple(value.strip() for value in part.split(":", maxsplit=1)) for part in output.split("|")]
            self.assertEqual(displayed, [("m", "1.0"), ("iteration", "1"), ("lr", "1.00e-03")])

    def test_explicit_none_restores_all_training_console_metrics(self) -> None:
        with capture_log_messages() as messages:
            algo = _DummyAlgorithm(policy=_DummyPolicy(), env=_DummyEnv(), learning_rate=1e-3)
            algo.learn(
                max_total_timesteps=1,
                logging_console_keys=None,
                save_optimizer=False,
                enable_command_prompt=False,
            )
            output = next(message for _level, function, message in messages if function == "_log_to_console")
            self.assertIn("learn_start:", output)
            self.assertIn("iteration:", output)
            self.assertIn("metric:", output)
            self.assertIn("timestamp:", output)

    def test_empty_console_selection_is_silent_and_keeps_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, capture_log_messages() as messages, enable_mock_wandb():
            wandb_run = mock.Mock()
            metrics_logger = MetricsLogger(log_dir=tmp_dir, wandb_run=wandb_run, console_keys=[])
            metrics_logger.log({"timesteps": 1, "diagnostic": 3.75})
            metrics_logger.close()
            self.assertEqual(messages, [])
            with (Path(tmp_dir) / "log.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle, delimiter=";"))
            self.assertEqual(row["diagnostic"], "3.75")
            self.assertEqual(wandb_run.log.call_args.args[0]["diagnostic"], 3.75)

    def test_console_and_persistence_levels_can_be_selected_independently(self) -> None:
        for console_level in LOGGING_LEVELS:
            for persistence_level in LOGGING_LEVELS:
                with (
                    self.subTest(console_level=console_level, persistence_level=persistence_level),
                    tempfile.TemporaryDirectory() as tmp_dir,
                    capture_log_messages() as messages,
                    enable_mock_wandb(),
                ):
                    wandb_run = mock.Mock()
                    metrics_logger = MetricsLogger(
                        log_dir=tmp_dir,
                        wandb_run=wandb_run,
                        console_level=console_level,
                        persistence_level=persistence_level,
                    )
                    metrics_logger.log({
                        "iteration": 2,
                        "timesteps": 100,
                        "total_updates": 3,
                        "learn_start": "2026-10-06_12-00-00",
                        "timestamp": "2026-10-06T12:00:00+00:00",
                        "ep_rew": compute_summary_statistics([10.0, 20.0]),
                        "ep_rew_ema": 12.0,
                        "best_ep_rew_ema": 14.0,
                        "ep_success_rate": 50.0,
                        "ep_success_rate_ema": 40.0,
                        "scenario/easy/ep_rew": compute_summary_statistics([2.0, 4.0]),
                        "scenario/easy/ep_success_rate": 75.0,
                        "scenario/easy/episodes": 2,
                        "critic_loss": compute_summary_statistics([1.0, 2.0]),
                        "clip_frac": compute_summary_statistics([0.1, 0.2]),
                        "expl_var": mean_std(0.75),
                        "q_pi": compute_summary_statistics([2.0, 4.0]),
                        "target_q": compute_summary_statistics([1.5, 2.5]),
                        "entropy": compute_summary_statistics([-0.4, -0.2]),
                        "target_entropy": compute_summary_statistics([-0.25]),
                        "actor_updated": compute_summary_statistics([0.0, 1.0]),
                        "replay_size": 10_000,
                        "random_actions": True,
                        "training_skipped": True,
                        "learning_rate": 1e-3,
                        "fps": 1200.0,
                        "detailed_diagnostic": 42.0,
                    })
                    metrics_logger.close()
                    output = next(message for _level, function, message in messages if function == "_log_to_console")
                    if console_level == "full":
                        self.assertIn("detailed_diagnostic:", output)
                        self.assertIn("critic_loss:", output)
                    else:
                        self.assertIn("steps: 100", output)
                        self.assertIn("ret: 12.00", output)
                        self.assertIn("succ%: 40.0", output)
                        self.assertNotIn("detailed_diagnostic", output)
                        if console_level == "minimal":
                            self.assertIn("q_loss: 1.500", output)
                            for field in (
                                "clip: 0.150", "ev: 0.750", "q_pi: 3.000", "q_tgt: 2.000",
                                "ent: -0.300", "ent_tgt: -0.250", "pi_upd: 0.50",
                                "replay: 10,000", "rnd: 1", "skip: 1",
                            ):
                                self.assertIn(field, output)
                        else:
                            self.assertIn("ret_mean: 15.00", output)
                            for diagnostic in (
                                "q_loss", "critic_loss", "lr:", "fps:", "clip:", "ev:",
                                "q_pi:", "q_tgt:", "ent:", "ent_tgt:", "pi_upd:",
                                "replay:", "rnd:", "skip:",
                            ):
                                self.assertNotIn(diagnostic, output)
                    with (Path(tmp_dir) / "log.csv").open(newline="", encoding="utf-8") as handle:
                        row = next(csv.DictReader(handle, delimiter=";"))
                    saved_wandb = wandb_run.log.call_args.args[0]
                    for saved in (row, saved_wandb):
                        for key in (
                            "timestamp", "iteration", "timesteps", "total_updates", "learn_start",
                            "ep_rew__mean", "ep_rew__std", "ep_rew__n", "ep_rew_ema",
                            "best_ep_rew_ema", "ep_success_rate", "ep_success_rate_ema",
                            "scenario/easy/ep_rew__mean", "scenario/easy/ep_success_rate",
                            "scenario/easy/episodes",
                        ):
                            self.assertIn(key, saved)
                        self.assertEqual("detailed_diagnostic" in saved, persistence_level == "full")
                        self.assertEqual("critic_loss__mean" in saved, persistence_level != "return_success")
                        self.assertEqual("learning_rate" in saved, persistence_level != "return_success")
                        self.assertEqual("fps" in saved, persistence_level != "return_success")
                        for key in (
                            "clip_frac__mean", "expl_var__mean", "q_pi__mean", "target_q__mean",
                            "entropy__mean", "target_entropy__mean", "actor_updated__mean",
                            "replay_size", "random_actions", "training_skipped",
                        ):
                            self.assertEqual(key in saved, persistence_level != "return_success")
                        self.assertNotIn("q_loss", saved)
                    self.assertEqual(wandb_run.log.call_args.kwargs["step"], 100)
                    self.assertFalse(any(level == "WARNING" for level, _function, _message in messages))

    def test_reduced_persistence_keeps_custom_wandb_step_and_applies_explicit_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, enable_mock_wandb():
            wandb_run = mock.Mock()
            metrics_logger = MetricsLogger(
                log_dir=tmp_dir,
                wandb_run=wandb_run,
                wandb_step_key="progress/updates",
                persistence_level="return_success",
                ignore_keys_for_persistence=["ep_rew_ema"],
            )
            metrics_logger.log({"progress/updates": 7, "ep_rew_ema": 12.0, "ep_success_rate_ema": 50.0, "loss": 2})
            metrics_logger.close()
            with (Path(tmp_dir) / "log.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle, delimiter=";"))
            self.assertEqual(row["progress/updates"], "7")
            self.assertNotIn("ep_rew_ema", row)
            self.assertNotIn("loss", row)
            self.assertIn("ep_success_rate_ema", row)
            self.assertEqual(wandb_run.log.call_args.kwargs["step"], 7)

    def test_minimal_console_formats_scalar_losses_without_errors(self) -> None:
        with capture_log_messages() as messages:
            metrics_logger = MetricsLogger()
            metrics_logger.log({"critic_loss": 1.25, "actor_loss": -0.5})
            metrics_logger.close()
            output = next(message for _level, function, message in messages if function == "_log_to_console")
            self.assertIn("q_loss: 1.250 | pi_loss: -0.500", output)
            self.assertFalse(any(level == "ERROR" for level, _function, _message in messages))

    def test_levels_preserve_buffered_return_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, capture_log_messages() as messages:
            metrics_logger = MetricsLogger(
                log_dir=tmp_dir, buffer_size=2, console_level="minimal", persistence_level="return_success"
            )
            for step, value in enumerate((2.0, 4.0), start=1):
                metrics_logger.log({
                    "iteration": step,
                    "timesteps": step * 10,
                    "ep_rew": compute_summary_statistics([value]),
                    "ep_rew_ema": value,
                    "critic_loss": compute_summary_statistics([value / 2]),
                })
            metrics_logger.close()
            output = next(message for _level, function, message in messages if function == "_log_to_console")
            self.assertIn("q_loss: 1.500", output)
            with (Path(tmp_dir) / "log.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter=";"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["iteration"], "2")
            self.assertEqual(rows[0]["timesteps"], "20")
            self.assertEqual(rows[0]["ep_rew__n"], "2")
            self.assertEqual(float(rows[0]["ep_rew__mean"]), 3.0)
            self.assertEqual(float(rows[0]["ep_rew__std"]), 1.0)
            self.assertNotIn("critic_loss__mean", rows[0])

    def test_reduced_persistence_preserves_prior_csv_rows_when_resuming(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_logger = MetricsLogger(log_dir=tmp_dir)
            metrics_logger.log({"timesteps": 1, "ep_rew_ema": 10.0, "diagnostic": 5.0})
            metrics_logger.close()
            resumed_logger = MetricsLogger(log_dir=tmp_dir, persistence_level="return_success")
            resumed_logger.log({"timesteps": 2, "ep_rew_ema": 20.0, "diagnostic": 6.0})
            resumed_logger.close()
            with (Path(tmp_dir) / "log.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter=";"))
            self.assertEqual(rows[0]["diagnostic"], "5.0")
            self.assertEqual(rows[1]["diagnostic"], "")
            self.assertEqual(rows[1]["ep_rew_ema"], "20.0")

    def test_unknown_logging_levels_fail_before_starting_the_logger(self) -> None:
        for option in ("console_level", "persistence_level"):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as tmp_dir:
                log_dir = Path(tmp_dir) / "unused"
                with self.assertRaisesRegex(ValueError, "Unknown logging level"):
                    MetricsLogger(log_dir=log_dir, **{option: "typo"})
                self.assertFalse(log_dir.exists())

    def test_learn_forwards_independent_logging_levels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, capture_log_messages() as messages:
            algo = _DiagnosticAlgorithm(policy=_DummyPolicy(), env=_DummyEnv(), learning_rate=3e-4)
            algo.learn(
                max_total_timesteps=1,
                run_dir=tmp_dir,
                logging_console_level="full",
                logging_persistence_level="return_success",
                save_optimizer=False,
                enable_command_prompt=False,
            )
            output = next(message for _level, function, message in messages if function == "_log_to_console")
            self.assertIn("critic_nop_scalar_loss", output)
            with (Path(tmp_dir) / "log.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle, delimiter=";"))
            self.assertIn("ep_rew_ema", row)
            self.assertIn("ep_success_rate_ema", row)
            self.assertIn("iteration", row)
            self.assertIn("timestamp", row)
            self.assertNotIn("critic_loss__mean", row)
            self.assertNotIn("learning_rate", row)

    def test_missing_custom_console_keys_warn_only_once(self) -> None:
        with capture_log_messages() as messages:
            metrics_logger = MetricsLogger(console_keys=["misspelled"])
            metrics_logger.log({"timesteps": 1})
            metrics_logger.log({"timesteps": 2})
            metrics_logger.close()
            warnings = [message for level, _function, message in messages if level == "WARNING"]
            self.assertEqual(len(warnings), 1)
            self.assertIn("misspelled", warnings[0])

    def test_compact_console_formats_multiple_learning_rates(self) -> None:
        for learning_rate, formatted in (
            ([1e-3, 2e-4], "[1.00e-03, 2.00e-04]"),
            ({"actor": 1e-3, "critic": 2e-4}, "{actor: 1.00e-03, critic: 2.00e-04}"),
        ):
            with self.subTest(learning_rate=learning_rate), capture_log_messages() as messages:
                algo = _DummyAlgorithm(policy=_DummyPolicy(), env=_DummyEnv(), learning_rate=learning_rate)
                algo.learn(max_total_timesteps=1, save_optimizer=False, enable_command_prompt=False)
                output = next(message for _level, function, message in messages if function == "_log_to_console")
                self.assertIn(f"lr: {formatted}", output)
                self.assertFalse(any(level == "ERROR" for level, _function, _message in messages))

    def test_full_buffer_is_processed_on_background_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir)
            metrics_logger = MetricsLogger(log_dir=log_dir, buffer_size=2)
            worker_started = Event()
            allow_worker_to_continue = Event()
            process_metrics_batch = metrics_logger._process_metrics_batch

            def wait_before_processing(metrics_batch: list[dict[str, Any]]) -> None:
                worker_started.set()
                if not allow_worker_to_continue.wait(timeout=5.0):
                    raise TimeoutError("Timed out waiting to continue metrics processing.")
                process_metrics_batch(metrics_batch)

            metrics_logger._process_metrics_batch = wait_before_processing
            try:
                metrics_logger.log({"iteration": 1})
                metrics_logger.log({"iteration": 2})

                self.assertTrue(worker_started.wait(timeout=5.0))
                self.assertFalse((log_dir / "log.csv").exists())
            finally:
                allow_worker_to_continue.set()
                metrics_logger.close()

            with (log_dir / "log.csv").open(newline="", encoding="utf-8") as log_file:
                rows = list(csv.DictReader(log_file, delimiter=";"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["iteration"], "2")

    def test_background_worker_failure_is_raised_on_close(self) -> None:
        metrics_logger = MetricsLogger(buffer_size=1)
        metrics_logger._process_metrics_batch = mock.Mock(side_effect=ValueError("write failed"))

        metrics_logger.log({"iteration": 1})

        with self.assertRaisesRegex(RuntimeError, "background worker failed") as raised:
            metrics_logger.close()
        self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_buffer_combines_explicit_reductions_and_summary_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir)
            metrics_logger = MetricsLogger(log_dir=log_dir, buffer_size=3)

            for index, (numerator, denominator) in enumerate(((10, 1), (30, 1), (20, 2)), start=1):
                stats = compute_summary_statistics([float(index), float(index + 1)], make_histogram=2)
                metrics_logger.log({
                    "last_value": index,
                    "sample": mean_std(float(index)),
                    "count": summed(index),
                    "throughput": rate(numerator, denominator),
                    "stats": stats,
                })
                if index < 3:
                    self.assertFalse((log_dir / "log.csv").exists())

            metrics_logger.close()

            with (log_dir / "log.csv").open(newline="", encoding="utf-8") as log_file:
                rows = list(csv.DictReader(log_file, delimiter=";"))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["last_value"], "3")
            self.assertAlmostEqual(float(row["sample__mean"]), 2.0)
            self.assertAlmostEqual(float(row["sample__std"]), 0.816497)
            self.assertEqual(row["sample__n"], "3")
            self.assertEqual(row["count"], "6")
            self.assertAlmostEqual(float(row["throughput"]), 15.0)
            self.assertAlmostEqual(float(row["stats__mean"]), 2.5)
            self.assertEqual(row["stats__n"], "6")
            self.assertAlmostEqual(sum(json.loads(row["stats__histogram_freqs"])), 1.0)

    def test_close_flushes_partial_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir)
            metrics_logger = MetricsLogger(log_dir=log_dir, buffer_size=3)
            metrics_logger.log({"iteration": 1, "updates": summed(2)})
            metrics_logger.log({"iteration": 2, "updates": summed(3)})

            metrics_logger.close()

            with (log_dir / "log.csv").open(newline="", encoding="utf-8") as log_file:
                rows = list(csv.DictReader(log_file, delimiter=";"))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["iteration"], "2")
            self.assertEqual(rows[0]["updates"], "5")

    def test_rejects_invalid_buffer_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "buffer_size"):
            MetricsLogger(buffer_size=0)

    def test_learn_aggregates_complete_and_partial_logging_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            algo = _DummyAlgorithm(policy=_DummyPolicy(), env=_DummyEnv(), learning_rate=1e-3)

            algo.learn(
                max_total_timesteps=3,
                run_dir=run_dir,
                logging_buffer_size=2,
                save_optimizer=False,
                enable_command_prompt=False,
            )

            with (run_dir / "log.csv").open(newline="", encoding="utf-8") as log_file:
                rows = list(csv.DictReader(log_file, delimiter=";"))

            self.assertEqual(len(rows), 2)
            self.assertEqual([row["iteration"] for row in rows], ["2", "3"])
            self.assertEqual([row["total_updates"] for row in rows], ["2", "3"])

    def test_learn_clears_active_state_when_metrics_worker_fails(self) -> None:
        algo = _DummyAlgorithm(policy=_DummyPolicy(), env=_DummyEnv(), learning_rate=1e-3)
        record_env_factory = mock.Mock()

        with mock.patch.object(
                MetricsLogger,
                "_process_metrics_batch",
                side_effect=ValueError("write failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "background worker failed"):
                algo.learn(
                    max_total_timesteps=1,
                    save_optimizer=False,
                    enable_command_prompt=False,
                    make_record_env=record_env_factory,
                    extra_run_metadata={"test": True},
                )

        self.assertIsNone(algo._active_run_dir)
        self.assertIsNone(algo._active_extra_run_metadata)
        self.assertTrue(algo._active_save_optimizer)
        self.assertIsNone(algo._active_max_total_timesteps)
        self.assertIsNone(algo._active_learn_started_monotonic)
        self.assertIsNone(algo._active_learn_started_timesteps)
        self.assertFalse(algo._stop_requested)
        self.assertTrue(algo._stop_should_save)
        self.assertIsNone(algo._stop_save_optimizer)
        self.assertIsNone(algo._last_return_ema)
        self.assertIsNone(algo._latest_hp_update)
        self.assertIsNone(algo._make_record_env)
        self.assertIsNone(algo._command_log_path)

    def test_compress_persisted_log_replaces_csv_with_gzip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir)
            logger = MetricsLogger(log_dir=log_dir)
            logger.log({"timesteps": 12, "value": 3.5})
            logger.close()

            original_path = log_dir / "log.csv"
            self.assertTrue(original_path.exists())

            compressed_path = logger.compress_persisted_log()

            self.assertEqual(compressed_path, log_dir / "log.csv.gz")
            assert compressed_path is not None
            self.assertFalse(original_path.exists())
            self.assertTrue(compressed_path.exists())

            with gzip.open(compressed_path, "rt", encoding="utf-8") as f:
                contents = f.read()

            self.assertIn("timesteps;value;timestamp", contents)
            self.assertIn("12;3.5;", contents)

    def test_csv_schema_expands_when_new_metrics_appear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir)
            metrics_logger = MetricsLogger(log_dir=log_dir)

            metrics_logger.log({"timesteps": 1})
            metrics_logger.log({"timesteps": 2, "loss": 0.25})
            metrics_logger.close()

            with (log_dir / "log.csv").open(newline="", encoding="utf-8") as log_file:
                rows = list(csv.DictReader(log_file, delimiter=";"))

            self.assertEqual(rows[0]["timesteps"], "1")
            self.assertEqual(rows[0]["loss"], "")
            self.assertEqual(rows[1]["timesteps"], "2")
            self.assertEqual(rows[1]["loss"], "0.25")

    def test_learn_keeps_csv_when_compression_flag_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            algo = _DummyAlgorithm(policy=_DummyPolicy(), env=_DummyEnv(), learning_rate=1e-3)

            algo.learn(
                max_total_timesteps=1,
                run_dir=run_dir,
                save_optimizer=False,
                compress_metrics_log_on_exit=False,
                enable_command_prompt=False,
            )

            self.assertTrue((run_dir / "log.csv").exists())
            self.assertFalse((run_dir / "log.csv.gz").exists())

    def test_run_metadata_includes_machine_specs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            algo = _DummyAlgorithm(policy=_DummyPolicy(), env=_DummyEnv(), learning_rate=1e-3)
            machine_specs = {
                "machine_name": "test-machine",
                "cpu": {"name": "test-cpu"},
                "memory": {"total_bytes": 123},
                "gpus": [{"index": 0, "name": "test-gpu"}],
            }

            with mock.patch(
                    "swarmbots.learn.algos.base_algorithm.collect_machine_specs",
                    return_value=machine_specs,
            ):
                algo.learn(
                    max_total_timesteps=1,
                    run_dir=run_dir,
                    save_optimizer=False,
                    compress_metrics_log_on_exit=False,
                    enable_command_prompt=False,
                )

            metadata_path = run_dir / "run_metadata_0.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertEqual(metadata["machine_specs"], machine_specs)

    def test_learn_compresses_csv_when_compression_flag_is_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            algo = _DummyAlgorithm(policy=_DummyPolicy(), env=_DummyEnv(), learning_rate=1e-3)

            algo.learn(
                max_total_timesteps=1,
                run_dir=run_dir,
                save_optimizer=False,
                compress_metrics_log_on_exit=True,
                enable_command_prompt=False,
            )

            self.assertFalse((run_dir / "log.csv").exists())
            self.assertTrue((run_dir / "log.csv.gz").exists())

    def test_learn_keeps_final_return_ema_after_cleanup(self) -> None:
        algo = _ReturnEmaAlgorithm(policy=_DummyPolicy(), env=_DummyEnv(), learning_rate=1e-3)

        algo.learn(
            max_total_timesteps=12,
            save_optimizer=False,
            enable_command_prompt=False,
        )

        self.assertIsNone(algo._last_return_ema)
        self.assertEqual(algo._final_return_ema, 15.0)

    def test_learn_logs_success_rate_ema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            algo = _SuccessRateEmaAlgorithm(policy=_DummyPolicy(), env=_DummyEnv(), learning_rate=1e-3)

            algo.learn(
                max_total_timesteps=11,
                run_dir=run_dir,
                save_optimizer=False,
                enable_command_prompt=False,
            )

            with (run_dir / "log.csv").open(newline="", encoding="utf-8") as log_file:
                rows = list(csv.DictReader(log_file, delimiter=";"))

            self.assertIn("ep_success_rate_ema", rows[-1])
            self.assertEqual(float(rows[-1]["ep_success_rate_ema"]), 50.0)

    def test_show_run_path_command_logs_active_run_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            algo = _DummyAlgorithm(policy=_DummyPolicy(), env=_DummyEnv(), learning_rate=1e-3)
            algo._active_run_dir = run_dir
            messages: list[str] = []
            sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}")

            try:
                updated = algo.execute_command("show_run_path", "", None)
            finally:
                logger.remove(sink_id)

            self.assertFalse(updated)
            log_output = "".join(messages)
            self.assertIn(run_dir.resolve().as_posix(), log_output)
            self.assertIn(run_dir.name, log_output)

    def test_show_id_alias_logs_active_run_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            algo = _DummyAlgorithm(policy=_DummyPolicy(), env=_DummyEnv(), learning_rate=1e-3)
            algo._active_run_dir = run_dir
            messages: list[str] = []
            sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}")

            try:
                updated = algo.execute_command("show_id", "", None)
            finally:
                logger.remove(sink_id)

            self.assertFalse(updated)
            self.assertIn(run_dir.resolve().as_posix(), "".join(messages))


if __name__ == "__main__":
    unittest.main()
