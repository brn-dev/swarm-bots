import gzip
import tempfile
import unittest
from pathlib import Path

import torch
from loguru import logger

from swarmbots.learn.algos.base_algorithm import BaseAlgorithm
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.metrics_logger import MetricsLogger


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

    def perform_iteration(self, episode_return_ema, update_ema: bool) -> tuple[dict[str, float], int]:
        _ = episode_return_ema
        _ = update_ema
        self.n_total_iterations += 1
        self.n_total_updates += 1
        self.n_total_timesteps += 1
        return {"metric": 1.0, "total_updates": self.n_total_updates}, 1


class _ReturnEmaAlgorithm(_DummyAlgorithm):
    def perform_iteration(self, episode_return_ema, update_ema: bool) -> tuple[dict[str, float], int]:
        if update_ema:
            for _ in range(100):
                episode_return_ema.update(15.0)
        self.n_total_iterations += 1
        self.n_total_updates += 1
        self.n_total_timesteps += 1
        return {"metric": 1.0, "total_updates": self.n_total_updates}, 1


class MetricsLoggerTests(unittest.TestCase):
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
            max_total_timesteps=6,
            save_optimizer=False,
            enable_command_prompt=False,
        )

        self.assertIsNone(algo._last_return_ema)
        self.assertEqual(algo._final_return_ema, 15.0)

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
