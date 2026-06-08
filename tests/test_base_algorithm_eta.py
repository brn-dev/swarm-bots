import io
import unittest
from datetime import datetime, timezone
from typing import Any

from loguru import logger
import torch
from unittest import mock

from swarmbots.learn.algos.base_algorithm import BaseAlgorithm
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage


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
            state_dict: dict[str, Any],
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
        self.n_total_timesteps += 100
        return {"metric": 1.0}, 100


class BaseAlgorithmEtaTests(unittest.TestCase):
    def test_build_run_eta_estimate_uses_iteration_throughput(self) -> None:
        algo = _DummyAlgorithm(policy=_DummyPolicy(), env=_DummyEnv(), learning_rate=1e-3)
        algo.n_total_timesteps = 900
        algo._active_max_total_timesteps = 1400
        algo._active_learn_started_monotonic = 10.0
        algo._active_learn_started_timesteps = 400

        eta = algo._build_run_eta_estimate(
            current_time=datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc),
            elapsed_seconds=20.0,
        )

        self.assertEqual(
            eta,
            {
                "estimated_finish_at": "2026-06-08T12:00:20+00:00",
                "remaining_duration": "20s",
                "remaining_timesteps": 500,
                "observed_throughput_tps": 25.0,
            },
        )

    def test_eta_command_logs_estimate(self) -> None:
        algo = _DummyAlgorithm(policy=_DummyPolicy(), env=_DummyEnv(), learning_rate=1e-3)
        algo.n_total_timesteps = 900
        algo._active_max_total_timesteps = 1400
        algo._active_learn_started_monotonic = 80.0
        algo._active_learn_started_timesteps = 400

        buffer = io.StringIO()
        sink_id = logger.add(buffer, format="{message}")
        try:
            with mock.patch("swarmbots.learn.algos.base_algorithm.time.monotonic", return_value=100.0):
                executed = algo._execute_command("eta", "", None)
        finally:
            logger.remove(sink_id)

        self.assertFalse(executed)
        output = buffer.getvalue()
        self.assertIn("estimated_finish_at", output)
        self.assertIn("remaining_duration", output)
        self.assertIn("remaining_timesteps", output)


if __name__ == "__main__":
    unittest.main()
