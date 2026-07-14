from pathlib import Path
from typing import Any

import torch

from swarmbots.learn.algos.base_algorithm import BaseAlgorithm, LearningRate
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage


class _Policy(BasePolicy):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

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

    def get_hyper_parameters(self) -> dict[str, int]:
        return {"timesteps_per_iteration": self.timesteps_per_iteration}

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
