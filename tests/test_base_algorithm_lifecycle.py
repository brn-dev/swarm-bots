from pathlib import Path
from typing import Any

import torch
import pytest

from swarmbots.learn.algos.base_algorithm import BaseAlgorithm, LearningRate
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage


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


@pytest.mark.parametrize("stop_params,save_optimizer,expected_save,expected_optimizer", [
    ("", True, True, {"optimizer_step": 1}),
    ('{"optimizer": false}', True, True, None),
    ('{"optimizer": true}', False, True, {"optimizer_step": 1}),
    ('{"save": false}', True, False, None),
])
def test_hook_stop_without_prompt_honors_checkpoint_options(
    tmp_path: Path, stop_params: str, save_optimizer: bool, expected_save: bool,
    expected_optimizer: dict[str, int] | None,
) -> None:
    algorithm = _Algorithm()

    def request_stop(trainer: BaseAlgorithm, metrics: dict[str, Any], rollout_steps: int) -> None:
        trainer.execute_command("stop", stop_params, None)

    algorithm.learn(
        max_total_timesteps=20, run_dir=tmp_path, log_interval=None,
        save_optimizer=save_optimizer, enable_command_prompt=False,
        post_iteration_hooks=[request_stop],
    )

    assert algorithm.n_total_iterations == 1
    assert algorithm.n_total_timesteps == 4
    expected_checkpoints = [] if not expected_save else [
        (tmp_path / "models" / "model_4_steps_stopped.pt", expected_optimizer, None),
    ]
    assert algorithm.saved_checkpoints == expected_checkpoints


@pytest.mark.parametrize("resume_mode", ["timeout", "unpause", "stop"])
def test_programmatic_pause_without_prompt_blocks_the_next_iteration(
    monkeypatch: pytest.MonkeyPatch, resume_mode: str,
) -> None:
    algorithm = _Algorithm()
    elapsed = 0.0
    iterations_while_waiting: list[int] = []

    def monotonic() -> float:
        return elapsed

    def advance_time(duration: float) -> None:
        nonlocal elapsed
        iterations_while_waiting.append(algorithm.n_total_iterations)
        elapsed += duration
        if resume_mode != "timeout":
            algorithm.execute_command(resume_mode, "", None)

    def control_training(trainer: BaseAlgorithm, metrics: dict[str, Any], rollout_steps: int) -> None:
        if trainer.n_total_iterations == 1:
            trainer.execute_command("pause", "1s", None)
        else:
            trainer.execute_command("stop", "", None)

    monkeypatch.setattr("swarmbots.learn.algos.base_algorithm.time.monotonic", monotonic)
    monkeypatch.setattr("swarmbots.learn.algos.base_algorithm.time.sleep", advance_time)
    algorithm.learn(
        max_total_timesteps=20, log_interval=None, enable_command_prompt=False,
        post_iteration_hooks=[control_training],
    )

    assert iterations_while_waiting
    assert set(iterations_while_waiting) == {1}
    if resume_mode == "timeout":
        assert elapsed >= 1.0
    assert algorithm.n_total_iterations == (1 if resume_mode == "stop" else 2)
    assert algorithm._pause_until_monotonic is None


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
    "policy_state_dict", "hyper_parameters", "n_total_iterations", "n_total_updates",
    "n_total_timesteps", "best_return_ema", "env_state",
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
