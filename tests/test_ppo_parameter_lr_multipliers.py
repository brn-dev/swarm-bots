from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch import nn

from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo import PPO, AutomaticLearningRate, AutomaticLearningRateUpdateResult, StepsRolloutMode
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples, PPOSamplerConfig
from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.scheduling.auto_lr_updater import make_auto_lr_updater
from swarmbots.learn.scheduling.schedulers import ScheduledHyperParameter, SchedulerManager


def test_checkpoint_restores_custom_scheduler_state_and_enabled_flag(tmp_path) -> None:
    def schedule(old_value, state, **kwargs):
        state["calls"] = state.get("calls", 0) + 1
        return {"new_value": old_value * 0.5 if state["calls"] % 2 == 0 else None}

    def configure(ppo):
        item = ScheduledHyperParameter("learning_rate", schedule, lambda: ppo.learning_rate, ppo.set_learning_rate)
        disabled = ScheduledHyperParameter("disabled", schedule, lambda: 1.0, lambda value: None)
        ppo.scheduler_manager = SchedulerManager([item, disabled])
        return item, disabled

    source, _ = _make_ppo()
    item, disabled = configure(source)
    disabled.enabled = False
    source.scheduler_manager.step(n_iterations=1, n_model_updates=1, n_timesteps=1, metrics={})
    path = tmp_path / "scheduler.pt"
    source.save(path, optimizer_state_dict=source._get_optimizer_state_dict())
    restored, _ = _make_ppo()
    restored_item, restored_disabled = configure(restored)
    restored.load(path)
    assert restored_item.state == item.state == {"calls": 1}
    assert not restored_disabled.enabled
    restored_item.state["extra"] = True
    assert "extra" not in item.state
    for algorithm in (source, restored):
        algorithm.scheduler_manager.step(n_iterations=2, n_model_updates=2, n_timesteps=2, metrics={})
    assert restored.learning_rate == source.learning_rate == pytest.approx(0.005)


class _DummyActionDist:
    def __init__(self) -> None:
        self.distributions = [object()]
        self.has_gsde = False


class _DummyPolicy(BasePPOPolicy[PPOSamples, PPOSamplerConfig]):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(3, 3, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(3, 3, bias=False),
            nn.Linear(3, 3, bias=False),
        )
        self.query_encoder = nn.Linear(3, 3, bias=False)
        self.context_encoder = nn.Linear(3, 3, bias=False)
        self.action_dist = _DummyActionDist()

    @property
    def gsde_enabled(self) -> bool:
        return False

    def get_hyper_parameters(self) -> dict[str, float]:
        return {}

    def get_grad_norms(self) -> dict[str, float]:
        return {
            "encoder": 0.0,
            "decoder": 0.0,
            "action_dist": 0.0,
            "critic": 0.0,
            "total": 0.0,
        }

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
        _ = global_obs, hidden_local_vars, hidden_global_vars, agent_mask, previous_actions, deterministic
        batch_size, n_agents = local_obs.shape[:2]
        return torch.zeros(batch_size, n_agents, 1, device=local_obs.device)

    def update_loss_weights(self, **weights: float) -> None:
        if weights:
            raise ValueError(f"Unexpected weights: {weights}")

    def requires_previous_actions(self) -> bool:
        return False

    def forward(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _ = hidden_local_vars, hidden_global_vars, agent_mask, previous_actions, deterministic
        batch_size, n_agents = local_obs.shape[:2]
        actions = torch.zeros(batch_size, n_agents, 1, device=local_obs.device)
        log_probs = torch.zeros(batch_size, n_agents, device=local_obs.device)
        values = torch.zeros(batch_size, device=local_obs.device)
        return actions, log_probs, values

    def predict_values(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = local_obs, global_obs, hidden_local_vars, hidden_global_vars, agent_mask, previous_actions
        return torch.zeros(1)

    def _evaluate_actions(
            self,
            batch,
            action_splitter: object = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, float], torch.Tensor]:
        _ = batch, action_splitter
        log_probs = torch.zeros(1)
        values = torch.zeros(1)
        local_latents = torch.zeros(1, 1, 1)
        return log_probs, values, {}, {}, local_latents

    def make_sampler(self, episodes, config):
        _ = episodes, config
        raise NotImplementedError


class _DummyEnv:
    def __init__(self) -> None:
        self.observation_space = spaces.Dict({
            "local_obs": spaces.Box(low=-1.0, high=1.0, shape=(2, 2, 3), dtype=np.float32),
            "global_obs": spaces.Box(low=-1.0, high=1.0, shape=(2, 1), dtype=np.float32),
            "hidden_local_vars": spaces.Box(low=-1.0, high=1.0, shape=(2, 2, 1), dtype=np.float32),
            "hidden_global_vars": spaces.Box(low=-1.0, high=1.0, shape=(2, 1), dtype=np.float32),
        })
        self.action_space = VectorHybridActionSpace({
            "move": spaces.Box(low=-1.0, high=1.0, shape=(2, 2, 1), dtype=np.float32),
        })


def _make_ppo(
    parameter_lr_multipliers: dict[str, float] | None = None,
    learning_rate: float | AutomaticLearningRate = 1e-2,
) -> tuple[PPO, _DummyPolicy]:
    policy = _DummyPolicy()
    ppo = PPO(
        policy=policy,
        env=_DummyEnv(),
        learning_rate=learning_rate,
        target_kl=0.002 if isinstance(learning_rate, AutomaticLearningRate) else None,
        rollout_mode=StepsRolloutMode(2),
        max_episode_length=2,
        n_epochs=1,
        parameter_lr_multipliers=parameter_lr_multipliers,
    )
    return ppo, policy


def _param_group_by_parameter_id(ppo: PPO) -> dict[int, dict[str, object]]:
    mapping: dict[int, dict[str, object]] = {}
    for group in ppo.optimizer.param_groups:
        for parameter in group["params"]:
            mapping[id(parameter)] = group
    return mapping


def test_ppo_uses_single_default_optimizer_group_when_no_multipliers_are_configured() -> None:
    ppo, _policy = _make_ppo()

    assert len(ppo.optimizer.param_groups) == 1
    assert ppo.optimizer.param_groups[0]["lr"] == pytest.approx(1e-2)
    assert "lr_multiplier" not in ppo.optimizer.param_groups[0]


@pytest.mark.parametrize("name", ["initial_lr", "max_lr"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_automatic_learning_rate_rejects_invalid_bounds(name: str, value: float) -> None:
    options = {"initial_lr": 1e-2, "max_lr": 1e-1}
    options[name] = value
    with pytest.raises(ValueError, match="must be finite and > 0"):
        AutomaticLearningRate(updater=make_auto_lr_updater(), **options)


@pytest.mark.parametrize("via_command", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_automatic_learning_rate_rejects_zero_without_changing_optimizer(via_command: bool, enabled: bool) -> None:
    ppo, _ = _make_ppo(learning_rate=AutomaticLearningRate(initial_lr=1e-2, updater=make_auto_lr_updater()))
    ppo._auto_lr_enabled = enabled
    original_rates = [group["lr"] for group in ppo.optimizer.param_groups]

    with pytest.raises(ValueError, match="must be finite and > 0"):
        if via_command:
            ppo._execute_command("set_lr", "0.0", None)
        else:
            ppo.set_learning_rate(0.0)

    assert ppo.learning_rate == 1e-2
    assert [group["lr"] for group in ppo.optimizer.param_groups] == original_rates
    ppo._maybe_update_automatic_lr(None, None, {})
    ppo._maybe_update_automatic_lr(None, None, {})


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_automatic_learning_rate_rejects_invalid_controller_requests(value: float) -> None:
    def request_rate(**kwargs: object) -> AutomaticLearningRateUpdateResult:
        return {"new_lr": value}

    ppo, _ = _make_ppo(learning_rate=AutomaticLearningRate(initial_lr=1e-2, updater=request_rate))
    with pytest.raises(ValueError, match="must be finite and > 0"):
        ppo._maybe_update_automatic_lr(None, None, {})

    assert ppo.learning_rate == 1e-2
    assert all(group["lr"] == 1e-2 for group in ppo.optimizer.param_groups)


def test_automatic_learning_rate_caps_finite_controller_requests() -> None:
    def request_rate(**kwargs: object) -> AutomaticLearningRateUpdateResult:
        return {"new_lr": 0.1}

    ppo, _ = _make_ppo(learning_rate=AutomaticLearningRate(initial_lr=1e-4, updater=request_rate, max_lr=1e-2))
    result = ppo._maybe_update_automatic_lr(None, None, {})

    assert result["auto_lr_event"] == "lr_increase_capped"
    assert ppo.learning_rate == 1e-2
    assert all(group["lr"] == 1e-2 for group in ppo.optimizer.param_groups)


@pytest.mark.parametrize("name", ["initial_lr", "max_lr"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_ppo_revalidates_mutated_automatic_learning_rate_config(name: str, value: float) -> None:
    config = AutomaticLearningRate(initial_lr=1e-2, updater=make_auto_lr_updater())
    setattr(config, name, value)
    with pytest.raises(ValueError, match="must be finite and > 0"):
        _make_ppo(learning_rate=config)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_automatic_learning_rate_revalidates_mutated_maximum(value: float) -> None:
    def request_rate(**kwargs: object) -> AutomaticLearningRateUpdateResult:
        return {"new_lr": 2e-2}

    config = AutomaticLearningRate(initial_lr=1e-2, updater=request_rate)
    ppo, _ = _make_ppo(learning_rate=config)
    config.max_lr = value
    with pytest.raises(ValueError, match="must be finite and > 0"):
        ppo._maybe_update_automatic_lr(None, None, {})
    assert ppo.learning_rate == 1e-2
    assert all(group["lr"] == 1e-2 for group in ppo.optimizer.param_groups)


@pytest.mark.parametrize("field", ["base", "optimizer", "secondary_optimizer", "secondary_multiplier", "both"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_public_load_rejects_invalid_automatic_rates_without_mutating_trainer(
    tmp_path: Path, field: str, value: float,
) -> None:
    multipliers = {"decoder": 0.5}
    source, _ = _make_ppo(
        parameter_lr_multipliers=multipliers,
        learning_rate=AutomaticLearningRate(initial_lr=1e-2, updater=make_auto_lr_updater()),
    )
    with torch.no_grad():
        for parameter in source.policy.parameters():
            parameter.fill_(2.0)
    source.n_total_timesteps = 42
    path = tmp_path / "invalid_rate.pt"
    source.save(path, optimizer_state_dict=source._get_optimizer_state_dict())
    checkpoint = torch.load(path, weights_only=False)
    if field in {"base", "both"}:
        checkpoint["hyper_parameters"]["learning_rate"] = value
    if field in {"optimizer", "both"}:
        checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"] = value
    if field == "secondary_optimizer":
        checkpoint["optimizer_state_dict"]["param_groups"][1]["lr"] = value
    if field == "secondary_multiplier":
        checkpoint["optimizer_state_dict"]["param_groups"][1]["lr_multiplier"] = value
    torch.save(checkpoint, path)
    restored, _ = _make_ppo(
        parameter_lr_multipliers=multipliers,
        learning_rate=AutomaticLearningRate(initial_lr=1e-3, updater=make_auto_lr_updater()),
    )
    original_weights = {name: tensor.clone() for name, tensor in restored.policy.state_dict().items()}
    original_rates = [group["lr"] for group in restored.optimizer.param_groups]

    with pytest.raises(ValueError, match="must be finite and > 0"):
        restored.load(path)

    torch.testing.assert_close(restored.policy.state_dict(), original_weights)
    assert restored.learning_rate == 1e-3
    assert [group["lr"] for group in restored.optimizer.param_groups] == original_rates
    assert restored.n_total_timesteps == 0
    assert restored._auto_lr_state == {}
    restored._maybe_update_automatic_lr(None, None, {})
    restored._maybe_update_automatic_lr(None, None, {})
    assert restored.learning_rate == pytest.approx(1.1e-3)


@pytest.mark.parametrize("include_optimizer", [False, True])
def test_public_load_ignores_saved_automatic_rates_when_optimizer_is_not_restored(
    tmp_path: Path, include_optimizer: bool,
) -> None:
    source, _ = _make_ppo(learning_rate=AutomaticLearningRate(initial_lr=1e-2, updater=make_auto_lr_updater()))
    path = tmp_path / "weights_only.pt"
    source.save(path, optimizer_state_dict=source._get_optimizer_state_dict() if include_optimizer else None)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["hyper_parameters"]["learning_rate"] = 0.0
    if include_optimizer:
        checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"] = 0.0
    torch.save(checkpoint, path)
    restored, _ = _make_ppo(learning_rate=AutomaticLearningRate(initial_lr=1e-3, updater=make_auto_lr_updater()))

    restored.load(path, restore_optimizer_state=not include_optimizer)

    torch.testing.assert_close(restored.policy.state_dict(), source.policy.state_dict())
    assert restored.learning_rate == 1e-3
    assert all(group["lr"] == 1e-3 for group in restored.optimizer.param_groups)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_parameter_lr_multipliers_reject_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite and > 0"):
        _make_ppo(parameter_lr_multipliers={"decoder": value})


def test_parameter_lr_multipliers_scale_matching_parameters_and_keep_longest_prefix() -> None:
    ppo, policy = _make_ppo({
        "decoder": 0.5,
        "decoder.1": 0.2,
        "query_encoder": 0.25,
        "context_encoder": 0.75,
    })

    param_groups = _param_group_by_parameter_id(ppo)

    assert len(ppo.optimizer.param_groups) == 5
    assert param_groups[id(policy.shared.weight)]["lr"] == pytest.approx(1e-2)
    assert param_groups[id(policy.shared.weight)]["lr_multiplier"] == pytest.approx(1.0)
    assert param_groups[id(policy.decoder[0].weight)]["lr"] == pytest.approx(5e-3)
    assert param_groups[id(policy.decoder[0].weight)]["lr_multiplier"] == pytest.approx(0.5)
    assert param_groups[id(policy.decoder[1].weight)]["lr"] == pytest.approx(2e-3)
    assert param_groups[id(policy.decoder[1].weight)]["lr_multiplier"] == pytest.approx(0.2)
    assert param_groups[id(policy.query_encoder.weight)]["lr"] == pytest.approx(2.5e-3)
    assert param_groups[id(policy.query_encoder.weight)]["lr_multiplier"] == pytest.approx(0.25)
    assert param_groups[id(policy.context_encoder.weight)]["lr"] == pytest.approx(7.5e-3)
    assert param_groups[id(policy.context_encoder.weight)]["lr_multiplier"] == pytest.approx(0.75)

    ppo.set_learning_rate(2e-2)
    param_groups = _param_group_by_parameter_id(ppo)

    assert ppo.learning_rate == pytest.approx(2e-2)
    assert param_groups[id(policy.shared.weight)]["lr"] == pytest.approx(2e-2)
    assert param_groups[id(policy.decoder[0].weight)]["lr"] == pytest.approx(1e-2)
    assert param_groups[id(policy.decoder[1].weight)]["lr"] == pytest.approx(4e-3)
    assert param_groups[id(policy.query_encoder.weight)]["lr"] == pytest.approx(5e-3)
    assert param_groups[id(policy.context_encoder.weight)]["lr"] == pytest.approx(1.5e-2)


def test_optimizer_state_dict_round_trip_recovers_base_learning_rate() -> None:
    ppo, policy = _make_ppo({
        "decoder": 0.5,
        "query_encoder": 0.25,
        "context_encoder": 0.75,
    })
    optimizer_state = ppo._get_optimizer_state_dict()

    restored_ppo, restored_policy = _make_ppo({
        "decoder": 0.5,
        "query_encoder": 0.25,
        "context_encoder": 0.75,
    }, learning_rate=0.123)
    restored_ppo._apply_optimizer_state_dict(
        optimizer_state,
        missing_keys=[],
        unexpected_keys=[],
    )

    restored_groups = _param_group_by_parameter_id(restored_ppo)

    assert restored_ppo.learning_rate == pytest.approx(1e-2)
    assert restored_groups[id(restored_policy.shared.weight)]["lr"] == pytest.approx(1e-2)
    assert restored_groups[id(restored_policy.decoder[0].weight)]["lr"] == pytest.approx(5e-3)
    assert restored_groups[id(restored_policy.query_encoder.weight)]["lr"] == pytest.approx(2.5e-3)
    assert restored_groups[id(restored_policy.context_encoder.weight)]["lr"] == pytest.approx(7.5e-3)


def test_ppo_rejects_optimizer_state_without_controller_metadata() -> None:
    original, _ = _make_ppo(learning_rate=1e-3)
    restored = PPO(
        policy=_DummyPolicy(),
        env=_DummyEnv(),
        learning_rate=AutomaticLearningRate(initial_lr=0.123, updater=make_auto_lr_updater()),
        target_kl=0.002,
    )
    with pytest.raises(KeyError, match="automatic_lr"):
        restored._apply_optimizer_state_dict(original.optimizer.state_dict(), missing_keys=[], unexpected_keys=[])


def test_automatic_lr_checkpoint_preserves_controller_state() -> None:
    def make_ppo() -> PPO:
        return PPO(
            policy=_DummyPolicy(), env=_DummyEnv(), target_kl=0.002,
            learning_rate=AutomaticLearningRate(initial_lr=0.01, updater=make_auto_lr_updater()),
        )

    source = make_ppo()
    source._maybe_update_automatic_lr(None, None, {})
    source._maybe_update_automatic_lr(None, None, {})
    source._auto_lr_enabled = False
    restored = make_ppo()

    restored._apply_optimizer_state_dict(source._get_optimizer_state_dict(), missing_keys=[], unexpected_keys=[])

    assert restored.learning_rate == pytest.approx(source.learning_rate)
    assert restored._auto_lr_state == source._auto_lr_state
    assert not restored._auto_lr_enabled
