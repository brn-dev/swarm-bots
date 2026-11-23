from pathlib import Path
from copy import deepcopy
from functools import partial
from dataclasses import dataclass
import json
from typing import Any

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
from swarmbots.learn.scheduling.linear_scheduler import LinearScheduler
from swarmbots.learn.scheduling.schedulers import ScheduledHyperParameter, SchedulerManager, ScheduleUnit


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


CHECKPOINT_CONFIGURATION_MISMATCHES = (
    "automatic_lr", "missing_automatic_lr", "schedulers", "missing_schedulers",
    "scheduler_names", "scheduler_order", "scheduler_duration", "scheduler_binding",
    "automatic_lr_cap", "automatic_lr_updater",
)


def _make_incompatible_trainers(mismatch: str) -> tuple[PPO, PPO]:
    source_rate = AutomaticLearningRate(initial_lr=0.02, updater=make_auto_lr_updater())
    target_rate = AutomaticLearningRate(
        initial_lr=0.001,
        max_lr=0.003 if mismatch == "automatic_lr_cap" else source_rate.max_lr,
        updater=make_auto_lr_updater(**({"max_kl_div": 0.05} if mismatch == "automatic_lr_updater" else {})),
    )
    source, _ = _make_ppo(learning_rate=source_rate if mismatch != "missing_automatic_lr" else 0.02)
    target, _ = _make_ppo(learning_rate=target_rate if mismatch != "automatic_lr" else 0.001)

    def configure_schedulers(algorithm: PPO, names: list[str]) -> None:
        def schedule(**kwargs: Any) -> dict[str, None]:
            return {"new_value": None}

        algorithm.scheduler_manager = SchedulerManager([
            ScheduledHyperParameter(name, schedule, lambda: algorithm.learning_rate, algorithm.set_learning_rate)
            for name in names
        ])

    if mismatch in {"schedulers", "scheduler_names", "scheduler_order"}:
        configure_schedulers(source, ["first", "second"])
    if mismatch in {"missing_schedulers", "scheduler_names", "scheduler_order"}:
        configure_schedulers(target, ["second", "first"] if mismatch == "scheduler_order" else ["other"])
    if mismatch in {"scheduler_duration", "scheduler_binding"}:
        for algorithm in (source, target):
            duration = 100 if algorithm is target and mismatch == "scheduler_duration" else 10
            attribute = "gae_lambda" if algorithm is target and mismatch == "scheduler_binding" else "gamma"
            algorithm.scheduler_manager = SchedulerManager([ScheduledHyperParameter(
                "discount",
                LinearScheduler(ScheduleUnit.ITERATIONS, duration, 0.02, 0.001),
                partial(getattr, algorithm, attribute),
                partial(setattr, algorithm, attribute),
            )])
    return source, target


@pytest.mark.parametrize("bound_methods", [False, True])
def test_checkpoint_restores_schedule_with_mutable_receiver_and_unavailable_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound_methods: bool,
) -> None:
    @dataclass
    class RuntimeValues:
        discount: float = 0.99

        def get_discount(self) -> float:
            return self.discount

        def set_discount(self, value: float) -> None:
            self.discount = value

    def make_trainer(*, shared_receiver: bool = True) -> tuple[PPO, RuntimeValues]:
        trainer, _ = _make_ppo()
        values = RuntimeValues()
        setter_values = values if shared_receiver else RuntimeValues()
        trainer.scheduler_manager = SchedulerManager([ScheduledHyperParameter(
            "discount", LinearScheduler(ScheduleUnit.ITERATIONS, 10, 0.99, 0.89),
            values.get_discount if bound_methods else partial(getattr, values, "discount").__call__,
            setter_values.set_discount if bound_methods else partial(setattr, setter_values, "discount"),
        )])
        return trainer, values

    source, source_values = make_trainer()
    source.scheduler_manager.step(n_iterations=2, n_model_updates=0, n_timesteps=0, metrics={})
    path = tmp_path / "scheduler.pt"
    source.save(path, optimizer_state_dict=source._get_optimizer_state_dict())
    target, target_values = make_trainer()
    assert target_values.discount != source_values.discount

    def unavailable_source(*_args: Any, **_kwargs: Any) -> str:
        raise OSError("source unavailable")

    monkeypatch.setattr("swarmbots.learn.serialization_utils.inspect.getsource", unavailable_source)
    mismatched_target, mismatched_values = make_trainer(shared_receiver=False)
    with pytest.raises(ValueError, match="Checkpoint scheduler configuration"):
        mismatched_target.load(path)
    assert mismatched_values.discount == pytest.approx(0.99)
    target.load(path)
    assert target_values.discount == source_values.discount == pytest.approx(0.97)


@pytest.mark.parametrize("stock_partials", [False, True])
def test_checkpoint_rejects_different_declared_schedule_selector(tmp_path: Path, stock_partials: bool) -> None:
    @dataclass
    class RuntimeBinding:
        field: str
        values: dict[str, float]

        def get_hyper_parameters(self) -> dict[str, str]:
            return {"field": self.field}

        def get(self) -> float:
            return self.values[self.field]

        def apply(self, value: float) -> None:
            self.values[self.field] = value

        value = property(get, apply)

    def make_trainer(binding: RuntimeBinding) -> PPO:
        trainer, _ = _make_ppo()
        get_value = partial(getattr, binding, "value").__call__ if stock_partials else binding.get
        apply = partial(setattr, binding, "value") if stock_partials else binding.apply
        trainer.scheduler_manager = SchedulerManager([ScheduledHyperParameter(
            "discount", LinearScheduler(ScheduleUnit.ITERATIONS, 10, 0.99, 0.89), get_value, apply,
        )])
        return trainer

    source = make_trainer(RuntimeBinding("gamma", {"gamma": 0.71, "gae_lambda": 0.8}))
    path = tmp_path / "scheduler.pt"
    source.save(path, optimizer_state_dict=source._get_optimizer_state_dict())

    different = RuntimeBinding("gae_lambda", {"gamma": 0.95, "gae_lambda": 0.81})
    with pytest.raises(ValueError, match="Checkpoint scheduler configuration"):
        make_trainer(different).load(path)
    assert different.values == {"gamma": 0.95, "gae_lambda": 0.81}

    matching = RuntimeBinding("gamma", {"gamma": 0.95, "gae_lambda": 0.81})
    make_trainer(matching).load(path)
    assert matching.values == {"gamma": 0.71, "gae_lambda": 0.81}


@pytest.mark.parametrize("mismatch", CHECKPOINT_CONFIGURATION_MISMATCHES)
@pytest.mark.parametrize(("strict_load_state_dict", "extra_policy_key"), [
    (True, False), (True, True), (False, False),
])
def test_rejected_checkpoint_configuration_preserves_trainer(
    tmp_path: Path, mismatch: str, strict_load_state_dict: bool, extra_policy_key: bool,
) -> None:
    source, target = _make_incompatible_trainers(mismatch)
    if extra_policy_key:
        target.policy.register_buffer("additional_features", torch.tensor([7.0]))
    with torch.no_grad():
        for parameter in source.policy.parameters():
            parameter.fill_(2.0)
    source.n_total_timesteps = 42
    path = tmp_path / "incompatible.pt"
    source.save(path, optimizer_state_dict=source._get_optimizer_state_dict())
    sum(parameter.square().sum() for parameter in target.policy.parameters()).backward()
    target.optimizer.step()
    target.optimizer.zero_grad(set_to_none=True)
    weights = deepcopy(target.policy.state_dict())
    optimizer = deepcopy(target._get_optimizer_state_dict())
    rollout_state = object()
    target._rollout_state = rollout_state

    with pytest.raises(ValueError, match="Checkpoint"):
        target.load(path, strict_load_state_dict=strict_load_state_dict)

    torch.testing.assert_close(target.policy.state_dict(), weights)
    current_optimizer = target._get_optimizer_state_dict()
    assert current_optimizer.pop("schedulers") == optimizer.pop("schedulers")
    assert current_optimizer.pop("schedule_configuration") == optimizer.pop("schedule_configuration")
    torch.testing.assert_close(current_optimizer, optimizer)
    assert target.learning_rate == 0.001
    assert target.n_total_timesteps == 0
    assert target._rollout_state is rollout_state


@pytest.mark.parametrize("mismatch", CHECKPOINT_CONFIGURATION_MISMATCHES)
@pytest.mark.parametrize("key_difference", ["missing", "unexpected"])
def test_partial_checkpoint_load_skips_incompatible_optimizer(
    tmp_path: Path, mismatch: str, key_difference: str,
) -> None:
    source, target = _make_incompatible_trainers(mismatch)
    expanded = target if key_difference == "missing" else source
    expanded.policy.register_buffer("additional_features", torch.tensor([7.0]))
    with torch.no_grad():
        for parameter in source.policy.parameters():
            parameter.fill_(2.0)
    path = tmp_path / "partial.pt"
    source.save(path, optimizer_state_dict=source._get_optimizer_state_dict())
    sum(parameter.square().sum() for parameter in target.policy.parameters()).backward()
    target.optimizer.step()
    target.optimizer.zero_grad(set_to_none=True)
    optimizer = deepcopy(target._get_optimizer_state_dict())
    source_weights = source.policy.state_dict()
    expected_weights = {
        name: source_weights.get(name, tensor).clone()
        for name, tensor in target.policy.state_dict().items()
    }

    target.load(path, strict_load_state_dict=False, restore_env_state=False)

    torch.testing.assert_close(target.policy.state_dict(), expected_weights)
    current_optimizer = target._get_optimizer_state_dict()
    assert current_optimizer.pop("schedulers") == optimizer.pop("schedulers")
    assert current_optimizer.pop("schedule_configuration") == optimizer.pop("schedule_configuration")
    torch.testing.assert_close(current_optimizer, optimizer)
    assert target.learning_rate == 0.001


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
        train_device="cpu",
        rollout_device="cpu",
    )
    return ppo, policy


def _step_with_distinct_gradients(algorithm: PPO) -> None:
    for index, parameter in enumerate(algorithm.policy.parameters(), start=1):
        parameter.grad = torch.full_like(parameter, float(index))
    algorithm.optimizer.step()
    algorithm.optimizer.zero_grad(set_to_none=True)


def _assert_optimizer_state_equal(algorithm: PPO, expected: dict[str, Any]) -> None:
    for key, value in algorithm._get_optimizer_state_dict().items():
        if key == "state":
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        else:
            assert value == expected[key]


def test_ppo_metadata_includes_automatic_lr_configuration() -> None:
    algorithm, _ = _make_ppo(learning_rate=AutomaticLearningRate(
        initial_lr=1e-4, updater=make_auto_lr_updater(max_kl_div=0.0031, early_stop_epoch_decay_limit=6),
    ))

    updater_metadata = algorithm.get_hyper_parameters()["automatic_learning_rate"]["updater"]

    assert updater_metadata["config"] == {
        "warm_scheduler_config": None, "early_stop_epoch_decay_limit": 6, "max_kl_div": 0.0031,
    }


@pytest.mark.parametrize("callback_kind", ["object", "method", "partial"])
def test_checkpoints_preserve_differing_scheduler_configuration(tmp_path: Path, callback_kind: str) -> None:
    saved = []
    for duration in (10, 100):
        algorithm, _ = _make_ppo()
        scheduler = LinearScheduler(ScheduleUnit.ITERATIONS, duration, 0.01, 0.001)
        callback = scheduler if callback_kind == "object" else scheduler.__call__
        if callback_kind == "partial":
            callback = partial(callback)
        algorithm.scheduler_manager = SchedulerManager([ScheduledHyperParameter(
            name="lr", scheduler=callback,
            get_value=lambda: algorithm.learning_rate, apply=algorithm.set_learning_rate,
        )])
        path = tmp_path / f"scheduler-{duration}.pt"
        algorithm.save(path, optimizer_state_dict=algorithm._get_optimizer_state_dict())
        checkpoint = torch.load(path, weights_only=False)
        metadata = checkpoint["hyper_parameters"]["schedulers"][0]["scheduler"]
        if callback_kind == "partial":
            metadata = metadata["function"]
        assert metadata["config"]["duration"] == duration
        assert metadata["config"]["unit"] == "ITERATIONS"
        assert metadata["config"]["start_value"] == 0.01
        assert metadata["config"]["final_value"] == 0.001
        saved.append(metadata)
    assert saved[0] != saved[1]


@pytest.mark.parametrize("host_kind", ["trainer", "cyclic_dataclass", "callback_cycle"])
@pytest.mark.parametrize("callback_kind", ["object", "method"])
def test_partial_attribute_callbacks_save_and_resume_checkpoints(
    tmp_path: Path, host_kind: str, callback_kind: str,
) -> None:
    @dataclass
    class RuntimeState:
        algorithm: PPO
        owner: Any = None

        @property
        def gamma(self) -> float:
            return self.algorithm.gamma

        @gamma.setter
        def gamma(self, value: float) -> None:
            self.algorithm.gamma = value

    def make_algorithm() -> PPO:
        algorithm, _ = _make_ppo()
        host: Any = algorithm
        if host_kind != "trainer":
            host = RuntimeState(algorithm)
            host.owner = host if host_kind == "cyclic_dataclass" else partial(getattr, host, "gamma")
        getter = partial(getattr, host, "gamma")
        applier = partial(setattr, host, "gamma")
        algorithm.scheduler_manager = SchedulerManager([ScheduledHyperParameter(
            name="gamma", scheduler=LinearScheduler(ScheduleUnit.ITERATIONS, 10, 0.99, 0.9),
            get_value=getter.__call__ if callback_kind == "method" else getter,
            apply=applier.__call__ if callback_kind == "method" else applier,
        )])
        return algorithm

    source = make_algorithm()
    source.scheduler_manager.step(n_iterations=1, n_model_updates=0, n_timesteps=0, metrics={})
    assert source.gamma == pytest.approx(0.981)
    path = tmp_path / "partial-callbacks.pt"

    source.save(path, optimizer_state_dict=source._get_optimizer_state_dict())
    checkpoint = torch.load(path, weights_only=False)
    metadata = checkpoint["hyper_parameters"]
    assert json.loads(json.dumps(metadata)) == metadata
    expected_host = {"type": "swarmbots.learn.algos.ppo.ppo.PPO", "opaque": True}
    for callback_name in ("get_value", "apply"):
        record = metadata["schedulers"][0][callback_name]
        if callback_kind == "method":
            record = record["config"]
        recorded_host = record["args"][0]
        if host_kind == "trainer":
            assert recorded_host == expected_host
        else:
            assert recorded_host["algorithm"] == expected_host
            back_reference = recorded_host["owner"]
            if host_kind == "callback_cycle":
                back_reference = back_reference["args"][0]
            assert back_reference == {
                "type": f"{RuntimeState.__module__}.{RuntimeState.__qualname__}", "opaque": True,
            }

    restored = make_algorithm()
    restored.load(path)
    assert restored.gamma == source.gamma
    for algorithm in (source, restored):
        algorithm.scheduler_manager.step(n_iterations=5, n_model_updates=0, n_timesteps=0, metrics={})
    assert restored.gamma == source.gamma == pytest.approx(0.945)


@pytest.mark.parametrize("source_groups,target_groups", [
    ({"shared": 0.5}, {"query_encoder": 0.5}),
    ({"shared": 0.5, "query_encoder": 0.25}, {"shared": 0.25, "query_encoder": 0.5}),
    ({"decoder": 0.5}, {"shared": 0.5}),
    ({}, {"shared": 0.5}),
    ({"shared": 0.5}, {"shared": 0.25}),
])
@pytest.mark.parametrize("strict", [False, True])
def test_checkpoint_rejects_changed_parameter_groups_before_mutating_trainer(
    tmp_path: Path, source_groups: dict[str, float], target_groups: dict[str, float], strict: bool,
) -> None:
    source, _ = _make_ppo(source_groups, learning_rate=0.02)
    target, _ = _make_ppo(target_groups, learning_rate=0.001)
    with torch.no_grad():
        for parameter in source.policy.parameters():
            parameter.fill_(2.0)
    _step_with_distinct_gradients(source)
    _step_with_distinct_gradients(target)
    source.n_total_timesteps = 42
    path = tmp_path / "different_groups.pt"
    source.save(path, optimizer_state_dict=source._get_optimizer_state_dict())
    weights = deepcopy(target.policy.state_dict())
    optimizer_state = deepcopy(target._get_optimizer_state_dict())
    rollout_state = object()
    target._rollout_state = rollout_state

    with pytest.raises(ValueError, match="parameter.*groups.*restore_optimizer_state=False"):
        target.load(path, strict_load_state_dict=strict)

    torch.testing.assert_close(target.policy.state_dict(), weights, rtol=0, atol=0)
    _assert_optimizer_state_equal(target, optimizer_state)
    assert target.learning_rate == 0.001
    assert target.parameter_lr_multipliers == target_groups
    assert target.n_total_timesteps == 0
    assert target._rollout_state is rollout_state


def test_matching_checkpoint_parameter_groups_restore_moments_to_their_parameters(tmp_path: Path) -> None:
    groups = {"decoder": 0.5, "query_encoder": 0.25}
    source, _ = _make_ppo(groups)
    _step_with_distinct_gradients(source)
    path = tmp_path / "matching_groups.pt"
    source.save(path, optimizer_state_dict=source._get_optimizer_state_dict())
    target, _ = _make_ppo(groups, learning_rate=0.123)

    target.load(path, restore_env_state=False)

    source_parameters = dict(source.policy.named_parameters())
    for name, parameter in target.policy.named_parameters():
        source_parameter = source_parameters[name]
        torch.testing.assert_close(target.optimizer.state[parameter], source.optimizer.state[source_parameter])
    _step_with_distinct_gradients(source)
    _step_with_distinct_gradients(target)
    torch.testing.assert_close(target.policy.state_dict(), source.policy.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize("partial_load", [False, True])
def test_changed_parameter_groups_allow_weights_only_checkpoint_loading(tmp_path: Path, partial_load: bool) -> None:
    source, _ = _make_ppo({"shared": 0.5})
    target, _ = _make_ppo({"query_encoder": 0.5}, learning_rate=0.001)
    if partial_load:
        target.policy.register_buffer("extra_features", torch.tensor([7.0]))
    _step_with_distinct_gradients(source)
    _step_with_distinct_gradients(target)
    optimizer_state = deepcopy(target._get_optimizer_state_dict())
    path = tmp_path / "weights_only_groups.pt"
    source.save(path, optimizer_state_dict=source._get_optimizer_state_dict())

    target.load(
        path, restore_optimizer_state=partial_load, strict_load_state_dict=not partial_load,
        restore_env_state=False,
    )

    source_parameters = dict(source.policy.named_parameters())
    for name, parameter in target.policy.named_parameters():
        torch.testing.assert_close(parameter, source_parameters[name], rtol=0, atol=0)
    _assert_optimizer_state_equal(target, optimizer_state)
    assert target.parameter_lr_multipliers == {"query_encoder": 0.5}
    assert target.learning_rate == 0.001


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


@pytest.mark.parametrize("enabled", [False, True])
def test_automatic_learning_rate_rejects_zero_without_changing_optimizer(enabled: bool) -> None:
    ppo, _ = _make_ppo(learning_rate=AutomaticLearningRate(initial_lr=1e-2, updater=make_auto_lr_updater()))
    ppo._auto_lr_enabled = enabled
    original_rates = [group["lr"] for group in ppo.optimizer.param_groups]

    with pytest.raises(ValueError, match="must be finite and > 0"):
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


@pytest.mark.parametrize("invalid", [{"n_epochs": 0}, {"max_grad_norm": -1.0}, {"max_grad_norm": float("nan")}])
def test_invalid_ppo_optimization_settings(invalid: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=next(iter(invalid))):
        PPO(_DummyPolicy(), _DummyEnv(), rollout_mode=StepsRolloutMode(2), max_episode_length=2, **invalid)


def test_equivalent_unit_groups_restore_in_either_direction(tmp_path: Path) -> None:
    path = tmp_path / "equivalent.pt"
    for rules, other in (({}, {"decoder": 1.0}), ({"decoder": 1.0}, {})):
        source, _ = _make_ppo(rules)
        target, _ = _make_ppo(other)
        _step_with_distinct_gradients(source)
        source.save(path, optimizer_state_dict=source._get_optimizer_state_dict())
        target.load(path)
        _step_with_distinct_gradients(source)
        _step_with_distinct_gradients(target)
        torch.testing.assert_close(source.policy.state_dict(), target.policy.state_dict())


def test_rejected_scheduler_callback_restores_prior_settings_and_weights(tmp_path: Path) -> None:
    source, _ = _make_ppo()
    target, _ = _make_ppo()
    for trainer in (source, target):
        trainer.scheduler_manager = SchedulerManager([
            ScheduledHyperParameter(name="gamma", scheduler=None, get_value=partial(getattr, trainer, "gamma"),
                                    apply=partial(setattr, trainer, "gamma")),
            ScheduledHyperParameter(name="lr", scheduler=None, get_value=partial(getattr, trainer, "learning_rate"),
                                    apply=trainer.set_learning_rate),
        ])
    path = tmp_path / "invalid-schedule.pt"
    source.save(path, optimizer_state_dict=source._get_optimizer_state_dict())
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["optimizer_state_dict"]["schedulers"][0]["value"] = 0.31
    checkpoint["optimizer_state_dict"]["schedulers"][1]["value"] = -1.0
    torch.save(checkpoint, path)
    weights = {key: val.clone() for key, val in target.policy.state_dict().items()}
    gamma, lr = target.gamma, target.learning_rate
    with pytest.raises(ValueError, match="learning_rate"):
        target.load(path)
    torch.testing.assert_close(weights, target.policy.state_dict())
    assert target.gamma == gamma and target.learning_rate == lr


def test_nested_unit_multiplier_restores_base_rate() -> None:
    ppo, policy = _make_ppo({"decoder": 0.5, "decoder.1": 1.0})
    groups = _param_group_by_parameter_id(ppo)
    assert groups[id(policy.decoder[0].weight)]["lr"] == pytest.approx(0.005)
    assert groups[id(policy.decoder[1].weight)]["lr"] == pytest.approx(0.01)


@pytest.mark.parametrize("corrupt", [
    "policy", "moment", "negative_variance", "nan_momentum", "overflow_momentum",
    "invalid_slots", "quantized_policy", "quantized_moment", "ema", "counter", "nan_parameter", "overflow_parameter",
])
def test_invalid_checkpoint_does_not_change_live_policy(tmp_path: Path, corrupt: str) -> None:
    source, _ = _make_ppo()
    _step_with_distinct_gradients(source)
    path = tmp_path / "malformed.pt"
    source.save(path, optimizer_state_dict=source._get_optimizer_state_dict())
    checkpoint = torch.load(path, weights_only=False)
    if corrupt == "policy":
        weights = checkpoint["policy_state_dict"]
        weights[next(reversed(weights))] = torch.zeros(23)
    elif corrupt in {"nan_parameter", "overflow_parameter"}:
        weights = checkpoint["policy_state_dict"]
        key = next(reversed(weights))
        value = float("nan") if corrupt == "nan_parameter" else 1e100
        weights[key] = torch.full_like(weights[key], value, dtype=torch.float64)
    elif corrupt == "moment":
        next(iter(checkpoint["optimizer_state_dict"]["state"].values()))["exp_avg"] = torch.zeros(23)
    elif corrupt in {"negative_variance", "nan_momentum", "overflow_momentum"}:
        slot = next(iter(checkpoint["optimizer_state_dict"]["state"].values()))
        if corrupt == "negative_variance":
            slot["exp_avg_sq"].fill_(-1)
        else:
            value = float("nan") if corrupt == "nan_momentum" else 1e100
            slot["exp_avg"] = torch.full_like(slot["exp_avg"], value, dtype=torch.float64)
    elif corrupt == "invalid_slots":
        slots = checkpoint["optimizer_state_dict"]["state"]
        slots[next(iter(slots))] = None
    elif corrupt.startswith("quantized"):
        if corrupt == "quantized_policy":
            container = checkpoint["policy_state_dict"]
            key = next(reversed(container))
        else:
            container = next(iter(checkpoint["optimizer_state_dict"]["state"].values()))
            key = "exp_avg"
        container[key] = torch.quantize_per_tensor(container[key], 0.1, 0, torch.qint8)
    elif corrupt == "ema":
        checkpoint["episode_return_ema"]["pre_exponential_count"] = -1
    else:
        del checkpoint["n_total_updates"]
    torch.save(checkpoint, path)
    target, _ = _make_ppo()
    original = {key: val.clone() for key, val in target.policy.state_dict().items()}
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        target.load(path)
    torch.testing.assert_close(original, target.policy.state_dict())
    assert not target.optimizer.state


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
