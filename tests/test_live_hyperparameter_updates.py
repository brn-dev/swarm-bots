from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from swarmbots.learn.algos.base_algorithm import BaseAlgorithm
from swarmbots.learn.algos.ppo.ppo import PPO
from swarmbots.learn.algos.sac.sac import SAC
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_ppo_wrapper import NextObsPredWrapper, NOPWorldModelConfig
from tests.test_off_policy_commands import _make_algorithm
from tests.test_ppo_policy_dropout import _make_env as _make_ppo_env, _make_policy, _make_ppo
from tests.test_sac_algorithm import _make_env


@pytest.fixture(scope="module", autouse=True)
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=("ppo", "sac", "td3", "ddpg"))
def algorithm(request: pytest.FixtureRequest) -> Iterator[BaseAlgorithm]:
    if request.param == "ppo":
        env = _make_ppo_env()
        yield _make_ppo(env, _make_policy(env))
    else:
        env = _make_env()
        try:
            yield _make_algorithm(env, request.param)
        finally:
            env.close()


def _optimizer_rates(algorithm: BaseAlgorithm) -> list[float]:
    optimizers = (
        (algorithm.optimizer,)
        if isinstance(algorithm, PPO)
        else (algorithm.actor_optimizer, algorithm.critic_optimizer)
    )
    if isinstance(algorithm, SAC) and algorithm.ent_coef_optimizer is not None:
        optimizers = (*optimizers, algorithm.ent_coef_optimizer)
    return [group["lr"] for optimizer in optimizers for group in optimizer.param_groups]


@pytest.mark.parametrize("value", [-0.001, float("nan"), float("inf")])
def test_invalid_learning_rate_updates_preserve_base_and_optimizer_rates(
    algorithm: BaseAlgorithm, value: float,
) -> None:
    before = algorithm.learning_rate, _optimizer_rates(algorithm)
    with pytest.raises(ValueError):
        algorithm.set_learning_rate(value)
    assert (algorithm.learning_rate, _optimizer_rates(algorithm)) == before


@pytest.mark.parametrize("learning_rate", [0.0, 0.002])
def test_fixed_learning_rate_updates_keep_base_and_optimizers_consistent(
    algorithm: BaseAlgorithm, learning_rate: float,
) -> None:
    algorithm.set_learning_rate(learning_rate)
    assert algorithm.learning_rate == learning_rate
    assert all(rate == learning_rate for rate in _optimizer_rates(algorithm))


@pytest.mark.parametrize("kind", ["sac", "td3", "ddpg"])
def test_live_learning_rate_update_uses_proposed_base_during_warmup(kind: str) -> None:
    env = _make_env()
    try:
        algorithm = _make_algorithm(env, kind)
        algorithm.learning_rate_warmup_updates = 4
        algorithm.learning_rate_warmup_start_factor = 0.25
        algorithm.n_total_updates = 2
        algorithm.set_learning_rate(0.002)
        assert algorithm.learning_rate == 0.002
        for optimizer in (algorithm.actor_optimizer, algorithm.critic_optimizer):
            assert all(group["lr"] == 0.00125 for group in optimizer.param_groups)
        if isinstance(algorithm, SAC):
            assert all(group["lr"] == 0.002 for group in algorithm.ent_coef_optimizer.param_groups)
    finally:
        env.close()


def _world_model_config() -> NOPWorldModelConfig:
    return NOPWorldModelConfig(
        n_agents=2, local_latent_dim=8, action_dim=1, d_model_transition_model=8,
        nhead_transition_model=2, num_layers_transition_model=1, dim_feedforward_transition_model=16,
        next_obs_pred_config=NextObsPredConfig(local_scalar_target_indices=[0]),
    )


@pytest.mark.parametrize("failure", ["negative", "nonfinite", "unknown", "aliases", "disabled_module"])
def test_loss_weight_requests_are_atomic(algorithm: BaseAlgorithm, failure: str) -> None:
    if isinstance(algorithm, PPO):
        policy = NextObsPredWrapper(algorithm.policy, world_model_config=_world_model_config())
        names = ("scalar_loss_weight", "angle_loss_weight", "world_model_loss_coef")
        before = tuple(getattr(policy, name) for name in names)
        requests = {
            "negative": {"scalar": 2.0, "angle": -1.0},
            "nonfinite": {"scalar": 2.0, "angle": float("nan")},
            "unknown": {"scalar": 2.0, "unknown": 1.0},
            "aliases": {"scalar": 2.0, "angle": 0.5, "angle_loss_weight": 0.25},
            "disabled_module": {"scalar": 2.0, "actor_nop_loss_coef": 0.5},
        }
    else:
        policy = algorithm.policy
        before = policy.critic_nop.nop_loss_coef
        requests = {
            "negative": {"nop_loss_coef": 0.5, "critic_nop_loss_coef": -1.0},
            "nonfinite": {"nop_loss_coef": 0.5, "critic_nop_loss_coef": float("nan")},
            "unknown": {"nop_loss_coef": 0.5, "unknown": 1.0},
            "aliases": {"nop_loss_coef": 0.5, "critic_nop_loss_coef": 0.2, "critic_world_model_loss_coef": 0.3},
            "disabled_module": {"nop_loss_coef": 0.5, "actor_nop_loss_coef": 0.2},
        }
    with pytest.raises(ValueError):
        policy.update_loss_weights(**requests[failure])
    if isinstance(algorithm, PPO):
        assert tuple(getattr(policy, name) for name in names) == before
    else:
        assert policy.critic_nop.nop_loss_coef == before


@pytest.mark.parametrize("kind", ["sac", "td3", "ddpg"])
def test_resume_restores_distinct_actor_and_critic_nop_coefficients(kind: str, tmp_path: Path) -> None:
    env = _make_env()
    try:
        trainer = _make_algorithm(env, kind, nop_source="both")
        trainer.policy.update_loss_weights(actor_nop_loss_coef=0, critic_nop_loss_coef=0.25)
        path = tmp_path / "checkpoint.pt"
        trainer.save(path)
        trainer.policy.update_loss_weights(nop_loss_coef=0.75)
        trainer.load(path, restore_optimizer_state=False)
        assert trainer.policy.actor_nop.nop_loss_coef == 0
        assert trainer.policy.critic_nop.nop_loss_coef == 0.25
    finally:
        env.close()


def test_ppo_checkpoint_restores_target_weights_across_compile_wrappers(tmp_path: Path) -> None:
    env = _make_ppo_env()
    policy = NextObsPredWrapper(_make_policy(env), world_model_config=_world_model_config())
    trainer = _make_ppo(env, policy)
    saved = dict(world_model_loss_coef=0.0, scalar_loss_weight=1.1, angle_loss_weight=0.4,
                 rot6d_loss_weight=0.7, binary_loss_weight=1.25,
                 global_scalar_loss_weight=2.5, global_rot6d_loss_weight=0.3)
    policy.update_loss_weights(**saved)
    path = tmp_path / "checkpoint.pt"
    trainer.save(path)
    policy.update_loss_weights(**dict.fromkeys(saved, 1.0))
    trainer.policy = torch.compile(policy, backend="eager")
    trainer.load(path, restore_optimizer_state=False)
    for name, value in saved.items():
        assert getattr(policy, name) == value


@pytest.mark.parametrize("corruption", ["nonfinite", "missing", "unknown"])
def test_bad_checkpoint_loss_weights_leave_policy_and_counters_intact(tmp_path: Path, corruption: str) -> None:
    env = _make_ppo_env()
    policy = NextObsPredWrapper(_make_policy(env), world_model_config=_world_model_config())
    trainer = _make_ppo(env, policy)
    path = tmp_path / "checkpoint.pt"
    trainer.save(path)
    checkpoint = torch.load(path, weights_only=False)
    weights = checkpoint["policy_loss_weights"]
    if corruption == "nonfinite":
        weights["angle_loss_weight"] = float("nan")
    elif corruption == "missing":
        del weights["scalar_loss_weight"]
    else:
        weights["unknown"] = 0.1
    torch.save(checkpoint, path)
    policy.update_loss_weights(world_model_loss_coef=2)
    parameter = next(policy.parameters())
    with torch.no_grad():
        parameter.fill_(5)
    trainer.n_total_updates = 7
    with pytest.raises(ValueError, match="loss weight|loss_weight"):
        trainer.load(path)
    assert policy.world_model_loss_coef == 2
    assert (parameter == 5).all()
    assert trainer.n_total_updates == 7

    policy.register_buffer("new_feature", torch.ones(1))
    trainer.load(path, strict_load_state_dict=False)
    assert policy.world_model_loss_coef == 2
    assert not (parameter == 5).all()


@pytest.mark.parametrize("name,value", [("world_model_loss_coef", float("inf")), ("angle_loss_weight", -0.1)])
def test_ppo_nop_configuration_rejects_invalid_loss_weights(name: str, value: float) -> None:
    config = _world_model_config()
    if name == "world_model_loss_coef":
        config = replace(config, world_model_loss_coef=value)
    else:
        config = replace(config, next_obs_pred_config=replace(config.next_obs_pred_config, angle_loss_weight=value))
    with pytest.raises(ValueError, match=name):
        NextObsPredWrapper(_make_policy(_make_ppo_env()), world_model_config=config)
