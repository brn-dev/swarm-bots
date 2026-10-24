from collections.abc import Iterator

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


@pytest.mark.parametrize("value", ["null", "0", "true", "[]", "{}", '"0.001"', "-0.001", "NaN", "Infinity"])
def test_rejected_learning_rate_command_preserves_base_and_optimizer_rates(
    algorithm: BaseAlgorithm, value: str,
) -> None:
    before = algorithm.learning_rate, _optimizer_rates(algorithm)
    assert not algorithm.execute_command("set_lr", value, None)
    assert (algorithm.learning_rate, _optimizer_rates(algorithm)) == before


@pytest.mark.parametrize("learning_rate", [0.0, 0.002])
def test_fixed_learning_rate_updates_keep_base_and_optimizers_consistent(
    algorithm: BaseAlgorithm, learning_rate: float,
) -> None:
    assert algorithm.execute_command("set_lr", str(learning_rate), None)
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


def _nop_wrapper(algorithm: PPO) -> NextObsPredWrapper:
    return NextObsPredWrapper(algorithm.policy, world_model_config=NOPWorldModelConfig(
        n_agents=2, local_latent_dim=8, action_dim=1, d_model_transition_model=8,
        nhead_transition_model=2, num_layers_transition_model=1, dim_feedforward_transition_model=16,
        next_obs_pred_config=NextObsPredConfig(local_scalar_target_indices=[0]),
    ))


@pytest.mark.parametrize("failure", ["negative", "nonfinite", "unknown", "aliases", "disabled_module"])
def test_loss_weight_requests_are_atomic(algorithm: BaseAlgorithm, failure: str) -> None:
    if isinstance(algorithm, PPO):
        policy = _nop_wrapper(algorithm)
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


def test_rejected_ppo_loss_weight_command_preserves_weights() -> None:
    env = _make_ppo_env()
    base = _make_ppo(env, _make_policy(env))
    wrapper = _nop_wrapper(base)
    algorithm = _make_ppo(env, wrapper)
    before = wrapper.scalar_loss_weight, wrapper.angle_loss_weight
    assert not algorithm.execute_command("set_extra_loss_weights", '{"scalar":2.0,"angle":-1.0}', None)
    assert (wrapper.scalar_loss_weight, wrapper.angle_loss_weight) == before
    assert algorithm.execute_command("set_extra_loss_weights", '{"scalar":2.0,"angle":0.5}', None)
    assert (wrapper.scalar_loss_weight, wrapper.angle_loss_weight) == (2.0, 0.5)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-1"])
def test_rejected_entropy_command_preserves_automatic_temperature(value: str) -> None:
    env = _make_env()
    try:
        algorithm = _make_algorithm(env, "sac")
        before = algorithm.ent_coef, algorithm.log_ent_coef, algorithm.ent_coef_optimizer
        assert before[2] is not None
        assert not algorithm.execute_command("set_ent_coef", value, None)
        assert algorithm.ent_coef == before[0]
        assert algorithm.log_ent_coef is before[1]
        assert algorithm.ent_coef_optimizer is before[2]
        assert algorithm.ent_coef_tensor is None
        assert algorithm.execute_command("set_ent_coef", "0", None)
        assert algorithm.ent_coef_tensor.item() == 0.0
        assert algorithm.ent_coef_optimizer is None
    finally:
        env.close()
