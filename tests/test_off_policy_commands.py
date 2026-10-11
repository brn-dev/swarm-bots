from collections.abc import Iterator
from typing import Literal

import pytest
import torch

from swarmbots.learn.algos.off_policy import collect_off_policy_steps
from swarmbots.learn.algos.off_policy.base_algorithm import OffPolicyAlgorithm
from swarmbots.learn.algos.off_policy.nop import SACNOPConfig
from swarmbots.learn.algos.sac import SAC
from swarmbots.learn.algos.td3 import DDPG, TD3
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.gsde_reset import GSDEEpisodeResetMode
from tests.test_deterministic_baselines import make_policy, nop_options
from tests.test_sac_algorithm import _make_env, _make_policy


@pytest.fixture(scope="module", autouse=True)
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def env() -> Iterator[BaseLearnEnvWrapper]:
    instance = _make_env()
    yield instance
    instance.close()


@pytest.fixture(params=("sac", "td3", "ddpg"))
def algorithm_kind(request: pytest.FixtureRequest) -> Literal["sac", "td3", "ddpg"]:
    return request.param


def _make_algorithm(
    env: BaseLearnEnvWrapper,
    kind: Literal["sac", "td3", "ddpg"],
    *,
    independent_nop_sampling: bool = False,
) -> OffPolicyAlgorithm:
    nop_config = SACNOPConfig(
        enabled=True,
        num_next_steps=1,
        next_obs_pred_config=NextObsPredConfig(local_scalar_target_indices=[0]),
        transition_model_d_model=8,
        transition_model_nhead=2,
        transition_model_num_layers=1,
        transition_model_dim_feedforward=16,
    )
    if kind == "sac":
        policy = _make_policy(env, nop_config=nop_config)
        algorithm_class = SAC
    else:
        variant = "matd3_deepset" if kind == "td3" else "maddpg_deepset"
        policy = make_policy(env, variant, **nop_options(steps=1))
        algorithm_class = TD3 if kind == "td3" else DDPG
    return algorithm_class(
        policy,
        env,
        buffer_capacity_per_env=32,
        learning_starts=0,
        batch_size=2,
        learning_rate_warmup_updates=0,
        nop_batch_size=3 if independent_nop_sampling else None,
        independent_nop_sampling=independent_nop_sampling,
        train_device="cpu",
        replay_storage_device="cpu",
    )


def test_episode_gsde_reset_mode_serializes_explicitly() -> None:
    assert OffPolicyAlgorithm._serialize_gsde_reset_mode(GSDEEpisodeResetMode()) == {"mode": "episode"}


@pytest.mark.parametrize(
    "command,value,attribute",
    (
        ("set_gamma", "-1", "gamma"),
        ("set_gamma", "1.01", "gamma"),
        ("set_gamma", "nan", "gamma"),
        ("set_gamma", "invalid", "gamma"),
        ("set_tau", "0", "tau"),
        ("set_tau", "2", "tau"),
        ("set_tau", "inf", "tau"),
        ("set_tau", "nan", "tau"),
    ),
)
def test_rejected_commands_preserve_the_active_setting(
    env: BaseLearnEnvWrapper,
    algorithm_kind: Literal["sac", "td3", "ddpg"],
    command: str,
    value: str,
    attribute: str,
) -> None:
    algorithm = _make_algorithm(env, algorithm_kind)
    previous = getattr(algorithm, attribute)

    assert not algorithm.execute_command(command, value, None)

    assert getattr(algorithm, attribute) == previous
    algorithm._validate_hyper_parameters()


@pytest.mark.parametrize("command,value", (("set_gamma", "0"), ("set_gamma", "1"), ("set_tau", "1")))
def test_valid_command_boundaries_update_the_active_setting(
    env: BaseLearnEnvWrapper,
    algorithm_kind: Literal["sac", "td3", "ddpg"],
    command: str,
    value: str,
) -> None:
    algorithm = _make_algorithm(env, algorithm_kind)

    assert algorithm.execute_command(command, value, None)

    assert getattr(algorithm, command.removeprefix("set_")) == float(value)
    algorithm._validate_hyper_parameters()


@pytest.mark.parametrize("command", ("set_batch_size", "batch_size"))
@pytest.mark.parametrize("independent_nop_sampling", (False, True))
def test_batch_size_command_keeps_nop_sampling_and_metadata_consistent(
    env: BaseLearnEnvWrapper,
    algorithm_kind: Literal["sac", "td3", "ddpg"],
    command: str,
    independent_nop_sampling: bool,
) -> None:
    algorithm = _make_algorithm(env, algorithm_kind, independent_nop_sampling=independent_nop_sampling)

    assert algorithm.execute_command(command, "4", None)

    expected_nop_batch_size = 3 if independent_nop_sampling else 4
    assert algorithm.get_hyper_parameters()["nop_batch_size"] == expected_nop_batch_size
    algorithm._validate_hyper_parameters()
    collect_off_policy_steps(env, algorithm.replay_buffer, n_steps=4, policy=algorithm.policy, random_actions=True)
    batch, nop_batch, _reuse_latents = algorithm._sample_training_batches()
    assert batch.actions.shape[0] == 4
    assert nop_batch is not None
    assert nop_batch.actions.shape[0] == expected_nop_batch_size
