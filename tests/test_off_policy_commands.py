from collections.abc import Iterator
from typing import Literal

import pytest
import torch

from swarmbots.learn.algos.off_policy.base_algorithm import OffPolicyAlgorithm
from swarmbots.learn.algos.off_policy.nop import SACNOPConfig
from swarmbots.learn.algos.sac import SAC
from swarmbots.learn.algos.td3 import DDPG, TD3
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.gsde_reset import GSDEEpisodeResetMode
from tests.test_deterministic_baselines import make_policy, nop_options
from tests.test_sac_algorithm import _make_policy


@pytest.fixture(scope="module", autouse=True)
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


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
        variant = "matd3__deepset_critic" if kind == "td3" else "maddpg__deepset_critic"
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
