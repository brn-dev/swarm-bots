from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.torch_record_episode_statistics_wrapper import TorchRecordEpisodeStatisticsWrapper
from swarmbots.learn.env_wrappers.torch_task_ep_stats_wrapper import TorchTaskEpisodeStatsWrapper
from swarmbots.learn.testing_env import TestingSwarmBotsEnv as SwarmTestEnv


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
            ],
        ),
    ]
)
def base_env(request: pytest.FixtureRequest) -> Iterator[SwarmBotsLearnEnvWrapper]:
    env = SwarmBotsLearnEnvWrapper(
        SyncVectorEnv(
            [lambda: SwarmTestEnv(2, 3, 1, 1, 1, max_steps=10) for _ in range(3)],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
    )
    env.set_device(request.param)
    try:
        yield env
    finally:
        env.close()


def test_episode_statistics_survive_partial_reset_after_simultaneous_completions(
    base_env: SwarmBotsLearnEnvWrapper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = TorchTaskEpisodeStatsWrapper(TorchRecordEpisodeStatisticsWrapper(base_env))
    observations, _ = env.reset()
    actions = torch.zeros((3, 2, 2), device=env.device)
    rewards = torch.tensor([1.0, 2.0, 3.0], device=env.device)
    progress = torch.tensor([0.1, 0.2, 0.3], device=env.device)
    terminations = torch.tensor([True, True, False], device=env.device)
    truncations = torch.zeros(3, device=env.device, dtype=torch.bool)

    def step(actions: torch.Tensor) -> tuple[Any, ...]:
        return observations, rewards, terminations, truncations, {"progress_reward": progress}

    monkeypatch.setattr(base_env, "step", step)
    _, _, _, _, first_infos = env.step(actions)
    torch.testing.assert_close(
        first_infos["episode"]["r"], torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64, device=env.device)
    )
    torch.testing.assert_close(first_infos["episode"]["l"], torch.tensor([1, 1, 0], device=env.device))
    env.reset(options={"reset_mask": np.array([True, False, False])})

    terminations.zero_()
    _, _, _, _, ordinary_infos = env.step(actions)
    assert "episode" not in ordinary_infos
    assert "_episode" not in ordinary_infos
    terminations.fill_(True)
    _, _, _, _, final_infos = env.step(actions)
    torch.testing.assert_close(
        final_infos["episode"]["r"], torch.tensor([2.0, 4.0, 9.0], dtype=torch.float64, device=env.device)
    )
    torch.testing.assert_close(final_infos["episode"]["l"], torch.tensor([2, 2, 3], device=env.device))
    torch.testing.assert_close(
        final_infos["episode"]["progress_reward"], torch.tensor([0.2, 0.4, 0.9], dtype=torch.float64, device=env.device)
    )
    torch.testing.assert_close(
        first_infos["episode"]["r"], torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64, device=env.device)
    )


@pytest.mark.parametrize("present_mask", [[True, False, True], [False, False, False]])
def test_task_statistics_exclude_absent_nonfinite_values(
    base_env: SwarmBotsLearnEnvWrapper,
    monkeypatch: pytest.MonkeyPatch,
    present_mask: list[bool],
) -> None:
    env = TorchTaskEpisodeStatsWrapper(base_env)
    observations, _ = env.reset()
    actions = torch.zeros((3, 2, 2), device=env.device)
    mask = torch.tensor(present_mask, device=env.device)
    values = torch.tensor([2.0, torch.nan, 4.0], device=env.device).masked_fill(~mask, torch.nan)

    def step(actions: torch.Tensor) -> tuple[Any, ...]:
        return (
            observations,
            torch.ones(3, device=env.device),
            torch.ones(3, device=env.device, dtype=torch.bool),
            torch.zeros(3, device=env.device, dtype=torch.bool),
            {
                "progress_reward": values,
                "_progress_reward": mask,
                "success": values,
                "_success": mask,
            },
        )

    monkeypatch.setattr(base_env, "step", step)
    _, _, _, _, infos = env.step(actions)
    expected = torch.where(mask, values, 0.0).to(torch.float64)
    torch.testing.assert_close(infos["episode"]["progress_reward"], expected)
    torch.testing.assert_close(infos["episode"]["success"], expected)
    assert torch.equal(infos["episode"]["_success"], mask)
    assert infos["progress_reward"] is values
    assert torch.equal(infos["_progress_reward"], mask)


def test_task_statistics_merge_final_values_without_mutating_step_values(
    base_env: SwarmBotsLearnEnvWrapper,
) -> None:
    env = TorchTaskEpisodeStatsWrapper(base_env)
    env.reset()
    step_values = torch.tensor([10.0, torch.nan, 30.0], device=env.device)
    final_values = torch.tensor([100.0, 200.0, torch.nan], device=env.device)
    infos = {
        "progress_reward": step_values,
        "_progress_reward": torch.tensor([True, False, True], device=env.device),
        "final_info": {
            "progress_reward": final_values,
            "_progress_reward": torch.tensor([True, True, False], device=env.device),
        },
        "_final_info": torch.tensor([False, True, True], device=env.device),
    }
    values, mask = env._extract_step_values_with_mask(infos, "progress_reward")
    torch.testing.assert_close(values, torch.tensor([10.0, 200.0, 30.0], dtype=torch.float64, device=env.device))
    assert torch.equal(mask, torch.ones(3, device=env.device, dtype=torch.bool))
    torch.testing.assert_close(step_values, torch.tensor([10.0, torch.nan, 30.0], device=env.device), equal_nan=True)
    torch.testing.assert_close(final_values, torch.tensor([100.0, 200.0, torch.nan], device=env.device), equal_nan=True)
