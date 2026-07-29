import numpy as np
import pytest
import torch

from swarmbots.learn.rollout_utils import append_episode_infos


def test_append_episode_infos_copies_top_level_scenario_metadata() -> None:
    episode_infos: list[dict[str, object]] = []
    infos = {
        "episode": {
            "r": np.asarray([2.5, 7.5]),
            "l": torch.tensor([3, 9]),
            "_private": np.asarray([10, 20]),
        },
        "_episode": np.asarray([True, False]),
        "scenario_id": torch.tensor([4, 5]),
        "scenario_name": np.asarray(["wall", "payload"], dtype=object),
    }

    append_episode_infos(
        episode_infos=episode_infos,
        infos=infos,
        dones=torch.tensor([True, False]),
    )

    assert episode_infos == [{
        "r": 2.5,
        "l": 3,
        "scenario_id": 4,
        "scenario_name": "wall",
    }]


def test_append_episode_infos_reads_episode_stats_from_final_info() -> None:
    episode_infos: list[dict[str, object]] = []
    infos = {
        "final_info": {
            "episode": {
                "r": np.asarray([1.0, 8.0]),
            },
            "_episode": np.asarray([False, True]),
        },
        "scenario_id": np.asarray([0, 1]),
        "scenario_name": np.asarray(["wall", "payload"], dtype=object),
    }

    append_episode_infos(
        episode_infos=episode_infos,
        infos=infos,
        dones=torch.tensor([False, True]),
    )

    assert episode_infos == [{
        "r": 8.0,
        "scenario_id": 1,
        "scenario_name": "payload",
    }]


def test_append_episode_infos_prefers_scenario_metadata_from_final_info() -> None:
    episode_infos: list[dict[str, object]] = []
    infos = {
        "final_info": {
            "episode": {"r": np.asarray([3.0])},
            "_episode": np.asarray([True]),
            "scenario_id": np.asarray([7]),
            "scenario_name": np.asarray(["terminal-name"], dtype=object),
        },
        "scenario_id": np.asarray([1]),
        "scenario_name": np.asarray(["reset-name"], dtype=object),
    }

    append_episode_infos(
        episode_infos=episode_infos,
        infos=infos,
        dones=torch.tensor([True]),
    )

    assert episode_infos[0]["scenario_id"] == 7
    assert episode_infos[0]["scenario_name"] == "terminal-name"


def test_append_episode_infos_handles_multiple_completed_scenarios() -> None:
    episode_infos: list[dict[str, object]] = []
    infos = {
        "episode": {
            "r": np.asarray([1.0, 2.0, 3.0]),
            "success": np.asarray([True, False, True]),
        },
        "_episode": np.asarray([True, False, True]),
        "scenario_id": np.asarray([0, 0, 1]),
        "scenario_name": np.asarray(["wall", "wall", "payload"], dtype=object),
    }

    append_episode_infos(
        episode_infos=episode_infos,
        infos=infos,
        dones=torch.tensor([True, False, True]),
    )

    assert episode_infos == [
        {
            "r": 1.0,
            "success": True,
            "scenario_id": 0,
            "scenario_name": "wall",
        },
        {
            "r": 3.0,
            "success": True,
            "scenario_id": 1,
            "scenario_name": "payload",
        },
    ]


def test_append_episode_infos_omits_stats_masked_out_for_a_scenario() -> None:
    episode_infos: list[dict[str, object]] = []
    infos = {
        "episode": {
            "r": np.asarray([1.0, 2.0]),
            "success": np.asarray([True, False]),
            "_success": np.asarray([True, False]),
            "guidance_reward": np.asarray([0.0, 5.0]),
            "_guidance_reward": np.asarray([False, True]),
        },
        "_episode": np.asarray([True, True]),
        "scenario_name": np.asarray(["wall", "payload"], dtype=object),
    }

    append_episode_infos(
        episode_infos=episode_infos,
        infos=infos,
        dones=torch.tensor([True, True]),
    )

    assert episode_infos == [
        {
            "r": 1.0,
            "success": True,
            "scenario_name": "wall",
        },
        {
            "r": 2.0,
            "guidance_reward": 5.0,
            "scenario_name": "payload",
        },
    ]


def test_append_episode_infos_keeps_legacy_behavior_without_scenario_metadata() -> None:
    episode_infos: list[dict[str, object]] = []

    append_episode_infos(
        episode_infos=episode_infos,
        infos={
            "episode": {"r": np.asarray([4.0])},
            "_episode": np.asarray([True]),
        },
        dones=torch.tensor([True]),
    )

    assert episode_infos == [{"r": 4.0}]


def test_append_episode_infos_rejects_episode_mask_disagreeing_with_dones() -> None:
    with pytest.raises(ValueError, match="match computed dones"):
        append_episode_infos(
            episode_infos=[],
            infos={
                "episode": {"r": np.asarray([1.0, 2.0])},
                "_episode": np.asarray([True, False]),
                "scenario_id": np.asarray([0, 1]),
            },
            dones=torch.tensor([False, True]),
        )
