from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pytest
import numpy as np
import torch
from torch import nn

from swarmbots.mj_env.quat_rot6d import quat_to_rot6d
from swarmbots.learn.masking import build_valid_mask, masked_mean, restrict_loss_agent_mask
from swarmbots.learn.polyak_update import polyak_update
from swarmbots.learn.serialization_utils import serialize_dataclass
from swarmbots.utils.recording_schedule import format_recording_percentage, install_scheduled_recordings
from swarmbots.utils.run_paths import get_run_id_from_checkpoint_path, make_run_dir


class _Mode(Enum):
    TRAIN = "train"


@dataclass
class _InnerConfig:
    width: int
    mode: _Mode


@dataclass
class _OuterConfig:
    inner: _InnerConfig
    dims: tuple[int, int]
    activation: type[nn.Module]
    labels: dict[int, _Mode]


class _RecordingAlgorithm:
    def __init__(self, *, n_total_timesteps: int) -> None:
        self.n_total_timesteps = int(n_total_timesteps)
        self.commands: list[tuple[str, str, Any]] = []

    def execute_command(self, command: str, payload: str, *, extra_run_metadata: Any) -> None:
        self.commands.append((command, payload, extra_run_metadata))


def test_build_valid_mask_combines_batch_agent_and_time_masks() -> None:
    agent_mask = torch.tensor([[True, False, True], [False, True, True]])
    time_mask = torch.tensor([[True, False], [True, True]])

    valid = build_valid_mask(
        base_shape=(2, 2, 3),
        device=torch.device("cpu"),
        agent_mask=agent_mask,
        time_mask=time_mask,
    )

    assert torch.equal(
        valid,
        torch.tensor(
            [
                [[True, False, True], [False, False, False]],
                [[False, True, True], [False, True, True]],
            ]
        ),
    )


def test_restrict_loss_agent_mask_broadcasts_and_restricts_masks() -> None:
    loss_agent_mask = torch.tensor([[True, False, True], [True, True, False]])
    agent_mask = torch.tensor(
        [
            [[True, True, False], [False, True, True]],
            [[True, False, True], [True, True, True]],
        ]
    )

    restricted = restrict_loss_agent_mask(
        base_shape=(2, 2, 3),
        loss_agent_mask=loss_agent_mask,
        agent_mask=agent_mask,
    )

    assert torch.equal(
        restricted,
        torch.tensor(
            [
                [[True, False, False], [False, False, True]],
                [[True, False, False], [True, True, False]],
            ]
        ),
    )


def test_masked_mean_ignores_invalid_entries_and_returns_zero_for_empty_mask() -> None:
    values = torch.tensor([1.0, 3.0, 100.0])

    assert torch.equal(masked_mean(values, torch.tensor([False, False, False])), torch.tensor(0.0))
    assert torch.equal(masked_mean(values, torch.tensor([True, True, False])), torch.tensor(2.0))

    with pytest.raises(ValueError, match="Expected valid mask shape"):
        masked_mean(values, torch.ones((3, 1), dtype=torch.bool))


def test_polyak_update_interpolates_parameters_and_copies_buffers() -> None:
    source = nn.BatchNorm1d(2)
    target = nn.BatchNorm1d(2)
    with torch.no_grad():
        source.weight.fill_(4.0)
        source.bias.fill_(6.0)
        source.running_mean.fill_(8.0)
        target.weight.fill_(2.0)
        target.bias.fill_(10.0)
        target.running_mean.fill_(1.0)

    polyak_update(source, target, tau=0.25)

    assert torch.equal(target.weight, torch.full_like(target.weight, 2.5))
    assert torch.equal(target.bias, torch.full_like(target.bias, 9.0))
    assert torch.equal(target.running_mean, source.running_mean)
    with pytest.raises(ValueError, match="must be in"):
        polyak_update(source, target, tau=1.01)


def test_scheduled_recordings_skip_past_and_duplicate_targets_then_fire_in_order() -> None:
    algorithm = _RecordingAlgorithm(n_total_timesteps=9)
    hook = install_scheduled_recordings(
        algorithm=algorithm,
        total_timesteps=20,
        schedule={25.0: 2, 50.0: 3, 50.4: 4, 100.0: 5},
    )

    hook(algorithm, {}, 1)
    assert algorithm.commands == []

    algorithm.n_total_timesteps = 11
    hook(algorithm, {}, 1)
    assert [command for command, _payload, _metadata in algorithm.commands] == ["record"]
    assert '"episodes": 3' in algorithm.commands[0][1]
    assert "record_050pct_11_steps" in algorithm.commands[0][1]

    algorithm.n_total_timesteps = 20
    hook(algorithm, {}, 1)
    assert len(algorithm.commands) == 2
    assert '"episodes": 5' in algorithm.commands[1][1]


def test_scheduled_recordings_zero_percent_fires_after_first_training_step() -> None:
    algorithm = _RecordingAlgorithm(n_total_timesteps=0)
    hook = install_scheduled_recordings(
        algorithm=algorithm,
        total_timesteps=100,
        schedule={0.0: 2},
    )

    hook(algorithm, {}, 1)
    assert algorithm.commands == []

    algorithm.n_total_timesteps = 1
    hook(algorithm, {}, 1)

    assert len(algorithm.commands) == 1
    assert '"episodes": 2' in algorithm.commands[0][1]
    assert "record_000pct_1_steps" in algorithm.commands[0][1]


def test_scheduled_recordings_reject_invalid_training_plan() -> None:
    algorithm = _RecordingAlgorithm(n_total_timesteps=0)

    with pytest.raises(ValueError, match="total_timesteps must be positive"):
        install_scheduled_recordings(algorithm=algorithm, total_timesteps=0, schedule={50.0: 1})

    with pytest.raises(ValueError, match=r"percentage must be in \[0, 100\]"):
        install_scheduled_recordings(algorithm=algorithm, total_timesteps=100, schedule={101.0: 1})

    with pytest.raises(ValueError, match="episode count must be positive"):
        install_scheduled_recordings(algorithm=algorithm, total_timesteps=100, schedule={50.0: 0})


def test_format_recording_percentage_is_stable_for_integer_and_fractional_values() -> None:
    assert format_recording_percentage(5.0) == "005"
    assert format_recording_percentage(12.5) == "12p5"


def test_make_run_dir_keeps_runs_under_repo_run_tree() -> None:
    run_dir = make_run_dir("group", "run-1")

    assert run_dir.parts[-3:] == ("runs", "group", "run-1")

    with pytest.raises(ValueError, match="run_group"):
        make_run_dir("../outside", "run-1")

    with pytest.raises(ValueError, match="run_id"):
        make_run_dir("group", "/tmp/run-1")


def test_get_run_id_from_checkpoint_path_uses_run_directory_name() -> None:
    checkpoint_path = Path("runs") / "group" / "2026-06-17_12-00-00" / "models" / "model.pt"

    assert get_run_id_from_checkpoint_path(checkpoint_path) == "2026-06-17_12-00-00"


def test_serialize_dataclass_recurses_into_enums_types_tuples_and_dict_keys() -> None:
    serialized = serialize_dataclass(
        _OuterConfig(
            inner=_InnerConfig(width=32, mode=_Mode.TRAIN),
            dims=(1, 2),
            activation=nn.ReLU,
            labels={1: _Mode.TRAIN},
        )
    )

    assert serialized == {
        "inner": {"width": 32, "mode": "TRAIN"},
        "dims": [1, 2],
        "activation": "torch.nn.modules.activation.ReLU",
        "labels": {"1": "TRAIN"},
    }


def test_quat_to_rot6d_handles_arbitrary_axis_and_is_sign_invariant() -> None:
    quats = np.asarray(
        [
            [[1.0, -1.0], [0.0, -0.0], [0.0, -0.0], [0.0, -0.0]],
            [[0.0, -0.0], [0.0, -0.0], [1.0, -1.0], [0.0, -0.0]],
        ],
        dtype=np.float32,
    )

    rot6d = quat_to_rot6d(quats, axis=1)

    assert rot6d.shape == (2, 6, 2)
    np.testing.assert_allclose(rot6d[0, :, 0], rot6d[0, :, 1], atol=1e-7)
    np.testing.assert_allclose(rot6d[0, :, 0], np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    np.testing.assert_allclose(rot6d[1, :, 0], rot6d[1, :, 1], atol=1e-7)

    with pytest.raises(ValueError, match="Expected quaternion size 4"):
        quat_to_rot6d(np.zeros((3, 3), dtype=np.float32), axis=1)
