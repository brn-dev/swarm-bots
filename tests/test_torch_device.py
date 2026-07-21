from unittest.mock import patch

import torch

from swarmbots.learn.torch_device import as_device


def test_bare_cuda_string_resolves_to_current_indexed_device() -> None:
    with patch.object(torch.cuda, "current_device", return_value=3) as current_device:
        assert as_device("cuda") == torch.device("cuda:3")

    current_device.assert_called_once_with()


def test_bare_cuda_device_resolves_to_current_indexed_device() -> None:
    with patch.object(torch.cuda, "current_device", return_value=2):
        assert as_device(torch.device("cuda")) == torch.device("cuda:2")


def test_explicit_cuda_index_is_preserved() -> None:
    with patch.object(torch.cuda, "current_device") as current_device:
        assert as_device("cuda:1") == torch.device("cuda:1")

    current_device.assert_not_called()


def test_auto_uses_current_indexed_cuda_device_when_available() -> None:
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "current_device", return_value=4),
    ):
        assert as_device("auto") == torch.device("cuda:4")


def test_auto_and_indexed_cpu_resolve_to_canonical_cpu_device() -> None:
    with patch.object(torch.cuda, "is_available", return_value=False):
        assert as_device("auto") == torch.device("cpu")

    assert as_device(torch.device("cpu:0")) == torch.device("cpu")
