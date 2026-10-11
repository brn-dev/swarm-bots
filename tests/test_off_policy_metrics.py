from collections.abc import Iterator
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from swarmbots.learn.algos.off_policy.base_algorithm import OffPolicyAlgorithm, TrainStepResult
from swarmbots.learn.metrics_list import MetricsLists
from swarmbots.learn.metrics_logger import MetricsLogger
from swarmbots.learn.torch_device import as_device
from tests.test_sac_nop import _make_flat_batch, _make_module, _move_batch_to_cuda, _nop_config


@pytest.fixture(scope="module", autouse=True)
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=(
    "cpu",
    pytest.param(
        "cuda",
        marks=(pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
    ),
))
def device(request: pytest.FixtureRequest) -> torch.device:
    return as_device(request.param)


def test_nop_metrics_are_detached_scalars_and_materialize_as_one_batch(device: torch.device) -> None:
    config = _nop_config(nop_loss_coef=0.25)
    weights = {
        "scalar": 0.5,
        "angle": 0.75,
        "rot6d": 1.25,
        "binary": 1.5,
        "global_scalar": 1.75,
        "global_rot6d": 2.0,
    }
    config = replace(config, next_obs_pred_config=replace(
        config.next_obs_pred_config,
        **{f"{name}_loss_weight": weight for name, weight in weights.items()},
    ))
    module = _make_module(config=config).to(device).eval()
    batch = _make_flat_batch()
    if device.type == "cuda":
        batch = _move_batch_to_cuda(batch)
    source_latents = torch.randn(4, 3, 6, device=device, requires_grad=True)
    results: list[TrainStepResult] = []
    metric_snapshots: list[dict[str, torch.Tensor]] = []
    losses: list[torch.Tensor] = []

    with (
        patch.object(torch.Tensor, "item", side_effect=AssertionError("metrics must remain on the device")),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("metrics must remain on the device")),
    ):
        for offset in (0.0, 0.5):
            loss, metrics = module.compute_loss(source_latents=source_latents + offset, batch=batch)
            losses.append(loss)
            metric_snapshots.append(metrics)
            results.append(OffPolicyAlgorithm._preserve_train_step_result((metrics, 0.0, 0.0)))

    for loss, metrics in zip(losses, metric_snapshots, strict=True):
        assert loss.requires_grad
        assert len(metrics) == 2 * len(weights) + 2
        for value in metrics.values():
            assert value.ndim == 0 and value.device == device
            assert not value.requires_grad and value.grad_fn is None
        for name, weight in weights.items():
            torch.testing.assert_close(metrics[f"test_nop_{name}_loss_scaled"], metrics[f"test_nop_{name}_loss"] * weight)
        total_component_loss = torch.stack([metrics[f"test_nop_{name}_loss_scaled"] for name in weights]).sum()
        torch.testing.assert_close(metrics["test_nop_loss"], total_component_loss)
        torch.testing.assert_close(metrics["test_nop_loss_scaled"], loss.detach())

    cpu_transfers: list[int] = []
    tensor_cpu = torch.Tensor.cpu

    def record_cpu_transfer(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        cpu_transfers.append(tensor.numel())
        return tensor_cpu(tensor, *args, **kwargs)

    with patch.object(torch.Tensor, "cpu", new=record_cpu_transfer):
        materialized = OffPolicyAlgorithm._materialize_train_step_results(results)

    assert cpu_transfers == [sum(len(metrics) for metrics in metric_snapshots)]
    for (metrics, _actor_norm, _critic_norm), expected in zip(materialized, metric_snapshots, strict=True):
        assert metrics.keys() == expected.keys()
        for name, value in metrics.items():
            assert isinstance(value, float)
            assert value == expected[name].item()
    sum(losses).backward()
    assert source_latents.grad is not None
    assert torch.isfinite(source_latents.grad).all()


def test_metric_lists_defer_tensor_values_and_keep_summary_logger_semantics(device: torch.device) -> None:
    metrics = MetricsLists[float | torch.Tensor]()
    value = torch.tensor(1.0, device=device, requires_grad=True).square()
    with (
        patch.object(torch.Tensor, "item", side_effect=AssertionError("collection must not materialize scalars")),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("collection must not transfer tensors")),
    ):
        metrics.add({"nop_loss": value})
        metrics.add({"nop_loss": 3.0})
        with torch.no_grad():
            value.fill_(10.0)

    snapshot = metrics.get()["nop_loss"][0]
    assert isinstance(snapshot, torch.Tensor)
    assert not snapshot.requires_grad and snapshot.grad_fn is None
    assert snapshot.device == device

    statistics = metrics.compute_summary_statistics(find_min=True, find_max=True)
    stats = statistics["nop_loss"]
    assert stats.n == 2 and stats.mean == 2.0 and stats.std == 1.0
    assert stats.min_value == 1.0 and stats.max_value == 3.0
    assert stats.data is None
    logged = MetricsLogger._aggregate_buffered_metrics([statistics, statistics])["nop_loss"]
    assert logged.n == 4 and logged.mean == 2.0 and logged.std == 1.0
