from collections import defaultdict
from typing import Any, TypeVar, Generic

import torch

from swarmbots.learn.summary_statistics import SummaryStatistics, TensorSummaryStatistics, compute_summary_statistics

T = TypeVar('T')
MetricKey = TypeVar('MetricKey')


def _tensor_metric_groups(
    metrics: dict[MetricKey, Any],
) -> dict[tuple[torch.device, torch.dtype], list[tuple[MetricKey, Any, torch.Tensor]]]:
    tensor_groups: dict[tuple[torch.device, torch.dtype], list[tuple[MetricKey, Any, torch.Tensor]]] = {}
    for name, value in metrics.items():
        if isinstance(value, TensorSummaryStatistics):
            tensor = value.moments.detach()
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            tensor = value.detach().reshape(1)
        else:
            continue
        tensor_groups.setdefault((tensor.device, tensor.dtype), []).append((name, value, tensor))
    return tensor_groups


def materialize_scalar_metrics(metrics: dict[MetricKey, Any]) -> dict[MetricKey, Any]:
    tensor_groups = _tensor_metric_groups(metrics)
    if not tensor_groups:
        return metrics
    materialized = dict(metrics)
    for fields in tensor_groups.values():
        values = torch.cat([tensor for _, _, tensor in fields]).cpu().tolist()
        offset = 0
        for name, value, tensor in fields:
            materialized[name] = (
                value.materialize(values[offset:offset + tensor.numel()])
                if isinstance(value, TensorSummaryStatistics) else values[offset]
            )
            offset += tensor.numel()
    return materialized


class MetricsLists(Generic[T]):

    def __init__(self) -> None:
        self.lists: dict[str, list[T]] = defaultdict(list)

    def add(self, metrics: dict[str, T]) -> None:
        snapshots = dict(metrics)
        for fields in _tensor_metric_groups(metrics).values():
            # One packed copy snapshots every compiled output without launching
            # a separate clone kernel for each scalar diagnostic.
            packed = torch.cat([tensor for _, _, tensor in fields])
            offset = 0
            for name, value, tensor in fields:
                snapshot = packed[offset:offset + tensor.numel()]
                snapshots[name] = (
                    TensorSummaryStatistics(value.n, snapshot) if isinstance(value, TensorSummaryStatistics)
                    else snapshot.reshape(value.shape)
                )
                offset += tensor.numel()
        for name, value in snapshots.items():
            if isinstance(value, torch.Tensor) and value.numel() != 1:
                value = value.detach().clone()
            self.lists[name].append(value)

    def get(self) -> dict[str, list[T]]:
        return self.lists

    def compute_summary_statistics(
            self,
            find_min: bool = False,
            find_max: bool = False,
            make_histogram: bool | int = False,
            compute_skewness: bool = False,
            compute_kurtosis: bool = False,
            keep_data: bool = False,
            prefix: str | None = None
    ) -> dict[str, SummaryStatistics | None]:
        materialized = materialize_scalar_metrics({
            (name, index): value for name, values in self.lists.items() for index, value in enumerate(values)
        })
        return {
            k if prefix is None else prefix + k: compute_summary_statistics(
                [materialized[k, index] for index in range(len(v))],
                find_min=find_min, find_max=find_max,
                compute_skewness=compute_skewness, compute_kurtosis=compute_kurtosis,
                make_histogram=make_histogram, keep_data=keep_data,
            ) for k, v in self.lists.items()
        }
