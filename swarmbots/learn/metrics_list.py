from collections import defaultdict
from typing import Any, TypeVar, Generic

import torch

from swarmbots.learn.summary_statistics import SummaryStatistics, compute_summary_statistics

T = TypeVar('T')


def materialize_scalar_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    tensor_metrics = {name: value for name, value in metrics.items() if isinstance(value, torch.Tensor)}
    if not tensor_metrics:
        return metrics
    values = torch.stack([value.detach().reshape(()) for value in tensor_metrics.values()]).cpu().tolist()
    return {**metrics, **dict(zip(tensor_metrics, values, strict=True))}


class MetricsLists(Generic[T]):

    def __init__(self) -> None:
        self.lists: dict[str, list[T]] = defaultdict(list)

    def add(self, metrics: dict[str, T]) -> None:
        for name, value in metrics.items():
            # Compiled updates can reuse scalar output storage before aggregation.
            self.lists[name].append(value.detach().clone() if isinstance(value, torch.Tensor) else value)

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

        return {
            k if prefix is None else prefix + k: compute_summary_statistics(
                v,
                find_min=find_min, find_max=find_max,
                compute_skewness=compute_skewness, compute_kurtosis=compute_kurtosis,
                make_histogram=make_histogram, keep_data=keep_data,
            ) for k, v in self.lists.items()
        }
