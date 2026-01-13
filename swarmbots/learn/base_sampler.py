import abc
from typing import TypeVar, Generic, Generator, Any

import torch

SamplesType = TypeVar('SamplesType')

class BaseSampler(Generic[SamplesType], abc.ABC):

    def __init__(self, n_samples: int):
        self.n_samples = n_samples

    @abc.abstractmethod
    def _fetch_samples(self, batch_indices: torch.Tensor) -> SamplesType:
        raise NotImplementedError()

    def sample(self, batch_size: int, drop_last: bool = True) -> Generator[SamplesType, Any, None]:
        indices = torch.randperm(self.n_samples)

        for start_idx in range(0, self.n_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]

            if drop_last and len(batch_indices) < batch_size:
                continue

            yield self._fetch_samples(batch_indices)