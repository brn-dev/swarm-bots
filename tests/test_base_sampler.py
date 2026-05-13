import unittest
from unittest.mock import patch

import torch

from swarmbots.learn.base_sampler import BaseSampler, BaseSamplerConfig, BatchIndices


class _TensorSampler(BaseSampler[torch.Tensor, BaseSamplerConfig]):
    def __init__(self, *, config: BaseSamplerConfig, n_samples: int):
        self.values = torch.arange(n_samples)
        super().__init__(config=config, n_samples=n_samples)

    def _fetch_samples(self, batch_indices: BatchIndices) -> torch.Tensor:
        return self.values[batch_indices]


class BaseSamplerTests(unittest.TestCase):
    def test_full_batch_does_not_permute_indices(self) -> None:
        sampler = _TensorSampler(config=BaseSamplerConfig(batch_size=5), n_samples=5)

        with patch("torch.randperm", side_effect=AssertionError("randperm should not be called")):
            batches = list(sampler.sample())

        self.assertEqual(len(batches), 1)
        torch.testing.assert_close(batches[0], sampler.values)


if __name__ == "__main__":
    unittest.main()
