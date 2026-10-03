import unittest

import torch

from swarmbots.learn.base_sampler import BaseSampler, BaseSamplerConfig, BatchIndices


class _TensorSampler(BaseSampler[torch.Tensor, BaseSamplerConfig]):
    def __init__(self, *, config: BaseSamplerConfig, n_samples: int):
        self.values = torch.arange(n_samples)
        super().__init__(config=config, n_samples=n_samples)

    def _fetch_samples(self, batch_indices: BatchIndices) -> torch.Tensor:
        return self.values[batch_indices]


class BaseSamplerTests(unittest.TestCase):
    def test_rejects_non_positive_batch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size must be > 0"):
            _TensorSampler(config=BaseSamplerConfig(batch_size=0), n_samples=5)

    def test_full_batch_yields_all_samples_once(self) -> None:
        sampler = _TensorSampler(config=BaseSamplerConfig(batch_size=5), n_samples=5)

        batches = list(sampler.sample())

        self.assertEqual(len(batches), 1)
        torch.testing.assert_close(batches[0].sort().values, sampler.values)

    def test_mini_batches_drop_incomplete_batch_by_default(self) -> None:
        sampler = _TensorSampler(config=BaseSamplerConfig(batch_size=2), n_samples=5)

        batches = list(sampler.sample())

        self.assertEqual(len(batches), 2)
        self.assertTrue(all(len(batch) == 2 for batch in batches))
        sampled_values = torch.cat(batches)
        self.assertEqual(len(sampled_values.unique()), 4)
        self.assertTrue(torch.isin(sampled_values, sampler.values).all())

    def test_drop_last_false_keeps_incomplete_final_batch(self) -> None:
        sampler = _TensorSampler(config=BaseSamplerConfig(batch_size=2), n_samples=5)

        batches = list(sampler.sample(drop_last=False))

        self.assertEqual(len(batches), 3)
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        torch.testing.assert_close(torch.cat(batches).sort().values, sampler.values)


if __name__ == "__main__":
    unittest.main()
