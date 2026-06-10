import unittest

import numpy as np
import torch

from swarmbots.learn.metrics_list import MetricsLists
from swarmbots.learn.summary_statistics import compute_summary_statistics


class SummaryStatisticsTests(unittest.TestCase):
    def test_compute_summary_statistics_does_not_keep_data_by_default(self) -> None:
        stats = compute_summary_statistics(torch.tensor([1.0, 2.0, 3.0]))

        assert stats is not None
        self.assertIsNone(stats.data)
        self.assertEqual(stats.n, 3)
        self.assertAlmostEqual(float(stats.mean), 2.0)

    def test_compute_summary_statistics_can_keep_data_when_requested(self) -> None:
        stats = compute_summary_statistics(torch.tensor([1.0, 2.0, 3.0]), keep_data=True)

        assert stats is not None
        self.assertIsNotNone(stats.data)
        self.assertEqual(stats.data.tolist(), [1.0, 2.0, 3.0])

    def test_histogram_does_not_require_retained_data(self) -> None:
        stats = compute_summary_statistics(torch.tensor([1.0, 2.0, 3.0]), make_histogram=3)

        assert stats is not None
        self.assertIsNone(stats.data)
        self.assertIsNotNone(stats.histogram)

    def test_torch_histogram_uses_tensor_values_without_retaining_data(self) -> None:
        stats = compute_summary_statistics(torch.tensor([0.0, 1.0, 2.0, 3.0]), make_histogram=2)

        assert stats is not None
        assert stats.histogram is not None
        self.assertIsNone(stats.data)
        self.assertEqual(stats.histogram.bin_frequencies, [0.5, 0.5])
        self.assertEqual(stats.histogram.bin_edges, [0.0, 1.5, 3.0])

    def test_std_matches_between_numpy_list_and_torch_inputs(self) -> None:
        values = [1.0, 2.0, 3.0]

        list_stats = compute_summary_statistics(values)
        numpy_stats = compute_summary_statistics(np.asarray(values))
        torch_stats = compute_summary_statistics(torch.tensor(values))

        assert list_stats is not None
        assert numpy_stats is not None
        assert torch_stats is not None
        self.assertAlmostEqual(float(list_stats.std), float(numpy_stats.std))
        self.assertAlmostEqual(float(list_stats.std), float(torch_stats.std))

    def test_single_value_torch_histogram_has_bounds(self) -> None:
        stats = compute_summary_statistics(torch.tensor([2.0]), make_histogram=3)

        assert stats is not None
        assert stats.histogram is not None
        self.assertEqual(stats.min_value, 2.0)
        self.assertEqual(stats.max_value, 2.0)
        self.assertEqual(stats.histogram.bin_frequencies, [1.0])

    def test_metrics_lists_does_not_keep_data_by_default(self) -> None:
        metrics = MetricsLists[float]()
        metrics.add({"loss": 1.0})
        metrics.add({"loss": 2.0})

        stats = metrics.compute_summary_statistics()["loss"]

        assert stats is not None
        self.assertIsNone(stats.data)
        self.assertEqual(stats.n, 2)


if __name__ == "__main__":
    unittest.main()
