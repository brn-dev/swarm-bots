import unittest

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
