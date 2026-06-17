import unittest

import numpy as np
import torch

from swarmbots.learn.metrics_list import MetricsLists
from swarmbots.learn.summary_statistics import (
    HISTOGRAM_DEFAULT_BINS,
    SummaryStatistics,
    combine_summary_statistics,
    compute_summary_statistics,
    maybe_compute_summary_statistics,
)


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

    def test_default_histogram_request_on_retained_stats_uses_default_bin_count(self) -> None:
        stats = compute_summary_statistics(np.arange(10.0), keep_data=True)

        assert stats is not None
        maybe_compute_summary_statistics(stats, make_histogram=True)

        assert stats.histogram is not None
        self.assertEqual(len(stats.histogram.bin_frequencies), HISTOGRAM_DEFAULT_BINS)

    def test_explicit_histogram_request_on_retained_stats_uses_requested_bin_count(self) -> None:
        stats = compute_summary_statistics(np.arange(10.0), keep_data=True)

        assert stats is not None
        maybe_compute_summary_statistics(stats, make_histogram=4)

        assert stats.histogram is not None
        self.assertEqual(len(stats.histogram.bin_frequencies), 4)

    def test_combined_summary_statistics_match_original_population(self) -> None:
        first = compute_summary_statistics(np.array([1.0, 2.0, 3.0]), find_min=True, find_max=True)
        second = compute_summary_statistics(np.array([10.0, 12.0]), find_min=True, find_max=True)
        expected = compute_summary_statistics(np.array([1.0, 2.0, 3.0, 10.0, 12.0]), find_min=True, find_max=True)

        assert first is not None
        assert second is not None
        assert expected is not None
        combined = combine_summary_statistics([first, second])

        self.assertEqual(combined.n, expected.n)
        self.assertAlmostEqual(float(combined.mean), float(expected.mean))
        self.assertAlmostEqual(float(combined.std), float(expected.std))
        self.assertEqual(combined.min_value, expected.min_value)
        self.assertEqual(combined.max_value, expected.max_value)

    def test_combined_summary_statistics_ignore_empty_shards(self) -> None:
        empty = compute_summary_statistics([], find_min=True, find_max=True)
        non_empty = compute_summary_statistics(np.array([3.0, 5.0]), find_min=True, find_max=True)

        assert empty is not None
        assert non_empty is not None
        combined = combine_summary_statistics([empty, non_empty])

        self.assertEqual(combined.n, 2)
        self.assertAlmostEqual(float(combined.mean), 4.0)
        self.assertAlmostEqual(float(combined.std), 1.0)
        self.assertEqual(combined.min_value, 3.0)
        self.assertEqual(combined.max_value, 5.0)

    def test_combined_summary_statistics_only_report_bounds_when_all_non_empty_shards_have_them(self) -> None:
        with_bounds = compute_summary_statistics(np.array([1.0, 2.0]), find_min=True, find_max=True)
        without_bounds = SummaryStatistics(n=1, mean=10.0, std=0.0)

        assert with_bounds is not None
        combined = combine_summary_statistics([with_bounds, without_bounds])

        self.assertIsNone(combined.min_value)
        self.assertIsNone(combined.max_value)


if __name__ == "__main__":
    unittest.main()
