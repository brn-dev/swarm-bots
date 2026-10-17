import unittest

import numpy as np
import torch

from swarmbots.learn.metrics_list import MetricsLists
from swarmbots.learn.summary_statistics import (
    HISTOGRAM_DEFAULT_BINS,
    NO_DATA,
    SummaryStatisticsFormat,
    format_summary_statistics,
    HistogramConfig,
    SummaryStatistics,
    combine_summary_statistics,
    compute_summary_statistics,
    maybe_compute_summary_statistics,
)


class SummaryStatisticsTests(unittest.TestCase):
    def test_integer_and_boolean_tensor_statistics_match_numpy(self) -> None:
        for values in (
            torch.tensor([True, False, True]),
            torch.tensor([1, 2, 4], dtype=torch.int16),
            torch.tensor([2**40, 2**40 + 1, 2**40 + 4]),
        ):
            with self.subTest(dtype=values.dtype):
                expected = compute_summary_statistics(
                    values.numpy().astype(np.float64), find_min=True, find_max=True, make_histogram=3,
                )
                actual = compute_summary_statistics(values, find_min=True, find_max=True, make_histogram=3)
                self.assertAlmostEqual(actual.mean, expected.mean)
                self.assertAlmostEqual(actual.std, expected.std)
                self.assertEqual(actual.min_value, expected.min_value)
                self.assertEqual(actual.max_value, expected.max_value)
                np.testing.assert_allclose(actual.histogram.bin_frequencies, expected.histogram.bin_frequencies)

    def test_combined_higher_moments_match_the_original_population_without_raw_samples(self) -> None:
        shards = [[1.0, 2.0, 4.0], [10.0, 10.0], [30.0]]
        stats = [compute_summary_statistics(values, compute_skewness=True, compute_kurtosis=True) for values in shards]
        combined = combine_summary_statistics(stats)
        expected = compute_summary_statistics(
            [value for shard in shards for value in shard], compute_skewness=True, compute_kurtosis=True,
        )
        self.assertAlmostEqual(combined.skewness, expected.skewness)
        self.assertAlmostEqual(combined.kurtosis, expected.kurtosis)
        self.assertIsNone(combined.data)

    def test_skewness_only_merge_is_scale_invariant_for_large_values(self) -> None:
        for shards in (([1.0], [2.0]), ([1.0, 2.0, 4.0], [10.0, 12.0, 19.0])):
            with self.subTest(shards=shards):
                population = np.concatenate(shards)
                expected_skewness = np.mean(((population - population.mean()) / population.std()) ** 3)
                stats = [
                    compute_summary_statistics(np.asarray(values) * 1e80, compute_skewness=True)
                    for values in shards
                ]
                combined = combine_summary_statistics(stats)
                self.assertAlmostEqual(combined.skewness, expected_skewness)
                self.assertIsNone(combined.kurtosis)
                self.assertIsNone(combined.data)

    def test_single_summary_preserves_independently_requested_higher_moments(self) -> None:
        for compute_skewness, compute_kurtosis in ((True, False), (False, True), (True, True)):
            original = compute_summary_statistics(
                [1.0, 2.0, 4.0], compute_skewness=compute_skewness, compute_kurtosis=compute_kurtosis,
            )
            combined = combine_summary_statistics([original])
            self.assertEqual(combined.skewness, original.skewness)
            self.assertEqual(combined.kurtosis, original.kurtosis)

    def test_combined_higher_moments_report_missing_inputs(self) -> None:
        known = compute_summary_statistics([1.0, 2.0, 4.0], compute_skewness=True, compute_kurtosis=True)
        missing = compute_summary_statistics([10.0, 12.0])
        combined = combine_summary_statistics([known, missing])
        self.assertIs(combined.skewness, NO_DATA)
        self.assertIs(combined.kurtosis, NO_DATA)

    def test_kurtosis_only_merge_preserves_equal_source_means(self) -> None:
        for shards in (([0.0, 0.2], [0.1]), ([0.1], [0.0, 0.2])):
            with self.subTest(shards=shards):
                stats = [compute_summary_statistics(values, compute_kurtosis=True) for values in shards]
                combined = combine_summary_statistics(stats)
                self.assertEqual(combined.mean, 0.1)
                self.assertAlmostEqual(combined.kurtosis, 1.5)
                self.assertIsNone(combined.skewness)

    def test_kurtosis_only_merge_requires_third_moments_for_different_means(self) -> None:
        stats = [
            compute_summary_statistics(values, compute_kurtosis=True)
            for values in ([0.0, 0.2], [0.4])
        ]
        self.assertIs(combine_summary_statistics(stats).kurtosis, NO_DATA)

    def test_extreme_aggregate_moments_remain_finite_or_unavailable(self) -> None:
        for scale, moment in ((1e80, "kurtosis"), (1e110, "skewness"), (1e-100, "kurtosis"), (1e-110, "skewness")):
            with self.subTest(scale=scale, moment=moment), np.errstate(all="raise"):
                stats = [
                    compute_summary_statistics([value * scale], **{f"compute_{moment}": True})
                    for value in (1.0, 2.0)
                ]
                result = getattr(combine_summary_statistics(stats), moment)
                self.assertTrue(result is NO_DATA or np.isfinite(result))

    def test_nonfinite_source_moments_are_unavailable(self) -> None:
        stats = [
            SummaryStatistics(n=2, mean=0.0, std=1.0, skewness=float("nan"), kurtosis=float("inf")),
            compute_summary_statistics([-1.0, 1.0], compute_skewness=True, compute_kurtosis=True),
        ]
        with np.errstate(all="raise"):
            combined = combine_summary_statistics(stats)
        self.assertIs(combined.skewness, NO_DATA)
        self.assertIs(combined.kurtosis, NO_DATA)
        self.assertEqual(combined.mean, 0.0)
        self.assertEqual(combined.std, 1.0)

    def test_deferred_bounds_are_available_to_console_formatting(self) -> None:
        stats = compute_summary_statistics([1.0, 2.0, 4.0], keep_data=True)
        formatted = format_summary_statistics(
            stats, SummaryStatisticsFormat(mean=".1f", min_value=".1f", max_value=".1f"),
        )
        self.assertEqual(formatted, "2.3 [1.0, 4.0]")
        self.assertEqual(stats.min_value, 1.0)
        self.assertEqual(stats.max_value, 4.0)

    def test_deferred_bounds_report_unavailable_samples(self) -> None:
        for values in ([], [1.0, 2.0]):
            stats = compute_summary_statistics(values)
            maybe_compute_summary_statistics(stats, find_min=True, find_max=True)
            self.assertIs(stats.min_value, NO_DATA)
            self.assertIs(stats.max_value, NO_DATA)

    def test_fixed_histogram_range_clamps_outliers_without_changing_summary_or_input(self) -> None:
        source = np.array([-2.0, 0.25, 0.75, 3.0])
        config = HistogramConfig(bins=2, low=0.0, high=1.0)
        for values in (source.copy(), torch.tensor(source)):
            stats = compute_summary_statistics(values, make_histogram=config, find_min=True, find_max=True, keep_data=True)

            self.assertEqual(stats.histogram.bin_frequencies, [0.5, 0.5])
            self.assertEqual(stats.histogram.bin_edges, [0.0, 0.5, 1.0])
            self.assertAlmostEqual(stats.mean, source.mean())
            self.assertAlmostEqual(stats.std, source.std())
            self.assertEqual(stats.min_value, -2.0)
            self.assertEqual(stats.max_value, 3.0)
            np.testing.assert_array_equal(stats.data, source)
            np.testing.assert_array_equal(np.asarray(values), source)

    def test_fixed_range_histograms_keep_outliers_when_merging_unequal_batches(self) -> None:
        config = HistogramConfig(bins=2, low=0.0, high=1.0)
        first_values = [-2.0, -2.0, -2.0, 0.25]
        second_values = [0.75]
        for convert in (np.asarray, torch.tensor):
            first = compute_summary_statistics(convert(first_values), make_histogram=config)
            second = compute_summary_statistics(convert(second_values), make_histogram=config)
            combined = combine_summary_statistics([first, second], combine_histograms=True)
            direct = compute_summary_statistics(convert(first_values + second_values), make_histogram=config)

            np.testing.assert_allclose(combined.histogram.bin_frequencies, [0.8, 0.2])
            np.testing.assert_allclose(combined.histogram.bin_frequencies, direct.histogram.bin_frequencies)

    def test_fixed_histogram_range_places_single_outlier_in_boundary_bin(self) -> None:
        config = HistogramConfig(bins=2, low=0.0, high=1.0)
        for value, expected in ((-2.0, [1.0, 0.0]), (3.0, [0.0, 1.0])):
            for values in (np.array([value]), torch.tensor([value])):
                stats = compute_summary_statistics(values, make_histogram=config)
                self.assertEqual(stats.histogram.bin_frequencies, expected)

    def test_single_sample_retains_data_for_deferred_histograms(self) -> None:
        for values in ([2.0], np.array([2.0]), torch.tensor([2.0])):
            stats = compute_summary_statistics(values, keep_data=True)
            self.assertEqual(stats.data.tolist(), [2.0])
            maybe_compute_summary_statistics(stats, make_histogram=True)
            self.assertIsNotNone(stats.histogram)
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

    def test_fixed_histogram_range_is_kept_for_constant_values(self) -> None:
        histogram_config = HistogramConfig(bins=2, low=0.0, high=1.0)

        stats = compute_summary_statistics(torch.zeros(4), make_histogram=histogram_config)

        assert stats is not None
        assert stats.histogram is not None
        self.assertEqual(stats.histogram.bin_edges, [0.0, 0.5, 1.0])
        self.assertEqual(stats.histogram.bin_frequencies, [1.0, 0.0])

    def test_histogram_config_requires_a_complete_valid_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "both be set"):
            HistogramConfig(bins=2, low=0.0)
        with self.assertRaisesRegex(ValueError, "less than high"):
            HistogramConfig(bins=2, low=1.0, high=1.0)

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

    def test_combined_empty_summary_statistics_preserve_requested_fields(self) -> None:
        first = compute_summary_statistics([], find_min=True, find_max=True, make_histogram=3)
        second = compute_summary_statistics([], find_min=True, find_max=True, make_histogram=3)

        assert first is not None
        assert second is not None
        combined = combine_summary_statistics([first, second], combine_histograms=True)

        self.assertEqual(combined.n, 0)
        self.assertIs(combined.mean, first.mean)
        self.assertIs(combined.std, first.std)
        self.assertIs(combined.min_value, first.min_value)
        self.assertIs(combined.max_value, first.max_value)
        self.assertIs(combined.histogram, first.histogram)

    def test_combined_histograms_are_weighted_by_sample_count(self) -> None:
        first = compute_summary_statistics(np.array([0.0, 1.0]), make_histogram=2)
        second = compute_summary_statistics(np.array([0.0, 1.0, 1.0, 1.0]), make_histogram=2)

        assert first is not None
        assert second is not None
        combined = combine_summary_statistics([first, second], combine_histograms=True)

        assert combined.histogram is not None
        self.assertEqual(combined.n, 6)
        self.assertAlmostEqual(sum(combined.histogram.bin_frequencies), 1.0)
        self.assertAlmostEqual(combined.histogram.bin_frequencies[0], 1.0 / 3.0)
        self.assertAlmostEqual(combined.histogram.bin_frequencies[1], 2.0 / 3.0)

    def test_fixed_range_histograms_combine_without_changing_edges(self) -> None:
        histogram_config = HistogramConfig(bins=2, low=0.0, high=1.0)
        first = compute_summary_statistics(np.array([0.0, 0.0]), make_histogram=histogram_config)
        second = compute_summary_statistics(np.array([1.0, 1.0]), make_histogram=histogram_config)

        assert first is not None
        assert second is not None
        combined = combine_summary_statistics([first, second], combine_histograms=True)

        assert combined.histogram is not None
        self.assertEqual(combined.histogram.bin_edges, [0.0, 0.5, 1.0])
        self.assertEqual(combined.histogram.bin_frequencies, [0.5, 0.5])

    def test_dynamic_histograms_with_different_edges_are_rebinned(self) -> None:
        first = compute_summary_statistics(np.array([0.0, 1.0]), make_histogram=2)
        second = compute_summary_statistics(np.array([2.0, 4.0]), make_histogram=2)

        assert first is not None
        assert second is not None
        combined = combine_summary_statistics([first, second], combine_histograms=True)

        assert combined.histogram is not None
        self.assertEqual(combined.histogram.bin_edges, [0.0, 2.0, 4.0])
        self.assertAlmostEqual(sum(combined.histogram.bin_frequencies), 1.0)


if __name__ == "__main__":
    unittest.main()
