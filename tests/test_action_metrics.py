import math
import unittest

import torch

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliActionDist
from swarmbots.learn.action_dists.continuous_action_dist import ContinuousActionDist


class _ContinuousMetricsFixture:
    action_dim = 1
    log_stds = torch.tensor([math.log(2.0)])


class ActionMetricsTests(unittest.TestCase):
    def test_continuous_action_and_std_histograms_use_separate_ranges(self) -> None:
        metrics = ContinuousActionDist.get_metrics(
            _ContinuousMetricsFixture(),
            torch.zeros(4),
        )

        action_histogram = metrics["act"].histogram
        std_histogram = metrics["std"].histogram
        assert action_histogram is not None
        assert std_histogram is not None
        self.assertEqual(len(action_histogram.bin_frequencies), 11)
        self.assertEqual(action_histogram.bin_edges[0], -1.0)
        self.assertEqual(action_histogram.bin_edges[-1], 1.0)
        self.assertEqual(len(std_histogram.bin_frequencies), 1)
        self.assertGreater(std_histogram.bin_edges[0], 1.0)

    def test_bernoulli_actions_use_two_fixed_bins_over_zero_one(self) -> None:
        action_dist = BernoulliActionDist(latent_dim=2, action_dim=1)

        metrics = action_dist.get_metrics(torch.tensor([0.0, 1.0, 1.0, 0.0]))

        histogram = metrics["act"].histogram
        assert histogram is not None
        self.assertEqual(histogram.bin_edges, [0.0, 0.5, 1.0])
        self.assertEqual(histogram.bin_frequencies, [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
