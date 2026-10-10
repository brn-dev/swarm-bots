import pytest
import torch

from swarmbots.learn.action_dists.predicted_std_gaussian_action_dist import PredictedStdGaussianActionDist


def test_runtime_std_survives_checkpoint_and_loads_old_checkpoints():
    source = PredictedStdGaussianActionDist(2, 1, base_std=0.2)
    source.set_std(0.4)
    source.scale_std(2.0)
    restored = PredictedStdGaussianActionDist(2, 1, base_std=0.2)
    restored.load_state_dict(source.state_dict())
    source.update_latent_features(torch.zeros(1, 2))
    restored.update_latent_features(torch.zeros(1, 2))
    assert restored.distribution.stddev.item() == pytest.approx(0.8)
    torch.testing.assert_close(source.distribution.stddev, restored.distribution.stddev)
    old_state = source.state_dict()
    del old_state["base_log_std"]
    old_restored = PredictedStdGaussianActionDist(2, 1, base_std=0.2)
    old_restored.load_state_dict(old_state)
    old_restored.update_latent_features(torch.zeros(1, 2))
    assert old_restored.distribution.stddev.item() == pytest.approx(0.2)
