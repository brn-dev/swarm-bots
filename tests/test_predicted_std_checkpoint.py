import pytest
import torch

from swarmbots.learn.action_dists.predicted_std_gaussian_action_dist import PredictedStdGaussianActionDist


def test_runtime_std_survives_checkpoint() -> None:
    source = PredictedStdGaussianActionDist(2, 1, base_std=0.2)
    source.set_std(0.4)
    source.scale_std(2.0)
    restored = PredictedStdGaussianActionDist(2, 1, base_std=0.2)
    restored.load_state_dict(source.state_dict())
    source.update_latent_features(torch.zeros(1, 2))
    restored.update_latent_features(torch.zeros(1, 2))
    assert restored.distribution.stddev.item() == pytest.approx(0.8)
    torch.testing.assert_close(source.distribution.stddev, restored.distribution.stddev)


def test_checkpoint_requires_saved_std() -> None:
    source = PredictedStdGaussianActionDist(2, 1, base_std=0.2)
    state = source.state_dict()
    del state["base_log_std"]
    with pytest.raises(RuntimeError, match="base_log_std"):
        source.load_state_dict(state)
