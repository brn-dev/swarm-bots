import numpy as np
import pytest
from gymnasium import spaces

from swarmbots.learn.hybrid_action_space import HybridActionSpace, VectorHybridActionSpace


@pytest.mark.parametrize("space_type, shape", [(HybridActionSpace, (2, 3)), (VectorHybridActionSpace, (1, 2, 3))])
def test_seed_reproduces_each_action_component(space_type, shape):
    def make(seed):
        return space_type({
            "actuators": spaces.Box(-1.0, 1.0, shape=shape, dtype=np.float32),
            "connectors": spaces.MultiBinary(shape),
        }, seed=seed)

    first, second = make(123), make(123)
    first_sample, second_sample = first.sample(), second.sample()
    for key in first_sample:
        np.testing.assert_array_equal(first_sample[key], second_sample[key])
    first.seed(123)
    for key, value in first.sample().items():
        np.testing.assert_array_equal(value, first_sample[key])
    generator_first = make(np.random.default_rng(12))
    generator_second = make(np.random.default_rng(12))
    first_sample, second_sample = generator_first.sample(), generator_second.sample()
    for key in first_sample:
        np.testing.assert_array_equal(first_sample[key], second_sample[key])
