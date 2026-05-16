from typing import Callable

from torch import nn

LinearInitialization = Callable[[nn.Linear], nn.Linear]
DEFAULT_ORTHOGONAL_GAIN = 0.01


def init_linear_orthogonal(module: nn.Linear, gain: float = DEFAULT_ORTHOGONAL_GAIN) -> nn.Linear:
    nn.init.orthogonal_(module.weight, gain=gain)
    nn.init.constant_(module.bias, 0)
    return module


def make_init_linear_orthogonal(gain: float) -> LinearInitialization:
    gain = float(gain)

    def init(module: nn.Linear) -> nn.Linear:
        return init_linear_orthogonal(module, gain=gain)

    return init


def init_transformer_feedforward(layer: nn.Module, *, gain: float) -> None:
    init_linear_orthogonal(layer.linear1, gain=gain)
    init_linear_orthogonal(layer.linear2, gain=1.0)
