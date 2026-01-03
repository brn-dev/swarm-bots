from typing import Callable

from torch import nn

LinearInitialization = Callable[[nn.Linear], nn.Linear]

def init_linear_orthogonal(module: nn.Linear, gain: float = 0.01) -> nn.Linear:
    nn.init.orthogonal_(module.weight, gain=gain)
    nn.init.constant_(module.bias, 0)
    return module
