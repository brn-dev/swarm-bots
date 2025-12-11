"""
https://github.com/adysonmaia/sb3-plus/blob/main/sb3_plus
"""


from functools import singledispatch

import numpy as np
from gymnasium import spaces


@singledispatch
def get_action_dim(space: spaces.Space) -> int:
    raise NotImplementedError(f"{space} space is not supported")


@get_action_dim.register
def _get_action_dim_box(space: spaces.Box) -> int:
    return int(np.prod(space.shape))


@get_action_dim.register
def _get_action_dim_discrete(space: spaces.Discrete) -> int:
    return 1


@get_action_dim.register
def _get_action_dim_multibinary(space: spaces.MultiBinary) -> int:
    return int(space.n)


@get_action_dim.register
def _get_action_dim_multidiscrete(space: spaces.MultiDiscrete) -> int:
    return int(len(space.nvec))


@get_action_dim.register
def _get_action_dim_dict(space: spaces.Dict) -> int:
    return int(sum([get_action_dim(s) for s in space.spaces.values()]))


@get_action_dim.register
def _get_action_dim_tuple(space: spaces.Tuple) -> int:
    return int(sum([get_action_dim(s) for s in space.spaces]))
