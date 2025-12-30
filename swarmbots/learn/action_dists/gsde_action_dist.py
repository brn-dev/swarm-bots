import math
from typing import Optional, Self

import torch
import torch.distributions as torchdist
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionNetInitialization
from swarmbots.learn.action_dists.continuous_action_dist import ContinuousActionDist
from swarmbots.learn.action_dists.tanh_bijector import TanhBijector
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal

LogStdNetInitialization = ActionNetInitialization


class GSDEActionDist(ContinuousActionDist):
    pass