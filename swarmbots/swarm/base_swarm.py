import abc
from dataclasses import dataclass

import mujoco
import numpy as np


class BaseSwarm(abc.ABC):

    def __init__(self, num_units: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.num_units = num_units

    @abc.abstractmethod
    def create_swarm_spec(self) -> mujoco.MjSpec:
        raise NotImplementedError()

    def reset_swarm(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict:
        """
        initialize state and potentially randomize swarm
        """
        return dict()

    def get_unit_prefixes(self) -> list[str]:
        return [f'Unit{i}--' for i in range(1, self.num_units + 1)]
