

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.logger import warn
from gymnasium.vector.vector_env import (
    AutoresetMode,
    VectorEnv,
    VectorObservationWrapper,
)
from gymnasium.wrappers.utils import RunningMeanStd


class NaiveNormalizeLocalGlobalObsWrapper(VectorObservationWrapper, gym.utils.RecordConstructorArgs):

    def __init__(self, env: VectorEnv, epsilon: float = 1e-8):
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        VectorObservationWrapper.__init__(self, env)

        if "autoreset_mode" not in self.env.metadata:
            warn(
                f"{self} is missing `autoreset_mode` data. Assuming that the vector environment it follows the "
                f"`NextStep` autoreset api or autoreset is disabled. Read https://farama.org/Vector-Autoreset-Mode "
                f"for more details."
            )
        else:
            assert self.env.metadata["autoreset_mode"] in {AutoresetMode.NEXT_STEP}

        assert isinstance(self.env.single_observation_space, gym.spaces.Dict), \
            f"Expected Dict observation space, got {type(self.env.single_observation_space)}"
        assert 'local_obs' in self.env.single_observation_space.spaces
        assert 'global_obs' in self.env.single_observation_space.spaces

        self.local_obs_space = self.env.single_observation_space['local_obs']
        self.global_obs_space = self.env.single_observation_space['global_obs']

        self.local_obs_rms = RunningMeanStd(
            shape=self.local_obs_space.shape,
            dtype=self.local_obs_space.dtype,
        )
        self.global_obs_rms = RunningMeanStd(
            shape=self.global_obs_space.shape,
            dtype=self.global_obs_space.dtype,
        )
        
        self.epsilon = epsilon
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the observation statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the observation statistics."""
        self._update_running_mean = setting

    def observations(self, observations: ObsType) -> ObsType:
        """
        Normalizes the 'local_obs' and 'global_obs' in the dictionary observation.
        """
        local_obs = observations['local_obs']
        global_obs = observations['global_obs']

        if self._update_running_mean:
            self.local_obs_rms.update(local_obs)
            self.global_obs_rms.update(global_obs)

        observations['local_obs'] = (local_obs - self.local_obs_rms.mean) / np.sqrt(
            self.local_obs_rms.var + self.epsilon
        )
        
        observations['global_obs'] = (global_obs - self.global_obs_rms.mean) / np.sqrt(
            self.global_obs_rms.var + self.epsilon
        )
        
        return observations


class NaiveNormalizeLocalObsWrapper(VectorObservationWrapper, gym.utils.RecordConstructorArgs):
    """
    A wrapper that normalizes ONLY the 'local_obs' of a dictionary observation space.
    Designed for SwarmBotsEnv wrapped in a VectorEnv.
    """

    def __init__(self, env: VectorEnv, epsilon: float = 1e-8):
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        VectorObservationWrapper.__init__(self, env)

        if "autoreset_mode" not in self.env.metadata:
            warn(
                f"{self} is missing `autoreset_mode` data. Assuming that the vector environment it follows the "
                f"`NextStep` autoreset api or autoreset is disabled. Read https://farama.org/Vector-Autoreset-Mode "
                f"for more details."
            )
        else:
            assert self.env.metadata["autoreset_mode"] in {AutoresetMode.NEXT_STEP}

        assert isinstance(self.env.single_observation_space, gym.spaces.Dict), \
            f"Expected Dict observation space, got {type(self.env.single_observation_space)}"
        assert 'local_obs' in self.env.single_observation_space.spaces

        self.local_obs_space = self.env.single_observation_space['local_obs']

        self.local_obs_rms = RunningMeanStd(
            shape=self.local_obs_space.shape,
            dtype=self.local_obs_space.dtype,
        )
        
        self.epsilon = epsilon
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the observation statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the observation statistics."""
        self._update_running_mean = setting

    def observations(self, observations: ObsType) -> ObsType:
        """
        Normalizes only the 'local_obs' in the dictionary observation.
        """
        local_obs = observations['local_obs']

        if self._update_running_mean:
            self.local_obs_rms.update(local_obs)

        observations['local_obs'] = (local_obs - self.local_obs_rms.mean) / np.sqrt(
            self.local_obs_rms.var + self.epsilon
        )
        
        return observations


class NaiveNormalizeGlobalWithLocalObsWrapper(VectorObservationWrapper, gym.utils.RecordConstructorArgs):
    """
    A wrapper that normalizes both 'local_obs' and 'global_obs' using statistics calculated ONLY from 'local_obs'
    """

    def __init__(self, env: VectorEnv, epsilon: float = 1e-8):
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        VectorObservationWrapper.__init__(self, env)

        if "autoreset_mode" not in self.env.metadata:
            warn(
                f"{self} is missing `autoreset_mode` data. Assuming that the vector environment it follows the "
                f"`NextStep` autoreset api or autoreset is disabled. Read https://farama.org/Vector-Autoreset-Mode "
                f"for more details."
            )
        else:
            assert self.env.metadata["autoreset_mode"] in {AutoresetMode.NEXT_STEP}

        assert isinstance(self.env.single_observation_space, gym.spaces.Dict), \
            f"Expected Dict observation space, got {type(self.env.single_observation_space)}"
        assert 'local_obs' in self.env.single_observation_space.spaces
        assert 'global_obs' in self.env.single_observation_space.spaces

        self.local_obs_space = self.env.single_observation_space['local_obs']

        self.local_obs_rms = RunningMeanStd(
            shape=self.local_obs_space.shape,
            dtype=self.local_obs_space.dtype,
        )
        
        self.epsilon = epsilon
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the observation statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the observation statistics."""
        self._update_running_mean = setting

    def observations(self, observations: ObsType) -> ObsType:
        """
        Normalizes 'local_obs' and 'global_obs' using statistics from 'local_obs'.
        """
        local_obs = observations['local_obs']
        global_obs = observations['global_obs']

        if self._update_running_mean:
            self.local_obs_rms.update(local_obs)

        observations['local_obs'] = (local_obs - self.local_obs_rms.mean) / np.sqrt(
            self.local_obs_rms.var + self.epsilon
        )
        
        observations['global_obs'] = (global_obs - self.local_obs_rms.mean) / np.sqrt(
            self.local_obs_rms.var + self.epsilon
        )
        
        return observations
