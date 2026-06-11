

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from swarmbots.utils.connector_actions import connector_action_space
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.tensor_conversion import to_backend_array


def _is_continuous_connector_action_space(space: spaces.Space) -> bool:
    if not isinstance(space, spaces.Box):
        return False
    if not np.issubdtype(space.dtype, np.floating):
        return False
    return bool(np.allclose(space.low, -1.0) and np.allclose(space.high, 1.0))


class VectorSwarmBotsActionSpace(VectorHybridActionSpace):
    def __init__(
        self,
        n_envs: int,
        n_agents: int,
        actuators_dim: int,
        connectors_dim: int,
        *,
        continuous_connector_actions: bool = False,
    ):
        super().__init__(spaces={
            "actuators": spaces.Box(low=-1.0, high=1.0, shape=(n_envs, n_agents, actuators_dim), dtype=np.float32),
            "connectors": connector_action_space(
                (n_envs, n_agents, connectors_dim),
                continuous=continuous_connector_actions,
            ),
        })


class SwarmBotsLearnEnvWrapper(BaseLearnEnvWrapper):

    def __init__(
        self,
        env: VectorEnv,
        *,
        device: torch.device | str = "cpu",
        obs_dtype: torch.dtype = torch.float32,
        reward_dtype: torch.dtype = torch.float32,
    ):
        assert isinstance(env.action_space, spaces.Dict)
        assert 'actuators' in env.action_space.keys()
        assert 'connectors' in env.action_space.keys()

        n_envs = env.action_space['actuators'].shape[0]
        n_agents = env.action_space['actuators'].shape[1]

        assert env.action_space['connectors'].shape[0] == n_envs
        assert env.action_space['connectors'].shape[1] == n_agents

        actuators_dim = env.action_space['actuators'].shape[2]
        connectors_dim = env.action_space['connectors'].shape[2]
        continuous_connector_actions = _is_continuous_connector_action_space(env.action_space['connectors'])

        action_space = VectorSwarmBotsActionSpace(
            n_envs=n_envs,
            n_agents=n_agents,
            actuators_dim=actuators_dim,
            connectors_dim=connectors_dim,
            continuous_connector_actions=continuous_connector_actions,
        )

        super().__init__(
            env=env,
            action_space=action_space,
            device=device,
            obs_dtype=obs_dtype,
            reward_dtype=reward_dtype,
        )

        self.actuators_dim = action_space['actuators'].shape[2]
        self.connectors_dim = action_space['connectors'].shape[2]
        self.continuous_connector_actions = continuous_connector_actions
        self.action_backend = str(getattr(env, "action_backend", "numpy")).lower()

    def _actions_to_env_dict(self, actions: torch.Tensor) -> dict[str, object]:
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        actions = actions.detach()
        actuators = to_backend_array(
            actions[..., :self.actuators_dim],
            backend=self.action_backend,
            dtype=np.float32 if self.action_backend == "numpy" else None,
        )
        connector_actions = actions[..., self.actuators_dim:]
        if self.continuous_connector_actions:
            connectors = to_backend_array(
                connector_actions,
                backend=self.action_backend,
                dtype=np.float32 if self.action_backend == "numpy" else None,
            )
        else:
            connectors = to_backend_array(
                connector_actions > 0.5,
                backend=self.action_backend,
                dtype=np.bool_ if self.action_backend == "numpy" else None,
            )

        return {"actuators": actuators, "connectors": connectors}


