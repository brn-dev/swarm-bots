

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace


class VectorSwarmBotsActionSpace(VectorHybridActionSpace):
    def __init__(self, n_envs: int, n_agents: int, actuators_dim: int, connectors_dim: int):
        super().__init__(spaces={
            "actuators": spaces.Box(low=-1.0, high=1.0, shape=(n_envs, n_agents, actuators_dim), dtype=np.float32),
            "connectors": spaces.MultiBinary((n_envs, n_agents, connectors_dim)),
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

        action_space = VectorSwarmBotsActionSpace(
            n_envs=n_envs,
            n_agents=n_agents,
            actuators_dim=actuators_dim,
            connectors_dim=connectors_dim,
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

    def _actions_to_env_dict(self, actions: torch.Tensor) -> dict[str, np.ndarray]:
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        actions = actions.detach().to("cpu")
        actuators = actions[..., :self.actuators_dim].numpy().astype(np.float32, copy=False)
        connectors = actions[..., self.actuators_dim:].numpy().astype(bool, copy=False)

        return {"actuators": actuators, "connectors": connectors}


