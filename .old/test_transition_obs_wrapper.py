

import numpy as np
from gymnasium.vector import SyncVectorEnv

from swarmbots.learn.env_wrappers.transition_obs_wrapper import TransitionObsWrapper
from swarmbots.learn.testing_env import TestingSwarmBotsEnv


def _make_env(*, max_steps: int) -> callable[[], TestingSwarmBotsEnv]:
    def _thunk() -> TestingSwarmBotsEnv:
        return TestingSwarmBotsEnv(
            n_agents=3,
            n_local_obs=4,
            n_global_obs=5,
            actuators_dim=2,
            connectors_dim=3,
            max_steps=max_steps,
        )

    return _thunk


def main() -> None:
    n_envs = 2
    n_agents = 3
    n_local_obs = 4
    n_global_obs = 5
    actuators_dim = 2
    connectors_dim = 3
    n_action_features = actuators_dim + connectors_dim

    env = SyncVectorEnv([_make_env(max_steps=10) for _ in range(n_envs)])
    wrapped = TransitionObsWrapper(env)

    obs, _info = wrapped.reset(seed=0)
    assert obs["local_obs"].shape == (n_envs, n_agents, 2 * n_local_obs + n_action_features)
    assert obs["global_obs"].shape == (n_envs, 2 * n_global_obs)
    assert np.all(obs["local_obs"] == 0)
    assert np.all(obs["global_obs"] == 0)

    actuators = np.arange(n_envs * n_agents * actuators_dim, dtype=np.float32).reshape(
        n_envs, n_agents, actuators_dim
    )
    connectors = np.zeros((n_envs, n_agents, connectors_dim), dtype=np.int8)
    connectors[:, :, 0] = 1

    next_obs, _rewards, _terminations, _truncations, _infos = wrapped.step(
        {"actuators": actuators, "connectors": connectors}
    )

    prev_local = next_obs["local_obs"][..., :n_local_obs]
    prev_actions = next_obs["local_obs"][..., n_local_obs : n_local_obs + n_action_features]
    new_local = next_obs["local_obs"][..., n_local_obs + n_action_features :]

    expected_prev_actions = np.concatenate((actuators, connectors.astype(np.float32)), axis=-1)

    assert np.all(prev_local == 0)
    assert np.all(new_local == 1)
    assert np.allclose(prev_actions, expected_prev_actions)

    prev_global = next_obs["global_obs"][..., :n_global_obs]
    new_global = next_obs["global_obs"][..., n_global_obs:]
    assert np.all(prev_global == 0)
    assert np.all(new_global == 1)

    print("OK: TransitionObsWrapper stacking behavior looks correct.")


if __name__ == "__main__":
    main()


