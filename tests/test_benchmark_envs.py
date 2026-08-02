from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv, VectorEnv

from experiments.external_benchmarks.benchmark_experiment_common import (
    _make_algorithm,
    _make_obs_indices,
    _wrap_ppo_policy_with_nop,
    _wrap_vector_env,
)
from experiments.external_benchmarks.scenario_experiment_common import (
    EXPERIMENT_CONFIGS,
)
from experiments.transformer_policy_common import (
    MATInitGains,
    MATNormalizationConfig,
    NOPInitGains,
    make_benchmark_transformer_policy,
)
from swarmbots.external_benchmark_envs.multi_agent_mujoco_env import MultiAgentMujocoEnv
from swarmbots.external_benchmark_envs.vmas_vector_env import VMASVectorEnv
from swarmbots.learn.env_wrappers.learn_wrappers.continuous_actions_learn_env_wrapper import (
    ContinuousActionsLearnEnvWrapper,
)

MAMUJOCO_EXPERIMENT_NAMES = [
    name for name, config in EXPERIMENT_CONFIGS.items() if config.suite == "mamujoco"
]
VMAS_EXPERIMENT_NAMES = [
    name for name, config in EXPERIMENT_CONFIGS.items() if config.suite == "vmas"
]


class _AsymmetricContinuousActionVectorEnv(VectorEnv):
    num_envs = 1
    action_backend = "numpy"

    def __init__(self) -> None:
        super().__init__()
        self.metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}
        self.observation_space = spaces.Dict(
            {
                "local_obs": spaces.Box(-np.inf, np.inf, shape=(1, 2, 1)),
                "global_obs": spaces.Box(-np.inf, np.inf, shape=(1, 0)),
                "hidden_local_vars": spaces.Box(-np.inf, np.inf, shape=(1, 2, 0)),
                "hidden_global_vars": spaces.Box(-np.inf, np.inf, shape=(1, 0)),
            }
        )
        self.action_space = spaces.Box(
            low=np.array([[[-2.0, 2.0], [10.0, -4.0]]], dtype=np.float32),
            high=np.array([[[2.0, 6.0], [14.0, 0.0]]], dtype=np.float32),
            dtype=np.float32,
        )


def test_continuous_action_wrapper_rescales_and_centers_native_actions() -> None:
    env = ContinuousActionsLearnEnvWrapper(_AsymmetricContinuousActionVectorEnv())

    assert np.all(env.action_space["actions"].low == -1.0)
    assert np.all(env.action_space["actions"].high == 1.0)
    native_actions = env._actions_to_env(torch.tensor([[[-1.0, 0.0], [1.0, -0.5]]]))
    np.testing.assert_allclose(
        native_actions,
        np.array([[[-2.0, 4.0], [14.0, -3.0]]], dtype=np.float32),
    )


def test_every_registered_external_benchmark_has_a_scenario_launcher() -> None:
    benchmark_root = (
        Path(__file__).resolve().parents[1] / "experiments" / "external_benchmarks"
    )
    launcher_names = {path.parent.name for path in benchmark_root.glob("*/run.py")}

    assert launcher_names == set(EXPERIMENT_CONFIGS)


@pytest.mark.parametrize("experiment_name", MAMUJOCO_EXPERIMENT_NAMES)
def test_registered_mamujoco_experiment_constructs(experiment_name: str) -> None:
    pytest.importorskip("gymnasium_robotics")
    config = EXPERIMENT_CONFIGS[experiment_name]
    env = MultiAgentMujocoEnv(
        scenario=config.scenario,
        agent_conf=config.mamujoco_agent_conf,
        agent_obsk=config.mamujoco_agent_obsk,
        episode_length=2,
    )
    try:
        observations, _ = env.reset(seed=3)

        assert observations["local_obs"].shape[0] == len(env.agent_names)
        assert env.action_space.shape[0] == len(env.agent_names)
    finally:
        env.close()


@pytest.mark.parametrize("experiment_name", VMAS_EXPERIMENT_NAMES)
def test_registered_vmas_experiment_constructs(experiment_name: str) -> None:
    pytest.importorskip("vmas")
    config = EXPERIMENT_CONFIGS[experiment_name]
    env = VMASVectorEnv(
        scenario=config.scenario,
        num_envs=2,
        device="cpu",
        episode_length=2,
    )
    try:
        observations, _ = env.reset(seed=5)

        assert observations["local_obs"].shape[:2] == (2, env.n_agents)
        assert env.action_space.shape[:2] == (2, env.n_agents)
    finally:
        env.close()


def test_mamujoco_adapter_exposes_homogeneous_team_observations_and_actions() -> None:
    pytest.importorskip("gymnasium_robotics")
    env = MultiAgentMujocoEnv(episode_length=1)
    try:
        observations, _ = env.reset(seed=7)
        next_observations, reward, terminated, truncated, _ = env.step(
            env.action_space.sample()
        )

        assert observations["local_obs"].shape == (2, 12)
        assert observations["hidden_global_vars"].shape == (17,)
        assert env.action_space.shape == (2, 3)
        assert next_observations["local_obs"].shape == observations["local_obs"].shape
        assert np.asarray(reward).shape == ()
        assert isinstance(terminated, bool)
        assert truncated
    finally:
        env.close()


def test_mamujoco_sync_vector_env_keeps_terminal_observation_for_same_step_autoreset() -> (
    None
):
    pytest.importorskip("gymnasium_robotics")
    vector_env = SyncVectorEnv(
        [partial(MultiAgentMujocoEnv, episode_length=1)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    try:
        vector_env.reset(seed=11)
        observations, _, _, truncations, infos = vector_env.step(
            vector_env.action_space.sample()
        )

        assert truncations.tolist() == [True]
        assert infos["_final_obs"].tolist() == [True]
        terminal_observation = infos["final_obs"][0]
        assert terminal_observation["local_obs"].shape == (2, 12)
        assert observations["local_obs"].shape == (1, 2, 12)
    finally:
        vector_env.close()


def test_vmas_adapter_performs_same_step_lane_resets_and_preserves_terminal_observations() -> (
    None
):
    pytest.importorskip("vmas")
    env = VMASVectorEnv(scenario="balance", num_envs=3, device="cpu", episode_length=1)
    try:
        observations, _ = env.reset(seed=13)
        next_observations, rewards, terminations, truncations, infos = env.step(
            env.action_space.sample()
        )

        assert observations["local_obs"].shape == (3, 3, 16)
        assert next_observations["hidden_global_vars"].shape == (3, 48)
        assert rewards.shape == (3,)
        assert not torch.any(terminations)
        assert torch.all(truncations)
        assert torch.all(infos["_final_obs"])
        assert infos["final_obs"]["local_obs"].shape == (3, 3, 16)
        assert not torch.equal(
            next_observations["local_obs"], infos["final_obs"]["local_obs"]
        )
    finally:
        env.close()


def test_vmas_native_action_range_is_exposed_as_normalized_actions() -> None:
    pytest.importorskip("vmas")
    vector_env = VMASVectorEnv(
        scenario="give_way", num_envs=2, device="cpu", episode_length=2
    )
    env = ContinuousActionsLearnEnvWrapper(vector_env)
    try:
        assert np.all(env.action_space["actions"].low == -1.0)
        assert np.all(env.action_space["actions"].high == 1.0)
        native_actions = env._actions_to_env(torch.ones(vector_env.action_space.shape))

        torch.testing.assert_close(native_actions, torch.full_like(native_actions, 0.5))
    finally:
        env.close()


@pytest.mark.parametrize("policy_variant", ["mat_qcx", "mat_dec", "tmasac"])
def test_benchmark_policy_factory_builds_requested_feedforward_variants(
    policy_variant: str,
) -> None:
    pytest.importorskip("vmas")
    vector_env = VMASVectorEnv(
        scenario="balance", num_envs=2, device="cpu", episode_length=2
    )
    obs_indices = _make_obs_indices(vector_env)
    env = _wrap_vector_env(
        vector_env=vector_env,
        obs_indices=obs_indices,
        rollout_device=torch.device("cpu"),
        use_popart=policy_variant != "tmasac",
    )
    try:
        policy = make_benchmark_transformer_policy(
            env=env,
            policy_variant=policy_variant,
            use_nop=policy_variant == "tmasac",
            obs_indices=obs_indices,
            compile_modules=False,
            mat_init_gains=MATInitGains(),
            nop_init_gains=NOPInitGains(),
            mat_normalization=MATNormalizationConfig(),
        )
        observations, _ = env.reset(seed=17)
        actions = policy.act(**observations)

        assert actions.shape == (2, 3, 2)
        if policy_variant == "tmasac":
            assert not policy.requires_recurrent_training()
            assert policy.config.nop_config.enabled
    finally:
        env.close()


@pytest.mark.parametrize("policy_variant", ["mat_qcx", "mat_dec"])
def test_external_benchmark_ppo_nop_uses_configured_encoder_width(
    policy_variant: str,
) -> None:
    pytest.importorskip("vmas")
    vector_env = VMASVectorEnv(
        scenario="balance", num_envs=2, device="cpu", episode_length=2
    )
    obs_indices = _make_obs_indices(vector_env)
    env = _wrap_vector_env(
        vector_env=vector_env,
        obs_indices=obs_indices,
        rollout_device=torch.device("cpu"),
        use_popart=True,
    )
    try:
        encoder_width = 32
        nop_init_gains = NOPInitGains()
        base_policy = make_benchmark_transformer_policy(
            env=env,
            policy_variant=policy_variant,
            use_nop=True,
            obs_indices=obs_indices,
            compile_modules=False,
            mat_init_gains=MATInitGains(),
            nop_init_gains=nop_init_gains,
            mat_normalization=MATNormalizationConfig(),
            enc_d_model=encoder_width,
            enc_nhead=4,
            dec_d_model=16,
            dec_nhead=2,
            world_model_num_next_steps=1,
            transition_model_d_model=16,
        )
        policy = _wrap_ppo_policy_with_nop(
            base_policy=base_policy,
            env=env,
            use_nop=True,
            obs_indices=obs_indices,
            compile_modules=False,
            local_latent_dim=encoder_width,
            nop_init_gains=nop_init_gains,
            world_model_loss_coef=0.1,
            world_model_num_next_steps=1,
            transition_model_d_model=16,
        )
        algorithm = _make_algorithm(
            policy=policy,
            env=env,
            policy_variant=policy_variant,
            use_nop=True,
            rollout_samples=2,
            rollout_steps_per_env=1,
            episode_length=2,
            world_model_num_next_steps=1,
            rollout_device=torch.device("cpu"),
            train_device=torch.device("cpu"),
            record_device=torch.device("cpu"),
        )
        algorithm.n_epochs = 1

        algorithm.learn(
            max_total_timesteps=2,
            log_interval=1,
            save_interval=None,
            enable_command_prompt=False,
            logging_console_keys=[],
        )

        assert algorithm.n_total_iterations == 1
    finally:
        env.close()


def test_external_benchmark_ppo_runs_without_swarmbots_episode_metrics() -> None:
    pytest.importorskip("vmas")
    vector_env = VMASVectorEnv(
        scenario="balance", num_envs=2, device="cpu", episode_length=1
    )
    obs_indices = _make_obs_indices(vector_env)
    env = _wrap_vector_env(
        vector_env=vector_env,
        obs_indices=obs_indices,
        rollout_device=torch.device("cpu"),
        use_popart=True,
    )
    try:
        policy = make_benchmark_transformer_policy(
            env=env,
            policy_variant="mat_dec",
            use_nop=False,
            obs_indices=obs_indices,
            compile_modules=False,
            mat_init_gains=MATInitGains(),
            nop_init_gains=NOPInitGains(),
            mat_normalization=MATNormalizationConfig(),
            enc_d_model=32,
            enc_nhead=4,
            dec_d_model=16,
            dec_nhead=2,
        )
        algorithm = _make_algorithm(
            policy=policy,
            env=env,
            policy_variant="mat_dec",
            use_nop=False,
            rollout_samples=2,
            rollout_steps_per_env=1,
            episode_length=1,
            world_model_num_next_steps=1,
            rollout_device=torch.device("cpu"),
            train_device=torch.device("cpu"),
            record_device=torch.device("cpu"),
        )
        algorithm.n_epochs = 1

        algorithm.learn(
            max_total_timesteps=2,
            log_interval=1,
            save_interval=None,
            enable_command_prompt=False,
            logging_console_keys=[],
        )

        assert algorithm.n_total_iterations == 1
    finally:
        env.close()
