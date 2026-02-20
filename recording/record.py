import sys
from typing import Any, Callable

import torch
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers.vector import RecordEpisodeStatistics, NormalizeReward
from loguru import logger
from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import GSDEParams
from swarmbots.learn.algos.mat.wm.mat_nop_policy import MATNOPPolicy
from swarmbots.learn.checkpointing import (
    apply_env_state,
    extract_env_state,
    extract_policy_state_dict,
    freeze_env_normalization,
    load_checkpoint,
)
from swarmbots.learn.env_wrappers.obs_normalization.feature_wise_obs_norm_wrapper import (
    FeatureWiseObsNormWrapper,
)
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.transition_obs_wrapper import TransitionObsWrapper
from swarmbots.learn.gsde_reset import GSDEProbabilityResetMode
from swarmbots.learn.obs_indices import ObsIndices
from swarmbots.learn.recording import record_policy
from swarmbots.learn.swarmbots_obs_indices import build_obs_indices
from swarmbots.mj_env.scenarios.scenario_presets import default_wall
from swarmbots.mj_env.swarm.homogeneous_swarm import (
    HomogeneousSwarm,
    PreConnectedUnitLocationsConfig,
)
from swarmbots.mj_env.swarm.unit_config import UNIT_CONFIG_TETRAHEDRON_YX
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv


def make_env_fn(
    episode_length: int,
    scenario_kwargs: dict[str, Any] | None = None,
    render_mode: str | None = None,
    first_episode_length: int | None = None,
) -> Callable[[], SwarmBotsEnv]:
    if scenario_kwargs is None:
        scenario_kwargs = {}

    def _init() -> SwarmBotsEnv:
        scenario = default_wall(
            swarm=HomogeneousSwarm(
                unit_config=UNIT_CONFIG_TETRAHEDRON_YX,
                unit_start_locations=PreConnectedUnitLocationsConfig(
                    num_units=4,
                    num_unit_probs={
                        2: 1.0,
                        3: 1.0,
                        4: 1.0,
                    },
                    max_radius=1.5,
                    z_pos=0.5,
                ),
                randomize_unit_orientations=True,
            ),
            first_wall_distance=2.0,
            **scenario_kwargs,
        )
        return SwarmBotsEnv(
            scenario=scenario,
            episode_length=episode_length,
            render_mode=render_mode,
            camera=0,
            first_episode_length=first_episode_length,
        )

    return _init


def wrap_vec_env(
    vector_env: SyncVectorEnv,
    obs_indices: ObsIndices,
    gamma: float,
    rollout_device: torch.device,
) -> SwarmBotsLearnEnvWrapper:
    vector_env = RecordEpisodeStatistics(vector_env)
    vector_env = FeatureWiseObsNormWrapper(
        vector_env,
        obs_key="local_obs",
        scalar_feature_indices=obs_indices.local_scalar_indices,
        quaternion_indices=obs_indices.local_quaternion_indices,
    )
    vector_env = FeatureWiseObsNormWrapper(
        vector_env,
        obs_key="global_obs",
        scalar_feature_indices=obs_indices.global_scalar_indices,
        quaternion_indices=obs_indices.global_quaternion_indices,
    )
    vector_env = FeatureWiseObsNormWrapper(
        vector_env,
        obs_key="hidden_vars",
        scalar_feature_indices=obs_indices.hidden_vars_scalar_indices,
        quaternion_indices=obs_indices.hidden_vars_quaternion_indices,
    )
    vector_env = TransitionObsWrapper(vector_env)
    vector_env = NormalizeReward(vector_env, gamma=gamma)
    return SwarmBotsLearnEnvWrapper(vector_env, device=rollout_device)


def main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    episode_length = 512
    load_path = "runs/mat_nop_swarm_bots_wall/2026-02-17_23-05-06/models/model_7231796_steps_stopped.pt"
    deterministic = False
    rollout_device = torch.device("cpu")
    gsde_reset_mode = GSDEProbabilityResetMode(probability=1 / 6)
    gamma = 0.99
    scenario_kwargs: dict[str, Any] = {}

    print("Creating dummy env for capturing settings...")
    dummy_env = make_env_fn(
        episode_length=episode_length,
        scenario_kwargs=scenario_kwargs,
        render_mode=None,
    )()
    env_settings = dummy_env.get_settings()
    local_obs_dim = int(dummy_env.observation_space["local_obs"].shape[-1])
    global_obs_dim = int(dummy_env.observation_space["global_obs"].shape[-1])
    hidden_vars_dim = int(dummy_env.observation_space["hidden_vars"].shape[-1])
    dummy_env.close()
    del dummy_env
    print("Env settings captured.")

    obs_indices = build_obs_indices(
        env_settings=env_settings,
        local_obs_dim=local_obs_dim,
        global_obs_dim=global_obs_dim,
        hidden_vars_dim=hidden_vars_dim,
    )

    print("Creating env...")
    record_env_fn = make_env_fn(
        episode_length=episode_length,
        scenario_kwargs=scenario_kwargs,
        render_mode="rgb_array",
    )

    record_vector_env = SyncVectorEnv([record_env_fn])
    record_env = wrap_vec_env(
        vector_env=record_vector_env,
        obs_indices=obs_indices,
        gamma=gamma,
        rollout_device=rollout_device,
    )

    print("Initializing Policy...")
    policy = MATNOPPolicy(
        env=record_env,
        local_obs_encoder_hidden_dims=[256, 256],
        action_encoder_hidden_dims=[64],
        d_model=128,
        d_model_decoder=64,
        nhead_encoder=2,
        nhead_decoder=2,
        num_layers_encoder=2,
        num_layers_decoder=2,
        dim_feedforward_encoder=256,
        dim_feedforward_decoder=128,
        dropout=0.0,
        n_critic_local_projection_hidden_layers=1,
        n_critic_value_regressor_hidden_layers=1,
        cross_attn_first=True,
        act_fn_cls=nn.GELU,
        continuous_config=GSDEParams(
            base_std=0.25,
            latent_sde_dim=None,
            std_learnable=True,
            full_std=True,
            sde_learn_features=False,
            log_std_clamp_range=(-20.0, 2.0),
            normalize_latent_sde_by_dim=True,
        ),
        bernoulli_initial_prob=0.75,
        max_agents=20,
        wm_pre_transition_dims=[128],
        d_model_transition_model=128,
        nhead_transition_model=2,
        num_layers_transition_model=2,
        dim_feedforward_transition_model=128,
        transition_model_coembed_hidden_dims=[128],
        wm_pre_predictors_dims=[128, 128],
        wm_scalar_predictor_hidden_dims=[],
        wm_angle_predictor_hidden_dims=[],
        wm_rot6d_predictor_hidden_dims=[],
        wm_binary_predictor_hidden_dims=[],
        local_scalar_target_indices=obs_indices.local_scalar_indices,
        local_angle_target_indices=obs_indices.local_angle_indices,
        local_rot6d_target_indices=obs_indices.local_rot6d_indices,
        local_binary_target_indices=obs_indices.local_binary_indices,
        scalar_loss_fn="smooth_l1",
        scalar_loss_weight=1.0,
        angle_loss_weight=1.0,
        rot6d_loss_weight=1.0,
        binary_loss_weight=1.0,
    )
    print(policy)

    logger.info(f"Loading model from {load_path}")
    checkpoint = load_checkpoint(load_path)
    apply_env_state(record_env, extract_env_state(checkpoint))
    freeze_env_normalization(record_env)
    policy.load_state_dict(extract_policy_state_dict(checkpoint), strict=True)

    print("Starting recording...")
    record_policy(
        env=record_env,
        policy=policy,
        video_folder="videos",
        video_name_prefix="mat_nop_wall",
        num_episodes=3,
        deterministic=deterministic,
        gsde_reset_mode=gsde_reset_mode,
        device=rollout_device,
    )

    record_env.close()


if __name__ == "__main__":
    main()
