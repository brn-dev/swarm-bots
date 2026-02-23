from __future__ import annotations

import argparse
import sys
from typing import Any, Callable

import torch
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers.vector import NormalizeReward, RecordEpisodeStatistics
from loguru import logger

from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.feature_wise_obs_norm_wrapper import FeatureWiseObsNormWrapper
from swarmbots.learn.env_wrappers.progress_guidance_ep_stats_wrapper import ProgressGuidanceEpisodeStatsWrapper
from swarmbots.learn.env_wrappers.transition_obs_wrapper import TransitionObsWrapper
from swarmbots.learn.obs_indices import ObsIndices
from swarmbots.learn.recording import record_policy
from swarmbots.learn.swarmbots_obs_indices import build_obs_indices
from swarmbots.mj_env.scenarios.scenario_presets import default_wall
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record wall rollouts with constant actuator actions.")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--episode-length", type=int, default=512)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video-folder", type=str, default="videos")
    parser.add_argument("--video-prefix", type=str, default="mat_nop_wall_actuators_one")
    parser.add_argument("--actuator-value", type=float, default=1.0)
    parser.add_argument("--connector-value", type=int, choices=(0, 1), default=1)
    parser.add_argument("--first-wall-distance", type=float, default=2.0)
    return parser.parse_args()


def make_env_fn(
    episode_length: int,
    first_wall_distance: float,
    camera: int,
    render_mode: str | None = None,
    first_episode_length: int | None = None,
) -> Callable[[], SwarmBotsEnv]:
    def _init() -> SwarmBotsEnv:
        scenario = default_wall(first_wall_distance=first_wall_distance)
        return SwarmBotsEnv(
            scenario=scenario,
            episode_length=episode_length,
            render_mode=render_mode,
            camera=camera,
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
    vector_env = ProgressGuidanceEpisodeStatsWrapper(vector_env)
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


class ConstantActuatorsPolicy(BasePolicy):
    def __init__(
        self,
        actuators_dim: int,
        connectors_dim: int,
        actuator_value: float,
        connector_value: int,
    ) -> None:
        super().__init__()
        self.actuators_dim = actuators_dim
        self.connectors_dim = connectors_dim
        self.actuator_value = float(actuator_value)
        self.connector_value = float(connector_value)

    @property
    def gsde_enabled(self) -> bool:
        return False

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            "actuator_value": self.actuator_value,
            "connector_value": self.connector_value,
        }

    def act(
        self,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        hidden_vars: torch.Tensor | None = None,
        agent_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        n_envs, n_agents = local_obs.shape[:2]
        actuators = torch.full(
            (n_envs, n_agents, self.actuators_dim),
            fill_value=self.actuator_value,
            device=local_obs.device,
            dtype=local_obs.dtype,
        )
        connectors = torch.full(
            (n_envs, n_agents, self.connectors_dim),
            fill_value=self.connector_value,
            device=local_obs.device,
            dtype=local_obs.dtype,
        )
        return torch.cat((actuators, connectors), dim=-1)


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    rollout_device = torch.device("cpu")

    dummy_env = make_env_fn(
        episode_length=args.episode_length,
        first_wall_distance=args.first_wall_distance,
        camera=args.camera,
        render_mode=None,
    )()
    env_settings = dummy_env.get_settings()
    local_obs_dim = int(dummy_env.observation_space["local_obs"].shape[-1])
    global_obs_dim = int(dummy_env.observation_space["global_obs"].shape[-1])
    hidden_vars_dim = int(dummy_env.observation_space["hidden_vars"].shape[-1])
    dummy_env.close()

    obs_indices = build_obs_indices(
        env_settings=env_settings,
        local_obs_dim=local_obs_dim,
        global_obs_dim=global_obs_dim,
        hidden_vars_dim=hidden_vars_dim,
    )

    record_env_fn = make_env_fn(
        episode_length=args.episode_length,
        first_wall_distance=args.first_wall_distance,
        camera=args.camera,
        render_mode="rgb_array",
    )
    record_vector_env = SyncVectorEnv([record_env_fn])
    record_env = wrap_vec_env(
        vector_env=record_vector_env,
        obs_indices=obs_indices,
        gamma=args.gamma,
        rollout_device=rollout_device,
    )

    policy = ConstantActuatorsPolicy(
        actuators_dim=record_env.actuators_dim,
        connectors_dim=record_env.connectors_dim,
        actuator_value=args.actuator_value,
        connector_value=args.connector_value,
    )

    record_policy(
        env=record_env,
        policy=policy,
        video_folder=args.video_folder,
        video_name_prefix=args.video_prefix,
        num_episodes=args.num_episodes,
        deterministic=True,
        gsde_reset_mode=None,
        fps=args.fps,
        device=rollout_device,
    )

    record_env.close()


if __name__ == "__main__":
    main()
