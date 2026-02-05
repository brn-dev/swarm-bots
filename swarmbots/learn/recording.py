import os
from typing import Any

import moviepy.video.io.ImageSequenceClip
import numpy as np
import torch

from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.gsde_reset import GSDEResetMode, GSDEIntervalResetMode, GSDEProbabilityResetMode
from swarmbots.learn.summary_statistics import SummaryStatistics, compute_summary_statistics, format_summary_statistics, \
    SummaryStatisticsFormat


def _maybe_reset_gsde_noise(
    *,
    policy: BasePolicy,
    local_obs: torch.Tensor,
    deterministic: bool,
    rollout_step_idx: int,
    gsde_reset_mode: GSDEResetMode | None,
) -> None:
    if deterministic or not bool(getattr(policy, "gsde_enabled", False)):
        return

    if gsde_reset_mode is None:
        raise RuntimeError("Policy reports gsde_enabled=True but gsde_reset_mode is None.")

    action_dist = getattr(policy, "action_dist", None)
    if action_dist is None or not hasattr(action_dist, "reset_noise"):
        raise RuntimeError("Policy reports gsde_enabled=True but has no action_dist.reset_noise().")

    batch_shape = tuple(local_obs.shape[:-1])
    if isinstance(gsde_reset_mode, GSDEIntervalResetMode):
        if (rollout_step_idx % gsde_reset_mode.interval) == 0:
            action_dist.reset_noise(batch_shape=batch_shape)
        return

    if isinstance(gsde_reset_mode, GSDEProbabilityResetMode):
        if not hasattr(action_dist, "reset_noise_masked"):
            raise RuntimeError(
                "GSDEProbabilityResetMode requires action_dist.reset_noise_masked(mask), but it's missing."
            )
        mask = torch.empty(batch_shape, device=local_obs.device, dtype=torch.bool).bernoulli_(gsde_reset_mode.probability)
        action_dist.reset_noise_masked(mask)
        return

    raise TypeError(f"Unknown gsde_reset_mode type: {type(gsde_reset_mode)}")


def record_policy(
    env: BaseLearnEnvWrapper,
    policy: BasePolicy,
    video_folder: str,
    video_name_prefix: str,
    num_episodes: int = 5,
    deterministic: bool = False,
    gsde_reset_mode: GSDEResetMode | None = None,
    fps: int = 30,
    device: torch.device = torch.device("cpu"),
):
    """
    Record episodes of a policy interacting with an environment.
    
    Args:
        env: The environment wrapper (BaseLearnEnvWrapper)
        policy: The policy to evaluate
        video_folder: Directory to save videos
        video_name_prefix: Prefix for video filenames
        num_episodes: Number of episodes to record
        deterministic: Whether to use deterministic actions
        gsde_reset_mode: Controls gSDE noise resampling strategy when deterministic=False.
        fps: Frames per second for the output video
        device: Torch device
    """
    os.makedirs(video_folder, exist_ok=True)
    
    policy.eval()
    policy.to(device)

    for episode_idx in range(num_episodes):
        obs, _ = env.reset()
        frames = []
                    
        try:
            first_frame = env.render()
        except Exception as e:
            print(f"Warning: Could not render environment. Error: {e}")
            first_frame = None

        if first_frame is None:
            print("Environment render returned None. Make sure render_mode='rgb_array' is set.")
            return

        done = False
        step_cnt = 0
        
        while not done:
            frame = env.render()
            
            if isinstance(frame, (list, tuple)):
                current_frame = frame[0]
            elif isinstance(frame, np.ndarray):
                if frame.ndim == 4:
                    current_frame = frame[0]
                else:
                    current_frame = frame
            else:
                current_frame = frame

            if current_frame is not None:
                frames.append(current_frame)

            with torch.no_grad():
                local_obs = obs['local_obs']
                global_obs = obs['global_obs']
                hidden_vars = obs["hidden_vars"]
                agent_mask = obs.get("agent_mask", None)
                
                _maybe_reset_gsde_noise(
                    policy=policy,
                    local_obs=local_obs,
                    deterministic=deterministic,
                    rollout_step_idx=step_cnt,
                    gsde_reset_mode=gsde_reset_mode,
                )
                actions = policy.act(
                    local_obs,
                    global_obs,
                    hidden_vars=hidden_vars,
                    agent_mask=agent_mask,
                    deterministic=deterministic,
                )
                print(format_summary_statistics(compute_summary_statistics(actions[:, :, :8], make_histogram=True), SummaryStatisticsFormat(histogram=True)))
            
            obs, _, term, trunc, _ = env.step(actions)
            
            if term[0] or trunc[0]:
                done = True
            
            step_cnt += 1
            
        print(f"Episode {episode_idx} finished after {step_cnt} steps.")

        if frames:
            video_path = os.path.join(video_folder, f"{video_name_prefix}_ep_{episode_idx}.mp4")
            try:
                clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, fps=fps)
                clip.write_videofile(video_path, logger=None)
                print(f"Saved video to {video_path}")
            except Exception as e:
                print(f"Failed to save video: {e}")
        else:
            print("No frames collected.")

