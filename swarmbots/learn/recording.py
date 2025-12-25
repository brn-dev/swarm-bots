import os
from typing import Optional

import moviepy.video.io.ImageSequenceClip
import numpy as np
import torch
from gymnasium.vector import VectorEnv

from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper


def record_policy(
    env: BaseLearnEnvWrapper,
    policy: BasePolicy,
    video_folder: str,
    video_name_prefix: str,
    num_episodes: int = 1,
    deterministic: bool = False,
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
                
                actions = policy.act(local_obs, global_obs, deterministic=deterministic)
            
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

