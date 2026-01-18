from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import moviepy.video.io.ImageSequenceClip

from swarmbots.mj_env.scenarios.obstacle_street_scenario import (
    ObstacleStreetScenario,
    PoleParams,
    UniformDistParams,
)
from swarmbots.mj_env.swarm.homogeneous_swarm import HomogeneousSwarm
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record random-action rollouts for ObstacleStreetScenario.")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--episode-length", type=int, default=300)
    p.add_argument("--action-repeat", type=int, default=15)
    p.add_argument("--num-walls", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out-dir", type=str, default="videos/_")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    swarm = HomogeneousSwarm(unit_start_locations="4:diamond")

    scenario = ObstacleStreetScenario(
        swarm=swarm,
        payload_type=None,
        seed=args.seed,
        num_walls=args.num_walls,
        no_initial_ramp=False,
        wall_height=[0.35] * args.num_walls,
        opening_width=[UniformDistParams(low=0.4, high=2.5)] * args.num_walls,
        first_wall_distance=UniformDistParams(low=1.5, high=3.0),
        inter_wall_distance=UniformDistParams(low=3.0, high=5.0),
        unusable_opening_offset=UniformDistParams(low=1.0, high=3.0),
        poles=[
            PoleParams(x=UniformDistParams(-2.0, 2.0), y=UniformDistParams(0.8, 1.2)),
        ],
        friction=[2.0, 1e-2, 2e-4],
        force_elliptic_cone=True,
        actuator_strength=5.0,
    )

    opt = mujoco.MjvOption()
    env = SwarmBotsEnv(
        scenario=scenario,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        render_mode="rgb_array",
        width=args.width,
        height=args.height,
        camera=args.camera,
        scene_option=opt,
    )

    rng = np.random.default_rng(args.seed + 12345)

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            frames: list[np.ndarray] = []

            first_frame = env.render()
            if first_frame is not None:
                frames.append(first_frame)

            done = False
            while not done:
                action = env.action_space.sample()

                if "connectors" in action and rng.random() < 0.15:
                    action["connectors"] = np.ones_like(action["connectors"], dtype=action["connectors"].dtype)

                obs, reward, terminated, truncated, info = env.step(action)

                frame = env.render()
                if frame is not None:
                    frames.append(frame)

                done = bool(terminated) or bool(truncated)

            if not frames:
                raise RuntimeError("No frames captured. Is render_mode='rgb_array' enabled?")

            video_path = out_dir / f"random_obstacle_street_{run_id}_ep_{ep}.mp4"
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, fps=args.fps)
            clip.write_videofile(video_path.as_posix(), logger=None)
            print(f"Saved {video_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()


