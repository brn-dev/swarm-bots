from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import moviepy.video.io.ImageSequenceClip

from swarmbots.mj_env.scenarios.scenario_presets import default_wall
from swarmbots.mj_env.swarm.homogeneous_swarm import HomogeneousSwarm, PreConnectedUnitLocationsConfig
from swarmbots.mj_env.swarm.unit_config import UNIT_CONFIG_TETRAHEDRON_YX
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record random-action rollouts with a preconnected swarm.")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--episode-length", type=int, default=200)
    p.add_argument("--action-repeat", type=int, default=15)
    p.add_argument("--num-units", type=int, default=50)
    p.add_argument("--max-radius", type=float, default=3.0)
    p.add_argument("--z-pos", type=float, default=1.5)
    p.add_argument("--no-center", action="store_true")
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

    swarm = HomogeneousSwarm(
        unit_start_locations=PreConnectedUnitLocationsConfig(
            num_units=args.num_units,
            unit_config=UNIT_CONFIG_TETRAHEDRON_YX,
            max_radius=args.max_radius,
            z_pos=args.z_pos,
            center=not args.no_center,
        )
    )

    scenario = default_wall(seed=args.seed if args.seed != 0 else None, swarm=swarm)

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

    rng = np.random.default_rng(args.seed if args.seed != 0 else None)

    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            if env.swarm_connections is None:
                raise RuntimeError("Swarm connections not initialized after reset.")
            connectors_mask = env.swarm_connections.get_is_active_mask().copy()
            print(f"Episode {ep}: active connectors = {connectors_mask.sum()}")

            frames: list[np.ndarray] = []
            first_frame = env.render()
            if first_frame is not None:
                frames.append(first_frame)

            done = False
            while not done:
                action = env.action_space.sample()
                action["connectors"] = connectors_mask

                if rng.random() < 0.1:
                    action["actuators"] *= 0.0

                obs, reward, terminated, truncated, info = env.step(action)

                frame = env.render()
                if frame is not None:
                    frames.append(frame)

                done = bool(terminated) or bool(truncated)

            if not frames:
                raise RuntimeError("No frames captured. Is render_mode='rgb_array' enabled?")

            video_path = out_dir / f"preconnected_swarm_{run_id}_ep_{ep}.mp4"
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, fps=args.fps)
            clip.write_videofile(video_path.as_posix(), logger=None)
            print(f"Saved {video_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
