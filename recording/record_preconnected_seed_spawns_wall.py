from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import moviepy.video.io.ImageSequenceClip
import numpy as np

from swarmbots.mj_env.scenarios.scenario_presets import default_wall
from swarmbots.mj_env.swarm.homogeneous_swarm import PreConnectedUnitLocationsConfig
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record one spawn-inspection video per preconnected swarm seed for the wall scenario."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(42_000, 42_005)),
        help="Seed values to record (default: 42000..42004).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("../runs/seed_spawn_checks_wall"))
    parser.add_argument("--episode-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def make_preconnected_unit_start_locations(pool_seeds: tuple[int, ...] | None) -> PreConnectedUnitLocationsConfig:
    return PreConnectedUnitLocationsConfig(
        num_units=5,
        num_unit_probs={
            # 2: 0.5,
            # 3: 0.5,
            4: 1.0,
            5: 1.0,
            # 6: 1.0,
        },
        max_radius=1.5,
        unconnected_prob=0.03,
        z_pos=0.5,
        pool_seeds=pool_seeds,
    )


def make_env(
    *,
    pool_seed: int,
    episode_length: int,
    width: int,
    height: int,
    camera: int,
) -> SwarmBotsEnv:
    scenario = default_wall(
        first_wall_distance=1.0,
        unit_start_locations=make_preconnected_unit_start_locations((pool_seed,)),
    )
    return SwarmBotsEnv(
        scenario=scenario,
        episode_length=episode_length,
        render_mode="rgb_array",
        width=width,
        height=height,
        camera=camera,
    )


def extract_render_frame(frame: Any) -> np.ndarray | None:
    if isinstance(frame, (list, tuple)):
        if len(frame) == 0:
            return None
        return extract_render_frame(frame[0])
    if isinstance(frame, np.ndarray):
        if frame.ndim == 4:
            return frame[0]
        return frame
    return None


def make_action(action_sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    actions: dict[str, np.ndarray] = {}

    for key, value in action_sample.items():
        if key == 'connectors':
            actions[key] = np.ones_like(value)
        else:
            actions[key] = np.zeros_like(value)
    return actions


def record_seed_video(
    *,
    seed: int,
    output_path: Path,
    episode_length: int,
    steps: int,
    fps: int,
    width: int,
    height: int,
    camera: int,
) -> None:
    env = make_env(
        pool_seed=seed,
        episode_length=episode_length,
        width=width,
        height=height,
        camera=camera,
    )
    try:
        env.reset(seed=seed)
        first_frame = extract_render_frame(env.render())
        if first_frame is None:
            raise RuntimeError("Environment render returned None. Ensure render_mode='rgb_array'.")

        frames: list[np.ndarray] = [first_frame]
        zero_action = make_action(env.action_space.sample())

        for _ in range(steps):
            _, _, terminated, truncated, _ = env.step(zero_action)
            frame = extract_render_frame(env.render())
            if frame is not None:
                frames.append(frame)
            if terminated or truncated:
                break

        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(str(output_path), logger=None)
        print(f"Saved {output_path}")
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seeds = tuple(sorted(set(args.seeds)))
    if not seeds:
        raise ValueError("Expected at least one seed.")

    print(f"Recording {len(seeds)} seed videos into: {args.out_dir}")
    for seed in seeds:
        output_path = args.out_dir / f"spawn_seed_{seed}.mp4"
        record_seed_video(
            seed=seed,
            output_path=output_path,
            episode_length=args.episode_length,
            steps=args.steps,
            fps=args.fps,
            width=args.width,
            height=args.height,
            camera=args.camera,
        )


if __name__ == "__main__":
    main()
