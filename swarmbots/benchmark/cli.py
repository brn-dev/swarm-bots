from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
import importlib
import json
from pathlib import Path
from typing import Any

from gymnasium import spaces
import torch

from swarmbots.benchmark.registry import get_benchmark_spec, list_benchmarks, make_env
from swarmbots.utils.recording_resolution import DEFAULT_RECORDING_HEIGHT, DEFAULT_RECORDING_WIDTH
from swarmbots.utils.torch_utils import resolve_torch_device


def _random_actions(env: Any, generator: torch.Generator) -> dict[str, torch.Tensor]:
    actions: dict[str, torch.Tensor] = {}
    for key, space in env.single_action_space.items():
        shape = (env.num_envs, *space.shape)
        if isinstance(space, spaces.Box):
            low = torch.as_tensor(space.low, device=env.device, dtype=torch.float32)
            high = torch.as_tensor(space.high, device=env.device, dtype=torch.float32)
            actions[key] = low + (high - low) * torch.rand(
                shape,
                device=env.device,
                dtype=torch.float32,
                generator=generator,
            )
        elif isinstance(space, spaces.MultiBinary):
            actions[key] = torch.randint(0, 2, shape, device=env.device, dtype=torch.bool, generator=generator)
        else:
            raise TypeError(f"Unsupported action space for {key!r}: {type(space).__name__}")
    return actions


def _list_command(*, as_json: bool) -> None:
    specs = list_benchmarks()
    if as_json:
        print(json.dumps([spec.to_dict() for spec in specs], indent=2))
        return
    for spec in specs:
        print(f"{spec.id:<38} [{spec.maturity}] {spec.description}")


def _describe_command(benchmark_id: str) -> None:
    print(json.dumps(get_benchmark_spec(benchmark_id).to_dict(), indent=2))


def _smoke_command(
    benchmark_id: str,
    *,
    device: str,
    num_envs: int,
    steps: int,
    seed: int,
    compiled: bool = False,
) -> None:
    env = make_env(
        benchmark_id,
        num_envs=num_envs,
        device=device,
        seed=seed,
        scenario_kwargs={"compile_reward_kernel": compiled},
        compile_tensor_operations=compiled,
    )
    try:
        observations, _ = env.reset(seed=seed)
        generator = torch.Generator(device=env.device).manual_seed(seed)
        total_reward = torch.zeros(env.num_envs, device=env.device)
        completed_episodes = 0
        for _ in range(steps):
            observations, rewards, terminations, truncations, _ = env.step(_random_actions(env, generator))
            total_reward += rewards
            completed_episodes += int((terminations | truncations).sum().item())

        summary = {
            "benchmark_id": benchmark_id,
            "device": str(env.device),
            "num_envs": env.num_envs,
            "steps": steps,
            "compiled": compiled,
            "completed_episodes": completed_episodes,
            "mean_accumulated_reward": float(total_reward.mean().item()),
            "observation_shapes": {key: list(value.shape) for key, value in observations.items()},
            "action_shapes": {key: list(space.shape) for key, space in env.action_space.items()},
        }
        print(json.dumps(summary, indent=2))
    finally:
        env.close()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def _camera(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _record_command(args: argparse.Namespace) -> None:
    video_folder = args.output or Path("recordings") / args.benchmark_id / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    device = resolve_torch_device(args.device)
    options = {
        "video_folder": video_folder,
        "num_episodes": args.episodes,
        "max_parallel_episodes": args.parallel,
        "device": device,
        "seed": args.seed,
        "episode_length": args.episode_length,
        "scenario_kwargs": args.scenario_kwargs,
        "env_kwargs": args.env_kwargs,
        "fps": args.fps,
        "fps_mode": args.fps_mode,
        "frame_stride": args.frame_stride,
        "width": args.width,
        "height": args.height,
        "camera": args.camera,
    }
    if args.prefix is not None:
        options["video_name_prefix"] = args.prefix
    if args.checkpoint is not None:
        from swarmbots.learn import record_checkpoint

        record_checkpoint(
            args.checkpoint,
            args.benchmark_id,
            args.variant,
            deterministic=not args.stochastic,
            exploration_noise=args.exploration_noise,
            continuous_action_dist=args.continuous_action_dist,
            use_nop=args.use_nop,
            model_scale=None if args.model_scale == "legacy" else args.model_scale,
            compile_modules=args.compile_policy,
            policy_kwargs=args.policy_kwargs,
            **options,
        )
    else:
        from swarmbots.benchmark.recording import record_policy

        module_name, factory_name = args.policy.split(":", maxsplit=1)
        factory = getattr(importlib.import_module(module_name), factory_name)
        torch.manual_seed(args.seed)
        policy = factory(
            benchmark_id=args.benchmark_id,
            device=device,
            seed=args.seed,
            deterministic=not args.stochastic,
        )
        record_policy(policy, args.benchmark_id, **options)
    print(f"Saved {args.episodes} episode recordings to {video_folder}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swarmbots", description="SwarmBots MJWarp benchmark tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list benchmark IDs")
    list_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    describe_parser = subparsers.add_parser("describe", help="describe one benchmark")
    describe_parser.add_argument("benchmark_id")

    smoke_parser = subparsers.add_parser("smoke", help="run random actions through an environment")
    smoke_parser.add_argument("benchmark_id")
    smoke_parser.add_argument("--device", default="auto")
    smoke_parser.add_argument("--num-envs", type=_positive_int, default=2)
    smoke_parser.add_argument("--steps", type=_positive_int, default=8)
    smoke_parser.add_argument("--seed", type=int, default=42)
    smoke_parser.add_argument("--compiled", action="store_true", help="compile tensor operations and rewards")

    record_parser = subparsers.add_parser("record", help="record a checkpoint or custom policy on a scenario")
    record_parser.add_argument("benchmark_id")
    policy_source = record_parser.add_mutually_exclusive_group(required=True)
    policy_source.add_argument("--checkpoint", type=Path, help="learning checkpoint (.pt); also requires --variant")
    policy_source.add_argument("--policy", help="custom policy factory as importable module:function")
    record_parser.add_argument("--variant", help="learning preset matching the checkpoint")
    record_parser.add_argument("--output", type=Path, help="video directory (default: recordings/task/timestamp)")
    record_parser.add_argument("--prefix", help="video filename prefix")
    record_parser.add_argument("--episodes", type=_positive_int, default=5)
    record_parser.add_argument("--parallel", type=_positive_int, default=4)
    record_parser.add_argument("--device", default="auto")
    record_parser.add_argument("--seed", type=int, default=1_000)
    record_parser.add_argument("--episode-length", type=_positive_int)
    record_parser.add_argument("--stochastic", action="store_true", help="sample actions instead of using modes")
    record_parser.add_argument(
        "--exploration-noise", type=float, help="override DDPG/TD3 checkpoint exploration noise for --stochastic"
    )
    record_parser.add_argument("--fps", type=_positive_int, default=30)
    record_parser.add_argument("--fps-mode", choices=("compensate_stride", "fixed"), default="compensate_stride")
    record_parser.add_argument("--frame-stride", type=_positive_int, default=1)
    record_parser.add_argument("--width", type=_positive_int, default=DEFAULT_RECORDING_WIDTH)
    record_parser.add_argument("--height", type=_positive_int, default=DEFAULT_RECORDING_HEIGHT)
    record_parser.add_argument(
        "--camera", type=_camera, default=-1, help="camera name or ID; -1 uses the scenario camera"
    )
    record_parser.add_argument("--scenario-kwargs", type=_json_object, default={}, help="scenario overrides as JSON")
    record_parser.add_argument("--env-kwargs", type=_json_object, default={}, help="simulator overrides as JSON")
    record_parser.add_argument(
        "--policy-kwargs", type=_json_object, default={}, help="checkpoint architecture overrides as JSON"
    )
    record_parser.add_argument("--continuous-action-dist", help="checkpoint action distribution override")
    record_parser.add_argument("--use-nop", action=argparse.BooleanOptionalAction, default=None)
    record_parser.add_argument("--model-scale", default="5M NOP1M", help="matching checkpoint scale; legacy uses explicit widths")
    record_parser.add_argument("--compile-policy", action="store_true", help="compile checkpoint policy modules")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        _list_command(as_json=args.json)
    elif args.command == "describe":
        _describe_command(args.benchmark_id)
    elif args.command == "smoke":
        _smoke_command(
            args.benchmark_id,
            device=args.device,
            num_envs=args.num_envs,
            steps=args.steps,
            seed=args.seed,
            compiled=args.compiled,
        )
    elif args.command == "record":
        if args.checkpoint is not None and args.variant is None:
            parser.error("--checkpoint requires --variant matching the saved policy")
        if args.policy is not None and ":" not in args.policy:
            parser.error("--policy must have the form module:function")
        if args.policy is not None and args.exploration_noise is not None:
            parser.error("--exploration-noise requires a DDPG/TD3 checkpoint")
        _record_command(args)


if __name__ == "__main__":
    main()
