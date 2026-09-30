from __future__ import annotations

import argparse
import json
from typing import Any

from gymnasium import spaces
import torch

from swarmbots.benchmark.registry import get_benchmark_spec, list_benchmarks, make_env


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
        print(f"{spec.id:<38} {spec.description}")


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
    return parser


def main() -> None:
    args = build_parser().parse_args()
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


if __name__ == "__main__":
    main()
