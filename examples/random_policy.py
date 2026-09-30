from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from collections.abc import Mapping

from gymnasium import spaces
import torch

from swarmbots import evaluate_policy, make_scenario
from swarmbots.benchmark.evaluation import EVALUATION_SEEDS


class RandomPolicy:
    def __init__(self, benchmark_id: str, *, seed: int) -> None:
        self.action_space = make_scenario(benchmark_id, seed=seed).get_single_action_space()
        self.seed = seed
        self.generator: torch.Generator | None = None

    def __call__(
        self,
        observations: Mapping[str, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _ = episode_starts
        device = observations["local_obs"].device
        if self.generator is None:
            self.generator = torch.Generator(device=device).manual_seed(self.seed)
        batch_size = observations["local_obs"].shape[0]
        actions: dict[str, torch.Tensor] = {}
        for key, space in self.action_space.items():
            shape = (batch_size, *space.shape)
            if isinstance(space, spaces.Box):
                low = torch.as_tensor(space.low, device=device, dtype=torch.float32)
                high = torch.as_tensor(space.high, device=device, dtype=torch.float32)
                actions[key] = low + (high - low) * torch.rand(
                    shape,
                    device=device,
                    generator=self.generator,
                )
            elif isinstance(space, spaces.MultiBinary):
                actions[key] = torch.randint(
                    0,
                    2,
                    shape,
                    device=device,
                    dtype=torch.bool,
                    generator=self.generator,
                )
            else:
                raise TypeError(type(space).__name__)
        return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_id", nargs="?", default="SwarmBots-WallEasy-v0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(EVALUATION_SEEDS))
    parser.add_argument("--output", type=Path, default=Path("runs/random-policy.json"))
    parser.add_argument("--source-revision", help="Git commit of the benchmark checkout")
    args = parser.parse_args()

    results = [
        evaluate_policy(
            RandomPolicy(args.benchmark_id, seed=seed),
            args.benchmark_id,
            device=args.device,
            num_envs=args.num_envs,
            num_episodes=args.num_envs,
            seed=seed,
            action_mode="stochastic",
            policy_metadata={"name": "uniform-random", "seed": seed},
            source_revision=args.source_revision,
        )
        for seed in args.seeds
    ]
    seed_returns = [result.mean_return for result in results]
    seed_successes = [result.success_rate for result in results if result.success_rate is not None]
    report = {
        "benchmark_id": args.benchmark_id,
        "seeds": args.seeds,
        "mean_return": statistics.fmean(seed_returns),
        "std_seed_mean_return": statistics.pstdev(seed_returns),
        "mean_success_rate": statistics.fmean(seed_successes) if seed_successes else None,
        "std_seed_success_rate": statistics.pstdev(seed_successes) if seed_successes else None,
        "results": [result.to_dict() for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(results)} seed results to {args.output}")


if __name__ == "__main__":
    main()
