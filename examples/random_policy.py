from __future__ import annotations

import argparse
from collections.abc import Mapping

from gymnasium import spaces
import torch

from swarmbots import evaluate_policy, make_scenario


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
    parser.add_argument("--num-episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    result = evaluate_policy(
        RandomPolicy(args.benchmark_id, seed=args.seed),
        args.benchmark_id,
        device=args.device,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        seed=args.seed,
    )
    print(result.to_dict())


if __name__ == "__main__":
    main()
