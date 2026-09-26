from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import statistics
from typing import Any, Protocol

import torch

from swarmbots.benchmark.registry import get_benchmark_spec, make_env

PROTOCOL_VERSION = "0.1"
POLICY_OBSERVATION_KEYS = ("local_obs", "global_obs", "agent_mask")


class Policy(Protocol):
    def __call__(
        self,
        observations: Mapping[str, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]: ...


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    benchmark_id: str
    seed: int
    episode_returns: tuple[float, ...]
    episode_lengths: tuple[int, ...]
    episode_successes: tuple[bool, ...] | None
    protocol_version: str = PROTOCOL_VERSION

    @property
    def mean_return(self) -> float:
        return statistics.fmean(self.episode_returns)

    @property
    def return_std(self) -> float:
        return statistics.pstdev(self.episode_returns)

    @property
    def mean_episode_length(self) -> float:
        return statistics.fmean(self.episode_lengths)

    @property
    def success_rate(self) -> float | None:
        if self.episode_successes is None:
            return None
        return statistics.fmean(self.episode_successes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "benchmark_id": self.benchmark_id,
            "seed": self.seed,
            "num_episodes": len(self.episode_returns),
            "mean_return": self.mean_return,
            "return_std": self.return_std,
            "mean_episode_length": self.mean_episode_length,
            "success_rate": self.success_rate,
            "episode_returns": list(self.episode_returns),
            "episode_lengths": list(self.episode_lengths),
            "episode_successes": None if self.episode_successes is None else list(self.episode_successes),
        }


def policy_observation(observations: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: observations[key] for key in POLICY_OBSERVATION_KEYS}


def evaluate_policy(
    policy: Policy,
    benchmark_id: str,
    *,
    num_episodes: int = 256,
    num_envs: int = 256,
    seed: int = 1_000,
    device: str | torch.device = "cuda",
    episode_length: int | None = None,
    scenario_kwargs: Mapping[str, object] | None = None,
    env_kwargs: Mapping[str, Any] | None = None,
) -> EvaluationResult:
    if num_episodes <= 0:
        raise ValueError(f"Expected num_episodes > 0, got {num_episodes}")
    if num_envs <= 0:
        raise ValueError(f"Expected num_envs > 0, got {num_envs}")

    spec = get_benchmark_spec(benchmark_id)
    env = make_env(
        benchmark_id,
        num_envs=min(num_envs, num_episodes),
        device=device,
        seed=seed,
        episode_length=episode_length,
        scenario_kwargs=scenario_kwargs,
        **dict(env_kwargs or {}),
    )
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[bool] | None = [] if spec.supports_success_metric else None

    try:
        observations, _ = env.reset(seed=seed)
        running_returns = torch.zeros(env.num_envs, device=env.device, dtype=torch.float64)
        running_lengths = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
        running_successes = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        episode_starts = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

        with torch.inference_mode():
            while len(returns) < num_episodes:
                actions = policy(policy_observation(observations), episode_starts)
                observations, rewards, terminations, truncations, infos = env.step(dict(actions))
                dones = terminations | truncations
                running_returns += rewards.to(dtype=torch.float64)
                running_lengths += 1
                if spec.supports_success_metric and "success" in infos:
                    running_successes |= infos["success"].to(dtype=torch.bool)

                for env_index in torch.nonzero(dones, as_tuple=True)[0].detach().cpu().tolist():
                    if len(returns) >= num_episodes:
                        break
                    returns.append(float(running_returns[env_index].item()))
                    lengths.append(int(running_lengths[env_index].item()))
                    if successes is not None:
                        successes.append(bool(running_successes[env_index].item()))

                running_returns[dones] = 0.0
                running_lengths[dones] = 0
                running_successes[dones] = False
                episode_starts = dones
    finally:
        env.close()

    return EvaluationResult(
        benchmark_id=benchmark_id,
        seed=seed,
        episode_returns=tuple(returns),
        episode_lengths=tuple(lengths),
        episode_successes=None if successes is None else tuple(successes),
    )
