from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import statistics
from typing import Any, Literal, Protocol

import torch

from swarmbots.benchmark.registry import get_benchmark_spec, make_env
from swarmbots.benchmark.metadata import runtime_metadata, serialize_settings

PROTOCOL_VERSION = "0.1"
POLICY_OBSERVATION_KEYS = ("local_obs", "global_obs", "agent_mask")
EVALUATION_SEEDS = tuple(range(1_000, 1_005))


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
    metadata: dict[str, Any] = field(default_factory=dict)

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
            "metadata": self.metadata,
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
    action_mode: Literal["deterministic", "stochastic", "unspecified"] = "unspecified",
    policy_metadata: Mapping[str, Any] | None = None,
    source_revision: str | None = None,
) -> EvaluationResult:
    """Evaluate one episode per lane, returning episodes in lane order.

    Action mode describes the caller's policy; it does not change its behavior.
    Use separate seeded calls for more samples instead of recycling early-finishing lanes.
    """
    if num_episodes <= 0:
        raise ValueError(f"Expected num_episodes > 0, got {num_episodes}")
    if num_envs <= 0:
        raise ValueError(f"Expected num_envs > 0, got {num_envs}")
    if num_episodes > num_envs:
        raise ValueError("num_episodes must be <= num_envs: evaluation accepts only the first episode per lane")
    if action_mode not in ("deterministic", "stochastic", "unspecified"):
        raise ValueError(f"Unknown action_mode: {action_mode!r}")

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
    try:
        canonical_scenario = not scenario_kwargs and not env_kwargs and episode_length in (None, spec.episode_length)
        metadata = serialize_settings({
            "runtime": runtime_metadata(env.device),
            "benchmark_maturity": spec.maturity,
            "source_revision": source_revision,
            "num_envs": env.num_envs,
            "episode_limit": spec.episode_length if episode_length is None else episode_length,
            "scenario_overrides": dict(scenario_kwargs or {}),
            "environment_overrides": dict(env_kwargs or {}),
            "resolved_settings": env.get_settings(),
            "canonical_scenario": canonical_scenario,
            "canonical_evaluation": (
                canonical_scenario and env.num_envs == 256 and seed in EVALUATION_SEEDS
                and action_mode == "deterministic"
            ),
            "action_mode": action_mode,
            "policy": dict(policy_metadata or {}),
        })
        observations, _ = env.reset(seed=seed)
        running_returns = torch.zeros(env.num_envs, device=env.device, dtype=torch.float64)
        running_lengths = torch.zeros(env.num_envs, device=env.device, dtype=torch.int64)
        running_successes = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        episode_starts = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        completed = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

        with torch.inference_mode():
            while not bool(completed.all()):
                actions = policy(policy_observation(observations), episode_starts)
                observations, rewards, terminations, truncations, infos = env.step(dict(actions))
                dones = terminations | truncations
                running_returns += torch.where(completed, 0.0, rewards.to(dtype=torch.float64))
                running_lengths += ~completed
                if spec.supports_success_metric:
                    running_successes |= infos["success"].to(dtype=torch.bool) & ~completed
                completed |= dones
                episode_starts = dones
    finally:
        env.close()

    return EvaluationResult(
        benchmark_id=benchmark_id,
        seed=seed,
        episode_returns=tuple(running_returns.cpu().tolist()),
        episode_lengths=tuple(running_lengths.cpu().tolist()),
        episode_successes=tuple(running_successes.cpu().tolist()) if spec.supports_success_metric else None,
        metadata=metadata,
    )
