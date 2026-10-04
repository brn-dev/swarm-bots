from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import torch

from swarmbots.mjw_env.mjw_swarm_bots_vector_env import MJWSwarmBotsVectorEnv
from swarmbots.mjw_env.scenarios.base_mjw_scenario import BaseMJWScenario
from swarmbots.mjw_env.scenarios.mjw_scenario_presets import (
    default_bridge,
    default_climb,
    default_dual_payload_plane,
    default_find_opening,
    default_move_to,
    default_multi_payload_goal,
    default_payload_plane,
    default_payload_step,
    default_vertical_reach,
    easy_partially_observable_wall,
    easy_wall,
    hard_partially_observable_wall,
    hard_wall,
    medium_partially_observable_wall,
    medium_wall,
)

ScenarioFactory = Callable[..., BaseMJWScenario]


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    id: str
    description: str
    category: str
    scenario_factory: ScenarioFactory
    episode_length: int = 500
    supports_success_metric: bool = True
    maturity: Literal["alpha", "beta", "stable"] = "alpha"

    def to_dict(self) -> dict[str, str | int | bool]:
        return {
            "id": self.id,
            "description": self.description,
            "category": self.category,
            "maturity": self.maturity,
            "episode_length": self.episode_length,
            "supports_success_metric": self.supports_success_metric,
        }


_SPECS = (
    BenchmarkSpec("SwarmBots-WallEasy-v0", "Fixed 0.2 m wall traversal.", "obstacle", easy_wall, maturity="beta"),
    BenchmarkSpec("SwarmBots-WallMedium-v0", "Fixed 0.3 m wall traversal.", "obstacle", medium_wall, maturity="beta"),
    BenchmarkSpec("SwarmBots-WallHard-v0", "Fixed 0.4 m wall traversal.", "obstacle", hard_wall, maturity="beta"),
    BenchmarkSpec(
        "SwarmBots-POWallEasy-v0",
        "Connected locomotion over a randomized hidden 0.25 m wall.",
        "obstacle",
        easy_partially_observable_wall,
        maturity="beta",
    ),
    BenchmarkSpec(
        "SwarmBots-POWallMedium-v0",
        "Connected locomotion over a randomized hidden 0.3 m wall.",
        "obstacle",
        medium_partially_observable_wall,
        maturity="beta",
    ),
    BenchmarkSpec(
        "SwarmBots-POWallHard-v0",
        "Connected locomotion over a randomized hidden 0.4 m wall.",
        "obstacle",
        hard_partially_observable_wall,
        maturity="beta",
    ),
    BenchmarkSpec("SwarmBots-Bridge-v0", "Narrow movable bridge traversal.", "obstacle", default_bridge),
    BenchmarkSpec(
        "SwarmBots-FindOpening-v0",
        "Exploration within an episode to locate and traverse a hidden opening.",
        "partial-observability",
        default_find_opening,
        maturity="beta",
    ),
    BenchmarkSpec("SwarmBots-Climb-v0", "Elevated platform climbing.", "locomotion", default_climb),
    BenchmarkSpec(
        "SwarmBots-VerticalReach-v0",
        "Reaching a goal volume above the swarm.",
        "locomotion",
        default_vertical_reach,
    ),
    BenchmarkSpec(
        "SwarmBots-PayloadPlane-v0",
        "Single-payload transport across a plane.",
        "transport",
        default_payload_plane,
        supports_success_metric=False,
    ),
    BenchmarkSpec(
        "SwarmBots-PayloadStep-v0",
        "Payload transport over a step.",
        "transport",
        default_payload_step,
    ),
    BenchmarkSpec(
        "SwarmBots-DualPayloadPlane-v0",
        "Coordinated transport of two payloads.",
        "transport",
        default_dual_payload_plane,
        supports_success_metric=False,
    ),
    BenchmarkSpec(
        "SwarmBots-MultiPayloadGoal-v0",
        "Transport of a variable payload set to assigned goals.",
        "transport",
        default_multi_payload_goal,
    ),
    BenchmarkSpec(
        "SwarmBots-MoveTo-v0",
        "Navigation toward a sampled planar goal.",
        "locomotion",
        default_move_to,
        supports_success_metric=False,
    ),
)

BENCHMARKS: Mapping[str, BenchmarkSpec] = MappingProxyType({spec.id: spec for spec in _SPECS})
ALL_BENCHMARK_IDS = tuple(BENCHMARKS)
CORE_BENCHMARK_IDS = (
    "SwarmBots-WallMedium-v0",
    "SwarmBots-POWallMedium-v0",
    "SwarmBots-Bridge-v0",
    "SwarmBots-FindOpening-v0",
    "SwarmBots-Climb-v0",
    "SwarmBots-VerticalReach-v0",
    "SwarmBots-PayloadStep-v0",
    "SwarmBots-MultiPayloadGoal-v0",
)


def get_benchmark_spec(benchmark_id: str) -> BenchmarkSpec:
    try:
        return BENCHMARKS[benchmark_id]
    except KeyError as error:
        available = ", ".join(ALL_BENCHMARK_IDS)
        raise KeyError(f"Unknown benchmark {benchmark_id!r}. Available benchmarks: {available}") from error


def list_benchmarks() -> tuple[BenchmarkSpec, ...]:
    return _SPECS


def make_scenario(
    benchmark_id: str,
    *,
    seed: int | None = None,
    **scenario_kwargs: object,
) -> BaseMJWScenario:
    spec = get_benchmark_spec(benchmark_id)
    return spec.scenario_factory(seed=seed, **scenario_kwargs)


def make_env(
    benchmark_id: str,
    *,
    num_envs: int,
    device: str | torch.device = "cuda",
    seed: int | None = None,
    episode_length: int | None = None,
    scenario_kwargs: Mapping[str, object] | None = None,
    **env_kwargs: Any,
) -> MJWSwarmBotsVectorEnv:
    spec = get_benchmark_spec(benchmark_id)
    scenario = make_scenario(benchmark_id, seed=seed, **dict(scenario_kwargs or {}))
    return MJWSwarmBotsVectorEnv(
        scenario,
        num_envs=num_envs,
        episode_length=spec.episode_length if episode_length is None else episode_length,
        device=device,
        **env_kwargs,
    )
