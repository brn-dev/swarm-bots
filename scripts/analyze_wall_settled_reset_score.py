from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "glfw")

from swarmbots.mj_env.scenarios.obstacle_street_scenario import ObstacleStreetScenario
from swarmbots.mj_env.scenarios.scenario_presets import default_wall
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv


@dataclass(frozen=True)
class ResetScore:
    seed: int
    progress: float
    available_forward_score: float
    available_wall_score: float
    available_progress_score: float
    forfeited_wall_score: float
    active_units: int
    passed_milestones: int
    remaining_milestones: int
    total_milestones: int


@dataclass(frozen=True)
class ScoreSummary:
    count: int
    mean: float
    std: float
    min: float
    max: float


@dataclass(frozen=True)
class AnalysisSummary:
    resets: int
    seed_start: int
    pool_size: int | None
    active_pool_size: int | None
    reset_settle_time: float
    reset_settle_timestep_scale: float
    unsettled_available_progress_score: ScoreSummary
    settled_available_progress_score: ScoreSummary
    unsettled_minus_settled_available_progress_score: ScoreSummary
    settled_available_forward_score: ScoreSummary
    settled_available_wall_score: ScoreSummary
    settled_forfeited_wall_score: ScoreSummary
    settled_passed_milestones: ScoreSummary
    settled_remaining_milestones: ScoreSummary


def make_wall_env(*, seed: int, reset_settle_time: float) -> SwarmBotsEnv:
    scenario = default_wall(seed=seed, reset_settle_time=reset_settle_time)
    return SwarmBotsEnv(scenario=scenario)


def summarize(values: Sequence[float]) -> ScoreSummary:
    return ScoreSummary(
        count=len(values),
        mean=float(mean(values)),
        std=float(pstdev(values)) if len(values) > 1 else 0.0,
        min=float(min(values)),
        max=float(max(values)),
    )


def compute_reset_score(env: SwarmBotsEnv, *, seed: int) -> ResetScore:
    env.reset(seed=seed)
    scenario = env.scenario
    if not isinstance(scenario, ObstacleStreetScenario):
        raise TypeError(f"Expected ObstacleStreetScenario, got {type(scenario).__name__}")
    state = env.scenario_state
    if state is None:
        raise RuntimeError("Environment reset did not create scenario state.")

    units_active_mask = np.asarray(state.get("units_active_mask"), dtype=bool)
    active_units = int(units_active_mask.sum())
    next_threshold_for_unit = np.asarray(state["next_threshold_for_unit"], dtype=int)
    active_next_thresholds = next_threshold_for_unit[units_active_mask]
    total_thresholds_per_unit = int(np.asarray(state["wall_pass_absolute_thresholds"]).size)
    total_milestones = active_units * total_thresholds_per_unit
    passed_milestones = int(active_next_thresholds.sum())
    remaining_milestones = total_milestones - passed_milestones

    thresholds_per_wall = int(scenario.wall_pass_thresholds.size)
    progress_reward_weight = float(scenario.reward_weights["progress_reward_weight"])
    wall_denominator = active_units * thresholds_per_wall
    available_wall_score = (
        (remaining_milestones / wall_denominator) * scenario.wall_pass_reward_weight * progress_reward_weight
        if wall_denominator > 0
        else 0.0
    )
    forfeited_wall_score = (
        (passed_milestones / wall_denominator) * scenario.wall_pass_reward_weight * progress_reward_weight
        if wall_denominator > 0
        else 0.0
    )

    progress = float(state["progress"])
    if scenario.forward_reward_max_y is None:
        raise ValueError("Wall score analysis needs a finite forward_reward_max_y.")
    available_forward_score = max(0.0, scenario.forward_reward_max_y - progress)
    available_forward_score *= scenario.forward_reward_weight * progress_reward_weight

    return ResetScore(
        seed=seed,
        progress=progress,
        available_forward_score=float(available_forward_score),
        available_wall_score=float(available_wall_score),
        available_progress_score=float(available_forward_score + available_wall_score),
        forfeited_wall_score=float(forfeited_wall_score),
        active_units=active_units,
        passed_milestones=passed_milestones,
        remaining_milestones=remaining_milestones,
        total_milestones=total_milestones,
    )


def collect_scores(*, resets: int, seed_start: int, reset_settle_time: float) -> tuple[list[ResetScore], SwarmBotsEnv]:
    env = make_wall_env(seed=seed_start, reset_settle_time=reset_settle_time)
    scores: list[ResetScore] = []
    for offset in range(resets):
        scores.append(compute_reset_score(env, seed=seed_start + offset))
    return scores, env


def build_summary(
    *,
    settled_scores: Sequence[ResetScore],
    unsettled_scores: Sequence[ResetScore],
    env: SwarmBotsEnv,
    seed_start: int,
) -> AnalysisSummary:
    unsettled_minus_settled_available_progress_score = [
        unsettled.available_progress_score - settled.available_progress_score
        for unsettled, settled in zip(unsettled_scores, settled_scores, strict=True)
    ]
    scenario = env.scenario
    return AnalysisSummary(
        resets=len(settled_scores),
        seed_start=seed_start,
        pool_size=env.get_swarm_pool_size(),
        active_pool_size=env.get_active_swarm_pool_size(),
        reset_settle_time=float(scenario.reset_settle_time),
        reset_settle_timestep_scale=float(scenario.reset_settle_timestep_scale),
        unsettled_available_progress_score=summarize([score.available_progress_score for score in unsettled_scores]),
        settled_available_progress_score=summarize([score.available_progress_score for score in settled_scores]),
        unsettled_minus_settled_available_progress_score=summarize(unsettled_minus_settled_available_progress_score),
        settled_available_forward_score=summarize([score.available_forward_score for score in settled_scores]),
        settled_available_wall_score=summarize([score.available_wall_score for score in settled_scores]),
        settled_forfeited_wall_score=summarize([score.forfeited_wall_score for score in settled_scores]),
        settled_passed_milestones=summarize([float(score.passed_milestones) for score in settled_scores]),
        settled_remaining_milestones=summarize([float(score.remaining_milestones) for score in settled_scores]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate how much weighted progress reward remains after default wall resets are settled. "
            "The default wall preset uses the regular 50-entry pre-connected swarm pool."
        )
    )
    parser.add_argument("--resets", type=int, default=1000, help="Number of paired reset seeds to sample.")
    parser.add_argument("--seed-start", type=int, default=1, help="First reset seed to evaluate.")
    parser.add_argument("--settle-time", type=float, default=1.0, help="Settled reset time for the wall preset.")
    parser.add_argument("--json", action="store_true", help="Print the full summary as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resets <= 0:
        raise ValueError(f"Expected --resets > 0, got {args.resets}")
    if args.settle_time < 0:
        raise ValueError(f"Expected --settle-time >= 0, got {args.settle_time}")

    settled_scores, settled_env = collect_scores(
        resets=args.resets,
        seed_start=args.seed_start,
        reset_settle_time=args.settle_time,
    )
    unsettled_scores, _unsettled_env = collect_scores(
        resets=args.resets,
        seed_start=args.seed_start,
        reset_settle_time=0.0,
    )
    summary = build_summary(
        settled_scores=settled_scores,
        unsettled_scores=unsettled_scores,
        env=settled_env,
        seed_start=args.seed_start,
    )

    if args.json:
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
        return

    print(f"resets: {summary.resets}")
    print(f"seed_start: {summary.seed_start}")
    print(f"pool_size: {summary.pool_size}")
    print(f"active_pool_size: {summary.active_pool_size}")
    print(f"reset_settle_time: {summary.reset_settle_time}")
    print(f"reset_settle_timestep_scale: {summary.reset_settle_timestep_scale}")
    print()
    print("weighted progress score available after settled reset:")
    print(f"  mean: {summary.settled_available_progress_score.mean:.6f}")
    print(f"  std:  {summary.settled_available_progress_score.std:.6f}")
    print(f"  min:  {summary.settled_available_progress_score.min:.6f}")
    print(f"  max:  {summary.settled_available_progress_score.max:.6f}")
    print("weighted progress score available without settling:")
    print(f"  mean: {summary.unsettled_available_progress_score.mean:.6f}")
    print("unsettled minus settled available weighted progress score:")
    print(f"  mean: {summary.unsettled_minus_settled_available_progress_score.mean:.6f}")
    print(f"  std:  {summary.unsettled_minus_settled_available_progress_score.std:.6f}")
    print("settled score components:")
    print(f"  forward mean:             {summary.settled_available_forward_score.mean:.6f}")
    print(f"  wall remaining mean:      {summary.settled_available_wall_score.mean:.6f}")
    print(f"  wall forfeited mean:       {summary.settled_forfeited_wall_score.mean:.6f}")
    print("settled wall milestones:")
    print(f"  passed mean:    {summary.settled_passed_milestones.mean:.6f}")
    print(f"  remaining mean: {summary.settled_remaining_milestones.mean:.6f}")


if __name__ == "__main__":
    main()
