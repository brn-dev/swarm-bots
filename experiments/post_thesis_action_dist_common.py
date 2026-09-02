from __future__ import annotations

from pathlib import Path
from typing import Literal

from experiments.mjw_experiment_common import MJWScenarioName
from experiments.tmasac_experiment_common import run_tmasac_experiment

PostThesisActionDistVariant = Literal[
    "bernstein_6",
    "bernstein_8",
    "rqs_4",
    "rqs_6",
]

ACTION_DIST_VARIANTS: tuple[PostThesisActionDistVariant, ...] = (
    "bernstein_6",
    "bernstein_8",
    "rqs_4",
    "rqs_6",
)
BASELINE_GROUP = "tmasac_baseline"


def action_dist_group_name(variant: PostThesisActionDistVariant) -> str:
    return f"{BASELINE_GROUP}_{variant}"


def run_post_thesis_action_dist_experiment(
        *,
        experiment_run_name: str,
        scenario_name: MJWScenarioName,
        scenario_kwargs: dict[str, object],
        continuous_action_dist: PostThesisActionDistVariant,
        entrypoint_path: Path,
) -> None:
    run_tmasac_experiment(
        experiment_run_name=experiment_run_name,
        scenario_name=scenario_name,
        scenario_kwargs=scenario_kwargs,
        variant="tmasac_baseline",
        entrypoint_path=entrypoint_path,
        continuous_action_dist=continuous_action_dist,
        use_nop=True,
        variant_name=action_dist_group_name(continuous_action_dist),
    )
