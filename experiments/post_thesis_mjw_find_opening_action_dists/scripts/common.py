from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.post_thesis_action_dist_common import (
    PostThesisActionDistVariant,
    run_post_thesis_action_dist_experiment,
)

EXPERIMENT_RUN_NAME = "post_thesis_mjw_find_opening_action_dists"
SCENARIO_KWARGS: dict[str, object] = {"continuous_connector_actions": True}


def run_experiment(
        *,
        continuous_action_dist: PostThesisActionDistVariant,
        entrypoint_path: Path,
) -> None:
    run_post_thesis_action_dist_experiment(
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_name="find_opening",
        scenario_kwargs=SCENARIO_KWARGS,
        continuous_action_dist=continuous_action_dist,
        entrypoint_path=entrypoint_path,
    )
