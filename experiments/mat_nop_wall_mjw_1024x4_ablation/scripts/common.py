from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mat_nop_wall_mjw_common import ContinuousActionDistVariant, run_experiment

EXPERIMENT_RUN_NAME = "mat_nop_swarm_bots_wall_mjw_1024x4_ablation"


def run_ablation(
        *,
        variant_name: str,
        entrypoint_path: Path,
        continuous_action_dist: ContinuousActionDistVariant = "sticky_lr_beta",
        use_nop: bool = True,
) -> None:
    run_experiment(
        num_envs=1024,
        rollout_steps_per_env=4,
        variant_name=variant_name,
        entrypoint_path=entrypoint_path,
        continuous_action_dist=continuous_action_dist,
        use_nop=use_nop,
        experiment_run_name=EXPERIMENT_RUN_NAME,
    )
