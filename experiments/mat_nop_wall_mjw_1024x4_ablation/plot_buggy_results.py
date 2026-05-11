from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import plot_experiment_results


EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / ".buggy" / "mat_nop_swarm_bots_wall_mjw_1024x4_ablation"
OUTPUT_DIR = Path(__file__).resolve().parent / "results_buggy"
GROUP_ORDER = (
    "sticky_lr_beta_nop",
    "gsde_nop",
    "lr_beta_nop",
    "sticky_lr_beta_no_nop",
)
DISPLAY_NAME_OVERRIDES = {
    "sticky_lr_beta_nop": "Sticky L/R Beta + NOP",
    "gsde_nop": "gSDE + NOP",
    "lr_beta_nop": "L/R Beta + NOP",
    "sticky_lr_beta_no_nop": "Sticky L/R Beta, no NOP",
}
EXTRA_GROUP_SOURCES = {
    "sticky_lr_beta_nop": (
        REPO_ROOT / "runs" / ".buggy" / "mat_nop_swarm_bots_wall_mjw_batch_env_sweep" / "1024x4",
    ),
}
THEORETICAL_MAXIMUM = 10.5


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        theoretical_maximum=THEORETICAL_MAXIMUM,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        extra_group_sources=EXTRA_GROUP_SOURCES,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
