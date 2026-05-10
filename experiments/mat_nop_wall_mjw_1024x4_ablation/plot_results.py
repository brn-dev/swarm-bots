from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import plot_experiment_results


EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / "mat_nop_swarm_bots_wall_mjw_1024x4_ablation"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
GROUP_ORDER = (
    "sticky_lr_beta_nop",
    "gsde_nop",
    "lr_beta_nop",
    "sticky_lr_beta_no_nop",
)
THEORETICAL_MAXIMUM = 10.5


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        theoretical_maximum=THEORETICAL_MAXIMUM,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
