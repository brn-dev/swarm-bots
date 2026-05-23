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
    "lr_beta_nop",
    "gsde_nop",
    "beta_nop",
    "squashed_diag_gaussian_nop",
    "lr_beta_nop_shuffle_agents",
    "lr_beta_nop_shuffle_agents_preserve_prefix",
    "lr_beta_no_nop",
)
DISPLAY_NAME_OVERRIDES = {
    "lr_beta_nop": "L/R Beta + NOP",
    "gsde_nop": "gSDE + NOP",
    "beta_nop": "Beta + NOP",
    "squashed_diag_gaussian_nop": "Squashed Diag Gaussian + NOP",
    "lr_beta_nop_shuffle_agents": "L/R Beta + NOP, shuffled agents",
    "lr_beta_nop_shuffle_agents_preserve_prefix": (
        "L/R Beta + NOP, shuffled active-prefix agents"
    ),
    "lr_beta_no_nop": "L/R Beta, no NOP",
}
EXTRA_GROUP_SOURCES = {
    "lr_beta_nop": (
        REPO_ROOT / "runs" / "mat_nop_swarm_bots_wall_mjw_batch_env_sweep" / "1024x4",
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
