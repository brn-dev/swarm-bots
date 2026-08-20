from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.thesis_mjw_po_wall_medium.plot_results import (
    EXTRA_GROUP_SOURCES as THESIS_PO_WALL_GROUP_SOURCES,
)
from plot_logs.experiment_results import plot_experiment_results

EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / "thesis_parallel_env_ablation_po_wall_medium"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
ROLLOUT_CONFIG_NAMES = ("1024x4", "512x8", "256x16", "128x32", "64x64")
# ALGORITHM_NAMES = ("mat_ind", "mat_qcx")
ALGORITHM_NAMES = ("mat_qcx",)
GROUP_ORDER = tuple(
    f"{algorithm_name}_{rollout_config_name}"
    for algorithm_name in ALGORITHM_NAMES
    for rollout_config_name in ROLLOUT_CONFIG_NAMES
)
DISPLAY_NAME_OVERRIDES = {
    f"{algorithm_name}_{rollout_config_name}": (
        f"{display_name}, {rollout_config_name}"
    )
    for algorithm_name, display_name in (
        # ("mat_ind", "MAT-IND"),
        ("mat_qcx", "MAT-QCX"),
    )
    for rollout_config_name in ROLLOUT_CONFIG_NAMES
}
EXTRA_GROUP_SOURCES = {
    # "mat_ind_1024x4": THESIS_PO_WALL_GROUP_SOURCES["mat_ind"],
    "mat_qcx_1024x4": THESIS_PO_WALL_GROUP_SOURCES["mat_qcx"],
}


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        extra_group_sources=EXTRA_GROUP_SOURCES,
        run_length_limit=100_000_000,
        cut_at_limit=True,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
