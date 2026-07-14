from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import plot_experiment_results


RUNS_DIR = REPO_ROOT / "runs"
EXPERIMENT_RUN_DIR = RUNS_DIR / "mjw_po_wall_medium_1024x1_250M_tmasac_cont_conn_act"
BASELINE_EXPERIMENT_RUN_DIR = RUNS_DIR / "mjw_po_wall_medium_1024x4_250M_cont_conn_act"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
MAT_DEC_BASELINE_GROUP = "mat_dec_baseline"
MAT_QCX_BASELINE_GROUP = "mat_qcx_baseline"
GROUP_ORDER = (
    MAT_DEC_BASELINE_GROUP,
    MAT_QCX_BASELINE_GROUP,
    "tmasac_rsmk",
    "tmasac_predicted_std",
)
DISPLAY_NAME_OVERRIDES = {
    MAT_DEC_BASELINE_GROUP: "MAT-Dec + NOP baseline",
    MAT_QCX_BASELINE_GROUP: "MAT-QCX + NOP baseline",
    "tmasac_rsmk": "TMASAC RSMK + NOP",
    "tmasac_predicted_std": "TMASAC Predicted Std + NOP",
}
EXTRA_GROUP_SOURCES = {
    MAT_DEC_BASELINE_GROUP: (BASELINE_EXPERIMENT_RUN_DIR / "mat_dec",),
    MAT_QCX_BASELINE_GROUP: (BASELINE_EXPERIMENT_RUN_DIR / "mat_qcx",),
}
THEORETICAL_MAXIMUM = None


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        theoretical_maximum=THEORETICAL_MAXIMUM,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        extra_group_sources=EXTRA_GROUP_SOURCES,
        run_length_limit=250_000_000,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
