from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import plot_experiment_results


EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / "mjw_po_wall_medium_1024x1_250M_tmasac_cont_conn_act"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
VARIANT_GROUPS = (
    "tmasac_rsmk",
    "tmasac_predicted_std",
)
DISPLAY_NAME_OVERRIDES = {
    "tmasac_rsmk": "TMASAC RSMK + NOP",
    "tmasac_predicted_std": "TMASAC Predicted Std + NOP",
}
THEORETICAL_MAXIMUM = None


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=VARIANT_GROUPS,
        theoretical_maximum=THEORETICAL_MAXIMUM,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        run_length_limit=250_000_000,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
