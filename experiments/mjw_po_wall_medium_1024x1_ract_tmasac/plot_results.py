from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import plot_experiment_results

EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / "mjw_po_wall_medium_1024x1_ract_tmasac"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
GROUP_ORDER = tuple(
    f"{cell}_{layout}"
    for cell in ("lstm", "slstm", "smlstm")
    for layout in ("small_end", "big_end", "two_small")
)
DISPLAY_NAME_OVERRIDES = {
    group: group.replace("smlstm", "sLSTM + mLSTM")
    .replace("slstm", "sLSTM")
    .replace("lstm", "LSTM")
    .replace("_", " ")
    for group in GROUP_ORDER
}


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        run_length_limit=100_000_000,
        cut_at_limit=True,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
