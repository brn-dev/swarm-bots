from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import ExperimentPlotSelection, plot_experiment_results


EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / "mjw_po_wall_medium_1024x4_250M_attention_heads"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
MAT_DEC_GROUP_ORDER = (
    "mat_dec_heads_1",
    "mat_dec_heads_2",
    "mat_dec_heads_4",
    "mat_dec_heads_8",
)
MAT_QCX_GROUP_ORDER = (
    "mat_qcx_heads_1",
    "mat_qcx_heads_2",
    "mat_qcx_heads_4",
    "mat_qcx_heads_8",
)
GROUP_ORDER = (*MAT_DEC_GROUP_ORDER, *MAT_QCX_GROUP_ORDER)
DISPLAY_NAME_OVERRIDES = {
    "mat_dec_heads_1": "MAT-Dec + NOP, 1 encoder attention head",
    "mat_dec_heads_2": "MAT-Dec + NOP, 2 encoder attention heads",
    "mat_dec_heads_4": "MAT-Dec + NOP, 4 encoder attention heads",
    "mat_dec_heads_8": "MAT-Dec + NOP, 8 encoder attention heads",
    "mat_qcx_heads_1": "MAT-QCX + NOP, 1 encoder attention head",
    "mat_qcx_heads_2": "MAT-QCX + NOP, 2 encoder attention heads",
    "mat_qcx_heads_4": "MAT-QCX + NOP, 4 encoder attention heads",
    "mat_qcx_heads_8": "MAT-QCX + NOP, 8 encoder attention heads",
}
EXTRA_PLOT_SELECTIONS = (
    ExperimentPlotSelection(
        name="mat_dec_attention_heads",
        group_names=MAT_DEC_GROUP_ORDER,
        title_suffix="MAT-Dec Encoder Attention Heads",
        output_subdir="selections",
    ),
    ExperimentPlotSelection(
        name="mat_qcx_attention_heads",
        group_names=MAT_QCX_GROUP_ORDER,
        title_suffix="MAT-QCX Encoder Attention Heads",
        output_subdir="selections",
    ),
)
THEORETICAL_MAXIMUM = None


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        theoretical_maximum=THEORETICAL_MAXIMUM,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        extra_plot_selections=EXTRA_PLOT_SELECTIONS,
        run_length_limit=250_000_000,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
