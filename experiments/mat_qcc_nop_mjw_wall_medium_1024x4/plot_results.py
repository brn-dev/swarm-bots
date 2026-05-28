from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import plot_experiment_results


EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / "mat_qcc_nop_mjw_wall_medium_1024x4"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
GROUP_ORDER = (
    "baseline",
    "mat_qcs_context_tokens_only_no_agent_embeddings",
    "mat_orig",
)
DISPLAY_NAME_OVERRIDES = {
    "baseline": "MAT-QCC + NOP, no agent embeddings, MJW wall medium, 1024x4",
    "mat_qcs_context_tokens_only_no_agent_embeddings": (
        "MAT-QCS context tokens only + NOP, no agent embeddings, MJW wall medium, 1024x4"
    ),
    "mat_orig": "MAT-Orig + NOP, no agent embeddings, MJW wall medium, 1024x4",
}
THEORETICAL_MAXIMUM = 10.5


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        theoretical_maximum=THEORETICAL_MAXIMUM,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
