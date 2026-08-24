from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.record_thesis_mjw_unseen_morphologies import main as _record_main

RECORDING_ROOT = (
    REPO_ROOT
    / "runs"
    / "thesis_mjw_po_wall_disconnected_finetune_50m"
    / "recordings"
)


def run_recording(
    argv: Sequence[str] | None = None,
    *,
    disable_policy_connector_actions: bool,
    variant_name: str,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Record a 50M disconnected-start PO-wall TMASAC checkpoint. "
            "Options after CHECKPOINT are forwarded to the shared thesis recorder."
        ),
        epilog=(
            "Example: CHECKPOINT --episodes 3 --frame-stride 2 --cuda-idx 0. "
            "Pass CHECKPOINT --help to list all forwarded recording options."
        ),
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("recording_options", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    recorder_argv = [
        "--target",
        "po_wall_tmasac",
        "--unit-counts",
        "4",
        "5",
        "--pool-seed-base",
        "3000000",
        "--po-wall-checkpoint",
        str(args.checkpoint),
        *args.recording_options,
    ]
    return _record_main(
        recorder_argv,
        unconnected_prob=1.0,
        disable_policy_connector_actions=disable_policy_connector_actions,
        default_output_root=RECORDING_ROOT / variant_name,
        morphology_description="fully disconnected 4- and 5-unit PO-wall pools",
    )
