from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def make_run_dir(run_group: str, run_id: str) -> Path:
    return REPO_ROOT / "runs" / run_group / run_id


def get_run_id_from_checkpoint_path(load_path: str | Path) -> str:
    return Path(load_path).parent.parent.name
