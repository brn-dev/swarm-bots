from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def make_run_dir(run_group: str, run_id: str) -> Path:
    _validate_run_path_part(run_group, name="run_group")
    _validate_run_path_part(run_id, name="run_id")
    return REPO_ROOT / "runs" / run_group / run_id


def get_run_id_from_checkpoint_path(load_path: str | Path) -> str:
    return Path(load_path).parent.parent.name


def _validate_run_path_part(value: str, *, name: str) -> None:
    path = Path(value)
    if not value or path.is_absolute() or path.name != value or value in {".", ".."}:
        raise ValueError(f"{name} must be a single relative path name, got {value!r}")
