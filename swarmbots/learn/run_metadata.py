"""Source provenance for training runs, including uncommitted source changes."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Any


def collect_source_revision(source_directory: Path | None = None) -> dict[str, Any]:
    directory = Path(__file__).resolve().parents[2] if source_directory is None else source_directory

    def git(*arguments: str) -> bytes:
        return subprocess.run(
            ["git", *arguments], cwd=directory, check=True, capture_output=True, timeout=10,
        ).stdout

    try:
        # An installed wheel can live inside another project's ignored .venv.
        git("ls-files", "--error-unmatch", "--", "swarmbots/__init__.py")
        revision = git("rev-parse", "HEAD").decode().strip()
        root = Path(git("rev-parse", "--show-toplevel").decode().strip())
        directory = root
        changes = git("diff", "--no-ext-diff", "--binary", "HEAD")
        untracked = git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        digest = hashlib.sha256(changes)
        for encoded_path in sorted(path for path in untracked if path):
            digest.update(encoded_path + b"\0")
            path = root / encoded_path.decode("utf-8")
            with path.open("rb") as source_file:
                digest.update(path.stat().st_size.to_bytes(8, "big"))
                for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                    digest.update(chunk)
        dirty = bool(changes or any(untracked))
        return {
            "commit": revision,
            "dirty": dirty,
            "working_tree_sha256": digest.hexdigest() if dirty else None,
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "working_tree_sha256": None}
