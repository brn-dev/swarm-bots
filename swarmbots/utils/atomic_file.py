from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def staged_output(path: Path) -> Iterator[Path]:
    """Stage on the destination filesystem, with the usual creation mode and umask."""
    with TemporaryDirectory(dir=path.parent, prefix=".swarmbots-") as temp:
        suffix = path.suffix
        staging = Path(temp) / ("output" + suffix if len(suffix) <= 16 else "output")
        staging.touch()
        yield staging


def publish_staged_output(staging: Path, path: Path) -> None:
    if os.name == "posix":
        try:
            mode = path.stat().st_mode
        except FileNotFoundError:
            pass
        else:
            staging.chmod(mode & 0o777)
    staging.replace(path)


@contextmanager
def atomic_output(path: Path) -> Iterator[Path]:
    with staged_output(path) as staging:
        yield staging
        publish_staged_output(staging, path)
