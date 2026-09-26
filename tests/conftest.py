from pathlib import Path

import pytest
import warp as wp


@pytest.fixture(scope="session", autouse=True)
def configure_warp_cache() -> None:
    cache_dir = Path(__file__).resolve().parents[1] / ".tmp" / "pytest-warp-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(cache_dir)
