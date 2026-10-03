import importlib
import pkgutil
from collections.abc import Iterable
from importlib import metadata
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import torch

import swarmbots
from swarmbots.learn import action_dists
from swarmbots.benchmark.metadata import runtime_metadata


PROJECT_ROOT = Path(__file__).parents[1]


def test_public_action_distributions_include_only_the_supported_families() -> None:
    assert {module.name for module in pkgutil.iter_modules(action_dists.__path__)} == {
        "action_dist",
        "action_sampling",
        "bernoulli_action_dist",
        "beta_action_dist",
        "continuous_action_dist",
        "diag_gaussian_action_dist",
        "discrete_action_dist",
        "entropy_utils",
        "gsde_action_dist",
        "gumbel_softmax_sign_magnitude_action_dist",
        "hybrid_action_dist",
        "predicted_std_gaussian_action_dist",
        "sign_magnitude_action_dist",
        "sign_magnitude_beta_action_dist",
        "squashed_diag_gaussian_action_dist",
        "tanh_bijector",
        "temporally_correlated_action_dist",
    }


def test_release_versions_match() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_version = pyproject["project"]["version"]

    citation = (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation_match = re.search(r"^version:\s*[\"']?([^\s\"']+)[\"']?\s*$", citation, flags=re.MULTILINE)
    assert citation_match is not None
    assert citation_match.group(1) == project_version

    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(rf"^## {re.escape(project_version)}(?:\s|$)", changelog, flags=re.MULTILINE)


@pytest.mark.parametrize("legacy_distribution_installed", [False, True])
def test_versions_identify_the_current_installed_distribution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    legacy_distribution_installed: bool,
) -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    distributions = {project["name"]: project["version"]}
    if legacy_distribution_installed:
        distributions["swarm-bots"] = "0.0.1"
    for name, version in distributions.items():
        dist_info = tmp_path / f"{name.replace('-', '_')}-{version}.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n", encoding="utf-8"
        )
    discover_distributions = metadata.Distribution.discover

    def isolated_distributions(**kwargs: Any) -> Iterable[metadata.Distribution]:
        return discover_distributions(path=[tmp_path], **kwargs)

    try:
        with monkeypatch.context() as isolated_installation:
            isolated_installation.setattr(metadata.Distribution, "discover", isolated_distributions)
            importlib.reload(swarmbots)
            assert swarmbots.__version__ == project["version"]
            assert runtime_metadata(torch.device("cpu"))["packages"][project["name"]] == project["version"]
    finally:
        importlib.reload(swarmbots)
