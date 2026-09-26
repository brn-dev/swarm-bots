import re
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def test_release_versions_match() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_version = pyproject["project"]["version"]

    citation = (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    citation_match = re.search(r"^version:\s*[\"']?([^\s\"']+)[\"']?\s*$", citation, flags=re.MULTILINE)
    assert citation_match is not None
    assert citation_match.group(1) == project_version

    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(rf"^## {re.escape(project_version)}(?:\s|$)", changelog, flags=re.MULTILINE)
