from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from swarmbots.tools.inspect_policy_parameters import allocation_doc_link


@pytest.mark.parametrize(
    "module",
    ["inspect_policy_parameters", "diagram_policy_architectures", "plot_logs", "plot_experiment_results"],
)
def test_tool_module_help_outside_checkout(tmp_path: Path, module: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", f"swarmbots.tools.{module}", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


@pytest.mark.parametrize("local_guide", [False, True])
def test_report_links_work_with_and_without_checkout_guides(tmp_path: Path, local_guide: bool) -> None:
    guide = tmp_path / "docs/policy_parameters/model_scale.md"
    if local_guide:
        guide.parent.mkdir(parents=True)
        guide.write_text("Sizing guide", encoding="utf-8")
    link = allocation_doc_link(tmp_path / "docs/policy_parameters/generated/5M", guide)
    if local_guide:
        assert link == "../../model_scale.md"
    else:
        assert link == "https://github.com/brn-dev/swarm-bots/blob/main/docs/policy_parameters/model_scale.md"
