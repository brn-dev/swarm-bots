import json
from pathlib import Path
import subprocess
from typing import Any
from unittest.mock import patch

import pytest
import torch

from swarmbots.learn.run_metadata import collect_source_revision
from swarmbots.learn.training import make_training
from tests.test_metrics_logger import _DummyAlgorithm, _DummyEnv, _DummyPolicy


def test_source_revision_identifies_tracked_and_untracked_changes(tmp_path: Path) -> None:
    def git(*arguments: str) -> None:
        subprocess.run(["git", *arguments], cwd=tmp_path, check=True, capture_output=True)

    git("init")
    package = tmp_path / "swarmbots"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("original", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    git("add", "swarmbots/__init__.py", ".gitignore")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "initial")
    clean = collect_source_revision(tmp_path)
    assert clean["commit"] and clean["dirty"] is False
    assert clean["working_tree_sha256"] is None

    source.write_text("changed", encoding="utf-8")
    modified = collect_source_revision(tmp_path)
    assert modified["commit"] == clean["commit"] and modified["dirty"] is True
    assert modified["working_tree_sha256"] == collect_source_revision(tmp_path)["working_tree_sha256"]

    (tmp_path / "new.py").write_text("new source", encoding="utf-8")
    untracked = collect_source_revision(tmp_path)
    assert untracked["working_tree_sha256"] != modified["working_tree_sha256"]

    installed = tmp_path / ".venv" / "Lib" / "site-packages"
    (installed / "swarmbots").mkdir(parents=True)
    (installed / "swarmbots" / "__init__.py").write_text("installed wheel", encoding="utf-8")
    assert collect_source_revision(installed)["commit"] is None


def test_source_revision_without_checkout_is_explicitly_unknown(tmp_path: Path) -> None:
    with patch("swarmbots.learn.run_metadata.subprocess.run", side_effect=FileNotFoundError):
        assert collect_source_revision(tmp_path) == {
            "commit": None, "dirty": None, "working_tree_sha256": None,
        }


def test_run_metadata_exists_before_training_and_preserves_resolved_settings(tmp_path: Path) -> None:
    class ConfiguredEnvironment:
        def get_settings(self) -> dict[str, Any]:
            return {"scenario": {"goal": torch.tensor([1.0, 2.0])}, "episode_length": 64}

    env = _DummyEnv()
    env.unwrapped = ConfiguredEnvironment()
    algorithm = _DummyAlgorithm(_DummyPolicy(), env, 1e-3)
    algorithm.training_metadata = {
        "training_seed": 73,
        "training_configuration": {"scenario_kwargs": {"goal": [1.0, 2.0]}},
    }

    def interrupt(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("interrupted before the first update")

    with patch.object(algorithm, "perform_iteration", side_effect=interrupt), pytest.raises(RuntimeError):
        algorithm.learn(max_total_timesteps=1, run_dir=tmp_path, enable_command_prompt=False)
    metadata = json.loads((tmp_path / "run_metadata_0.json").read_text(encoding="utf-8"))
    assert metadata["training_seed"] == 73
    assert metadata["environment_settings"] == {
        "scenario": {"goal": [1.0, 2.0]}, "episode_length": 64,
    }
    assert metadata["source_revision"]["commit"]


def test_make_training_retains_original_construction_options() -> None:
    from types import SimpleNamespace

    env = SimpleNamespace(unwrapped=SimpleNamespace(device=torch.device("cpu"), episode_length=500))
    captured = SimpleNamespace()
    with patch("swarmbots.learn.training._make_policy_env", return_value=(env, SimpleNamespace())), patch(
        "swarmbots.learn.training.PPO", return_value=captured,
    ):
        algorithm = make_training(
            "SwarmBots-WallEasy-v0", seed=73, scenario_kwargs={"wall_height": 0.4},
            env_kwargs={"first_episode_lengths": None}, algorithm_kwargs={"gamma": 0.98},
        )
    assert algorithm.training_metadata["training_seed"] == 73
    options = algorithm.training_metadata["training_configuration"]
    assert options["scenario_kwargs"] == {"wall_height": 0.4}
    assert options["env_kwargs"] == {"first_episode_lengths": None}
    assert options["algorithm_kwargs"] == {"gamma": 0.98}
    assert options["num_envs"] == 1024 and options["rollout_steps_per_env"] == 4
