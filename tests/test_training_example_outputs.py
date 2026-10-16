from pathlib import Path

import pytest

from examples.train_policy import create_run_directory, make_parser, run_experiment


@pytest.mark.parametrize("scale", ["5M-2L", "5M-3L", "legacy"])
def test_fresh_runs_keep_scale_seed_and_repetitions_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scale: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    args = make_parser().parse_args(["--variant", "tmasac", "--model-scale", scale, "--seed", "7", "--no-nop"])
    first = create_run_directory(args)
    second = create_run_directory(args)
    assert first != second
    assert first.is_dir() and second.is_dir()
    assert first.parent == Path("runs/SwarmBots-WallEasy-v0/tmasac") / scale / "seed-7"
    assert first.name.startswith("no-nop-")


def test_explicit_run_directory_cannot_overwrite_an_existing_run(tmp_path: Path) -> None:
    args = make_parser().parse_args(["--run-dir", str(tmp_path / "experiment")])
    run_dir = create_run_directory(args)
    result = run_dir / "evaluation.json"
    result.write_text("saved result", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_experiment(args)
    assert result.read_text(encoding="utf-8") == "saved result"
