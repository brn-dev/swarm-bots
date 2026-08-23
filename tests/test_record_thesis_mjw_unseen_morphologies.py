from pathlib import Path

import pytest

from experiments.record_thesis_mjw_unseen_morphologies import (
    DEFAULT_OUTPUT_ROOT,
    _parse_args,
    _prepare_combination_output,
    _validate_args,
    checkpoint_output_name,
    combination_output_dir,
    make_combination_rollout_seed,
)


def test_default_episode_count_is_three() -> None:
    config = _validate_args(_parse_args([]))

    assert config.episodes_per_combination == 3
    assert config.frame_stride == 1
    assert "runs" in DEFAULT_OUTPUT_ROOT.parts
    assert "experiments" not in DEFAULT_OUTPUT_ROOT.relative_to(DEFAULT_OUTPUT_ROOT.parents[2]).parts


def test_episodes_alias_sets_episode_count() -> None:
    config = _validate_args(_parse_args(["--episodes", "4"]))

    assert config.episodes_per_combination == 4


def test_combination_rollout_seeds_are_repeatable_and_distinct() -> None:
    first_rollout_seed = make_combination_rollout_seed(
        rollout_seed_base=2_000_000,
        combination_index=0,
    )
    second_rollout_seed = make_combination_rollout_seed(
        rollout_seed_base=2_000_000,
        combination_index=1,
    )

    assert first_rollout_seed == 2_000_000
    assert second_rollout_seed == 2_000_001


def test_combination_output_dir_separates_target_agent_count_and_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run name" / "models" / "model_100_steps_final.pt"

    output_dir = combination_output_dir(
        output_root=tmp_path / "recordings",
        target_key="po_wall_tmasac",
        unit_count=4,
        checkpoint_path=checkpoint,
    )

    assert output_dir.parent.name == "4_agents"
    assert output_dir.parent.parent.name == "po_wall_tmasac"
    assert output_dir.name == checkpoint_output_name(checkpoint)
    assert output_dir.name.startswith("run_name-")


def test_prepare_combination_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "combination"
    output_dir.mkdir()
    existing_video = output_dir / "old.mp4"
    existing_video.touch()

    with pytest.raises(FileExistsError, match="--overwrite"):
        _prepare_combination_output(output_dir, overwrite=False)

    _prepare_combination_output(output_dir, overwrite=True)

    assert output_dir.is_dir()
    assert not existing_video.exists()
