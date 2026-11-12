import json
from dataclasses import replace

import pytest

from swarmbots.learn.scheduling.auto_lr_updater import make_auto_lr_updater
from swarmbots.learn.scheduling.cosine_scheduler import CosineSchedulerConfig
from swarmbots.learn.scheduling.schedulers import ScheduleUnit
from swarmbots.learn.serialization_utils import serialize_fn
from swarmbots.learn.summary_statistics import SummaryStatistics


def test_auto_lr_updater_uses_configured_max_kl_threshold() -> None:
    default_updater = make_auto_lr_updater()
    default_result = default_updater(
        old_lr=1e-3,
        state={},
        n_iterations=0,
        n_model_updates=0,
        n_timesteps=0,
        early_stop_kl_div=0.008,
        early_stop_epoch=None,
        metrics={},
    )

    tuned_updater = make_auto_lr_updater(max_kl_div=0.007)
    tuned_result = tuned_updater(
        old_lr=1e-3,
        state={},
        n_iterations=0,
        n_model_updates=0,
        n_timesteps=0,
        early_stop_kl_div=0.008,
        early_stop_epoch=None,
        metrics={},
    )

    assert default_result["new_lr"] is None
    assert tuned_result["event"] == "max_kl_hit"
    assert tuned_result["new_lr"] is not None
    assert tuned_result["new_lr"] < 1e-3


def test_auto_lr_updater_decays_on_high_clip_fraction_and_resets_growth_counter() -> None:
    updater = make_auto_lr_updater()
    state = {"counter": 1}

    result = updater(
        old_lr=1e-3,
        state=state,
        n_iterations=0,
        n_model_updates=0,
        n_timesteps=0,
        early_stop_kl_div=None,
        early_stop_epoch=None,
        metrics={"clip_frac": SummaryStatistics(n=4, mean=0.25, std=0.0)},
    )

    assert result["event"] == "max_clip_frac_hit"
    assert result["new_lr"] == pytest.approx(7.5e-4)
    assert state["counter"] == 0
    assert state["warmup"] is False


def test_auto_lr_updater_increases_after_two_clean_iterations() -> None:
    updater = make_auto_lr_updater()
    state: dict[str, object] = {}

    first_result = updater(
        old_lr=1e-3,
        state=state,
        n_iterations=0,
        n_model_updates=0,
        n_timesteps=0,
        early_stop_kl_div=None,
        early_stop_epoch=None,
        metrics={},
    )
    second_result = updater(
        old_lr=1e-3,
        state=state,
        n_iterations=1,
        n_model_updates=0,
        n_timesteps=0,
        early_stop_kl_div=None,
        early_stop_epoch=None,
        metrics={},
    )

    assert first_result["new_lr"] is None
    assert second_result["new_lr"] == pytest.approx(1.1e-3)
    assert state["counter"] == 0


def test_updater_metadata_is_stable_and_records_behavior_configuration() -> None:
    warmup = CosineSchedulerConfig(
        unit=ScheduleUnit.ITERATIONS, duration=10, start_value=1e-4, final_value=1e-3,
    )
    options = {"warm_scheduler_config": warmup, "early_stop_epoch_decay_limit": 6, "max_kl_div": 0.0031}
    updater = make_auto_lr_updater(**options)
    metadata = serialize_fn(updater)

    assert metadata == serialize_fn(make_auto_lr_updater(**options))
    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata["config"] == {
        "warm_scheduler_config": {
            "unit": "ITERATIONS", "duration": 10, "start_value": 1e-4, "final_value": 1e-3,
            "bias": 1.0, "sharpness": 1.0,
        },
        "early_stop_epoch_decay_limit": 6,
        "max_kl_div": 0.0031,
    }
    for changed in (
        {"max_kl_div": 0.1},
        {"early_stop_epoch_decay_limit": 3},
        {"warm_scheduler_config": replace(warmup, duration=20)},
    ):
        assert serialize_fn(make_auto_lr_updater(**(options | changed))) != metadata

    state: dict[str, object] = {}
    old_lr = 1e-4
    for iteration, expected_lr in ((0, 1e-4), (5, 5.5e-4), (10, 1e-3), (11, 1e-3), (12, 1.1e-3)):
        result = updater(
            old_lr=old_lr, state=state, n_iterations=iteration, n_model_updates=0, n_timesteps=0,
            early_stop_kl_div=None, early_stop_epoch=None, metrics={},
        )
        old_lr = old_lr if result["new_lr"] is None else result["new_lr"]
        assert old_lr == pytest.approx(expected_lr)
    assert serialize_fn(updater) == metadata
