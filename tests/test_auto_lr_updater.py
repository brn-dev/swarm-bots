import pytest

from swarmbots.learn.scheduling.auto_lr_updater import make_auto_lr_updater
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
