from swarmbots.learn.scheduling.auto_lr_updater import make_auto_lr_updater


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
