"""Resolve replay exposure into a whole number of optimizer updates."""

import math


def resolve_replay_updates(
    *, replay_exposure: float, rollout_steps: int, learning_positions_per_batch: int,
) -> int:
    if not math.isfinite(replay_exposure) or replay_exposure <= 0:
        raise ValueError(f"replay_exposure must be finite and > 0, got {replay_exposure}")
    if rollout_steps <= 0:
        raise ValueError(f"rollout_steps must be > 0, got {rollout_steps}")
    if learning_positions_per_batch <= 0:
        raise ValueError(f"learning_positions_per_batch must be > 0, got {learning_positions_per_batch}")
    updates = replay_exposure * rollout_steps / learning_positions_per_batch
    nearest_integer = round(updates)
    # Accept floating-point roundoff when the mathematical update count is integral.
    if nearest_integer < 1 or not math.isclose(updates, nearest_integer, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "replay_exposure must produce a positive whole number of updates per collection cycle: "
            f"{replay_exposure} * {rollout_steps} / {learning_positions_per_batch} = {updates}. "
            "Adjust replay_exposure, rollout size, or batch size (including recurrent learning_steps)."
        )
    return nearest_integer
