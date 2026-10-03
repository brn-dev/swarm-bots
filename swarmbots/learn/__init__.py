from typing import Any

__all__ = ["as_benchmark_policy", "list_variants", "make_training", "record_checkpoint", "train"]


def __getattr__(name: str) -> Any:
    if name == "record_checkpoint":
        from swarmbots.learn.checkpoint_recording import record_checkpoint

        return record_checkpoint
    if name == "as_benchmark_policy":
        from swarmbots.learn.benchmark_policy import as_benchmark_policy

        return as_benchmark_policy
    if name in ("list_variants", "make_training", "train"):
        from swarmbots.learn import training

        return getattr(training, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
