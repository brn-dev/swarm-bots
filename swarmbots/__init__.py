from importlib.metadata import PackageNotFoundError, version

from swarmbots.benchmark import (
    ALL_BENCHMARK_IDS,
    CORE_BENCHMARK_IDS,
    EvaluationResult,
    evaluate_policy,
    get_benchmark_spec,
    list_benchmarks,
    make_env,
    make_scenario,
    record_policy,
)

try:
    __version__ = version("swarmbots")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "ALL_BENCHMARK_IDS",
    "CORE_BENCHMARK_IDS",
    "EvaluationResult",
    "__version__",
    "evaluate_policy",
    "get_benchmark_spec",
    "list_benchmarks",
    "make_env",
    "make_scenario",
    "record_policy",
]
