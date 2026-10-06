"""Shared console and persistence presets for training metrics."""

from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

from swarmbots.learn.summary_statistics import SummaryStatisticsFormat

LoggingLevel = Literal["full", "minimal", "return_success"]
LOGGING_LEVELS: tuple[LoggingLevel, ...] = ("full", "minimal", "return_success")
ConsoleMetricFormat = str | SummaryStatisticsFormat | None
ConsoleMetricSpec = str | tuple[str, ConsoleMetricFormat] | tuple[str, ConsoleMetricFormat, str]
ConsoleKeys = Collection[ConsoleMetricSpec]
ConsoleSelection = ConsoleKeys | Literal["default"] | None
FormattedConsoleMetric = tuple[str, ConsoleMetricFormat, str]

PROGRESS_CONSOLE_KEYS: tuple[FormattedConsoleMetric, ...] = (
    ("iteration", "d", "it"),
    ("timesteps", ",d", "steps"),
    ("total_updates", ",d", "upd"),
)
EMA_CONSOLE_KEYS: tuple[FormattedConsoleMetric, ...] = (
    ("ep_rew_ema", ".2f", "ret"),
    ("ep_success_rate_ema", ".1f", "succ%"),
)
RETURN_SUCCESS_CONSOLE_KEYS: tuple[FormattedConsoleMetric, ...] = (
    *PROGRESS_CONSOLE_KEYS,
    *EMA_CONSOLE_KEYS,
    ("ep_rew", SummaryStatisticsFormat(mean=".2f"), "ret_mean"),
    ("best_ep_rew_ema", ".2f", "best_ret"),
    ("ep_success_rate", ".1f", "succ_batch%"),
)
MINIMAL_CONSOLE_KEYS: tuple[FormattedConsoleMetric, ...] = (
    *PROGRESS_CONSOLE_KEYS,
    *EMA_CONSOLE_KEYS,
    ("critic_loss", SummaryStatisticsFormat(mean=".3f"), "q_loss"),
    ("actor_loss", SummaryStatisticsFormat(mean=".3f"), "pi_loss"),
    ("act_loss", SummaryStatisticsFormat(mean=".3f"), "pi_loss"),
    ("val_loss_scaled", SummaryStatisticsFormat(mean=".3f"), "v_loss"),
    ("approx_kl", SummaryStatisticsFormat(mean=".2e"), "kl"),
    ("clip_frac", SummaryStatisticsFormat(mean=".3f"), "clip"),
    ("expl_var", SummaryStatisticsFormat(mean=".3f"), "ev"),
    ("q_pi", SummaryStatisticsFormat(mean=".3f"), "q_pi"),
    ("target_q", SummaryStatisticsFormat(mean=".3f"), "q_tgt"),
    ("ent_coef", SummaryStatisticsFormat(mean=".3f"), "alpha"),
    ("entropy", SummaryStatisticsFormat(mean=".3f"), "ent"),
    ("target_entropy", SummaryStatisticsFormat(mean=".3f"), "ent_tgt"),
    ("actor_updated", SummaryStatisticsFormat(mean=".2f"), "pi_upd"),
    ("wm_loss_scaled", SummaryStatisticsFormat(mean=".3f"), "nop"),
    ("critic_nop_loss_scaled", SummaryStatisticsFormat(mean=".3f"), "c_nop"),
    ("actor_nop_loss_scaled", SummaryStatisticsFormat(mean=".3f"), "a_nop"),
    ("replay_size", ",d", "replay"),
    ("random_actions", "d", "rnd"),
    ("training_skipped", "d", "skip"),
    ("learning_rate", ".2e", "lr"),
    ("fps", ".0f", "fps"),
)
CONTEXT_METRIC_KEYS = frozenset({"timestamp", "learn_start", "episodes"})
RETURN_SUCCESS_METRIC_KEYS = (
    frozenset(key for key, _format, _alias in RETURN_SUCCESS_CONSOLE_KEYS) | CONTEXT_METRIC_KEYS
)
MINIMAL_METRIC_KEYS = frozenset(key for key, _format, _alias in MINIMAL_CONSOLE_KEYS) | RETURN_SUCCESS_METRIC_KEYS


@dataclass(frozen=True, slots=True)
class LoggingProfile:
    console_keys: tuple[FormattedConsoleMetric, ...] | None
    persistence_keys: frozenset[str] | None


LOGGING_PROFILES: dict[LoggingLevel, LoggingProfile] = {
    "full": LoggingProfile(console_keys=None, persistence_keys=None),
    "minimal": LoggingProfile(console_keys=MINIMAL_CONSOLE_KEYS, persistence_keys=MINIMAL_METRIC_KEYS),
    "return_success": LoggingProfile(
        console_keys=RETURN_SUCCESS_CONSOLE_KEYS, persistence_keys=RETURN_SUCCESS_METRIC_KEYS
    ),
}


def get_logging_profile(level: LoggingLevel) -> LoggingProfile:
    try:
        return LOGGING_PROFILES[level]
    except KeyError as error:
        raise ValueError(f"Unknown logging level {level!r}; choose from {', '.join(LOGGING_LEVELS)}") from error
