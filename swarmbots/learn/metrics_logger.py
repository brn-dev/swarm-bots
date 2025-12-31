import csv
from collections.abc import Collection, Iterable
import json
from pathlib import Path
from typing import Any
import numpy as np
from loguru import logger

from swarmbots.learn.summary_statistics import (
    SummaryStatistics,
    SummaryStatisticsFormat,
    format_summary_statistics,
    maybe_make_wandb_histogram,
)

class MetricsLogger:
    def __init__(
            self,
            log_dir: str | Path | None = None,
            filename: str = "log.csv",
            wandb_run: Any | None = None,
            wandb_project: str | None = None,
            wandb_entity: str | None = None,
            wandb_run_name: str | None = None,
            wandb_group: str | None = None,
            wandb_tags: list[str] | None = None,
            wandb_config: dict[str, Any] | None = None,
            wandb_mode: str | None = None,
            wandb_kwargs: dict[str, Any] | None = None,
            wandb_step_key: str | None = "timesteps",
            ignore_keys_for_persistence: Collection[str] | None = None,
            console_keys: list[str] | list[tuple[str, str | SummaryStatisticsFormat | None]] | None = None,
    ) -> None:
        self.log_dir = Path(log_dir) if log_dir else None
        self.file_path = self.log_dir / filename if self.log_dir else None
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.file = None
        self.writer = None
        self._csv_fieldnames: list[str] | None = None
        self._warned_csv_extra_keys = False

        self._wandb_run = wandb_run
        self._wandb_managed_run = False
        self._wandb_step_key = wandb_step_key

        self._ignore_keys_for_persistence = set(ignore_keys_for_persistence or [])
        self._console_key_specs = self._normalize_console_keys(console_keys)

        if self._wandb_run is None and wandb_project is not None:
            self._init_wandb(
                project=wandb_project,
                entity=wandb_entity,
                run_name=wandb_run_name,
                group=wandb_group,
                tags=wandb_tags,
                config=wandb_config,
                mode=wandb_mode,
                wandb_kwargs=wandb_kwargs,
            )

    def log(self, metrics: dict[str, Any]) -> None:
        self._log_to_console(metrics)

        persistence_metrics = {
            k: v for k, v in metrics.items()
            if not self._ignore_keys_for_persistence or k not in self._ignore_keys_for_persistence
        }

        if self.file_path and persistence_metrics:
            self._log_to_csv(persistence_metrics)

        if self._wandb_run is not None and persistence_metrics:
            self._log_to_wandb(persistence_metrics)

    def _log_to_console(self, metrics: dict[str, Any]) -> None:
        parts = []

        for key, value, fmt in self._iter_console_metrics(metrics):
            val_str = self._format_console_value(value, fmt=fmt)
            parts.append(f"<underline>{key}</underline>: {val_str}")

        if not parts:
            logger.warning('Nothing to log?')
            return

        logger.opt(colors=True).info(" | ".join(parts))

    def _log_to_csv(self, metrics: dict[str, Any]) -> None:
        csv_metrics: dict[str, Any] = {}

        for k, v in metrics.items():
            if isinstance(v, SummaryStatistics):
                csv_metrics[k + '__mean'] = round(v.mean, 6)
                if v.std is not None:
                    csv_metrics[k + '__std'] = round(v.std, 6)
                if v.min_value is not None:
                    csv_metrics[k + '__min'] = round(v.min_value, 6)
                if v.max_value is not None:
                    csv_metrics[k + '__max'] = round(v.max_value, 6)
                if v.histogram is not None:
                    csv_metrics[k + '__histogram_freqs'] = json.dumps([round(x, 6) for x in v.histogram.bin_frequencies])
                    csv_metrics[k + '__histogram_edges'] = json.dumps([round(x, 6) for x in v.histogram.bin_edges])
            else:
                csv_metrics[k] = v

        if self.file is None:
            file_exists = self.file_path.exists()
            self.file = open(self.file_path, mode='a', newline='', encoding="utf-8")
            
            self._csv_fieldnames = list(csv_metrics.keys())
            self.writer = csv.DictWriter(
                self.file,
                fieldnames=self._csv_fieldnames,
                delimiter=';',
                extrasaction="ignore",
            )
            
            if not file_exists:
                self.writer.writeheader()
        else:
            assert self._csv_fieldnames is not None
            extra_keys = set(csv_metrics.keys()) - set(self._csv_fieldnames)
            if extra_keys and not self._warned_csv_extra_keys:
                self._warned_csv_extra_keys = True
                logger.warning(
                    "MetricsLogger: ignoring new CSV metric keys not present in the header: "
                    f"{sorted(extra_keys)}"
                )
        
        self.writer.writerow(csv_metrics)
        self.file.flush()

    def _init_wandb(
            self,
            project: str,
            entity: str | None,
            run_name: str | None,
            group: str | None,
            tags: list[str] | None,
            config: dict[str, Any] | None,
            mode: str | None,
            wandb_kwargs: dict[str, Any] | None,
    ) -> None:
        try:
            import wandb  # type: ignore
        except Exception as e:
            logger.warning(f"wandb is not available ({e}); continuing without wandb logging.")
            return

        init_kwargs: dict[str, Any] = dict(wandb_kwargs or {})
        init_kwargs.setdefault("project", project)
        if entity is not None:
            init_kwargs.setdefault("entity", entity)
        if run_name is not None:
            init_kwargs.setdefault("name", run_name)
        if group is not None:
            init_kwargs.setdefault("group", group)
        if tags is not None:
            init_kwargs.setdefault("tags", tags)
        if config is not None:
            init_kwargs.setdefault("config", config)
        if self.log_dir is not None:
            init_kwargs.setdefault("dir", str(self.log_dir))
        if mode is not None:
            init_kwargs.setdefault("mode", mode)

        self._wandb_run = wandb.init(**init_kwargs)
        self._wandb_managed_run = True

    def _log_to_wandb(self, metrics: dict[str, Any]) -> None:
        metrics = {k: v for k, v in metrics.items() if v is not None}

        wandb_metrics: dict[str, Any] = {}
        for k, v in metrics.items():
            if isinstance(v, SummaryStatistics):
                if v.data:
                    wandb_metrics[k] = v.data
                else:
                    wandb_metrics[k + "__mean"] = v.mean
                    if v.std is not None:
                        wandb_metrics[k + "__std"] = v.std
                    if v.min_value is not None:
                        wandb_metrics[k + "__min"] = v.min_value
                    if v.max_value is not None:
                        wandb_metrics[k + "__max"] = v.max_value

                    if v.histogram is not None:
                        wandb_metrics[k + "__histogram_freqs"] = v.histogram.bin_frequencies
                        wandb_metrics[k + "__histogram_edges"] = v.histogram.bin_edges
                        wandb_hist = maybe_make_wandb_histogram(v.histogram)
                        if wandb_hist is not None:
                            wandb_metrics[k + "__histogram"] = wandb_hist
            else:
                wandb_metrics[k] = v

        step = None
        if self._wandb_step_key is not None:
            value = wandb_metrics.get(self._wandb_step_key)
            if isinstance(value, (int, float, np.number)) and not np.isnan(value):
                step = int(value)

        if step is None:
            self._wandb_run.log(wandb_metrics)
        else:
            self._wandb_run.log(wandb_metrics, step=step)

    def close(self) -> None:
        if self.file:
            self.file.close()
            self.file = None
            self.writer = None
            self._csv_fieldnames = None

        if self._wandb_run is not None and self._wandb_managed_run:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
            finally:
                self._wandb_run = None
                self._wandb_managed_run = False

    def __del__(self) -> None:
        self.close()

    def _normalize_console_keys(
        self,
        console_keys: Collection[str] | Collection[tuple[str, str | None]] | None,
    ) -> list[tuple[str, str | None]] | None:
        if console_keys is None:
            return None

        items = list(console_keys)
        if not items:
            return []

        if all(isinstance(item, str) for item in items):
            key_specs = [(key, None) for key in items]
            return key_specs

        if all(isinstance(item, tuple) and len(item) == 2 for item in items):
            key_specs: list[tuple[str, str | None]] = []
            for raw_key, raw_fmt in items:
                key_specs.append((raw_key, raw_fmt))
            return key_specs

        raise TypeError(console_keys)

    def _iter_console_metrics(self, metrics: dict[str, Any]) -> Iterable[tuple[str, Any, str | None]]:
        if self._console_key_specs is None:
            for key, value in metrics.items():
                yield key, value, None
            return

        for key, fmt in self._console_key_specs:
            yield key, metrics[key], fmt

    def _format_console_value(self, value: Any, fmt: str | SummaryStatisticsFormat | None) -> str:

        if isinstance(value, SummaryStatistics):
            assert fmt is None or isinstance(fmt, SummaryStatisticsFormat), 'supply a summary statistics format'
            return format_summary_statistics(value, fmt)

        if value is not None and fmt is not None:
            try:
                return f"{value:{fmt}}"
            except Exception:
                logger.exception(f"Error formatting console value: {value} with format: {fmt}")
                return str(value)

        if isinstance(value, (int, float, np.number)):
            if np.isclose(value % 1.0, 0):
                return f"{int(value):>2}"
            return f"{value:.3f}"

        return str(value)
