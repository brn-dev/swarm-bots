import csv
from pathlib import Path
from typing import Any, Dict, Union, Optional
import numpy as np
from loguru import logger

class MetricsLogger:
    def __init__(
            self,
            log_dir: Optional[Union[str, Path]] = None,
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
    ):
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

    def log(self, metrics: Dict[str, Any]):
        self._log_to_console(metrics)

        if self.file_path:
            self._log_to_csv(metrics)

        if self._wandb_run is not None:
            self._log_to_wandb(metrics)

    def _log_to_console(self, metrics: Dict[str, Any]):
        parts = []
        
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                if np.isclose(value % 1.0, 0):
                    val_str = f'{int(value):>2}'
                else:
                    val_str = f'{value: .3f}'
            else:
                val_str = str(value)
                
            parts.append(f"{key}: {val_str}")
            
        logger.info(" | ".join(parts))

    def _log_to_csv(self, metrics: Dict[str, Any]):
        if self.file is None:
            file_exists = self.file_path.exists()
            self.file = open(self.file_path, mode='a', newline='', encoding="utf-8")
            
            self._csv_fieldnames = list(metrics.keys())
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
            extra_keys = set(metrics.keys()) - set(self._csv_fieldnames)
            if extra_keys and not self._warned_csv_extra_keys:
                self._warned_csv_extra_keys = True
                logger.warning(
                    "MetricsLogger: ignoring new CSV metric keys not present in the header: "
                    f"{sorted(extra_keys)}"
                )
        
        self.writer.writerow(metrics)
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

    def _log_to_wandb(self, metrics: Dict[str, Any]) -> None:
        metrics = {k: v for k, v in metrics.items() if v is not None}
        step = None
        if self._wandb_step_key is not None:
            value = metrics.get(self._wandb_step_key)
            if isinstance(value, (int, float, np.number)) and not np.isnan(value):
                step = int(value)

        if step is None:
            self._wandb_run.log(metrics)
        else:
            self._wandb_run.log(metrics, step=step)

    def close(self):
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

    def __del__(self):
        self.close()
