import csv
from pathlib import Path
from typing import Any, Dict, Union, Optional
import numpy as np
from loguru import logger

class MetricLogger:
    def __init__(self, log_dir: Optional[Union[str, Path]] = None, filename: str = "log.csv"):
        self.log_dir = Path(log_dir) if log_dir else None
        self.file_path = self.log_dir / filename if self.log_dir else None
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.file = None
        self.writer = None

    def log(self, metrics: Dict[str, Any]):
        self._log_to_console(metrics)

        if self.file_path:
            self._log_to_csv(metrics)

    def _log_to_console(self, metrics: Dict[str, Any]):
        parts = []
        
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                if np.isclose(value % 1.0, 0):
                    val_str = f'{int(value):>2}'
                else:
                    val_str = f'{value: .4f}'
            else:
                val_str = str(value)
                
            parts.append(f"{key}: {val_str}")
            
        logger.info(" | ".join(parts))

    def _log_to_csv(self, metrics: Dict[str, Any]):
        if self.file is None:
            file_exists = self.file_path.exists()
            self.file = open(self.file_path, mode='a', newline='')
            
            fieldnames = list(metrics.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
            
            if not file_exists:
                self.writer.writeheader()
        
        self.writer.writerow(metrics)
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
            self.file = None

    def __del__(self):
        self.close()
