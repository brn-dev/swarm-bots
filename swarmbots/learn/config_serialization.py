from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any


def serialize_config_value(value: Any) -> Any:
    if is_dataclass(value):
        return serialize_config_value(asdict(value))
    if isinstance(value, dict):
        return {k: serialize_config_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_config_value(v) for v in value]
    if isinstance(value, tuple):
        return [serialize_config_value(v) for v in value]
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, type):
        return value.__name__
    if callable(value):
        return value.__name__ if hasattr(value, "__name__") else str(value)
    return value


def serialize_dataclass_config(config: Any) -> dict[str, Any]:
    if not is_dataclass(config):
        raise TypeError(f"Expected dataclass instance, got {type(config)}")
    serialized = serialize_config_value(asdict(config))
    if not isinstance(serialized, dict):
        raise TypeError(f"Expected serialized dataclass to be dict, got {type(serialized)}")
    return serialized
