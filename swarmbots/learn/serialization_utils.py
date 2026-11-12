import inspect
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import partial
from operator import attrgetter, itemgetter, methodcaller
from typing import Any

import numpy as np
import torch


_OPERATOR_HELPERS = (attrgetter, itemgetter, methodcaller)
_CALLABLE_HELPERS = (partial, *_OPERATOR_HELPERS)


def _type_identity(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _opaque_record(value: Any) -> dict[str, Any]:
    return {"type": _type_identity(value), "opaque": True}


def _callable_identity(fn: Callable[..., Any]) -> str:
    owner = getattr(fn, "__self__", None)
    if inspect.isclass(owner):
        return f"{owner.__module__}.{owner.__qualname__}.{fn.__name__}"
    module = getattr(fn, "__module__", None)
    if module is None:
        if owner is None:
            owner = getattr(fn, "__objclass__", None)
        if owner is not None:
            owner_identity = _callable_identity(owner) if callable(owner) else _type_identity(owner)
            return f"{owner_identity}.{fn.__name__}"
    qualified_name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    if qualified_name is None:
        return _type_identity(fn)
    return f"{module or type(fn).__module__}.{qualified_name}"


def _callable_configuration_owner(fn: Callable[..., Any]) -> Any | None:
    if inspect.isclass(fn):
        return None
    if callable(getattr(fn, "get_hyper_parameters", None)) or is_dataclass(fn) or isinstance(fn, Enum):
        return fn
    receiver = getattr(fn, "__self__", None)
    if getattr(fn, "__name__", None) == "__call__" and inspect.isroutine(receiver):
        return receiver
    # Runtime hosts also expose get_hyper_parameters(), which can recurse into these callbacks.
    if not inspect.isclass(receiver):
        if is_dataclass(receiver) or isinstance(receiver, (Enum, *_CALLABLE_HELPERS)) or (
            callable(receiver) and callable(getattr(receiver, "get_hyper_parameters", None))
        ):
            return receiver
    return None


class _Serializer:
    def __init__(self, *, opaque_objects: bool) -> None:
        self.opaque_objects = opaque_objects
        self._active_path: ContextVar[frozenset[tuple[str, int]]] = ContextVar("active_metadata_path", default=frozenset())

    @contextmanager
    def _visit(self, value: Any, *, role: str = "value") -> Iterator[bool]:
        if not self.opaque_objects:
            yield True
            return
        # A hook can project a callable dataclass's fields without creating a graph cycle.
        key = (role, id(value))
        active_path = self._active_path.get()
        if key in active_path:
            yield False
            return
        token = self._active_path.set(active_path | {key})
        try:
            yield True
        finally:
            self._active_path.reset(token)

    def function(self, fn: Callable[..., Any]) -> dict[str, Any]:
        with self._visit(fn, role="callable") as first_visit:
            if not first_visit:
                return _opaque_record(fn)
            result: dict[str, Any] = {"repr": _callable_identity(fn), "source": None}
            # Bound arguments have no declared enum type, so member names alone can collide.
            if isinstance(fn, partial):
                result.update(
                    function=self.function(fn.func),
                    args=self.value(fn.args, preserve_enum_identity=True),
                    keywords=self.value(fn.keywords, preserve_enum_identity=True),
                )
            elif isinstance(fn, _OPERATOR_HELPERS):
                constructor, arguments = fn.__reduce__()
                result.update(constructor=self.function(constructor), args=self.value(arguments, preserve_enum_identity=True))
            else:
                source_target = fn if getattr(fn, "__qualname__", None) is not None else type(fn)
                try:
                    result["source"] = inspect.getsource(source_target)
                except (OSError, TypeError):
                    pass
            owner = _callable_configuration_owner(fn)
            if owner is not None:
                if owner is not fn:
                    result["receiver"] = _type_identity(owner)
                scope = nullcontext(True) if owner is fn else self._visit(owner, role="configuration")
                with scope as first_configuration:
                    if not first_configuration:
                        result["config"] = _opaque_record(owner)
                    elif owner is not fn and (isinstance(owner, _CALLABLE_HELPERS) or inspect.isroutine(owner)):
                        result["config"] = self.function(owner)
                    else:
                        hook = getattr(owner, "get_hyper_parameters", None)
                        if callable(hook):
                            result["config"] = self.value(hook())
                        elif is_dataclass(owner):
                            result["config"] = self.dataclass(owner)
                        else:
                            result["config"] = {"name": owner.name, "value": self.value(owner.value)}
            return result

    def value(self, value: Any, *, preserve_enum_identity: bool = False) -> Any:
        if isinstance(value, type):
            return f"{value.__module__}.{value.__qualname__}"
        if is_dataclass(value):
            return self.dataclass(value, preserve_enum_identity=preserve_enum_identity)
        with self._visit(value) as first_visit:
            return self._value(value, preserve_enum_identity=preserve_enum_identity) if first_visit else _opaque_record(value)

    def _value(self, value: Any, *, preserve_enum_identity: bool) -> Any:
        if isinstance(value, torch.Tensor):
            return self.value(value.detach().cpu().tolist(), preserve_enum_identity=preserve_enum_identity)
        if isinstance(value, np.ndarray):
            return self.value(value.tolist(), preserve_enum_identity=preserve_enum_identity)
        if isinstance(value, np.generic):
            decoded = value.item()
            if not self.opaque_objects:
                return decoded
            return _opaque_record(decoded) if isinstance(decoded, np.generic) else self.value(
                decoded, preserve_enum_identity=preserve_enum_identity,
            )
        if isinstance(value, Enum):
            if preserve_enum_identity:
                return {
                    "type": _type_identity(value), "name": value.name,
                    "value": self.value(value.value, preserve_enum_identity=True),
                }
            return value.name
        if isinstance(value, (tuple, list)):
            return [self.value(item, preserve_enum_identity=preserve_enum_identity) for item in value]
        if isinstance(value, dict):
            if self.opaque_objects and any(type(key) is not str for key in value):
                return {
                    "type": _type_identity(value),
                    "entries": [
                        {
                            "key_type": _type_identity(key),
                            "key": self.value(key, preserve_enum_identity=preserve_enum_identity),
                            "value": self.value(inner, preserve_enum_identity=preserve_enum_identity),
                        }
                        for key, inner in value.items()
                    ],
                }
            return {str(key): self.value(inner, preserve_enum_identity=preserve_enum_identity) for key, inner in value.items()}
        if self.opaque_objects:
            if isinstance(value, (bytes, bytearray)):
                return {"type": _type_identity(value), "hex": value.hex()}
            if isinstance(value, (slice, range)):
                return {
                    "type": _type_identity(value),
                    **{
                        name: self.value(getattr(value, name), preserve_enum_identity=preserve_enum_identity)
                        for name in ("start", "stop", "step")
                    },
                }
        if callable(value):
            if isinstance(value, _CALLABLE_HELPERS) or _callable_configuration_owner(value) is not None:
                return serialize_fn(value)
            return _callable_identity(value)
        if self.opaque_objects and value is not None and type(value) not in (str, bool, int, float):
            return _opaque_record(value)
        return value

    def dataclass(self, config: Any, *, preserve_enum_identity: bool = False) -> dict[str, Any]:
        if not is_dataclass(config):
            raise TypeError(f"Expected dataclass instance, got {type(config)}")
        with self._visit(config) as first_visit:
            if not first_visit:
                return _opaque_record(config)
            return {
                field.name: self.value(getattr(config, field.name), preserve_enum_identity=preserve_enum_identity)
                for field in fields(config)
            }


@dataclass
class _SerializationScope:
    serializer: _Serializer
    parent: "_SerializationScope | None" = None
    inherited_path: frozenset[tuple[str, int]] = frozenset()
    closed: bool = False


_metadata_scope: ContextVar[_SerializationScope | None] = ContextVar("metadata_scope", default=None)


@contextmanager
def _serialization_context(*, opaque_objects: bool) -> Iterator[_Serializer]:
    current = _metadata_scope.get()
    inherited_path = None
    while current is not None and current.closed:
        inherited_path = current.inherited_path
        current = current.parent
    if current is not None:
        serializer = current.serializer
    else:
        serializer = _Serializer(opaque_objects=opaque_objects)
    if not serializer.opaque_objects:
        yield serializer
        return
    path_token = None
    if current is not None and inherited_path is not None:
        path_token = serializer._active_path.set(inherited_path)
    scope = _SerializationScope(serializer, parent=current, inherited_path=serializer._active_path.get())
    token = _metadata_scope.set(scope)
    try:
        yield serializer
    finally:
        # Copied entries can share a traversal while outliving the entry they inherited.
        scope.closed = True
        _metadata_scope.reset(token)
        if path_token is not None:
            serializer._active_path.reset(path_token)


def serialize_fn(fn: Callable[..., Any]) -> dict[str, Any]:
    with _serialization_context(opaque_objects=True) as serializer:
        return serializer.function(fn)


def serialize_value(value: Any, *, opaque_objects: bool = False) -> Any:
    with _serialization_context(opaque_objects=opaque_objects) as serializer:
        return serializer.value(value)


def serialize_dataclass(config: Any, *, opaque_objects: bool = False) -> dict[str, Any]:
    with _serialization_context(opaque_objects=opaque_objects) as serializer:
        return serializer.dataclass(config)
