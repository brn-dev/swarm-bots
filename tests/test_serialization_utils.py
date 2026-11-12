from collections.abc import Callable
from functools import partial
from dataclasses import dataclass
from enum import Enum, IntEnum, IntFlag
from threading import Barrier, Event, Lock
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from types import SimpleNamespace
from operator import attrgetter, itemgetter, methodcaller
import json
from typing import Any

import numpy as np
import pytest
import torch

from swarmbots.learn.scheduling.chainable_scheduler import ChainableScheduler
from swarmbots.learn.scheduling.cosine_scheduler import CosineScheduler
from swarmbots.learn.scheduling.exponential_scheduler import ExponentialScheduler
from swarmbots.learn.scheduling.linear_scheduler import LinearScheduler
from swarmbots.learn.scheduling.schedulers import ScheduledHyperParameter, SchedulerManager, ScheduleResult, ScheduleUnit
from swarmbots.learn.serialization_utils import serialize_dataclass, serialize_fn, serialize_value


@pytest.mark.parametrize("function", [len, [].append, object().__str__, dict, dict.fromkeys])
def test_native_callable_metadata_has_stable_identity_without_source(function: Callable[..., Any]) -> None:
    assert serialize_fn(function) == {
        "repr": f"builtins.{function.__qualname__}",
        "source": None,
    }


def test_function_metadata_preserves_source_without_instance_addresses() -> None:
    def make_function() -> Callable[[], float]:
        def scheduled_value() -> float:
            return 0.5

        return scheduled_value

    first = make_function()
    second = make_function()
    metadata = serialize_fn(first)

    assert metadata == serialize_fn(second)
    assert metadata["repr"] == f"{first.__module__}.{first.__qualname__}"
    assert "return 0.5" in metadata["source"]


@pytest.mark.parametrize(("factory", "shape_options"), [
    (LinearScheduler, {}),
    (CosineScheduler, {"bias": 1.5, "sharpness": 2.0}),
    (ExponentialScheduler, {"base": 2.0}),
])
@pytest.mark.parametrize("callback_kind", ["object", "method", "partial"])
def test_scheduler_manager_records_every_behavior_setting(
    factory: Callable[..., ChainableScheduler], shape_options: dict[str, float], callback_kind: str,
) -> None:
    options = {
        "unit": ScheduleUnit.ITERATIONS, "duration": 10, "start_value": 0.001,
        "final_value": 0.01, "name": "warmup", **shape_options,
    }

    def metadata(overrides: dict[str, Any]) -> dict[str, Any]:
        scheduler = factory(**(options | overrides))
        callback = scheduler if callback_kind == "object" else scheduler.__call__
        if callback_kind == "partial":
            callback = partial(callback)
        manager = SchedulerManager([ScheduledHyperParameter(
            name="lr", scheduler=callback,
            get_value=lambda: 0.001, apply=lambda value: None,
        )])
        record = manager.serialize()[0]["scheduler"]
        assert serialize_value(callback) == record
        if callback_kind == "partial":
            record = record["function"]
        if callback_kind != "object":
            assert record["receiver"] == f"{factory.__module__}.{factory.__qualname__}"
        return record

    original = metadata({})
    assert original["config"] == {**options, "unit": "ITERATIONS"}
    changes = {
        "unit": ScheduleUnit.TIMESTEPS, "duration": 100, "start_value": 0.002,
        "final_value": 0.02, "name": "cooldown",
        **{key: value + 1.0 for key, value in shape_options.items()},
    }
    for key, value in changes.items():
        assert metadata({key: value}) != original, key


def test_partial_metadata_records_target_and_bound_arguments() -> None:
    first = partial(pow, exp=2)
    same = partial(pow, exp=2)
    changed_keyword = partial(pow, exp=3)
    changed_target = partial(abs)
    changed_argument = partial(pow, 2)

    assert first(-3) != changed_target(-3)
    metadata = serialize_fn(first)
    assert metadata == serialize_fn(same) == serialize_value(first)
    assert metadata["function"]["repr"] == "builtins.pow"
    assert metadata["keywords"] == {"exp": 2}
    assert serialize_fn(changed_argument)["args"] == [2]
    for changed in (changed_keyword, changed_target, changed_argument):
        assert serialize_fn(changed) != metadata


@pytest.mark.parametrize("helper_kind", ["partial", "attribute", "item", "method"])
@pytest.mark.parametrize("wrap_getter", [False, True])
def test_bound_helper_methods_keep_their_configuration_in_scheduler_metadata(
    helper_kind: str, wrap_getter: bool,
) -> None:
    values = {"gamma": 0.99, "gae_lambda": 0.95}
    host = SimpleNamespace(**values)

    def getter(key: str) -> Callable[[], float]:
        if helper_kind == "partial":
            callback = partial(dict.get, values, key).__call__
        elif helper_kind == "attribute":
            callback = partial(attrgetter(key).__call__, host)
        elif helper_kind == "item":
            callback = partial(itemgetter(key).__call__, values)
        else:
            callback = partial(methodcaller("get", key).__call__, values)
        return partial(callback) if wrap_getter else callback

    def metadata(key: str) -> dict[str, Any]:
        callback = getter(key)
        manager = SchedulerManager([ScheduledHyperParameter(
            name="value", scheduler=lambda **kwargs: {"new_value": None},
            get_value=callback, apply=lambda value: None,
        )])
        step = manager.step(n_iterations=1, n_model_updates=0, n_timesteps=0, metrics={})[0]
        assert step.old_value == step.new_value == values[key]
        result = manager.serialize()[0]["get_value"]
        assert result == serialize_fn(callback) == serialize_value(callback)
        assert json.loads(json.dumps(result)) == result
        return result

    first = metadata("gamma")
    assert first == metadata("gamma")
    assert first != metadata("gae_lambda")


def test_bound_partial_method_with_recursive_arguments_terminates() -> None:
    def constant_value(**kwargs: Any) -> float:
        return 0.5

    callback = partial(constant_value)
    callback.keywords["recursive"] = callback.__call__
    assert callback.__call__() == 0.5

    metadata = serialize_fn(callback.__call__)
    encoded = json.dumps(metadata)
    assert '"opaque": true' in encoded
    assert json.loads(encoded) == metadata


def test_bound_partial_override_keeps_method_source_and_receiver_configuration() -> None:
    class OffsetPartial(partial):
        def __call__(self) -> float:
            return super().__call__() + 0.1

        def get_hyper_parameters(self) -> dict[str, float]:
            return {"offset": 0.1}

    getter = OffsetPartial(dict.get, {"gamma": 0.9}, "gamma")
    assert getter.__call__() == 1.0

    metadata = serialize_fn(getter.__call__)
    assert "super().__call__() + 0.1" in metadata["source"]
    assert metadata["config"] == serialize_fn(getter)
    assert metadata["config"]["config"] == {"offset": 0.1}


@pytest.mark.parametrize("target_kind", ["python", "native", "constructor"])
def test_routine_and_constructor_call_wrappers_keep_target_identity(target_kind: str) -> None:
    def first_value(values: list[int]) -> int:
        return len(values)

    def second_value(values: list[int]) -> int:
        return sum(values)

    if target_kind == "python":
        targets, argument = (first_value, second_value), [1, 2]
    elif target_kind == "native":
        targets, argument = (len, sum), [1, 2]
    else:
        targets, argument = (int, float), 1.9

    records = []
    values = []
    for target in targets:
        getter = partial(target.__call__, argument)
        manager = SchedulerManager([ScheduledHyperParameter(
            name="value", scheduler=lambda **kwargs: {"new_value": None},
            get_value=getter, apply=lambda value: None,
        )])
        values.append(manager.state_dict()[0]["value"])
        record = manager.serialize()[0]["get_value"]
        assert record == serialize_fn(getter)
        assert json.loads(json.dumps(record)) == record
        records.append(record)
    assert values[0] != values[1]
    assert records[0] != records[1]


@pytest.mark.parametrize("target_kind", ["classmethod", "metaclass"])
@pytest.mark.parametrize("callback_kind", ["method", "partial", "call_wrapper", "partial_call_wrapper"])
def test_class_bound_callbacks_keep_concrete_receiver_identity(target_kind: str, callback_kind: str) -> None:
    class Settings:
        @classmethod
        def get_value(cls) -> float:
            return cls.rate

        @classmethod
        def get_hyper_parameters(cls) -> dict[str, Any]:
            raise AssertionError("Class configuration hooks must not be invoked")

    class Factory(type):
        def __call__(cls) -> float:
            return cls.rate

    class Warmup(Settings, metaclass=Factory):
        rate = 0.1

    class Cooldown(Settings, metaclass=Factory):
        rate = 0.2

    def metadata(owner: type) -> dict[str, Any]:
        getter = owner.get_value if target_kind == "classmethod" else owner.__call__
        if "call_wrapper" in callback_kind:
            getter = getter.__call__
        if callback_kind.startswith("partial"):
            getter = partial(getter)
        manager = SchedulerManager([ScheduledHyperParameter(
            name="value", scheduler=lambda **kwargs: {"new_value": None},
            get_value=getter, apply=lambda value: None,
        )])
        assert manager.state_dict()[0]["value"] == owner.rate
        record = manager.serialize()[0]["get_value"]
        assert record == serialize_fn(getter)
        assert json.loads(json.dumps(record)) == record
        return record

    first = metadata(Warmup)
    assert first == metadata(Warmup)
    assert first != metadata(Cooldown)


@pytest.mark.parametrize("callback_kind", ["reduction", "reduction_call", "operator_call"])
def test_numpy_ufunc_receivers_keep_their_intrinsic_identity(callback_kind: str) -> None:
    values = []
    records = []
    for operation in (np.add, np.multiply):
        if callback_kind == "reduction":
            getter = partial(operation.reduce, [2.0, 3.0, 4.0])
        elif callback_kind == "reduction_call":
            getter = partial(operation.reduce.__call__, [2.0, 3.0, 4.0])
        else:
            getter = partial(operation.__call__, 2.0, 3.0)
        manager = SchedulerManager([ScheduledHyperParameter(
            name="value", scheduler=lambda **kwargs: {"new_value": None},
            get_value=getter, apply=lambda value: None,
        )])
        values.append(manager.state_dict()[0]["value"])
        record = manager.serialize()[0]["get_value"]
        assert record == serialize_fn(getter)
        assert json.loads(json.dumps(record)) == record
        records.append(record)
    assert values[0] != values[1]
    assert records[0] != records[1]


def test_partial_mapping_arguments_preserve_key_types_and_collisions() -> None:
    numeric = partial(dict.get, {1: 0.99}, 1, 0.95)
    textual = partial(dict.get, {"1": 0.99}, 1, 0.95)
    mixed = partial(dict.get, {1: 0.99, "1": 0.95}, 1)

    assert numeric() == 0.99 and textual() == 0.95
    first, second, both = (serialize_fn(callback) for callback in (numeric, textual, mixed))
    assert first != second
    assert both["args"][0] == {
        "type": "builtins.dict", "entries": [
            {"key_type": "builtins.int", "key": 1, "value": 0.99},
            {"key_type": "builtins.str", "key": "1", "value": 0.95},
        ],
    }
    assert json.loads(json.dumps(both)) == both
    assert second["args"][0] == {"1": 0.99}


@pytest.mark.parametrize(("container_type", "contents", "expected"), [
    (dict, {"gamma": 0.99}, {"gamma": 0.99}),
    (list, [0.99], [0.99]),
    (tuple, (0.99,), [0.99]),
])
@pytest.mark.parametrize("opaque_objects", [False, True])
def test_callable_containers_keep_their_data(
    container_type: type, contents: Any, expected: Any, opaque_objects: bool,
) -> None:
    class CallableContainer(container_type):
        def __call__(self) -> Any:
            return self

    assert serialize_value(CallableContainer(contents), opaque_objects=opaque_objects) == expected


def test_partial_arguments_keep_callable_mapping_contents() -> None:
    class CallableMapping(dict[str, float]):
        def __call__(self, key: str) -> float:
            return self[key]

    first = partial(dict.get, CallableMapping(gamma=0.99), "gamma")
    same = partial(dict.get, CallableMapping(gamma=0.99), "gamma")
    changed = partial(dict.get, CallableMapping(gamma=0.9), "gamma")

    assert first() == 0.99 and changed() == 0.9
    metadata = serialize_fn(first)
    assert metadata["args"] == [{"gamma": 0.99}, "gamma"]
    assert metadata == serialize_fn(same)
    assert metadata != serialize_fn(changed)
    assert json.loads(json.dumps(metadata)) == metadata


def test_callable_container_cycles_still_terminate() -> None:
    class CallableList(list):
        def __call__(self) -> Any:
            return self[0]

    circular = CallableList()
    circular.append(circular)

    assert serialize_value(circular, opaque_objects=True) == [
        {"type": f"{CallableList.__module__}.{CallableList.__qualname__}", "opaque": True},
    ]


@pytest.mark.parametrize("callback_kind", ["object", "method", "partial_object", "partial_method"])
def test_enum_scheduler_members_remain_distinct_in_metadata(callback_kind: str) -> None:
    class ConstantScheduler(Enum):
        LOW = 0.1
        HIGH = 0.2

        def __call__(self, **kwargs: Any) -> ScheduleResult:
            return {"new_value": self.value}

    records = []
    for scheduler in ConstantScheduler:
        callback = scheduler.__call__ if callback_kind.endswith("method") else scheduler
        if callback_kind.startswith("partial"):
            callback = partial(callback)
        values = {"value": 1.0}
        manager = SchedulerManager([ScheduledHyperParameter(
            name="value", scheduler=callback,
            get_value=lambda: values["value"], apply=lambda value: values.update(value=value),
        )])
        manager.step(n_iterations=1, n_model_updates=0, n_timesteps=0, metrics={})

        assert values["value"] == scheduler.value
        for opaque_objects in (False, True):
            assert serialize_value(scheduler, opaque_objects=opaque_objects) == scheduler.name
        metadata = manager.serialize()[0]["scheduler"]
        assert metadata == serialize_fn(callback)
        assert json.loads(json.dumps(metadata)) == metadata
        records.append(metadata)
    assert records[0] != records[1]


def test_unnamed_callable_flag_members_remain_distinct_in_metadata() -> None:
    class ConstantFlags(IntFlag):
        FIRST = 1

        def __call__(self, **kwargs: Any) -> ScheduleResult:
            return {"new_value": float(self.value)}

    first, second = ConstantFlags(2), ConstantFlags(4)
    assert first.name is None and second.name is None
    assert first()["new_value"] != second()["new_value"]
    assert serialize_fn(first) != serialize_fn(second)


@pytest.mark.parametrize("enum_kind", ["unnamed_flags", "named_integers", "named_floats"])
@pytest.mark.parametrize("argument_kind", ["positional", "keyword", "nested", "operator"])
def test_bound_enum_arguments_preserve_type_name_and_value(enum_kind: str, argument_kind: str) -> None:
    if enum_kind == "unnamed_flags":
        class Bits(IntFlag):
            FIRST = 1

        members = (Bits(2), Bits(4))
    elif enum_kind == "named_integers":
        class LowInteger(IntEnum):
            RATE = 1

        class HighInteger(IntEnum):
            RATE = 2

        members = (LowInteger.RATE, HighInteger.RATE)
    else:
        class LowFloat(float, Enum):
            RATE = 0.1

        class HighFloat(float, Enum):
            RATE = 0.2

        members = (LowFloat.RATE, HighFloat.RATE)

    @dataclass
    class Settings:
        rates: tuple[Enum, ...]

    def keyword_value(*, rate: Enum) -> float:
        return float(rate)

    def nested_value(options: dict[str, Settings]) -> float:
        return float(options["settings"].rates[0])

    records = []
    shared_values = {member: float(member) for member in members}
    for member in members:
        if argument_kind == "positional":
            getter = partial(float, member)
        elif argument_kind == "keyword":
            getter = partial(keyword_value, rate=member)
        elif argument_kind == "nested":
            getter = partial(nested_value, {"settings": Settings((member,))})
        else:
            getter = partial(itemgetter(member), shared_values)
        manager = SchedulerManager([ScheduledHyperParameter(
            name="value", scheduler=lambda **kwargs: {"new_value": None},
            get_value=getter, apply=lambda value: None,
        )])
        step = manager.step(n_iterations=1, n_model_updates=0, n_timesteps=0, metrics={})[0]
        assert step.old_value == step.new_value == float(member)
        record = manager.serialize()[0]["get_value"]
        assert json.loads(json.dumps(record)) == record
        assert record == serialize_fn(getter)
        for opaque_objects in (False, True):
            assert serialize_value(member, opaque_objects=opaque_objects) == member.name
        records.append(record)
    assert records[0] != records[1]


@pytest.mark.parametrize("callback_kind", ["object", "method", "partial"])
def test_dataclass_scheduler_fields_survive_without_a_configuration_hook(callback_kind: str) -> None:
    @dataclass
    class ConstantScheduler:
        final_value: float

        def __call__(self, **kwargs: Any) -> ScheduleResult:
            return {"new_value": self.final_value}

    records = []
    for configured_value in (0.1, 0.2):
        scheduler = ConstantScheduler(configured_value)
        callback = scheduler if callback_kind == "object" else scheduler.__call__
        if callback_kind == "partial":
            callback = partial(callback)
        values = {"value": 1.0}
        manager = SchedulerManager([ScheduledHyperParameter(
            name="value", scheduler=callback,
            get_value=lambda: values["value"], apply=lambda value: values.update(value=value),
        )])
        manager.step(n_iterations=1, n_model_updates=0, n_timesteps=0, metrics={})
        assert values["value"] == configured_value
        record = manager.serialize()[0]["scheduler"]
        if callback_kind == "partial":
            record = record["function"]
        assert record["config"] == {"final_value": configured_value}
        records.append(record)
    assert records[0] != records[1]


def test_dataclass_method_receiver_preserves_fields_and_explicit_hook_precedence() -> None:
    @dataclass
    class Settings:
        value: float

        def read(self) -> float:
            return self.value

    @dataclass
    class ConfiguredSettings(Settings):
        def get_hyper_parameters(self) -> dict[str, Any]:
            return {"effective_value": self.value}

    assert serialize_fn(Settings(0.1).read)["config"] == {"value": 0.1}
    assert serialize_fn(ConfiguredSettings(0.2).read)["config"] == {"effective_value": 0.2}


@pytest.mark.parametrize("runtime_kind", ["object", "numeric_subclass"])
def test_partial_metadata_detaches_runtime_arguments_and_keywords(runtime_kind: str) -> None:
    @dataclass
    class Options:
        target: Any
        scale: float

    class RuntimeNumber(float):
        pass

    target = SimpleNamespace() if runtime_kind == "object" else RuntimeNumber(1.0)
    target.value = 1.0
    target.runtime_lock = Lock()
    options = Options(target=target, scale=0.5)

    def value_from_context(items: list[Any], *, context: dict[str, Any]) -> float:
        return items[1]["options"].scale * context["target"].value

    callback = partial(value_from_context, [target, {"options": options}], context={"target": target})
    assert callback() == 0.5

    metadata = serialize_fn(callback)

    target_type = "types.SimpleNamespace" if runtime_kind == "object" else (
        f"{RuntimeNumber.__module__}.{RuntimeNumber.__qualname__}"
    )
    opaque_target = {"type": target_type, "opaque": True}
    assert metadata["args"] == [[opaque_target, {"options": {"target": opaque_target, "scale": 0.5}}]]
    assert metadata["keywords"] == {"context": {"target": opaque_target}}
    target.value = 2.0
    assert metadata == serialize_fn(callback)


def test_numpy_scalars_without_a_python_equivalent_do_not_recurse() -> None:
    value = np.longdouble(1)

    assert serialize_value(value) == value
    metadata = serialize_fn(partial(float, value))
    assert metadata["args"] == [{"type": "numpy.longdouble", "opaque": True}]


def test_partial_metadata_cuts_cycles_without_discarding_shared_data() -> None:
    shared = {"value": 0.5}
    recursive_dict: dict[str, Any] = {}
    recursive_dict["self"] = recursive_dict
    recursive_list: list[Any] = []
    recursive_list.append(recursive_list)
    recursive_array = np.empty(1, dtype=object)
    recursive_array[0] = recursive_array

    def read_value(items: list[Any]) -> float:
        return items[0]["value"]

    callback = partial(read_value, [shared, shared, recursive_dict, recursive_list, recursive_array])
    assert callback() == 0.5
    metadata = serialize_fn(callback)

    assert metadata["args"] == [[
        {"value": 0.5}, {"value": 0.5},
        {"self": {"type": "builtins.dict", "opaque": True}},
        [{"type": "builtins.list", "opaque": True}],
        [{"type": "numpy.ndarray", "opaque": True}],
    ]]
    assert json.loads(json.dumps(metadata)) == metadata


@pytest.mark.parametrize("helper_kind", ["attribute", "item", "method"])
def test_operator_getters_keep_constructor_settings_in_scheduler_metadata(helper_kind: str) -> None:
    values = {"gamma": 0.99, "gae_lambda": 0.95}

    def getter(key: str) -> Callable[[], float]:
        if helper_kind == "attribute":
            return partial(attrgetter(key), SimpleNamespace(**values))
        if helper_kind == "item":
            return partial(itemgetter(key), values)
        return partial(methodcaller("get", key), values)

    records = []
    for key, expected in values.items():
        callback = getter(key)
        manager = SchedulerManager([ScheduledHyperParameter(
            name="value", scheduler=LinearScheduler(ScheduleUnit.ITERATIONS, 10, 0.99, 0.9),
            get_value=callback, apply=lambda value: None,
        )])
        assert manager.state_dict()[0]["value"] == expected
        record = manager.serialize()[0]["get_value"]
        assert record == serialize_value(callback)
        assert json.loads(json.dumps(record)) == record
        records.append(record)
    assert records[0] != records[1]


def test_methodcaller_keyword_configuration_remains_distinct() -> None:
    class Values:
        def read(self, *, key: str, fallback: float = 0.0) -> float:
            return {"gamma": 0.99, "gae_lambda": 0.95}.get(key, fallback)

    host = Values()
    first = partial(methodcaller("read", key="gamma", fallback=0.1), host)
    same = partial(methodcaller("read", key="gamma", fallback=0.1), host)
    changed_key = partial(methodcaller("read", key="gae_lambda", fallback=0.1), host)
    changed_fallback = partial(methodcaller("read", key="gamma", fallback=0.2), host)

    assert first() == 0.99 and changed_key() == 0.95
    metadata = serialize_fn(first)
    assert metadata == serialize_fn(same)
    assert metadata != serialize_fn(changed_key)
    assert metadata != serialize_fn(changed_fallback)


@pytest.mark.parametrize("key", [slice(0, 1), slice(1, 2), b"gamma", b"gae_lambda"])
def test_itemgetter_structured_keys_are_explicit(key: Any) -> None:
    metadata = serialize_fn(itemgetter(key))
    argument = metadata["args"][0]
    if isinstance(key, slice):
        assert argument == {"type": "builtins.slice", "start": key.start, "stop": key.stop, "step": key.step}
    else:
        assert argument == {"type": "builtins.bytes", "hex": key.hex()}


def test_callable_dataclass_can_export_its_fields_through_a_metadata_hook() -> None:
    @dataclass
    class ConfiguredCallable:
        value: float

        def __call__(self) -> float:
            return self.value

        def get_hyper_parameters(self) -> dict[str, Any]:
            return serialize_dataclass(self)

    assert serialize_fn(ConfiguredCallable(0.5))["config"] == {"value": 0.5}


def test_failed_configuration_hook_does_not_leak_metadata_context() -> None:
    class ConfiguredCallable:
        fail = True

        def __call__(self) -> float:
            return 0.5

        def get_hyper_parameters(self) -> dict[str, Any]:
            if self.fail:
                raise ValueError("configuration failure")
            return {"value": 0.5}

    callback = ConfiguredCallable()
    with pytest.raises(ValueError, match="configuration failure"):
        serialize_fn(callback)
    callback.fail = False
    assert serialize_fn(callback)["config"] == {"value": 0.5}
    runtime_object = SimpleNamespace(value=0.5)
    assert serialize_value(runtime_object) is runtime_object


def test_concurrent_metadata_collections_do_not_share_active_objects() -> None:
    barrier = Barrier(2)

    class ConfiguredCallable:
        def __call__(self) -> float:
            return 0.5

        def get_hyper_parameters(self) -> dict[str, Any]:
            barrier.wait(timeout=5)
            return {"value": 0.5}

    callback = ConfiguredCallable()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(serialize_fn, (callback, callback)))
    assert results[0] == results[1]
    assert results[0]["config"] == {"value": 0.5}


def test_parallel_copied_contexts_do_not_mark_shared_children_as_cycles() -> None:
    barrier = Barrier(2)

    class Child:
        def __call__(self) -> float:
            return 0.5

        def get_hyper_parameters(self) -> dict[str, Any]:
            barrier.wait(timeout=5)
            return {"value": 0.5}

    class Parent:
        def __call__(self) -> float:
            return 0.5

        def get_hyper_parameters(self) -> dict[str, Any]:
            child = Child()
            contexts = [copy_context(), copy_context()]
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(context.run, serialize_fn, child) for context in contexts]
                return {"children": [future.result() for future in futures]}

    children = serialize_fn(Parent())["config"]["children"]
    assert children[0] == children[1]
    assert children[0]["config"] == {"value": 0.5}


def test_copied_context_can_outlive_its_original_metadata_collection() -> None:
    contexts = []

    class Exporter:
        def __call__(self) -> float:
            return 0.5

        def get_hyper_parameters(self) -> dict[str, Any]:
            contexts.append(copy_context())
            return {"value": 0.5}

    callback = Exporter()
    original = serialize_fn(callback)
    assert contexts[0].run(serialize_fn, callback) == original
    runtime_object = SimpleNamespace(value=0.5)
    assert contexts[0].run(serialize_value, runtime_object) is runtime_object


@pytest.mark.parametrize("parent_fails", [False, True])
def test_active_copied_collection_retains_metadata_safety_after_parent_exits(parent_fails: bool) -> None:
    @dataclass
    class Settings:
        value: float = 0.5
        self_reference: Any = None

    settings = Settings()
    settings.self_reference = settings
    child_started = Event()
    release_child = Event()
    parent_contexts = []
    child_contexts = []
    futures = []

    class Child:
        def __call__(self) -> float:
            return settings.value

        def get_hyper_parameters(self) -> dict[str, Any]:
            child_contexts.append(copy_context())
            child_started.set()
            assert release_child.wait(timeout=5)
            return serialize_dataclass(settings)

    child = Child()
    with ThreadPoolExecutor(max_workers=1) as executor:
        class Parent:
            def __call__(self) -> float:
                return 1.0

            def get_hyper_parameters(self) -> dict[str, Any]:
                parent_contexts.append(copy_context())
                futures.append(executor.submit(copy_context().run, serialize_fn, child))
                assert child_started.wait(timeout=5)
                if parent_fails:
                    raise ValueError("parent configuration failed")
                return {"value": 1.0}

        try:
            if parent_fails:
                with pytest.raises(ValueError, match="parent configuration failed"):
                    serialize_fn(Parent())
            else:
                assert serialize_fn(Parent())["config"] == {"value": 1.0}
            runtime_object = SimpleNamespace()
            assert parent_contexts[0].run(serialize_value, runtime_object) is runtime_object
        finally:
            release_child.set()
        metadata = futures[0].result(timeout=5)

    assert metadata["config"] == {
        "value": 0.5,
        "self_reference": {"type": f"{Settings.__module__}.{Settings.__qualname__}", "opaque": True},
    }
    assert child_contexts[0].run(serialize_fn, child) == metadata
    assert child_contexts[0].run(serialize_value, runtime_object) is runtime_object


@pytest.mark.parametrize("nested_fails", [False, True])
def test_expired_nested_copy_rejoins_its_live_enclosing_collection(nested_fails: bool) -> None:
    contexts = []
    runtime_object = SimpleNamespace()

    class Inner:
        fail = nested_fails

        def __call__(self) -> float:
            return 0.5

        def get_hyper_parameters(self) -> dict[str, Any]:
            contexts.append(copy_context())
            if self.fail:
                raise ValueError("nested configuration failed")
            return {"value": 0.5}

    inner = Inner()

    class Outer:
        def __call__(self) -> float:
            return 1.0

        def get_hyper_parameters(self) -> dict[str, Any]:
            if nested_fails:
                with pytest.raises(ValueError, match="nested configuration failed"):
                    serialize_fn(inner)
            else:
                serialize_fn(inner)
            inner.fail = False
            context = contexts[0]
            repeated_inner = context.run(serialize_fn, inner)
            return context.run(serialize_value, {
                "outer": self, "runtime": runtime_object, "inner": repeated_inner,
            })

    metadata = serialize_fn(Outer())
    assert metadata["config"]["outer"] == {
        "type": f"{Outer.__module__}.{Outer.__qualname__}", "opaque": True,
    }
    assert metadata["config"]["runtime"] == {"type": "types.SimpleNamespace", "opaque": True}
    assert metadata["config"]["inner"]["config"] == {"value": 0.5}
    assert contexts[0].run(serialize_value, runtime_object) is runtime_object


def test_descriptor_scheduler_can_execute_and_serialize() -> None:
    class DescriptorScheduler:
        def __get__(self, instance: Any, owner: type | None = None) -> "DescriptorScheduler":
            return self

        def __call__(self, **kwargs: Any) -> ScheduleResult:
            return {"new_value": 0.25}

    class Holder:
        scheduler = DescriptorScheduler()

    values = {"lr": 1.0}
    manager = SchedulerManager([ScheduledHyperParameter(
        name="lr", scheduler=Holder().scheduler,
        get_value=lambda: values["lr"], apply=lambda value: values.update(lr=value),
    )])
    manager.step(n_iterations=1, n_model_updates=0, n_timesteps=0, metrics={})

    assert values["lr"] == 0.25
    metadata = manager.serialize()[0]["scheduler"]
    assert metadata["repr"].endswith(".DescriptorScheduler")
    assert serialize_value(Holder().scheduler) == metadata["repr"]


@pytest.mark.parametrize(("factory", "owner_name", "setter"), [
    (lambda: torch.tensor(1.0), "torch.Tensor", "fill_"),
    (lambda: torch.nn.Parameter(torch.tensor(1.0), requires_grad=False), "torch.nn.parameter.Parameter", "fill_"),
    (lambda: np.array(1.0), "numpy.ndarray", "fill"),
])
def test_native_schedule_methods_record_their_receiver_class(
    factory: Callable[[], Any], owner_name: str, setter: str,
) -> None:
    value = factory()
    manager = SchedulerManager([ScheduledHyperParameter(
        name="lr", scheduler=lambda **kwargs: {"new_value": 0.25},
        get_value=value.item, apply=getattr(value, setter),
    )])
    manager.step(n_iterations=1, n_model_updates=0, n_timesteps=0, metrics={})

    assert value.item() == 0.25
    metadata = manager.serialize()[0]
    assert metadata["get_value"]["repr"] == f"{owner_name}.item"
    assert metadata["apply"]["repr"] == f"{owner_name}.{setter}"
    assert serialize_value(value.item) == f"{owner_name}.item"
