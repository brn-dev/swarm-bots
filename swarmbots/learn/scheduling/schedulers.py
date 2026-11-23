from dataclasses import dataclass, field
from copy import deepcopy
from enum import Enum
import math
from typing import Any, NotRequired, Optional, Protocol, TypedDict

from swarmbots.learn.serialization_utils import bound_accessor_receiver, serialize_fn


class ScheduleResult(TypedDict):
    new_value: Optional[float]
    msg: NotRequired[str]
    event: NotRequired[str]


class Scheduler(Protocol):
    def __call__(
            self,
            old_value: float,
            state: dict[str, Any],
            n_iterations: int,
            n_model_updates: int,
            n_timesteps: int,
            metrics: dict[str, Any],
    ) -> ScheduleResult:
        ...


class ScheduleApplier(Protocol):
    def __call__(self, new_value: float) -> None:
        ...


class ScheduleValueGetter(Protocol):
    def __call__(self) -> float:
        ...


@dataclass(slots=True)
class ScheduledHyperParameter:
    name: str
    scheduler: Scheduler
    get_value: ScheduleValueGetter
    apply: ScheduleApplier
    enabled: bool = True
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SchedulerStepResult:
    name: str
    old_value: float
    new_value: float
    updated: bool
    event: str | None
    msg: str | None


class SchedulerManager:
    def __init__(
            self,
            scheduled_hyper_parameters: list[ScheduledHyperParameter] | None = None,
    ) -> None:
        self.scheduled_hyper_parameters: list[ScheduledHyperParameter] = (
            [] if scheduled_hyper_parameters is None else list(scheduled_hyper_parameters)
        )

    def add(self, scheduled_hyper_parameter: ScheduledHyperParameter) -> None:
        self.scheduled_hyper_parameters.append(scheduled_hyper_parameter)

    def state_dict(self) -> list[dict[str, Any]]:
        return [
            {"name": item.name, "enabled": item.enabled, "value": float(item.get_value()), "state": deepcopy(item.state)}
            for item in self.scheduled_hyper_parameters
        ]

    def validate_state_dict(self, state_dict: list[dict[str, Any]]) -> None:
        names = [item.name for item in self.scheduled_hyper_parameters]
        if [item["name"] for item in state_dict] != names:
            raise ValueError("Checkpoint schedulers must match the configured scheduler names and order")

        for saved in state_dict:
            if type(saved["enabled"]) is not bool or not isinstance(saved["state"], dict) or not math.isfinite(saved["value"]):
                raise ValueError("Invalid checkpoint scheduler state")

    def load_state_dict(self, state_dict: list[dict[str, Any]]) -> None:
        self.validate_state_dict(state_dict)
        staged_states = [deepcopy(saved["state"]) for saved in state_dict]
        # Custom callbacks can reject a value even when the serialized state is well-formed.
        previous_values = [float(item.get_value()) for item in self.scheduled_hyper_parameters]
        attempted = 0
        try:
            for item, saved in zip(self.scheduled_hyper_parameters, state_dict, strict=True):
                attempted += 1
                item.apply(saved["value"])
        except Exception:
            for index in reversed(range(attempted)):
                self.scheduled_hyper_parameters[index].apply(previous_values[index])
            raise
        for item, saved, state in zip(self.scheduled_hyper_parameters, state_dict, staged_states, strict=True):
            item.enabled = saved["enabled"]
            item.state = state

    def step(
            self,
            *,
            n_iterations: int,
            n_model_updates: int,
            n_timesteps: int,
            metrics: dict[str, Any],
    ) -> list[SchedulerStepResult]:
        step_results: list[SchedulerStepResult] = []
        for scheduled in self.scheduled_hyper_parameters:
            if not scheduled.enabled:
                continue

            old_value = float(scheduled.get_value())
            update_result = scheduled.scheduler(
                old_value=old_value,
                state=scheduled.state,
                n_iterations=n_iterations,
                n_model_updates=n_model_updates,
                n_timesteps=n_timesteps,
                metrics=metrics,
            )

            requested_value = update_result.get("new_value")
            event = update_result.get("event")
            msg = update_result.get("msg")

            if requested_value is None:
                step_results.append(
                    SchedulerStepResult(
                        name=scheduled.name,
                        old_value=old_value,
                        new_value=old_value,
                        updated=False,
                        event=event,
                        msg=msg,
                    )
                )
                continue

            scheduled.apply(float(requested_value))
            new_value = float(scheduled.get_value())
            step_results.append(
                SchedulerStepResult(
                    name=scheduled.name,
                    old_value=old_value,
                    new_value=new_value,
                    updated=(new_value != old_value),
                    event=event,
                    msg=msg,
                )
            )

        return step_results

    def serialize(self, *, for_checkpoint_match: bool = False) -> list[dict[str, Any]]:
        return [
            {
                "name": scheduled.name,
                "enabled": scheduled.enabled,
                "scheduler": serialize_fn(scheduled.scheduler, for_checkpoint_match=for_checkpoint_match),
                "get_value": serialize_fn(
                    scheduled.get_value, for_checkpoint_match=for_checkpoint_match,
                    include_receiver_config=not for_checkpoint_match,
                ),
                "apply": serialize_fn(
                    scheduled.apply, for_checkpoint_match=for_checkpoint_match,
                    include_receiver_config=not for_checkpoint_match,
                ),
            }
            for scheduled in self.scheduled_hyper_parameters
        ]

    def checkpoint_configuration(self) -> list[dict[str, Any]]:
        configuration = self.serialize(for_checkpoint_match=True)
        for scheduled, entry in zip(self.scheduled_hyper_parameters, configuration, strict=True):
            del entry["enabled"]
            getter_receiver = bound_accessor_receiver(scheduled.get_value)
            applier_receiver = bound_accessor_receiver(scheduled.apply)
            if getter_receiver is not None and applier_receiver is not None:
                entry["shared_receiver"] = getter_receiver is applier_receiver
        return configuration


class ScheduleUnit(Enum):
    ITERATIONS = 1
    MODEL_UPDATES = 2
    TIMESTEPS = 3

    @staticmethod
    def get(
            unit: 'ScheduleUnit',
            n_iterations: int,
            n_model_updates: int,
            n_timesteps: int
    ) -> int:
        if unit == ScheduleUnit.ITERATIONS:
            return n_iterations
        if unit == ScheduleUnit.MODEL_UPDATES:
            return n_model_updates
        if unit == ScheduleUnit.TIMESTEPS:
            return n_timesteps
        raise ValueError(f'Invalid unit {unit}')
