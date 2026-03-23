import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NotRequired, Optional, Protocol, TypedDict

import numpy as np

from swarmbots.learn.serialization_utils import serialize_fn


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

    def serialize(self) -> list[dict[str, Any]]:
        return [
            {
                "name": scheduled.name,
                "enabled": scheduled.enabled,
                "scheduler": serialize_fn(scheduled.scheduler),
                "get_value": serialize_fn(scheduled.get_value),
                "apply": serialize_fn(scheduled.apply),
            }
            for scheduled in self.scheduled_hyper_parameters
        ]

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

class ChainableScheduler(abc.ABC):

    def __init__(
            self,
            unit: ScheduleUnit,
    ):
        self.unit = unit

    def __call__(
            self,
            old_value: float,
            state: dict[str, Any],
            n_iterations: int,
            n_model_updates: int,
            n_timesteps: int,
            metrics: dict[str, Any],
    ) -> ScheduleResult:
        t = ScheduleUnit.get(
            self.unit,
            n_iterations=n_iterations,
            n_model_updates=n_model_updates,
            n_timesteps=n_timesteps
        )
        return self.schedule(
            progress=t / self.get_duration(),
            old_value=old_value,
            state=state,
            metrics=metrics,
        )

    @abc.abstractmethod
    def schedule(
            self,
            progress: float,
            old_value: float,
            state: dict[str, Any],
            metrics: dict[str, Any],
    ) -> ScheduleResult:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_duration(self) -> int:
        raise NotImplementedError()


# class ScheduleChain(ChainableScheduler):
#
#     def __init__(self, unit: ScheduleUnit, schedulers: list[ChainableScheduler]):
#         super().__init__(unit)
#         self.schedulers = schedulers
#
#         self.scheduler_durations = [s.get_duration() for s in schedulers]
#         self.scheduler_starts = [0] + np.cumsum(self.scheduler_durations).tolist()[:-1]
#
#         self.duration = sum(self.scheduler_durations)
#
#         self.scheduler_idx = 0
#
#     def get_duration(self) -> int:
#         return self.duration
#
#     def schedule(
#             self,
#             progress: float,
#             old_value: float,
#             state: dict[str, Any],
#             metrics: dict[str, Any],
#     ) -> ScheduleResult:
#
#
#
#
# class ScheduleChainBuilder:
#
#     def __init__(self, unit: ScheduleUnit):
#         super().__init__(unit)
#         self.schedulers: list[tuple[int, ChainableScheduler]] = []
#
#     def chain(self, scheduler: ChainableScheduler):
#         if len(self.schedulers) == 0:
#             self.schedulers.append((0, scheduler))
#             return
#
#         prev_start, prev_scheduler = self.schedulers[-1]
#         prev_end = prev_start + prev_scheduler.get_duration()
#         self.schedulers.append((prev_end, scheduler))



class LinearScheduler(ChainableScheduler):

    def __init__(
            self,
            unit: ScheduleUnit,
            duration: int,
            start_value: float,
            end_value: float
    ):
        super().__init__(unit=unit)

        self.duration = duration
        self.start_value = start_value
        self.final_value = end_value

    def get_duration(self) -> int:
        return self.duration

    def schedule(
            self,
            progress: float,
            old_value: float,
            state: dict[str, Any],
            metrics: dict[str, Any],
    ) -> ScheduleResult:
        if self.duration <= 0:
            new_value = self.final_value
        else:
            progress = min(progress, 1.0)
            new_value = self.start_value + (self.final_value - self.start_value) * progress

        if abs(new_value - old_value) < 1e-6:
            return {"new_value": None, "event": "hold"}
        return {"new_value": new_value, "event": "linear"}
