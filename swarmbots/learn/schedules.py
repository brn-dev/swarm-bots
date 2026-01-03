import abc
import enum
import math


class ScheduleUnit(enum.Enum):
    ITERATIONS = 'ITERATIONS'
    TIMESTEPS = 'TIMESTEPS'
    UPDATES = 'UPDATES'

class Schedule(abc.ABC):

    def __call__(
            self,
            iterations: int,
            timesteps: int,
            updates: int,
            score: float
    ) -> float:
        return self.get_value(iterations, timesteps, updates, score)

    @abc.abstractmethod
    def get_value(
            self,
            iterations: int,
            timesteps: int,
            updates: int,
            score: float
    ) -> float:
        raise NotImplementedError()

    def _get_x(
            self,
            iterations: int,
            timesteps: int,
            updates: int,
            unit: ScheduleUnit
    ) -> int:
        if unit == ScheduleUnit.ITERATIONS:
            return iterations
        if unit == ScheduleUnit.TIMESTEPS:
            return timesteps
        if unit == ScheduleUnit.UPDATES:
            return updates
        raise ValueError(unit)


class ExponentialSchedule(Schedule):

    def __init__(
            self,
            initial_value: float,
            base: float,
            unit: ScheduleUnit,
            x_start: float = 0.0,
            x_scale: float = 1.0,
    ) -> None:
        if x_scale == 0:
            raise ValueError('x_scale must be non-zero')
        if base <= 0:
            raise ValueError('base must be > 0')

        self.initial_value = initial_value
        self.base = base
        self.unit = unit
        self.start = x_start
        self.scale = x_scale

    def get_value(
            self,
            iterations: int,
            timesteps: int,
            updates: int,
            score: float
    ) -> float:
        x = self._get_x(iterations, timesteps, updates, self.unit)
        return self.initial_value * self.base ** ((x - self.start) / self.scale)


class LinearSchedule(Schedule):

    def __init__(
            self,
            initial_value: float,
            final_value: float,
            x_start: float,
            x_end: float,
            unit: ScheduleUnit,
            clamp: bool = True,
    ) -> None:
        if x_end <= x_start:
            raise ValueError('x_end must be greater than x_start')

        self.initial_value = initial_value
        self.final_value = final_value
        self.unit = unit
        self.start = x_start
        self.end = x_end
        self.scale = x_end - x_start
        self.clamp = clamp

    def get_value(
            self,
            iterations: int,
            timesteps: int,
            updates: int,
            score: float
    ) -> float:
        x = self._get_x(iterations, timesteps, updates, self.unit)
        progress = (x - self.start) / self.scale
        if self.clamp:
            progress = max(0.0, min(1.0, float(progress)))
        return self.initial_value + (self.final_value - self.initial_value) * float(progress)


class CosineAnnealingSchedule(Schedule):

    def __init__(
            self,
            initial_value: float,
            final_value: float,
            x_start: float,
            x_end: float,
            unit: ScheduleUnit,
            clamp: bool = True,
    ) -> None:
        if x_end <= x_start:
            raise ValueError('x_end must be greater than x_start')

        self.initial_value = initial_value
        self.final_value = final_value
        self.unit = unit
        self.start = x_start
        self.end = x_end
        self.scale = x_end - x_start
        self.clamp = clamp

    def get_value(
            self,
            iterations: int,
            timesteps: int,
            updates: int,
            score: float
    ) -> float:
        x = self._get_x(iterations, timesteps, updates, self.unit)
        progress = (x - self.start) / self.scale
        if self.clamp:
            progress = max(0.0, min(1.0, float(progress)))

        cosine = 0.5 * (1.0 + math.cos(math.pi * float(progress)))
        return self.final_value + (self.initial_value - self.final_value) * cosine


class StepwiseSchedule(Schedule):

    def __init__(
            self,
            initial_value: float,
            milestones: list[float],
            values: list[float],
            unit: ScheduleUnit
    ) -> None:
        if not milestones:
            raise ValueError('milestones must not be empty')
        if len(values) != len(milestones):
            raise ValueError('values must have the same length as milestones')
        for previous, current in zip(milestones[:-1], milestones[1:]):
            if current < previous:
                raise ValueError('milestones must be sorted in non-decreasing order')

        self.initial_value = initial_value
        self.milestones = milestones
        self.values = values
        self.unit = unit

    def get_value(
            self,
            iterations: int,
            timesteps: int,
            updates: int,
            score: float
    ) -> float:
        x = self._get_x(iterations, timesteps, updates, self.unit)

        if x < self.milestones[0]:
            return self.initial_value

        i = 0
        while i < len(self.milestones) - 1 and x >= self.milestones[i + 1]:
            i += 1

        return self.values[i]


