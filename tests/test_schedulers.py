import math
import unittest

from swarmbots.learn.scheduling.cosine_scheduler import CosineScheduler, CosineSchedulerConfig
from swarmbots.learn.scheduling.exponential_scheduler import ExponentialScheduler, ExponentialSchedulerConfig
from swarmbots.learn.scheduling.linear_scheduler import LinearScheduler, LinearSchedulerConfig
from swarmbots.learn.scheduling.scheduler_factory import make_scheduler
from swarmbots.learn.scheduling.schedulers import (
    ScheduledHyperParameter,
    SchedulerManager,
    ScheduleUnit,
)


class SchedulerTests(unittest.TestCase):
    def test_linear_scheduler_uses_selected_unit_and_holds_when_value_is_unchanged(self) -> None:
        scheduler = LinearScheduler(
            unit=ScheduleUnit.TIMESTEPS,
            duration=100,
            start_value=1.0,
            final_value=0.0,
            name="lr",
        )

        result = scheduler(
            old_value=0.75,
            state={},
            n_iterations=999,
            n_model_updates=999,
            n_timesteps=25,
            metrics={},
        )
        hold_result = scheduler(
            old_value=0.75,
            state={},
            n_iterations=0,
            n_model_updates=0,
            n_timesteps=25,
            metrics={},
        )

        self.assertEqual(result, {"new_value": None, "event": "lr-hold"})
        self.assertEqual(hold_result, {"new_value": None, "event": "lr-hold"})

    def test_chainable_schedulers_saturate_after_duration(self) -> None:
        schedulers = [
            LinearScheduler(
                unit=ScheduleUnit.ITERATIONS,
                duration=10,
                start_value=1.0,
                final_value=0.25,
            ),
            ExponentialScheduler(
                unit=ScheduleUnit.ITERATIONS,
                duration=10,
                start_value=1.0,
                final_value=0.25,
                base=2.0,
            ),
            CosineScheduler(
                unit=ScheduleUnit.ITERATIONS,
                duration=10,
                start_value=1.0,
                final_value=0.25,
            ),
        ]

        for scheduler in schedulers:
            with self.subTest(scheduler=type(scheduler).__name__):
                result = scheduler(
                    old_value=1.0,
                    state={},
                    n_iterations=100,
                    n_model_updates=0,
                    n_timesteps=0,
                    metrics={},
                )

                self.assertAlmostEqual(result["new_value"], 0.25)

    def test_chainable_schedulers_with_non_positive_duration_jump_to_final_value(self) -> None:
        schedulers = [
            LinearScheduler(
                unit=ScheduleUnit.ITERATIONS,
                duration=0,
                start_value=1.0,
                final_value=0.25,
            ),
            ExponentialScheduler(
                unit=ScheduleUnit.ITERATIONS,
                duration=-1,
                start_value=1.0,
                final_value=0.25,
            ),
            CosineScheduler(
                unit=ScheduleUnit.ITERATIONS,
                duration=0,
                start_value=1.0,
                final_value=0.25,
            ),
        ]

        for scheduler in schedulers:
            with self.subTest(scheduler=type(scheduler).__name__):
                result = scheduler(
                    old_value=1.0,
                    state={},
                    n_iterations=0,
                    n_model_updates=0,
                    n_timesteps=0,
                    metrics={},
                )

                self.assertAlmostEqual(result["new_value"], 0.25)

    def test_scheduler_factory_builds_all_supported_schedulers_and_rejects_unknown_config(self) -> None:
        linear = make_scheduler(LinearSchedulerConfig(
            unit=ScheduleUnit.ITERATIONS,
            duration=10,
            start_value=0.0,
            final_value=1.0,
        ))
        exponential = make_scheduler(ExponentialSchedulerConfig(
            unit=ScheduleUnit.MODEL_UPDATES,
            duration=10,
            start_value=1.0,
            final_value=0.0,
            base=2.0,
        ))
        cosine = make_scheduler(CosineSchedulerConfig(
            unit=ScheduleUnit.TIMESTEPS,
            duration=10,
            start_value=1.0,
            final_value=0.0,
            bias=1.0,
            sharpness=1.0,
        ))

        self.assertEqual(linear.get_duration(), 10)
        self.assertAlmostEqual(
            exponential(
                old_value=-1.0,
                state={},
                n_iterations=0,
                n_model_updates=5,
                n_timesteps=0,
                metrics={},
            )["new_value"],
            1.0 - (math.sqrt(2.0) - 1.0),
        )
        self.assertAlmostEqual(
            cosine(
                old_value=-1.0,
                state={},
                n_iterations=0,
                n_model_updates=0,
                n_timesteps=5,
                metrics={},
            )["new_value"],
            0.5,
        )

        with self.assertRaisesRegex(TypeError, "Unsupported scheduler config"):
            make_scheduler(object())

    def test_scheduler_manager_applies_requested_value_and_reports_noops(self) -> None:
        current = {"value": 1.0}
        calls: list[tuple[float, int, dict[str, float]]] = []

        def scheduler(old_value, state, n_iterations, n_model_updates, n_timesteps, metrics):
            calls.append((old_value, n_timesteps, metrics))
            state["last_old"] = old_value
            return {"new_value": old_value * 0.5, "event": "decay", "msg": "half"}

        manager = SchedulerManager([
            ScheduledHyperParameter(
                name="entropy",
                scheduler=scheduler,
                get_value=lambda: current["value"],
                apply=lambda new_value: current.update(value=new_value),
            ),
            ScheduledHyperParameter(
                name="disabled",
                scheduler=lambda **kwargs: {"new_value": 0.0},
                get_value=lambda: 123.0,
                apply=lambda new_value: current.update(disabled=new_value),
                enabled=False,
            ),
        ])

        results = manager.step(
            n_iterations=3,
            n_model_updates=4,
            n_timesteps=5,
            metrics={"loss": 2.0},
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "entropy")
        self.assertEqual(results[0].old_value, 1.0)
        self.assertEqual(results[0].new_value, 0.5)
        self.assertTrue(results[0].updated)
        self.assertEqual(results[0].event, "decay")
        self.assertEqual(results[0].msg, "half")
        self.assertEqual(current, {"value": 0.5})
        self.assertEqual(calls, [(1.0, 5, {"loss": 2.0})])

    def test_scheduler_manager_does_not_apply_when_scheduler_returns_none(self) -> None:
        current = {"value": 1.0}

        manager = SchedulerManager([
            ScheduledHyperParameter(
                name="lr",
                scheduler=lambda **kwargs: {"new_value": None, "event": "hold"},
                get_value=lambda: current["value"],
                apply=lambda new_value: current.update(value=new_value),
            )
        ])

        results = manager.step(n_iterations=0, n_model_updates=0, n_timesteps=0, metrics={})

        self.assertEqual(current["value"], 1.0)
        self.assertFalse(results[0].updated)
        self.assertEqual(results[0].new_value, 1.0)
        self.assertEqual(results[0].event, "hold")

    def test_schedule_unit_rejects_invalid_unit(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid unit"):
            ScheduleUnit.get("iterations", n_iterations=1, n_model_updates=2, n_timesteps=3)


if __name__ == "__main__":
    unittest.main()
