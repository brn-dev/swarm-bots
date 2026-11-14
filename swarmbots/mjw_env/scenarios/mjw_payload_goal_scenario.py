from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from swarmbots.mjw_env.scenarios.base_mjw_scenario import MJWRecordingCameraConfig, MJWRuntimeBindings
from swarmbots.mjw_env.scenarios.mjw_multi_payload_goal_scenario import MJWMultiPayloadGoalScenario
from swarmbots.mjw_env.scenarios.mjw_payload_goal_sampling import PayloadGoalLayout
from swarmbots.mjw_env.scenarios.scenario_validation import finite_positive


@dataclass
class MJWPayloadGoalScenario(MJWMultiPayloadGoalScenario):
    goal_rect: tuple[float, float, float, float] | None = field(default=None, kw_only=True)
    goal_arc_radius: float = 3.0
    goal_arc_degrees: float = 120.0
    goal_noise_std: float = 0.15
    towards_payload_reward_weight: float = 1.0
    towards_payload_goal_radius: float = 1.25

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_payloads != 1:
            raise ValueError("PayloadGoal requires exactly one payload slot")
        self.towards_payload_reward_weight = float(self.towards_payload_reward_weight)
        self.towards_payload_goal_radius = float(self.towards_payload_goal_radius)
        if not math.isfinite(self.towards_payload_goal_radius) or self.towards_payload_goal_radius < 0.0:
            raise ValueError("towards_payload_goal_radius must be finite and non-negative")

    def _make_goal_sampling_layout(self, max_active_payloads: int) -> PayloadGoalLayout | None:
        if self.goal_rect is not None:
            raise ValueError("PayloadGoal uses an arc; configure goal_arc_radius, goal_arc_degrees and goal_noise_std")
        self.goal_arc_radius = finite_positive(self.goal_arc_radius, name="goal_arc_radius")
        self.goal_arc_degrees = finite_positive(self.goal_arc_degrees, name="goal_arc_degrees")
        if self.goal_arc_degrees > 360.0:
            raise ValueError("goal_arc_degrees must not exceed 360")
        self.goal_noise_std = float(self.goal_noise_std)
        if not math.isfinite(self.goal_noise_std) or self.goal_noise_std < 0.0:
            raise ValueError("goal_noise_std must be finite and non-negative")
        return None

    def _goal_sampling_settings(self) -> dict[str, Any]:
        return {
            "algorithm": "payload_relative_arc_gaussian_v1",
            "origin": "payload_spawn_xy_before_settling",
            "forward_direction": (0.0, 1.0),
            "radius": self.goal_arc_radius,
            "arc_degrees": self.goal_arc_degrees,
            "noise_std_xy": self.goal_noise_std,
        }

    def get_settings(self) -> dict[str, Any]:
        settings = super().get_settings()
        settings.update(
            scenario_type="payload_goal",
            towards_payload_reward_weight=self.towards_payload_reward_weight,
            towards_payload_goal_radius=self.towards_payload_goal_radius,
        )
        return settings

    def get_default_recording_camera_config(self) -> MJWRecordingCameraConfig:
        return MJWRecordingCameraConfig(
            lookat=(0.0, self.payload_spawn_y + self.goal_arc_radius / 2.0, max(0.5, self.payload_radius * 4.0)),
            distance=max(8.0, min(16.0, self.goal_arc_radius * 2.0 + self.swarm.max_unit_extent * 6.0)),
            azimuth=180.0,
            elevation=-35.0,
        )

    def create_runtime(self, *, bindings: MJWRuntimeBindings, runtime_metadata: Any) -> Any:
        from swarmbots.mjw_env.scenarios.mjw_payload_goal_runtime import PayloadGoalMJWScenarioRuntime

        return PayloadGoalMJWScenarioRuntime(scenario=self, bindings=bindings, runtime_metadata=runtime_metadata)
