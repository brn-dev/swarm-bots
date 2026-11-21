from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import mujoco
from gymnasium import spaces

from swarmbots.mjw_env.scenarios.base_mjw_scenario import MJWRecordingCameraConfig, MJWRuntimeBindings
from swarmbots.mjw_env.scenarios.mjw_payload_goal_sampling import PayloadGoalLayout
from swarmbots.mjw_env.scenarios.mjw_payload_goal_scenario import MJWPayloadGoalScenario
from swarmbots.mjw_env.scenarios.scenario_spaces import make_observation_space
from swarmbots.scenario_presets.scenario_obs_layouts import PAYLOAD_GOAL_SEQUENCE_GLOBAL_OBS_LAYOUT


@dataclass
class MJWPayloadGoalSequenceScenario(MJWPayloadGoalScenario):
    goal_arc_radius: float = field(default=3., init=False, repr=False)
    goal_arc_degrees: float = field(default=120., init=False, repr=False)
    goal_noise_std: float = field(default=.15, init=False, repr=False)
    goal_distance_ranges: tuple[tuple[float, float], ...] = ((.9, 1.1), (1., 2.), (1., 2.))
    goal_turn_arc_degrees: tuple[float, ...] = (10., 90., 180.)
    goal_hold_time: float = .5
    waypoint_reward: float = 2.
    randomize_initial_swarm_z_rotation: bool = True
    include_goal_delta_obs: bool = False
    include_payload_velocity_obs: bool = False
    include_goal_inside_obs: bool = False
    goal_hold_reward_fraction: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.goal_hold_time = float(self.goal_hold_time)
        if not math.isfinite(self.goal_hold_time) or self.goal_hold_time < 0.:
            raise ValueError("goal_hold_time must be finite and non-negative")
        self.waypoint_reward = float(self.waypoint_reward)
        if not math.isfinite(self.waypoint_reward) or self.waypoint_reward < 0.:
            raise ValueError("waypoint_reward must be finite and non-negative")
        self.goal_hold_reward_fraction = float(self.goal_hold_reward_fraction)
        if not math.isfinite(self.goal_hold_reward_fraction) or not 0. <= self.goal_hold_reward_fraction <= 1.:
            raise ValueError("goal_hold_reward_fraction must be finite and between 0 and 1")

    @property
    def num_goals(self) -> int:
        return len(self.goal_distance_ranges)

    @property
    def goal_hold_steps(self) -> int:
        return max(0, math.ceil(self.goal_hold_time / (self.timestep * self.action_repeat)))

    @property
    def global_obs_dim(self) -> int:
        return (13 + 3 * self.num_goals + 2 * (self.include_goal_delta_obs + self.include_payload_velocity_obs)
                + self.include_goal_inside_obs)

    def _make_goal_sampling_layout(self, max_active_payloads: int) -> PayloadGoalLayout | None:
        if self.goal_rect is not None:
            raise ValueError("PayloadGoalSequence uses goal_distance_ranges and goal_turn_arc_degrees")
        self.goal_distance_ranges = tuple(tuple(float(v) for v in bounds) for bounds in self.goal_distance_ranges)
        self.goal_turn_arc_degrees = tuple(float(v) for v in self.goal_turn_arc_degrees)
        if not self.num_goals or len(self.goal_turn_arc_degrees) != self.num_goals:
            raise ValueError("Provide one turn arc for each non-empty goal distance range")
        for bounds in self.goal_distance_ranges:
            if len(bounds) != 2:
                raise ValueError("Each goal distance range must contain a minimum and maximum")
            low, high = bounds
            if not (math.isfinite(low) and math.isfinite(high) and 2 * self.goal_radius < low <= high):
                raise ValueError("Goal distances must be finite, ordered, and exceed twice goal_radius")
        if any(not math.isfinite(v) or not 0. <= v <= 360. for v in self.goal_turn_arc_degrees):
            raise ValueError("Goal turn arcs must be finite and between 0 and 360 degrees")
        return None

    def _goal_sampling_settings(self) -> dict[str, Any]:
        return {
            "algorithm": "ordered_relative_turns_v1",
            "origin": "payload_spawn_xy_before_settling",
            "distance_ranges": self.goal_distance_ranges,
            "turn_arc_degrees": self.goal_turn_arc_degrees,
            "turn_reference": "previous_segment_heading",
            "arrangement_rotation": "shared_yaw_about_swarm_start",
            "yaw_range": (0., 2. * math.pi) if self.randomize_initial_swarm_z_rotation else (0., 0.),
        }

    def get_settings(self) -> dict[str, Any]:
        settings = super().get_settings()
        settings.update(
            scenario_type="payload_goal_sequence",
            global_obs_layout=PAYLOAD_GOAL_SEQUENCE_GLOBAL_OBS_LAYOUT,
            num_goals=self.num_goals,
            goal_distance_ranges=self.goal_distance_ranges,
            goal_turn_arc_degrees=self.goal_turn_arc_degrees,
            goal_hold_time=self.goal_hold_time,
            goal_hold_steps=self.goal_hold_steps,
            goal_hold_samples=self.goal_hold_steps + 1,
            effective_goal_hold_time=self.goal_hold_steps * self.timestep * self.action_repeat,
            waypoint_reward=self.waypoint_reward,
            include_goal_delta_obs=self.include_goal_delta_obs,
            include_payload_velocity_obs=self.include_payload_velocity_obs,
            include_goal_inside_obs=self.include_goal_inside_obs,
            goal_hold_reward_fraction=self.goal_hold_reward_fraction,
            terminal_info_keys=("goals_completed", *(f"goal_{i + 1}_reached" for i in range(self.num_goals))),
        )
        return settings

    def _add_goal_markers(self, *, worldbody: mujoco.MjsBody, payload_idx: int) -> None:
        for goal_idx in range(self.num_goals):
            super()._add_goal_markers(worldbody=worldbody, payload_idx=goal_idx)

    def get_single_observation_space(self) -> spaces.Dict:
        return make_observation_space(self, global_obs_dim=self.global_obs_dim)

    def get_default_recording_camera_config(self) -> MJWRecordingCameraConfig:
        route_extent = self.payload_spawn_y + sum(high for _, high in self.goal_distance_ranges)
        return MJWRecordingCameraConfig(
            lookat=(0., 0., max(.5, self.payload_radius * 4.)),
            distance=max(8., 3.5 * route_extent), azimuth=180., elevation=-75.,
        )

    def create_runtime(self, *, bindings: MJWRuntimeBindings, runtime_metadata: Any) -> Any:
        from swarmbots.mjw_env.scenarios.mjw_payload_goal_sequence_runtime import PayloadGoalSequenceMJWScenarioRuntime

        return PayloadGoalSequenceMJWScenarioRuntime(scenario=self, bindings=bindings, runtime_metadata=runtime_metadata)
