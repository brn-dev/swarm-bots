from pathlib import Path

import mujoco
from PIL import Image

from swarmbots.mj_env.scenarios.wall_scenario import WallScenario
from swarmbots.mj_env.scenarios.scenario_presets import DEFAULT_KWARGS, WALL_SCENARIO_KWARGS, default_wall
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv
from swarmbots.scenario_presets.scenario_presets_kwargs import make_scenario_kwargs


class RewardIndicatorWallScenario(WallScenario):
    def _create_scenario_spec(self) -> mujoco.MjSpec:
        spec = super()._create_scenario_spec()
        self._add_reward_indicator_geoms(spec)
        return spec

    def _add_reward_indicator_geoms(self, spec: mujoco.MjSpec) -> None:
        self._add_visual_box(
            spec,
            name="RewardForwardRegion",
            size=[self.side_wall_x, 0.5, 0.010],
            rgba=[0.0, 0.85, 0.20, 0.2],
        )

        for threshold_idx in range(int(self.wall_pass_thresholds.size)):
            self._add_visual_box(
                spec,
                name=f"RewardWallPassThreshold_{threshold_idx}",
                size=[self.side_wall_x, 0.016, 0.25],
                rgba=[0.35, 0.0, 0.45, 1.0],
            )

    @staticmethod
    def _add_visual_box(spec: mujoco.MjSpec, *, name: str, size: list[float], rgba: list[float]) -> None:
        body = spec.worldbody.add_body(name=name, mocap=True, pos=[0.0, 0.0, 0.0])
        geom = body.add_geom(
            name=f"{name}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            rgba=rgba,
        )
        geom.contype = 0
        geom.conaffinity = 0

    def reset_wall(
        self,
        data: mujoco.MjData,
        model: mujoco.MjModel,
        hidden_global_vars: list[float],
    ) -> float:
        wall_y = super().reset_wall(data, model, hidden_global_vars)
        self._position_reward_indicators(data, model, wall_y)
        return wall_y

    def _position_reward_indicators(
        self,
        data: mujoco.MjData,
        model: mujoco.MjModel,
        wall_y: float,
    ) -> None:
        forward_reward_cap_y = wall_y + self.wall_success_threshold
        self._set_visual_box_y_size(model, "RewardForwardRegion", forward_reward_cap_y / 2.0)
        self._set_mocap_pos(model, data, "RewardForwardRegion", [0.0, forward_reward_cap_y / 2.0, 0.018])

        for threshold_idx, threshold_y in enumerate(self._compute_wall_pass_thresholds(wall_y)):
            self._set_mocap_pos(
                model,
                data,
                f"RewardWallPassThreshold_{threshold_idx}",
                [0.0, float(threshold_y), 0.180],
            )

    @staticmethod
    def _set_mocap_pos(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, pos: list[float]) -> None:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id >= 0:
            data.mocap_pos[mocap_id] = pos

    @staticmethod
    def _set_visual_box_y_size(model: mujoco.MjModel, body_name: str, y_size: float) -> None:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{body_name}_geom")
        if geom_id >= 0:
            model.geom_size[geom_id, 1] = y_size


output_path = Path("wall_scenario_reward_indicators.png")
output_path.parent.mkdir(parents=True, exist_ok=True)

base = default_wall(seed=42)
scenario_kwargs = make_scenario_kwargs(DEFAULT_KWARGS, WALL_SCENARIO_KWARGS)
scenario = RewardIndicatorWallScenario(swarm=base.swarm, seed=42, **scenario_kwargs)

env = SwarmBotsEnv(
    scenario=scenario,
    episode_length=512,
    render_mode="rgb_array",
    width=640,
    height=480,
    camera=0,
)

try:
    env.reset(seed=47)
    frame = env.render()
    if frame is None:
        raise RuntimeError("render() returned None")
    Image.fromarray(frame).save(output_path)
finally:
    env.close()

print(output_path)
