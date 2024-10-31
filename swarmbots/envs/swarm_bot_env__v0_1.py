from typing import SupportsFloat, Any

import gymnasium.spaces
import numpy as np
from dm_control import mjcf
from dm_control.rl.control import PhysicsError
from gymnasium import Env
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.envs.registration import EnvSpec

from swarmbots.unit import Unit


class SwarmBotEnv(Env):

    def __init__(
            self,
            physics_steps_per_step: int = 1,
            max_time: float = 10.0,
            action_scale: float = 1.0,
            hip_range: float = np.pi / 3,
            forward_reward_weight: float = 1.0,
            ctrl_cost_weight: float = 0.01,
            physics_error_reward: float = -100.0
    ):
        self.physics_steps_per_step = physics_steps_per_step
        self.max_time = max_time
        self.action_scale = float(action_scale)
        self.hip_range = hip_range

        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.physics_error_reward = physics_error_reward

        self.physics = self.setup_physics()

        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, self.get_obs().shape)
        self.action_space = gymnasium.spaces.Box(-1, 1, self.physics.data.actuator_velocity.shape)

        self.spec = EnvSpec(
            id='SwarmBot-v0.1',
            kwargs={
                'physics_steps_per_step': physics_steps_per_step,
                'max_time': max_time,
                'action_scale': action_scale,
                'hip_range': hip_range,
                'forward_reward_weight': forward_reward_weight,
                'ctrl_cost_weight': ctrl_cost_weight,
                'physics_error_reward': physics_error_reward,
            }
        )

    def setup_physics(self):
        model = mjcf.RootElement()

        chequered = model.asset.add('texture', type='2d', builtin='checker', width=300,
                                    height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
        grid = model.asset.add('material', name='grid', texture=chequered,
                               texrepeat=[5, 5], reflectance=.2)
        model.worldbody.add('geom', type='plane', size=[2, 2, .1], material=grid)

        for x in [-2, 2]:
            model.worldbody.add('light', pos=[x, -1, 3], dir=[-x, 1, -2])

        unit1 = Unit(0.1, 0.2, 0.025, self.hip_range)
        spawn_site = model.worldbody.add('site', pos=[0.305, 0, 0.2], euler=[0, -np.pi / 2, 0])
        spawn_site.attach(unit1.model).add('freejoint')

        unit2 = Unit(0.1, 0.2, 0.025, self.hip_range)
        spawn_site = model.worldbody.add('site', pos=[-0.305, 0, 0.2], euler=[0, np.pi / 2, np.pi])
        spawn_site.attach(unit2.model).add('freejoint')
        model.equality.add(
            'weld',
            body1=f'unnamed_model/unnamed_model/{unit1.name}-leg0-foot',
            body2=f'unnamed_model_1/unnamed_model/{unit2.name}-leg0-foot',
            torquescale=10_000
        )

        cam = model.worldbody.add(
            'camera',
            mode='targetbody',
            target=f'unnamed_model_1/unnamed_model/{unit2.name}-leg0',
            pos=[0, 1.5, 1]
        )

        return mjcf.Physics.from_mjcf_model(model)

    def get_obs(self):
        # return np.concatenate((
        #     self.physics.data.qpos,
        #     self.physics.data.qvel,
        #     self.physics.data.xpos.reshape(-1),
        # )).copy()
        return np.concatenate((
            self.physics.data.qpos,
            self.physics.data.qvel,
        )).copy()

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.physics.reset()
        return self.get_obs(), {}

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        try:
            xpos_before = self.physics.data.xpos.copy()

            self.physics.set_control(np.asarray(action) * self.action_scale)
            self.physics.step(self.physics_steps_per_step)

            forward_reward = self.forward_reward_weight * (
                        self.physics.data.xpos[:, 0].mean() - xpos_before[:, 0].mean())
            ctrl_cost = self.ctrl_cost_weight * -np.sum(np.square(action))
            reward = forward_reward + ctrl_cost

            truncated = self.physics.data.time >= self.max_time

            return self.get_obs(), reward, False, truncated, {}
        except PhysicsError as pe:
            print(f'Encountered PhysicsError, terminating env: {pe}')

            obs = self.get_obs()

            if np.isnan(obs).any() or np.isinf(obs).any():
                obs = np.zeros_like(obs)

            return obs, self.physics_error_reward, True, False, {'physics_error': pe}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.physics.render(camera_id=0)
