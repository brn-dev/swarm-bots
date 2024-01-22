from typing import Literal, SupportsFloat, Any

import gymnasium as gym
import numpy as np
from dm_control import mjcf


class CartPole3D(gym.Env):

    def __init__(
            self,
            movement_type: Literal['1d', '2d'],
            render_mode='human',
            render_width=640,
            render_height=480
    ):
        self.movement_type = movement_type

        self.render_mode = render_mode
        self.render_width = render_width
        self.render_height = render_height

        self.physics = self.create_physics()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        self.physics.set_control(action)
        self.physics.step()

        observations = self.get_observations()
        reward, terminated, truncated = self.calc_reward_terminated_and_truncated()

        return observations, reward, terminated, truncated, dict()

    def render(self) -> np.ndarray | None:
        if self.render_mode == 'human':
            return self.physics.render(width=self.render_width, height=self.render_height, camera_id=-1)
        if self.render_mode is None:
            return None
        raise ValueError(f'Unknown render mode "{self.render_mode}"')

    def reset(
            self,
            randomize_initial_position=False,
            seed: int | None = None,
            options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.physics.reset()

        if randomize_initial_position:
            self.physics.data.qpos = (np.random.random(self.physics.data.qpos.size) - 0.5) / 10

        return self.get_observations(), dict()

    def get_observations(self):
        return np.concatenate([self.physics.data.qpos, self.physics.data.qvel])

    def calc_reward_terminated_and_truncated(self):
        if np.any(np.abs(self.physics.data.qpos) > 0.8):
            return -10, True, False

        if self.physics.data.time > 10:
            return 1000, True, True

        return 1, False, False

    def create_physics(self):
        env = mjcf.RootElement()

        getattr(env.visual, 'global').offwidth = self.render_width
        getattr(env.visual, 'global').offheight = self.render_height

        chequered = env.asset.add('texture', type='2d', builtin='checker', width=300,
                                  height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
        grid = env.asset.add('material', name='grid', texture=chequered,
                             texrepeat=[5, 5], reflectance=.2)
        env.worldbody.add('geom', type='plane', size=[2, 2, .1], material=grid)

        for x in [-2, 2]:
            env.worldbody.add('light', pos=[x, -1, 3], dir=[-x, 1, -2])

        cart = mjcf.RootElement()
        cart.compiler.angle = 'radian'

        base = cart.worldbody.add('body')
        base.add('geom', type='box', size=[0.25, 0.25, 0.05])

        appendage = base.add('body', pos=[0, 0, 0.0])
        appendage.add('geom', type='cylinder', fromto=[0, 0, 0, 0, 0, 0.5], size=[0.1])
        appendage.add('geom', type='cylinder', fromto=[-0.25, 0, 0.5, 0.25, 0, 0.5], size=[0.02])
        appendage.add('geom', type='cylinder', fromto=[0, -0.25, 0.5, 0, 0.25, 0.5], size=[0.02])

        appendage.add('joint', type='hinge', axis=[0, 1, 0], range=[-np.pi / 3, np.pi / 3])
        slide1 = base.add('joint', type='slide', axis=[1, 0, 0], name='s1')
        cart.actuator.add('motor', joint=slide1)

        if self.movement_type == '2d':
            appendage.add('joint', type='hinge', axis=[1, 0, 0], range=[-np.pi / 3, np.pi / 3])
            slide2 = base.add('joint', type='slide', axis=[0, 1, 0], name='s2')
            cart.actuator.add('motor', joint=slide2)

        spawn_site = env.worldbody.add('site', pos=[0, 0, 0.25])
        spawn_site.attach(cart)

        return mjcf.Physics.from_mjcf_model(env)
