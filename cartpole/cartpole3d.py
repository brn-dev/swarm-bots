from typing import Literal, SupportsFloat, Any, Callable

import gymnasium as gym
import PIL.Image
import numpy as np
from dm_control import mjcf


class CartPole3D(gym.Env):

    def __init__(
            self,
            movement_type: Literal['1d', '2d'],
            reset_randomization_magnitude=0.1,
            slide_range=0.8,
            hinge_range=0.8,
            time_limit=10.0,
            step_reward_function: Callable[[float, np.ndarray], float] = lambda time, state: 1.0,
            out_ouf_range_reward=-10,
            time_limit_reward=1000,
            render_mode='human',
            render_width=640,
            render_height=480,
    ):
        self.movement_type = movement_type.lower()

        self.reset_randomization_magnitude = reset_randomization_magnitude

        self.slide_range = slide_range
        self.hinge_range = hinge_range
        self.time_limit = time_limit

        self.step_reward_function = step_reward_function
        self.time_limit_reward = time_limit_reward
        self.out_ouf_range_reward = out_ouf_range_reward

        self.render_mode = render_mode
        self.render_width = render_width
        self.render_height = render_height

        self.physics = self.create_physics()

    def step(self, action: np.ndarray, nstep: int = 1) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        self.physics.set_control(action)
        self.physics.step(nstep=nstep)

        observations = self.get_observations()
        time = self.get_time()

        reward = self.step_reward_function(time, observations)
        terminated, truncated, info = False, False, dict()

        slide_pos, hinge_pos = np.split(self.physics.data.qpos, 2)

        if np.any(np.abs(slide_pos) > self.hinge_range):
            reward = self.out_ouf_range_reward
            terminated = True
            info['termination_reason'] = 'slide_out_of_range'

        if np.any(np.abs(hinge_pos) > self.slide_range):
            reward = self.out_ouf_range_reward
            terminated = True
            info['termination_reason'] = 'hinge_out_of_range'

        if time > self.time_limit:
            reward = self.time_limit_reward
            terminated, truncated = True, True
            info['termination_reason'] = 'time_limit_reached'

        return observations, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == 'human':
            return PIL.Image.fromarray(
                self.physics.render(width=self.render_width, height=self.render_height, camera_id=-1)
            )
        if self.render_mode == 'numpy':
            return self.physics.render(width=self.render_width, height=self.render_height, camera_id=-1)
        if self.render_mode is None:
            return None
        raise ValueError(f'Unknown render mode "{self.render_mode}"')

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.physics.reset()

        qpos = self.physics.data.qpos

        qpos[:int(len(qpos)/2)] = self.np_random.uniform(-self.slide_range, self.slide_range, int(qpos.size/2))
        qpos[int(len(qpos)/2):] = self.np_random.uniform(-self.hinge_range, self.hinge_range, int(qpos.size/2))

        self.physics.data.qpos *= self.reset_randomization_magnitude

        return self.get_observations(), dict()

    def get_observations(self):
        return np.concatenate([self.physics.data.qpos, self.physics.data.qvel])

    def get_time(self):
        return self.physics.time()

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
        if self.movement_type == '2d':
            appendage.add('joint', type='hinge', axis=[1, 0, 0], range=[-np.pi / 3, np.pi / 3])

        slide1 = base.add('joint', type='slide', axis=[1, 0, 0], name='s1')
        cart.actuator.add('motor', joint=slide1)
        if self.movement_type == '2d':
            slide2 = base.add('joint', type='slide', axis=[0, 1, 0], name='s2')
            cart.actuator.add('motor', joint=slide2)

        spawn_site = env.worldbody.add('site', pos=[0, 0, 0.25])
        spawn_site.attach(cart)

        return mjcf.Physics.from_mjcf_model(env)
