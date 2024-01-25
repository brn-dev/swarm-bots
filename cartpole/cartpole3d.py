from typing import Literal, SupportsFloat, Any, Callable

import gymnasium as gym
import PIL.Image
import numpy as np
from dm_control import mjcf


RewardFunction = Callable[[float, np.ndarray, np.ndarray], float]


class CartPole3D(gym.Env):

    def __init__(
            self,
            nr_movement_dimensions: Literal[1, 2, 3],
            force_magnitude=5000,
            physics_steps_per_step=1,
            reset_randomization_magnitude=0.1,
            slide_range=0.8,
            hinge_range=0.8,
            time_limit=10.0,
            step_reward_function: RewardFunction = lambda time, action, state: 1.0,
            out_ouf_range_reward_function: RewardFunction = lambda time, action, state: 1.0,
            time_limit_reward_function: RewardFunction = lambda time, action, state: 1.01000,
            render_mode='human',
            render_width=640,
            render_height=480,
    ):
        self.nr_movement_dimensions = nr_movement_dimensions
        self.nr_topple_dimensions = min(nr_movement_dimensions, 2)

        self.force_magnitude = force_magnitude
        self.physics_steps_per_step = physics_steps_per_step

        self.reset_randomization_magnitude = reset_randomization_magnitude

        self.slide_range = slide_range
        self.hinge_range = hinge_range
        self.time_limit = time_limit

        self.step_reward_function = step_reward_function
        self.time_limit_reward_function = time_limit_reward_function
        self.out_ouf_range_reward_function = out_ouf_range_reward_function

        self.render_mode = render_mode
        self.render_width = render_width
        self.render_height = render_height

        self.physics = self.create_physics()

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.nr_movement_dimensions,))
        obs_range = np.array(
            [self.slide_range] * self.nr_movement_dimensions
            + [self.hinge_range] * self.nr_topple_dimensions
            + [1.0e20] * self.nr_movement_dimensions
            + [1.0e20] * self.nr_topple_dimensions
        )
        self.observation_space = gym.spaces.Box(low=-obs_range, high=obs_range)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        self.physics.set_control(action * self.force_magnitude)
        self.physics.step(nstep=self.physics_steps_per_step)

        observations = self.get_observations()
        time = self.get_time()

        reward = self.step_reward_function(time, action, observations)
        terminated, truncated, info = False, False, dict()

        slide_pos, hinge_pos = np.split(self.physics.data.qpos, [self.nr_movement_dimensions])

        if np.any(np.abs(slide_pos) > self.hinge_range):
            reward = self.out_ouf_range_reward_function(time, action, observations)
            terminated = True
            info['termination_reason'] = 'slide_out_of_range'

        if np.any(np.abs(hinge_pos) > self.slide_range):
            reward = self.out_ouf_range_reward_function(time, action, observations)
            terminated = True
            info['termination_reason'] = 'hinge_out_of_range'

        if time > self.time_limit:
            reward = self.time_limit_reward_function(time, action, observations)
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

        qpos[:self.nr_movement_dimensions] = \
            self.np_random.uniform(-self.slide_range, self.slide_range, self.nr_movement_dimensions)
        qpos[self.nr_movement_dimensions:] = \
            self.np_random.uniform(-self.hinge_range, self.hinge_range, qpos.size - self.nr_movement_dimensions)

        self.physics.data.qpos *= self.reset_randomization_magnitude

        return self.get_observations(), dict()

    def get_observations(self):
        return np.concatenate([self.physics.data.qpos, self.physics.data.qvel])

    def get_time(self):
        return self.physics.time()

    def get_timesteps_per_second(self):
        return self.physics.timestep()

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

        appendage.add('joint', type='hinge', axis=[0, 1, 0], range=[-np.pi / 2, np.pi / 2])
        if self.nr_movement_dimensions >= 2:
            appendage.add('joint', type='hinge', axis=[1, 0, 0], range=[-np.pi / 2, np.pi / 2])

        for i in range(self.nr_movement_dimensions):
            slide_joint = base.add('joint', type='slide', axis=np.eye(3)[i], name=f's{i}')
            cart.actuator.add('motor', joint=slide_joint)

        spawn_site = env.worldbody.add('site', pos=[0, 0, 0.25])
        spawn_site.attach(cart)

        return mjcf.Physics.from_mjcf_model(env)
