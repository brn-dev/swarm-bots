from typing import Literal, SupportsFloat, Any, Callable

import gymnasium as gym
import PIL.Image
import numpy as np
from dm_control import mjcf


StepRewardFunction = Callable[[float, np.ndarray, np.ndarray, np.ndarray], float]
RewardFunction = Callable[[float, np.ndarray, np.ndarray], float]


class MultiAgentCartPole3D(gym.Env):

    def __init__(
            self,
            nr_carts=2,
            cart_size=0.25,
            force_magnitude=5000,
            physics_steps_per_step=1,
            reset_position_radius=1,
            reset_randomize_position_angle_offset=True,
            reset_position_randomization_magnitude=0.1,
            reset_hinge_randomization_magnitude=0.1,
            slide_range_enforcement: Literal['termination', 'walls'] = 'walls',
            slide_range=0.8,
            hinge_range=0.8,
            time_limit=10.0,
            step_reward_function: StepRewardFunction = lambda time, action, state, previous_state: 1.0,
            out_ouf_range_reward_function: RewardFunction = lambda time, action, state: 1.0,
            time_limit_reward_function: RewardFunction = lambda time, action, state: 100.0,
            render_mode='human',
            render_width=640,
            render_height=480,
    ):
        self.nr_carts = nr_carts

        self.cart_size = cart_size
        self.force_magnitude = force_magnitude
        self.physics_steps_per_step = physics_steps_per_step

        self.reset_position_radius = reset_position_radius
        self.reset_randomize_position_angle_offset = reset_randomize_position_angle_offset
        self.reset_position_randomization_magnitude = reset_position_randomization_magnitude
        self.reset_hinge_randomization_magnitude = reset_hinge_randomization_magnitude

        self.slide_range_enforcement = slide_range_enforcement
        self.slide_range = slide_range
        self.hinge_range = hinge_range
        self.time_limit = time_limit

        self.step_reward_function = step_reward_function
        self.time_limit_reward_function = time_limit_reward_function
        self.out_ouf_range_reward_function = out_ouf_range_reward_function

        self.render_mode = render_mode
        self.render_width = render_width
        self.render_height = render_height

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.nr_carts, 2))
        obs_range = np.array([
            [self.slide_range] * 2 + [self.hinge_range] * 2,
            [1.0e20] * 2 + [1.0e20] * 2
        ])[np.newaxis, :, :].repeat(self.nr_carts, axis=0)
        self.observation_space = gym.spaces.Box(low=-obs_range, high=obs_range)

        self.physics = self.create_physics()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        previous_observations = self.get_observations()

        self.physics.set_control(action.flatten() * self.force_magnitude)
        self.physics.step(nstep=self.physics_steps_per_step)

        observations = self.get_observations()
        time = self.get_time()

        reward = self.step_reward_function(time, action, observations, previous_observations)
        terminated, truncated, info = False, False, dict()

        slide_pos = observations[:, 0, :2]
        hinge_pos = observations[:, 0, 2:]

        if np.any(np.abs(slide_pos) > self.slide_range):
            reward = self.out_ouf_range_reward_function(time, action, observations)
            terminated = True
            info['termination_reason'] = 'slide_out_of_range'

        if np.any(np.abs(hinge_pos) > self.hinge_range):
            reward = self.out_ouf_range_reward_function(time, action, observations)
            terminated = True
            info['termination_reason'] = 'hinge_out_of_range'

        if time > self.time_limit:
            reward = self.time_limit_reward_function(time, action, observations)
            terminated, truncated = True, True
            info['termination_reason'] = 'time_limit_reached'

        return observations, reward, terminated, truncated, info

    def render(self, camera_id=-1) -> np.ndarray | None:
        if self.render_mode == 'human':
            return PIL.Image.fromarray(
                self.physics.render(width=self.render_width, height=self.render_height, camera_id=camera_id)
            )
        if self.render_mode == 'numpy':
            return self.physics.render(width=self.render_width, height=self.render_height, camera_id=camera_id)
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

        angle_offset = 0.0
        if self.reset_randomize_position_angle_offset:
            angle_offset = self.np_random.uniform(0, 2 * np.pi)

        for i in range(self.nr_carts):
            angle = 2 * np.pi * i / self.nr_carts + angle_offset
            x = np.cos(angle) * self.reset_position_radius + self.np_random.uniform(
                -self.reset_position_randomization_magnitude, self.reset_position_randomization_magnitude)
            y = np.sin(angle) * self.reset_position_radius + self.np_random.uniform(
                -self.reset_position_randomization_magnitude, self.reset_position_randomization_magnitude)
            qpos[4 * i:4 * i + 2] = [x, y]

            qpos[4 * i + 2:4 * i + 4] = self.np_random.uniform(
                -self.reset_hinge_randomization_magnitude, self.reset_hinge_randomization_magnitude, size=2)

        self.physics.step()
        self.physics.data.qvel[:] = 0.0
        self.physics.step()

        return self.get_observations(), dict()

    def get_observations(self):
        return np.concatenate([
            self.physics.data.qpos.reshape((self.nr_carts, -1)),
            self.physics.data.qvel.reshape((self.nr_carts, -1))
        ], axis=-1).reshape((self.nr_carts, 2, -1))

    def get_time(self):
        return self.physics.time()

    def get_timestep(self):
        return self.physics.timestep()

    def get_steps_per_second(self):
        return 1 / self.physics.timestep()

    def create_cart(self):
        cart = mjcf.RootElement()
        cart.compiler.angle = 'radian'

        base = cart.worldbody.add('body')
        base.add('geom', type='box', size=[self.cart_size, self.cart_size, self.cart_size / 5])

        appendage = base.add('body', pos=[0, 0, self.cart_size / 5])
        appendage.add('geom', type='sphere', size=[self.cart_size / 2], pos=[0, 0, 0])
        appendage.add('geom', type='cylinder', fromto=[0, 0, 0, 0, 0, self.cart_size * 2], size=[self.cart_size / 3])

        appendage.add('joint', type='hinge', axis=[0, 1, 0], range=[-np.pi / 2, np.pi / 2])
        appendage.add('joint', type='hinge', axis=[-1, 0, 0], range=[-np.pi / 2, np.pi / 2])

        for i in range(2):
            slide_joint = base.add('joint', type='slide', axis=np.eye(3)[i], name=f'slide_joint_{i}')
            cart.actuator.add('motor', joint=slide_joint)

        return cart

    def create_physics(self):
        env = mjcf.RootElement()
        env.compiler.angle = 'radian'

        getattr(env.visual, 'global').offwidth = self.render_width
        getattr(env.visual, 'global').offheight = self.render_height

        chequered = env.asset.add('texture', type='2d', builtin='checker', width=300,
                                  height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
        grid = env.asset.add('material', name='grid', texture=chequered,
                             texrepeat=[5, 5], reflectance=.2)
        env.worldbody.add('geom', type='plane', size=[2, 2, .1], material=grid)

        for x in [-3, 3]:
            env.worldbody.add('light', pos=[x, 0, 10], dir=[-x/5, 0, -1])

        if self.slide_range_enforcement == 'walls':
            wall_width = 0.1
            wall_height = 0.2
            for i in range(4):
                wall_position = np.array([
                    (self.slide_range + wall_width) * (i % 2),
                    (self.slide_range + wall_width) * ((i + 1) % 2),
                    0.15
                ], dtype=float)
                if i > 1:
                    wall_position[:2] *= -1

                env.worldbody.add(
                    'geom',
                    type='box',
                    size=[self.slide_range + wall_width, wall_width, wall_height],
                    pos=wall_position,
                    euler=[0, 0, i * np.pi / 2]
                )

        carts = env.worldbody.add('body')
        for _ in range(self.nr_carts):
            cart = self.create_cart()
            spawn_site = carts.add('site', pos=[0, 0, self.cart_size])
            spawn_site.attach(cart)

        env.worldbody.add('camera', pos=[0, -4, 3], euler=[np.pi/3.5, 0, 0])
        env.worldbody.add('camera', pos=[0, -4, 3], mode='targetbodycom', target=carts)

        return mjcf.Physics.from_mjcf_model(env)
