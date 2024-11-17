from typing import SupportsFloat, Any, Callable, NamedTuple

import gymnasium.spaces
import numpy as np
from dm_control import mjcf
from dm_control.rl.control import PhysicsError
from gymnasium import Env
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.envs.registration import EnvSpec

from swarmbots.swarm_model import SwarmModel, Connection, ConnectionSite
from swarmbots.unit_model import UnitModel


UnitModelProvider = Callable[[], UnitModel]


class SwarmBotEnv(Env):

    def __init__(
            self,
            physics_steps_per_step: int = 1,
            max_time: float = 10.0,
            action_scale: float = 1.0,
            hip_range: float = np.pi / 3,
            forward_reward_weight: float = 1.0,
            ctrl_cost_weight: float = 0.01,
            physics_error_reward: float = -100.0,
            # init_unit_model: UnitModelProvider | list[UnitModelProvider],
            obs_fields: list[str] = None
    ):
        if obs_fields is None:
            obs_fields = ['qpos', 'qvel']

        self.physics_steps_per_step = physics_steps_per_step
        self.max_time = max_time
        self.action_scale = float(action_scale)
        self.hip_range = hip_range

        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.physics_error_reward = physics_error_reward

        # TODO also add to envspec?
        self.unit_models = [
            UnitModel(
                body_radius=0.1,
                leg_length=0.2,
                leg_radius=0.025,
                hip_range=self.hip_range
            )
            for _ in range(2)
        ]
        self.unit_ids = [
            um.unit_id for um in self.unit_models
        ]
        self.num_units = len(self.unit_ids)

        self.enabled_connections = [
            Connection(ConnectionSite(0, 0), ConnectionSite(1, 0))
        ]

        self.model = mjcf.RootElement()
        self.physics = self.setup_physics()

        self.obs_fields = obs_fields
        self.unit_field_indices = {
            field: np.stack([
                 self.collect_unit_field_indices(unit_id, field) for unit_id in self.unit_ids
            ])
            for field in obs_fields
        }

        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, self.get_obs().shape)
        self.action_space = gymnasium.spaces.Box(-1, 1, self.physics.data.actuator_velocity.shape)

        self.spec = EnvSpec(
            id='SwarmBot-v0.2',
            kwargs={
                'physics_steps_per_step': physics_steps_per_step,
                'max_time': max_time,
                'action_scale': action_scale,
                'hip_range': hip_range,
                'forward_reward_weight': forward_reward_weight,
                'ctrl_cost_weight': ctrl_cost_weight,
                'physics_error_reward': physics_error_reward,
                'obs_fields': obs_fields,
            }
        )

    def collect_unit_field_indices(self, unit_id: str, field_key: str):
        indices = np.arange(len(getattr(self.physics.data, field_key)), dtype=int)

        field_indices = np.empty((0,), dtype=int)
        field_axis = getattr(self.physics.named.data, field_key).axes.row
        for key in [n for n in field_axis.names if n.startswith(unit_id)]:
            field_indices = np.concatenate((
                field_indices,
                np.atleast_1d(indices[field_axis.convert_key_item(key)])
            ))

        return field_indices

    def setup_physics(self):
        chequered = self.model.asset.add('texture', type='2d', builtin='checker', width=300,
                                         height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
        grid = self.model.asset.add('material', name='grid', texture=chequered,
                                    texrepeat=[5, 5], reflectance=.2)
        self.model.worldbody.add('geom', type='plane', size=[2, 2, .1], material=grid)

        for x in [-2, 2]:
            self.model.worldbody.add('light', pos=[x, -1, 3], dir=[-x, 1, -2])

        SwarmModel.attach_to(
            self.model,
            unit_models=self.unit_models,
            positions=[
                (0.305, 0, 0.2),
                (-0.305, 0, 0.2),
            ],
            eulers=[
                (0, -np.pi / 2, 0),
                (0, np.pi / 2, np.pi),
            ],
            enabled_connections=self.enabled_connections,
        )

        cam = self.model.worldbody.add(
            'camera',
            mode='targetbody',
            target=f'{self.unit_models[0].unit_id}/leg3/~',
            pos=[0, 1.5, 1]
        )

        return mjcf.Physics.from_mjcf_model(self.model)

    def get_obs(self):
        return np.concatenate([
            getattr(self.physics.data, field)[self.unit_field_indices[field]].reshape((self.num_units, -1))
            for field in self.obs_fields
        ], axis=1).copy()

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
