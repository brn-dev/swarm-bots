from collections import Iterable
import uuid
from typing import Optional

import numpy as np
from dm_control import mjcf


class Unit:

    class Leg:
        def __init__(self, length: float, radius: float, hip_range: float, rgba: Iterable[float], name: str):
            self.model = mjcf.RootElement()

            self.leg = self.model.worldbody.add('body', name=name)
            self.hip = self.leg.add('joint', type='ball', range=f'0 {hip_range}')
            self.leg.add(
                'geom',
                type='cylinder',
                fromto=[0, 0, 0, 0, 0, length],
                size=[radius],
                rgba=rgba
            )
            self.foot = self.leg.add('body', name=f'{name}-foot')
            self.foot.add(
                'geom',
                type='cylinder',
                fromto=[0, 0, length*0.99, 0, 0, length*1.01],
                size=[radius*1.1],
                rgba=rgba
            )

            # TODO: actuator

    def __init__(
            self,
            body_radius: float,
            leg_length: float,
            leg_radius: float,
            hip_range: float,
            body_rgba=(0.75, 0, 0, 0.1),
            leg_rgba=(0, 0, 0, 1),
            name: Optional[str] = None
    ):
        if name is None:
            name = f'Unit{str(uuid.uuid4())[0:8]}'
        self.name = name

        self.model = mjcf.RootElement()
        self.model.compiler.angle = 'radian'

        self.body = self.model.worldbody.add('body', name=name)

        self.body.add('geom', type='sphere', size=[body_radius], rgba=body_rgba)

        hip_site = self.body.add('site', pos=[0, 0, body_radius], euler=[0, 0, 0])
        leg = Unit.Leg(leg_length, leg_radius, hip_range, rgba=(0, 1, 0, 1), name=f'{name}-leg0')
        hip_site.attach(leg.model)

        lower_leg_positions = [
            [np.sqrt(8/9), 0, -1/3],
            [-np.sqrt(2/9), np.sqrt(2/3), -1/3],
            [-np.sqrt(2/9), -np.sqrt(2/3), -1/3]
        ]

        for (i, leg_rgba) in zip(range(3), [(0, 1, 1, 1), (0, 0, 1, 1), (1, 1, 0, 1)]):
            hip_pos = body_radius * np.array(lower_leg_positions[i])
            hip_site = self.body.add('site', pos=hip_pos, zaxis=hip_pos)
            leg = Unit.Leg(leg_length, leg_radius, hip_range, rgba=leg_rgba, name=f'{name}-leg{i+1}')
            hip_site.attach(leg.model)
