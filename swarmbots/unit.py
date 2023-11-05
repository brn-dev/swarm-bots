import numpy as np
from dm_control import mjcf

class Unit:

    class Leg:
        def __init__(self, length: float, radius: float, hip_range: float, rgba: list[float]):
            self.model = mjcf.RootElement()

            self.leg = self.model.worldbody.add('body')
            self.hip = self.leg.add('joint', type='ball', range=f'0 {hip_range}')
            self.leg.add('geom', type='cylinder', fromto=[0, 0, 0, 0, 0, length], size=[radius], rgba=rgba)

            # TODO: actuator

    def __init__(
            self,
            body_radius: float,
            leg_length: float,
            leg_radius: float,
            hip_range: float,
            body_rgba=(0.75, 0, 0, 1),
            leg_rgba=(0, 0, 0, 1)
    ):
        self.model = mjcf.RootElement()
        self.model.compiler.angle = 'radian'

        self.model.worldbody.add('geom', name='body', type='sphere', size=[body_radius], rgba=body_rgba)

        hip_site = self.model.worldbody.add('site', pos=[0, 0, body_radius], euler=[0, 0, 0])
        leg = Unit.Leg(leg_length, leg_radius, hip_range, rgba=leg_rgba)
        hip_site.attach(leg.model)

        rotations = [
            [np.sqrt(8/9), 0, -1/3],
            [-np.sqrt(2/9), np.sqrt(2/3), -1/3],
            [-np.sqrt(2/9), -np.sqrt(2/3), -1/3]
        ]

        for i in range(3):
            theta = i * 2 * np.pi / 3
            hip_pos = body_radius * np.cos(np.pi / 6) * np.array([np.cos(theta), np.sin(theta), -np.sin(np.pi / 6)])
            hip_site = self.model.worldbody.add('site', pos=hip_pos, zaxis=rotations[i])
            leg = Unit.Leg(leg_length, leg_radius, hip_range, rgba=leg_rgba)
            hip_site.attach(leg.model)





