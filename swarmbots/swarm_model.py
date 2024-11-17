from typing import Callable, NamedTuple

from dm_control import mjcf

from swarmbots.unit_model import UnitModel


class ConnectionSite(NamedTuple):
    unit_idx: int
    leg_index: int


class Connection(NamedTuple):
    a: ConnectionSite
    b: ConnectionSite


class SwarmModel:

    @staticmethod
    def attach_to(
            model: mjcf.RootElement,
            unit_models: list[UnitModel],
            positions: list[tuple[float, float, float]],
            eulers: list[tuple[float, float, float]],
            enabled_connections: list[Connection],
    ):

        unit: UnitModel
        pos: tuple[float, float, float]
        positions: tuple[float, float, float]
        for unit, pos, euler in zip(unit_models, positions, eulers, strict=True):

            spawn_site = model.worldbody.add('site', pos=pos, euler=euler, name=f'{unit.unit_id}-site')
            spawn_site.attach(unit.model).add('freejoint')

        for connection in enabled_connections:
            assert connection.a.unit_idx < connection.b.unit_idx

            unit_a_name = unit_models[connection.a.unit_idx].unit_id
            unit_b_name = unit_models[connection.b.unit_idx].unit_id

            model.equality.add(
                'weld',
                body1=f'{unit_a_name}/leg{connection.a.leg_index}/foot',
                body2=f'{unit_b_name}/leg{connection.b.leg_index}/foot',
                torquescale=10_000,
                name=f'{unit_a_name}_{connection.a.leg_index}--{unit_b_name}_{connection.b.leg_index}'
            )




