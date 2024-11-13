from typing import Callable

from dm_control import mjcf

from swarmbots.unit_model import UnitModel


UnitModelProvider = Callable[[], UnitModel]


class SwarmModel:

    def __init__(
            self,
            num_units: int,
            init_unit_model: UnitModelProvider | list[UnitModelProvider],
            positions: list[tuple[float, float, float]],
            eulers: list[tuple[float, float, float]]
    ):
        if isinstance(init_unit_model, list):
            if len(init_unit_model) != num_units:
                raise ValueError(
                    f'Invalid number of UnitProviders at init_unit: {len(init_unit_model) = } vs {num_units = }'
                )
            init_unit_models = init_unit_model
        else:
            init_unit_models = [init_unit_model] * num_units

        self.unit_models: list[UnitModel] = []
        self.model = mjcf.RootElement()

        for init_unit_model, pos, euler in zip(init_unit_models, positions, eulers, strict=True):
            unit = init_unit_model()
            self.unit_models.append(unit)

            spawn_site = self.model.worldbody.add('site', pos=pos, euler=euler)
            spawn_site.attach(unit.model).add('freejoint')

        # self.model.equality.add(
        #     'weld',
        #     body1=f'unnamed_model/unnamed_model/{unit1.name}-leg0-foot',
        #     body2=f'unnamed_model_1/unnamed_model/{unit2.name}-leg0-foot',
        #     torquescale=10_000
        # )




