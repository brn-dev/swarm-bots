from typing import Generic, TypeVar

from swarmbots.learn.base_sampler import BaseSamplerConfig

WMSamplesType = TypeVar("WMSamplesType", covariant=True)
WMSamplerConfigType = TypeVar("WMSamplerConfigType", covariant=True, bound=BaseSamplerConfig)


class BaseWMSampler(Generic[WMSamplesType, WMSamplerConfigType]):
    pass
