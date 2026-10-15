from dataclasses import dataclass, replace
from typing import Any

from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat.mat_ind_policy import MATIndPolicy, MATIndPolicyConfig
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper


@dataclass(frozen=True)
class MATDecPolicyConfig(MATIndPolicyConfig):
    pass


class MATDecPolicy(MATIndPolicy):
    def __init__(self, env: BaseLearnEnvWrapper, config: MATDecPolicyConfig | None = None) -> None:
        super().__init__(env=env, config=MATDecPolicyConfig() if config is None else config)

    def _build_encoder_config(self) -> MATEncoderConfig:
        return replace(super()._build_encoder_config(), use_agent_attention=False)

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {"mat_dec_policy_config": super().get_hyper_parameters()["mat_ind_policy_config"]}
