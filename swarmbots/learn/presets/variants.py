"""Canonical learning presets exposed by the benchmark training helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import get_args

from swarmbots.learn.presets.policy_factory import PolicyVariant
from swarmbots.learn.presets.tmasac import TMASACVariant


@dataclass(frozen=True)
class LearningVariantConfig:
    policy_variant: PolicyVariant | None = None
    tmasac_variant: TMASACVariant | None = None
    use_nop: bool = True
    use_agent_attention: bool = True
    hidden: bool = False


VARIANT_CONFIGS: dict[str, LearningVariantConfig] = {
    name: LearningVariantConfig(policy_variant=name)
    for name in (
        "ppo",
        "mappo",
        "mappo_mlp",
        "mat_orig",
        "mat_ind",
        "mat_dec",
        "mat_qcx",
        "mat_ind_lstm",
        "mat_qcx_lstm",
    )
}
VARIANT_CONFIGS.update({
    name: LearningVariantConfig(policy_variant=name, use_nop=not name.endswith("_mlp"))
    for name in (
        "maddpg_mlp", "maddpg_deepset", "matd3_mlp", "matd3_deepset",
        "masac_mlp", "masac_deepset", "tmatd3", "tmatd3_dec",
    )
})
VARIANT_CONFIGS.update({
    name: LearningVariantConfig(tmasac_variant=name)
    for name in get_args(TMASACVariant)
})
VARIANT_CONFIGS.update({
    "mat_ind_no_attention": LearningVariantConfig(policy_variant="mat_ind", use_agent_attention=False),
    "mat_qcx_no_nop": LearningVariantConfig(policy_variant="mat_qcx", use_nop=False),
    "tmasac_no_nop": LearningVariantConfig(tmasac_variant="tmasac", use_nop=False),
    "tmasac_slstm_no_nop": LearningVariantConfig(tmasac_variant="tmasac_slstm", use_nop=False),
    "tmasac_lstm_no_actor_state": LearningVariantConfig(policy_variant="tmasac_recurrent"),
    "tmasac_segment": LearningVariantConfig(policy_variant="tmasac_segment"),
})

# Keep optional baselines/controls usable for explicit experiments and old
# checkpoints, while keeping the normal selection and count sweep focused.
_DEFAULT_VARIANTS = {
    "ppo", "mappo", "mat_orig", "mat_ind", "mat_dec", "mat_qcx",
    "mat_ind_lstm", "mat_qcx_lstm", "maddpg_deepset", "matd3_deepset",
    "masac_deepset", "tmatd3", "tmatd3_dec", "tmasac", "tmasac_dec",
    "tmasac_shared_encoder", "tmasac_slstm", "tmasac_lstm",
}
VARIANT_CONFIGS = {
    name: replace(config, hidden=name not in _DEFAULT_VARIANTS)
    for name, config in VARIANT_CONFIGS.items()
}
