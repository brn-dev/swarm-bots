"""Canonical learning presets exposed by the benchmark training helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace

from swarmbots.learn.presets.policy_factory import PolicyVariant
from swarmbots.learn.presets.tmasac import TMASACVariant


@dataclass(frozen=True)
class LearningVariantConfig:
    policy_variant: PolicyVariant | None = None
    tmasac_variant: TMASACVariant | None = None
    use_nop: bool = True
    use_agent_attention: bool = True
    hidden: bool = False
    gated_deepset: bool = False


VARIANT_CONFIGS: dict[str, LearningVariantConfig] = {
    name: LearningVariantConfig(policy_variant=name)
    for name in (
        "ppo",
        "mappo",
        "mat_orig",
        "mat_ind",
        "mat_dec",
        "mat_qcx",
        "mat_ind__lstm",
        "mat_qcx__lstm",
    )
}
VARIANT_CONFIGS.update({
    name: LearningVariantConfig(policy_variant=name)
    for name in (
        "maddpg__deepset_critic", "matd3__deepset_critic", "masac__deepset_critic", "tmatd3", "tmatd3_dec",
    )
})
VARIANT_CONFIGS.update({
    name: LearningVariantConfig(tmasac_variant=name)
    for name in ("tmasac", "tmasac_dec", "tmasac__slstm", "tmasac__lstm")
})
VARIANT_CONFIGS.update({
    "mat_qcx__no_nop": LearningVariantConfig(policy_variant="mat_qcx", use_nop=False),
    "tmasac__no_nop": LearningVariantConfig(tmasac_variant="tmasac", use_nop=False),
})
VARIANT_CONFIGS.update({
    f"{name}__gated": LearningVariantConfig(policy_variant=name, gated_deepset=True)
    for name in ("maddpg__deepset_critic", "matd3__deepset_critic", "masac__deepset_critic")
})

# Auxiliary-loss and attention controls remain explicit selections.
_DEFAULT_VARIANTS = {
    "ppo", "mappo", "mat_orig", "mat_ind", "mat_dec", "mat_qcx",
    "mat_ind__lstm", "mat_qcx__lstm", "maddpg__deepset_critic", "matd3__deepset_critic",
    "masac__deepset_critic", "tmatd3", "tmatd3_dec", "tmasac", "tmasac_dec",
    "tmasac__slstm", "tmasac__lstm",
}
VARIANT_CONFIGS = {
    name: replace(config, hidden=name not in _DEFAULT_VARIANTS)
    for name, config in VARIANT_CONFIGS.items()
}
