"""Canonical learning presets exposed by the benchmark training helpers."""

from __future__ import annotations

from dataclasses import dataclass

from swarmbots.learn.presets.policy_variants import PolicyVariant
from swarmbots.learn.presets.tmasac import TMASACVariant


DEFAULT_LEARNING_VARIANT = "mappo__deepset_critic"


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
        "mappo__deepset_critic",
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
    "mat_qcx__no_nop": LearningVariantConfig(policy_variant="mat_qcx", use_nop=False, hidden=True),
    "tmasac__no_nop": LearningVariantConfig(tmasac_variant="tmasac", use_nop=False, hidden=True),
})
VARIANT_CONFIGS.update({
    f"{algorithm}__gated_deepset_critic": LearningVariantConfig(
        policy_variant=f"{algorithm}__deepset_critic", gated_deepset=True, hidden=True,
    )
    for algorithm in ("maddpg", "matd3", "masac")
})
