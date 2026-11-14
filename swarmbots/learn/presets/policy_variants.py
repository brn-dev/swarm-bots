"""Shared family and recurrence classification for resolved policy variants."""

from typing import Literal


PolicyVariant = Literal[
    "maddpg__mlp_critic", "maddpg__deepset_critic", "matd3__mlp_critic", "matd3__deepset_critic",
    "masac__mlp_critic", "masac__deepset_critic", "tmatd3", "tmatd3_dec",
    "mat_qcx",
    "mat_dec",
    "mat_ind",
    "mat_orig",
    "mat_qcx__lstm",
    "mat_ind__lstm",
    "ppo",
    "mappo__deepset_critic",
    "tmasac",
    "tmasac_dec",
    "tmasac__recurrent",
    "tmasac__segment",
]


def _is_sac_policy_variant(policy_variant: PolicyVariant) -> bool:
    return policy_variant in {
        "masac__mlp_critic", "masac__deepset_critic", "tmasac", "tmasac_dec",
        "tmasac__recurrent", "tmasac__segment",
    }


def _is_ddpg_policy_variant(policy_variant: PolicyVariant) -> bool:
    return policy_variant in {"maddpg__mlp_critic", "maddpg__deepset_critic"}


def _is_deterministic_policy_variant(policy_variant: PolicyVariant) -> bool:
    return _is_ddpg_policy_variant(policy_variant) or policy_variant in {
        "matd3__mlp_critic", "matd3__deepset_critic", "tmatd3", "tmatd3_dec",
    }


def _is_off_policy_variant(policy_variant: PolicyVariant) -> bool:
    return _is_sac_policy_variant(policy_variant) or _is_deterministic_policy_variant(policy_variant)


def _is_recurrent_policy_variant(policy_variant: PolicyVariant, *, td3_recurrent_actor: bool = False) -> bool:
    """Resolve recurrence while leaving unsupported TD3 overrides to factory validation."""
    return policy_variant in {
        "mat_ind__lstm", "mat_qcx__lstm", "tmasac__recurrent", "tmasac__segment",
    } or (_is_off_policy_variant(policy_variant) and td3_recurrent_actor)
