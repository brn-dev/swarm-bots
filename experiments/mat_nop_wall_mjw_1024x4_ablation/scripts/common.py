from __future__ import annotations

import sys
from pathlib import Path

from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swarmbots.learn.algos.mat.mat_decoder import MATDecoderSelfAttentionMode
from swarmbots.learn.nn_components.activations import ActivationFactory

from experiments.mat_nop_wall_mjw_common import ContinuousActionDistVariant, PolicyVariant, run_experiment

EXPERIMENT_RUN_NAME = "mat_nop_swarm_bots_wall_mjw_1024x4_ablation"


def run_ablation(
        *,
        variant_name: str,
        entrypoint_path: Path,
        continuous_action_dist: ContinuousActionDistVariant = "sticky_lr_beta",
        policy_variant: PolicyVariant = "mat",
        mat_add_agent_embeddings: bool = True,
        mat_decoder_self_attention_mode: MATDecoderSelfAttentionMode = MATDecoderSelfAttentionMode.FULL_AUTOREGRESSIVE,
        act_fn_cls: ActivationFactory = nn.GELU,
        use_nop: bool = True,
) -> None:
    run_experiment(
        num_envs=1024,
        rollout_steps_per_env=4,
        variant_name=variant_name,
        entrypoint_path=entrypoint_path,
        continuous_action_dist=continuous_action_dist,
        policy_variant=policy_variant,
        mat_add_agent_embeddings=mat_add_agent_embeddings,
        mat_decoder_self_attention_mode=mat_decoder_self_attention_mode,
        act_fn_cls=act_fn_cls,
        use_nop=use_nop,
        experiment_run_name=EXPERIMENT_RUN_NAME,
    )
