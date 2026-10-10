from __future__ import annotations

import copy
import csv
import gzip
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from swarmbots.tools.inspect_policy_parameters import layer_layout, module_parameter_counts, save_report
from swarmbots.learn.algos.mat_orig.mat_orig_encoder import MATOrigSelfAttention
from swarmbots.learn.algos.xlstm.slstm.slstm_temporal_sequence_model import (
    SLSTMTemporalSequenceModel,
    SLSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.nn_components.feed_forward import SwiGLU
from swarmbots.learn.nn_components.popart import PopArtLinear
from swarmbots.learn.parameter_counts import count_policy_parameters


def test_counts_separate_targets_nop_and_buffers() -> None:
    policy = nn.Module()
    policy.encoder = nn.Linear(3, 4)  # 16, shared by actor/critic
    policy.actor = nn.Linear(4, 2)  # 10
    policy.action_dist = nn.Linear(2, 2, bias=False)  # 4
    policy.critic = nn.Linear(4, 1)  # 5
    policy.critic_target = copy.deepcopy(policy.critic).requires_grad_(False)  # 5
    policy.actor_nop = nn.Module()
    policy.actor_nop.transition_model = nn.Linear(4, 3)  # 15
    policy.temperature = nn.Parameter(torch.ones(1))
    policy.register_buffer("normalization", torch.zeros(1000))

    result = count_policy_parameters(policy)

    assert result["total"] == 56
    assert result["trainable_minus_nop"] == 36  # excludes NOP and frozen targets
    assert result["trainable"] == 51
    assert result["frozen"] == 5
    assert result["parameter_bytes"] == 56 * 4
    assert {role: counts["total"] for role, counts in result["roles"].items()} == {
        "actor": 14, "critic": 5, "shared_encoder": 16,
        "next_obs_prediction": 15, "targets": 5, "other": 1,
    }
    assert sum(component["total"] for component in result["components"]) == 56


def test_wrapped_policy_and_world_model_use_the_same_breakdown() -> None:
    policy = nn.Module()
    policy.actor = nn.Linear(2, 1)  # 3
    wrapper = nn.Module()
    wrapper.policy = policy
    wrapper.pre_transition_transform = nn.Linear(1, 2)  # 4
    wrapper.local_scalars_predictor = nn.Linear(2, 3)  # 9
    result = count_policy_parameters(wrapper)
    assert result["roles"]["actor"]["total"] == 3
    assert result["roles"]["next_obs_prediction"]["total"] == 13
    assert result["total"] == 16


def test_separate_actor_encoder_keeps_mat_dec_encoder_in_critic() -> None:
    policy = nn.Module()
    policy.encoder = nn.Linear(3, 4)  # 16
    policy.actor_encoder = nn.Linear(3, 4)  # 16
    result = count_policy_parameters(policy)
    assert result["roles"]["actor"]["total"] == 16
    assert result["roles"]["critic"]["total"] == 16
    assert result["roles"]["shared_encoder"]["total"] == 0


def test_parameter_aliases_count_once_and_cross_role_ties_are_shared() -> None:
    policy = nn.Module()
    policy.actor = nn.Linear(3, 2)  # 8
    policy.critic = policy.actor
    result = count_policy_parameters(policy)
    assert result["total"] == 8
    assert result["roles"]["actor"]["total"] == 0
    assert result["roles"]["critic"]["total"] == 0
    assert result["roles"]["shared_encoder"]["total"] == 8
    assert len(result["shared_parameter_aliases"]) == 2


def test_tied_twin_critic_encoder_is_not_counted_twice() -> None:
    policy = nn.Module()
    policy.critic = nn.Module()
    policy.critic.encoder = nn.Linear(3, 2)  # 8
    policy.critic.encoder2 = policy.critic.encoder
    policy.critic.q1 = nn.Linear(2, 1)  # 3
    policy.critic.q2 = nn.Linear(2, 1)  # 3
    result = count_policy_parameters(policy)
    assert result["total"] == result["roles"]["critic"]["total"] == 14


def test_inclusive_module_counts_deduplicate_aliases_and_omit_targets() -> None:
    policy = nn.Module()
    policy.critic = nn.Module()
    policy.critic.encoder = nn.Linear(3, 2)  # 8
    policy.critic.encoder2 = policy.critic.encoder
    policy.critic.q1 = nn.Linear(2, 1)  # 3
    policy.critic.q2 = nn.Linear(2, 1)  # 3
    policy.critic_target = copy.deepcopy(policy.critic).requires_grad_(False)
    policy.critic.register_buffer("statistics", torch.zeros(100))
    wrapper = nn.Module()
    wrapper.policy = policy
    counts = module_parameter_counts(wrapper)
    assert counts["critic"] == 14
    assert counts["critic.encoder"] == counts["critic.encoder2"] == 8
    assert counts["critic.q1"] == counts["critic.q2"] == 3
    assert not any("target" in path or path.startswith("policy.") for path in counts)


def test_recurrent_weights_are_split_from_backbone() -> None:
    policy = nn.Module()
    policy.encoder = nn.Module()
    policy.encoder.projection = nn.Linear(3, 2)  # 8
    policy.encoder.layers = nn.ModuleList([nn.Module(), nn.Module()])
    for layer in policy.encoder.layers:
        layer.temporal_model = nn.Linear(2, 2)  # 6 each
    result = count_policy_parameters(policy)
    assert {component["module"]: component["total"] for component in result["components"]} == {
        "encoder.backbone": 8, "encoder.temporal_model": 12,
    }


def test_incompatible_target_aliases_are_reported() -> None:
    policy = nn.Module()
    policy.actor = nn.Linear(3, 2)
    policy.actor_target = policy.actor
    with pytest.raises(ValueError, match="incompatible roles"):
        count_policy_parameters(policy)


def test_processing_excludes_nop_targets_and_attention_projections() -> None:
    policy = nn.Module()
    policy.actor = nn.Module()
    policy.actor.attention = nn.MultiheadAttention(4, 2, batch_first=True)  # 80, includes out_proj
    policy.actor.feedforward = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))  # 76
    policy.actor.input_projection = nn.Linear(3, 4)  # 16
    policy.actor.norm = nn.LayerNorm(4)  # 8
    policy.actor.embedding = nn.Embedding(3, 4)  # 12
    policy.actor_target = copy.deepcopy(policy.actor).requires_grad_(False)
    policy.actor_nop = nn.Linear(4, 30)  # 150; fully excluded

    counts = count_policy_parameters(policy)
    processing = counts["processing"]
    assert processing["total"] == 192
    assert processing["mlp_and_linear"]["total"] == 92
    assert {kind: value["total"] for kind, value in processing["kinds"].items()} == {
        "mlp": 76, "linear_projection": 16, "attention": 80, "recurrent": 0,
        "normalization": 8, "embedding": 12, "other": 0,
    }
    assert processing["roles"]["actor"]["total"] == 192
    assert sum(component["total"] for component in processing["components"]) == 192


def test_custom_mat_attention_is_not_mistaken_for_an_mlp() -> None:
    policy = nn.Module()
    policy.actor = MATOrigSelfAttention(4, 2, max_agents=3, masked=False)  # four 4x4+bias projections
    processing = count_policy_parameters(policy)["processing"]
    assert processing["total"] == processing["kinds"]["attention"]["total"] == 80
    assert processing["mlp_and_linear"]["total"] == 0


def test_slstm_gate_projections_and_internal_norm_are_recurrent() -> None:
    policy = nn.Module()
    policy.actor = nn.Module()
    policy.actor.temporal_model = SLSTMTemporalSequenceModel(8, SLSTMTemporalSequenceModelConfig(num_heads=2))
    # Two heads: input projection 128, recurrent kernel 128, gate biases 32, norm 8.
    policy.actor.temporal_output_projection = nn.Linear(8, 8, bias=False)  # outside temporal model
    processing = count_policy_parameters(policy)["processing"]
    assert processing["kinds"]["recurrent"]["total"] == 296
    assert processing["kinds"]["normalization"]["total"] == 0
    assert processing["mlp_and_linear"]["total"] == 64
    assert processing["total"] == 360


def test_swiglu_gate_value_and_output_weights_are_in_the_mlp_budget() -> None:
    policy = nn.Module()
    policy.actor = SwiGLU(input_dim=4, hidden_dim=6, output_dim=4)  # 30 + 30 + 28
    processing = count_policy_parameters(policy)["processing"]
    assert processing["kinds"]["mlp"]["total"] == 88
    assert processing["mlp_and_linear"]["total"] == 88


def test_popart_affine_weights_count_as_a_projection_and_exclude_running_stats() -> None:
    policy = nn.Module()
    policy.critic = PopArtLinear(4, 1)
    processing = count_policy_parameters(policy)["processing"]
    assert processing["total"] == 5
    assert processing["roles"]["critic"]["kinds"]["linear_projection"]["total"] == 5


def test_transformer_feedforward_linears_are_in_the_mlp_budget() -> None:
    policy = nn.Module()
    policy.actor = nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, batch_first=True)
    processing = count_policy_parameters(policy)["processing"]
    assert processing["kinds"]["mlp"]["total"] == 76
    assert processing["kinds"]["attention"]["total"] == 80
    assert processing["kinds"]["normalization"]["total"] == 16
    assert processing["total"] == 172


def test_processing_shared_encoder_counts_once_for_tied_actor_critic() -> None:
    policy = nn.Module()
    policy.actor = nn.Linear(3, 2)  # 8
    policy.critic = policy.actor
    processing = count_policy_parameters(policy)["processing"]
    assert processing["total"] == processing["mlp_and_linear"]["total"] == 8
    assert processing["roles"]["shared_encoder"]["mlp_and_linear"]["total"] == 8
    assert processing["roles"]["actor"]["total"] == processing["roles"]["critic"]["total"] == 0


@pytest.mark.parametrize("nop_trainable", [False, True])
def test_report_exports_trainable_minus_nop_and_lossless_compressed_json(tmp_path: Path, nop_trainable: bool) -> None:
    policy = nn.Module()
    policy.actor = nn.Linear(2, 1)  # 3
    policy.actor_target = copy.deepcopy(policy.actor).requires_grad_(False)  # 3
    policy.actor_nop = nn.Linear(1, 2).requires_grad_(nop_trainable)  # 4, only subtract when trainable
    report = {
        "benchmark_id": "example", "device": "cpu", "generated_at": "2026-10-06",
        "requested_variants": ["example"], "overrides": {}, "errors": [],
        "policies": [{"variant": "example", "base_policy_class": "Module", "shape": {}, **count_policy_parameters(policy)}],
    }
    (tmp_path / "counts.json").write_text("obsolete export", encoding="utf-8")
    save_report(report, tmp_path)
    assert not (tmp_path / "counts.json").exists()
    with gzip.open(tmp_path / "counts.json.gz", "rt", encoding="utf-8") as stream:
        assert json.load(stream) == report
    with (tmp_path / "counts.csv").open(newline="", encoding="utf-8") as stream:
        row, = csv.DictReader(stream)
    assert row["total"] == "10"
    assert row["trainable_minus_nop"] == "3"
    assert "total_minus_nop" not in row
    assert row["main_policy_total"] == "3"  # online-only processing excludes targets too
    assert "Trainable − NOP" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_layer_layout_shows_actual_dimensions_without_nested_duplicates() -> None:
    policy = nn.Module()
    policy.actor = nn.Sequential(nn.Linear(5, 128), nn.ReLU(), nn.Linear(128, 64))
    policy.critic = nn.MultiheadAttention(64, 4, batch_first=True)
    policy.critic_target = copy.deepcopy(policy.critic).requires_grad_(False)
    rows = layer_layout(policy)
    assert {row["module"] for row in rows} == {"actor", "critic"}
    assert rows[0]["dimensions"] == "5 -> 128 -> 64"
    assert rows[0]["parameters"] == sum(parameter.numel() for parameter in policy.actor.parameters())
    assert rows[1]["dimensions"] == "d_model=64, heads=4"


def test_layer_layout_includes_nested_regressor_trunk_and_head() -> None:
    policy = nn.Module()
    policy.critic = nn.Sequential(nn.Sequential(nn.Linear(384, 512), nn.ReLU(), nn.Linear(512, 256)), nn.Linear(256, 1))
    rows = layer_layout(policy)
    assert len(rows) == 1
    assert rows[0]["dimensions"] == "384 -> 512 -> 256 -> 1"
    assert rows[0]["parameters"] == sum(parameter.numel() for parameter in policy.critic.parameters())
