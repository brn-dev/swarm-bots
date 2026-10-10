from __future__ import annotations

import gzip
import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from swarmbots.tools.diagram_policy_architectures import (
    build_diagram,
    configuration,
    mermaid_diagram,
    svg_diagram,
    dot_diagram,
    generate,
)
from swarmbots.learn import list_model_scales, list_variants

ROOT = Path(__file__).resolve().parents[1]
REPORTS = [ROOT / "docs/policy_parameters" / folder / "counts.json.gz" for folder in ["", "2.5M", "10M"]]
ROWS = [row for path in REPORTS for row in json.loads(gzip.decompress(path.read_bytes()))["policies"]]


def assert_line_order(node, *prefixes):
    positions = [next(i for i, line in enumerate(node.lines) if line.startswith(prefix)) for prefix in prefixes]
    assert positions == sorted(positions), (node.id, prefixes, node.lines)


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["variant"] + "-" + row["model_scale"]["name"])
def test_all_core_diagrams_render_with_valid_dimensions_and_privilege_boundary(row):
    d = build_diagram(row)
    root = ElementTree.fromstring(svg_diagram(d))
    assert root.tag.endswith("svg")
    assert mermaid_diagram(d).startswith("flowchart LR")
    assert dot_diagram(d).startswith("digraph policy")
    assert all("see layer table" not in " ".join(node.lines) for node in d.nodes)
    destinations = {"privileged"}
    for _ in range(len(d.nodes)):
        destinations |= {edge.target for edge in d.edges if edge.source in destinations}
    assert not destinations & {"actor", "actor_encoder", "actions"}
    assert ("future_actions", "transition") in {(edge.source, edge.target) for edge in d.edges}
    assert ("actions", "transition") not in {(edge.source, edge.target) for edge in d.edges}
    for role in ("actor", "critic", "shared_encoder", "next_obs_prediction"):
        assert (
            sum(node.parameters or 0 for node in d.nodes if node.parameter_role == role) == row["roles"][role]["total"]
        )
    assert sum(node.parameters or 0 for node in d.nodes) == row["trainable"]
    assert d.parameter_summary in "".join(root.itertext())
    for node in d.nodes:
        if node.parameters is not None:
            assert node.parameter_label in mermaid_diagram(d)
            assert node.parameter_label in dot_diagram(d)
        if any(line.startswith("obs MLP (once):") for line in node.lines):
            assert_line_order(node, "obs MLP (once):", "repeat ", "FF / block:", "output:")
        if any(line.startswith("local/action MLP (once):") for line in node.lines):
            assert_line_order(
                node,
                "local/action MLP (once):",
                "global MLP (once):",
                "add local",
                "repeat ",
                "attention / block:",
                "FF / block:",
                "output:",
            )
        temporal = next(
            (line.split(" / block:")[0] for line in node.lines if line.startswith(("LSTM / block:", "sLSTM / block:"))),
            None,
        )
        if temporal:
            cfg = configuration(row)["actor_encoder_config" if node.id == "actor_encoder" else "encoder_config"]
            first, second = (
                (temporal, "attention") if cfg["temporal_model_order"] == "temporal_first" else ("attention", temporal)
            )
            assert_line_order(node, first + " / block:", "inter FF / block:", second + " / block:", "FF / block:")
        if node.id == "transition":
            assert_line_order(
                node, "coembed (per step):", "repeat ", "attention / block:", "FF / block:", "head:", "latent residual;"
            )
        if node.id == "state_bridge":
            assert node.lines[0] == "detach last actor state"
    if row["variant"] == "tmasac_shared_encoder":
        actor = next(node for node in d.nodes if node.id == "actor_encoder")
        assert_line_order(actor, "FF / block:", "head:", "output:")
        assert actor.lines[-1] == f"output: N × {configuration(row)['actor_head_config']['hidden_dims'][-1]}"


@pytest.mark.parametrize(
    "row",
    [
        row
        for row in ROWS
        if row["variant"] in ["tmasac", "tmatd3", "tmasac_slstm", "tmasac_lstm", "tmasac_shared_encoder"]
    ],
    ids=lambda row: row["variant"] + "-" + row["model_scale"]["name"],
)
def test_transformer_q_heads_share_one_encoder_and_recurrent_inputs_are_detached(row):
    d = build_diagram(row)
    edges = {(edge.source, edge.target): edge for edge in d.edges}
    assert ("critic_encoder", "head1") in edges and ("critic_encoder", "head2") in edges
    assert len([node for node in d.nodes if node.id == "critic_encoder"]) == 1
    if row["variant"] in ["tmasac_lstm", "tmasac_slstm"]:
        assert edges["state_bridge", "critic_encoder"].kind == "state"
        assert ("actor_encoder", "actor_encoder") in edges
    else:
        assert "state_bridge" not in {node.id for node in d.nodes}


@pytest.mark.parametrize("row", [row for row in ROWS if row["variant"] in ["matd3_deepset", "masac_deepset"]])
def test_independent_deepset_critics_concatenate_element_features_for_nop(row):
    d = build_diagram(row)
    edges = {(edge.source, edge.target) for edge in d.edges}
    assert ("q_inputs", "q1_network") in edges and ("q_inputs", "q2_network") in edges
    assert ("q1_network", "q_features") in edges and ("q2_network", "q_features") in edges
    assert ("q_features", "nop_bridge") in edges
    assert "critic_encoder" not in {node.id for node in d.nodes}


@pytest.mark.parametrize("row", [row for row in ROWS if row["variant"] in ["mat_orig", "mat_qcx", "mat_qcx_lstm"]])
def test_autoregressive_decoders_only_feedback_previous_agent_actions(row):
    d = build_diagram(row)
    feedback = next(edge for edge in d.edges if edge.source == "actions" and edge.target == "actor")
    assert feedback.kind == "causal"
    assert feedback.label == "previous agents only"
    actor = next(node for node in d.nodes if node.id == "actor")
    assert_line_order(
        actor,
        "query projection (once):" if "qcx" in row["variant"] else "obs projection (once):",
        "action embedding (once):",
        "repeat ",
        "FF / block:",
        "output:",
    )
    if "qcx" in row["variant"] and row["model_scale"]["name"] == "5M NOP1M":
        assert "FF / block: 192 → 256 → 192 (98,752)" in actor.lines


def test_default_tmasac_labels_show_exact_subtotals_and_whole_components():
    row = next(row for row in ROWS if row["variant"] == "tmasac" and row["model_scale"]["name"] == "5M NOP1M")
    d = build_diagram(row)
    nodes = {node.id: node for node in d.nodes}
    assert "element MLP: 384 → 384 → 384 (295,680)" in nodes["head1"].lines
    assert "regressor: 384 → 384 → 1 (148,225)" in nodes["head1"].lines
    assert nodes["head1"].parameters == 295_680 + 148_225
    assert nodes["head2"].parameters == nodes["head1"].parameters
    assert nodes["critic_encoder"].parameters == 2_305_024
    assert nodes["critic_encoder"].parameters + 2 * nodes["head1"].parameters == row["roles"]["critic"]["total"]
    assert "block total: 987,008 per block" in nodes["critic_encoder"].lines


def test_deepset_feature_concatenation_adds_no_parameters():
    row = next(row for row in ROWS if row["variant"] == "matd3_deepset" and row["model_scale"]["name"] == "5M NOP1M")
    d = build_diagram(row)
    assert next(node for node in d.nodes if node.id == "q_features").parameters == 0


def test_refresh_keeps_compact_inputs_and_writes_verbose_reports_to_generated(tmp_path, monkeypatch):
    from swarmbots.tools import diagram_policy_architectures as diagrams

    report = json.loads(gzip.decompress(REPORTS[0].read_bytes()))
    guide = tmp_path / "docs/policy_parameters/model_scale.md"
    guide.parent.mkdir(parents=True)
    guide.write_text("Sizing guide", encoding="utf-8")
    def inspect(benchmark_id, variants, *, model_scale):
        assert benchmark_id == "SwarmBots-WallEasy-v0"
        assert variants == list_variants()
        assert model_scale == "5M NOP1M"
        return report

    monkeypatch.setattr(diagrams, "ROOT", tmp_path)
    monkeypatch.setattr(diagrams, "inspect_variants", inspect)
    rows = diagrams.load_reports(["5M NOP1M"], ["tmasac"], refresh=True)
    assert [row["variant"] for row in rows] == ["tmasac"]
    docs = tmp_path / "docs/policy_parameters"
    assert json.loads(gzip.decompress((docs / "counts.json.gz").read_bytes())) == report
    assert not (docs / "report.md").exists()
    assert not (docs / "counts.csv").exists()
    generated = docs / "generated/5M"
    assert (generated / "counts.csv").exists()
    assert (generated / "layer_layout.csv").exists()
    assert "[model-scale allocation](../../model_scale.md)" in (generated / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("report_path", REPORTS, ids=["5M", "2.5M", "10M"])
@pytest.mark.parametrize("refresh", [False, True])
def test_partial_audit_rebuild_preserves_core_and_cached_hidden_presets(tmp_path, monkeypatch, report_path, refresh):
    from swarmbots.tools import diagram_policy_architectures as diagrams
    from swarmbots.tools.inspect_policy_parameters import write_compressed_json

    report = json.loads(gzip.decompress(report_path.read_bytes()))
    scale = report["policies"][0]["model_scale"]["name"]
    hidden_name = "tmasac_slstm_no_residual"
    hidden_row = {**next(row for row in report["policies"] if row["variant"] == "tmasac_slstm"), "variant": hidden_name}
    report["policies"].append(hidden_row)
    report["requested_variants"].append(hidden_name)
    cached = {
        **report,
        "policies": [row for row in report["policies"] if refresh or row["variant"] != "tmasac"],
    }
    path = tmp_path / report_path.relative_to(ROOT)
    path.parent.mkdir(parents=True)
    write_compressed_json(cached, path)
    calls = []

    def inspect(benchmark_id, variants, *, model_scale):
        calls.append(variants)
        assert model_scale == scale
        by_name = {row["variant"]: row for row in report["policies"]}
        return {
            **report,
            "requested_variants": list(variants),
            "policies": [{**by_name[name], "refreshed": True} for name in variants],
        }

    monkeypatch.setattr(diagrams, "ROOT", tmp_path)
    monkeypatch.setattr(diagrams, "inspect_variants", inspect)
    rows = diagrams.load_reports([scale], ["tmasac"], refresh=refresh)
    assert [row["variant"] for row in rows] == ["tmasac"]
    assert calls == [(*list_variants(), hidden_name)]
    saved = json.loads(gzip.decompress(path.read_bytes()))
    assert saved["requested_variants"] == [*list_variants(), hidden_name]
    assert {row["variant"] for row in saved["policies"]} == {*list_variants(), hidden_name}
    assert all(row["refreshed"] for row in saved["policies"])
    # The complete rebuilt audit is reusable for unselected presets without
    # another inspection or snapshot rewrite.
    before = path.read_bytes()
    rows = diagrams.load_reports([scale], ["mappo", hidden_name], refresh=False)
    assert {row["variant"] for row in rows} == {"mappo", hidden_name}
    assert len(calls) == 1
    assert path.read_bytes() == before


@pytest.mark.integration
@pytest.mark.parametrize("scale", list_model_scales())
def test_stacked_swiglu_presets_generate_diagrams_from_actual_audits(scale, tmp_path):
    import torch
    from swarmbots.tools.inspect_policy_parameters import inspect_variants

    variants = ("tmasac_swiglu", "tmasac_slstm_swiglu")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = inspect_variants("SwarmBots-WallEasy-v0", variants, model_scale=scale)
    finally:
        torch.set_num_threads(previous_threads)
    assert not report["errors"]
    diagrams = generate(report["policies"], tmp_path)
    assert {d.variant for d in diagrams} == set(variants)
    for d, row in zip(diagrams, report["policies"], strict=True):
        assert sum(node.parameters or 0 for node in d.nodes) == row["trainable"]
        stacked_lines = [line for node in d.nodes for line in node.lines if "2 × SwiGLU" in line]
        assert len(stacked_lines) == (2 if d.variant == "tmasac_swiglu" else 1)
        svg_path, = tmp_path.glob(f"*/{d.variant}.svg")
        assert ElementTree.fromstring(svg_path.read_text(encoding="utf-8")).tag.endswith("svg")
        assert "2 × SwiGLU" in svg_path.with_suffix(".mmd").read_text(encoding="utf-8")
        assert "2 × SwiGLU" in svg_path.with_suffix(".dot").read_text(encoding="utf-8")
    assert (tmp_path / "index.html").exists()
