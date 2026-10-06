from __future__ import annotations

import gzip
import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from examples.diagram_policy_architectures import (
    build_diagram,
    configuration,
    mermaid_diagram,
    svg_diagram,
    dot_diagram,
)

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
    from examples import diagram_policy_architectures as diagrams

    report = json.loads(gzip.decompress(REPORTS[0].read_bytes()))
    monkeypatch.setattr(diagrams, "ROOT", tmp_path)
    monkeypatch.setattr(diagrams, "inspect_variants", lambda *args, **kwargs: report)
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
