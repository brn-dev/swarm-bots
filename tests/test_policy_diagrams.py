from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest

from swarmbots.tools.diagram_policy_architectures import (
    ComponentBlock,
    Node,
    build_diagram,
    configuration,
    mermaid_diagram,
    svg_diagram,
    dot_diagram,
    generate,
    write_gallery,
)
from swarmbots.learn import list_model_scales, list_variants
from swarmbots.tools.policy_diagram_layout import crosses_bounds, edge_points, layout_diagram, node_bounds

ROOT = Path(__file__).resolve().parents[1]
REPORTS = [ROOT / "docs/policy_parameters" / folder / "counts.json.gz" for folder in ["", "2.5M", "10M"]]
ROWS = [row for path in REPORTS for row in json.loads(gzip.decompress(path.read_bytes()))["policies"]]


def test_gallery_resolves_local_or_published_reading_guide(tmp_path: Path) -> None:
    diagram = build_diagram(ROWS[0])
    write_gallery([diagram], tmp_path)
    assert 'href="https://github.com/brn-dev/swarm-bots/blob/main/docs/policy_parameters/diagrams/README.md"' in (
        tmp_path / "index.html"
    ).read_text(encoding="utf-8")
    (tmp_path / "README.md").write_text("Local guide", encoding="utf-8")
    write_gallery([diagram], tmp_path)
    assert 'href="README.md"' in (tmp_path / "index.html").read_text(encoding="utf-8")


def assert_line_order(node: Node, *prefixes: str) -> None:
    positions = [next(i for i, line in enumerate(node.lines) if line.startswith(prefix)) for prefix in prefixes]
    assert positions == sorted(positions), (node.id, prefixes, node.lines)


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["variant"] + "-" + row["model_scale"]["name"])
def test_all_core_diagrams_render_with_valid_dimensions_and_privilege_boundary(row: dict[str, Any]) -> None:
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
            assert_line_order(node, "obs MLP (once):", "Encoder block ×", "FF / block:", "output:")
        if any(line.startswith("local/action MLP (once):") for line in node.lines):
            assert_line_order(
                node,
                "local/action MLP (once):",
                "global MLP (once):",
                "add local",
                "Encoder block ×",
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
                node, "coembed (per step):", "Transition encoder block ×", "attention / block:", "FF / block:", "head:", "latent residual;"
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
def test_autoregressive_decoders_only_feedback_previous_agent_actions(row: dict[str, Any]) -> None:
    d = build_diagram(row)
    feedback = next(edge for edge in d.edges if edge.source == "actions" and edge.target == "actor")
    assert feedback.kind == "causal"
    assert feedback.label == "previous agents only"
    actor = next(node for node in d.nodes if node.id == "actor")
    assert_line_order(
        actor,
        "query projection (once):" if "qcx" in row["variant"] else "obs projection (once):",
        "action embedding (once):",
        "Decoder block ×",
        "FF / block:",
        "output:",
    )
    if "qcx" in row["variant"] and row["model_scale"]["name"] == "5M NOP1M":
        assert "FF / block: 192 → 256 → 192 (98,752)" in actor.lines


def test_default_tmasac_labels_show_exact_subtotals_and_whole_components() -> None:
    row = next(row for row in ROWS if row["variant"] == "tmasac" and row["model_scale"]["name"] == "5M NOP1M")
    d = build_diagram(row)
    nodes = {node.id: node for node in d.nodes}
    assert "element MLP (once): 384 → 128 (49,280)" in nodes["head1"].lines
    assert "Set interaction × 2" in nodes["head1"].lines
    assert "u = LayerNorm(h)" in nodes["head1"].lines
    assert "l = h + local MLP(u)" in nodes["head1"].lines
    assert "regressor: 128 → 128 → 128 → 1 (33,153)" in nodes["head1"].lines
    assert nodes["head1"].parameters == 49_280 + 2 * 99_328 + 33_153
    assert nodes["head2"].parameters == nodes["head1"].parameters
    assert nodes["critic_encoder"].parameters == 2_305_408
    assert nodes["critic_encoder"].parameters + 2 * nodes["head1"].parameters == row["roles"]["critic"]["total"]
    assert "Per block: 987,008 parameters" in nodes["critic_encoder"].lines


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["variant"] + "-" + row["model_scale"]["name"])
def test_repeated_operations_are_scoped_separately_from_projections_and_readouts(row: dict[str, Any]) -> None:
    diagram = build_diagram(row)
    for node in diagram.nodes:
        repeated = [item for item in node.content if isinstance(item, ComponentBlock) and item.repeat is not None]
        for block in repeated:
            assert block.repeat >= 1
            assert block.parameters is not None
            assert all("(once)" not in line and not line.startswith(("output:", "regressor:")) for line in block.lines)
        if any("element MLP (once):" in line for line in node.lines):
            interaction, = repeated
            assert interaction.title == "Set interaction"
            assert interaction.repeat == (4 if row["variant"].endswith("_deepset") else 2)
            local, messages = [item for item in interaction.content if isinstance(item, ComponentBlock)]
            assert "local MLP:" in local.content[0]
            assert "masked mean of message MLP(u)" in messages.content
            assert any(line.startswith("broadcast g:") for line in messages.content)
            assert "merge: h_next = l + g" in interaction.content
            readout = next(item for item in node.content if isinstance(item, ComponentBlock) and item.repeat is None)
            assert "final masked mean across agents" in readout.content
            assert readout.content[-1].startswith("regressor:")
            assert node.content.index(interaction) < node.content.index(readout)


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["variant"] + "-" + row["model_scale"]["name"])
def test_connections_and_labels_leave_components_unobstructed(row: dict[str, Any]) -> None:
    diagram = build_diagram(row)
    nodes = {node.id: node for node in diagram.nodes}
    labels = [edge.label_box for edge in diagram.edges if edge.label_box is not None]
    for edge in diagram.edges:
        points = edge_points(edge, nodes)
        assert all(0 <= x <= diagram.width and 95 <= y < diagram.height - 20 for x, y in points)
        assert points[0] != points[-1]
        for start, end in zip(points, points[1:]):
            assert start != end
            assert start[0] == end[0] or start[1] == end[1], (edge, start, end)
            assert not any(crosses_bounds(start, end, node_bounds(node)) for node in diagram.nodes), edge
            assert not any(crosses_bounds(start, end, label) for label in labels if label is not edge.label_box), edge
        assert bool(edge.label) == (edge.label_box is not None)
    for index, label in enumerate(labels):
        assert 0 <= label.x and label.x + label.width <= diagram.width
        assert 95 <= label.y and label.y + label.height < diagram.height - 20
        assert not any(label.overlaps(node_bounds(node)) for node in diagram.nodes)
        assert not any(label.overlaps(other) for other in labels[index + 1:])


def test_layout_expands_for_a_critic_encoder_taller_than_its_readout() -> None:
    diagram = build_diagram(next(row for row in ROWS if row["variant"] == "mat_dec"))
    nodes = {node.id: node for node in diagram.nodes}
    encoder = nodes["representation"]
    encoder.content.extend(["Additional architecture detail"] * 40)
    layout_diagram(diagram)
    assert encoder.height > nodes["value"].height
    assert nodes["actor_encoder"].y + nodes["actor_encoder"].height < encoder.y
    assert encoder.y + encoder.height < nodes["nop_bridge"].y
    for edge in diagram.edges:
        points = edge_points(edge, nodes)
        assert not any(
            crosses_bounds(start, end, node_bounds(node))
            for start, end in zip(points, points[1:]) for node in diagram.nodes
        )


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["variant"] + "-" + row["model_scale"]["name"])
def test_major_paths_follow_separate_actor_critic_and_prediction_rows(row: dict[str, Any]) -> None:
    diagram = build_diagram(row)
    nodes = {node.id: node for node in diagram.nodes}
    actor_ids = {"public", "actor_encoder", "actor", "actions"}
    if "representation" in nodes and nodes["representation"].role == "shared":
        actor_ids.add("representation")
    actor_bottom = max(node.y + node.height for node in diagram.nodes if node.id in actor_ids)
    critic_nodes = [node for node in diagram.nodes if node.role == "critic" and node.id != "state_bridge"]
    assert actor_bottom < min(node.y for node in critic_nodes)
    if row["use_nop"]:
        assert max(node.y + node.height for node in critic_nodes) < min(
            node.y for node in diagram.nodes if node.role == "nop"
        )
    for encoder, output in (("actor", "actions"), ("value", "v"), ("head1", "q1"), ("head2", "q2")):
        if encoder in nodes:
            assert nodes[encoder].x + nodes[encoder].width < nodes[output].x


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
