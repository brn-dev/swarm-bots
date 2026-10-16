from __future__ import annotations

from collections.abc import Iterator
import gzip
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest

from swarmbots.tools.diagram_policy_architectures import (
    ComponentBlock,
    ContentText,
    Diagram,
    Node,
    build_diagram,
    configuration,
    mermaid_diagram,
    svg_diagram,
    dot_diagram,
    generate,
    write_gallery,
)
from swarmbots.learn import list_variants
from swarmbots.learn.presets.model_scale import DEFAULT_MODEL_SCALE, SCALE_LAYOUTS
from swarmbots.tools.policy_diagram_layout import crosses_bounds, edge_points, layout_diagram, node_bounds

ROOT = Path(__file__).resolve().parents[1]
REPORTS = [
    ROOT / "docs/policy_parameters" / SCALE_LAYOUTS[scale].slug / "counts.json.gz"
    for scale in (DEFAULT_MODEL_SCALE, *(scale for scale in SCALE_LAYOUTS if scale != DEFAULT_MODEL_SCALE))
]
ROWS = [row for path in REPORTS for row in json.loads(gzip.decompress(path.read_bytes()))["policies"]]


@pytest.fixture(scope="module")
def single_thread() -> Iterator[None]:
    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


def test_gallery_resolves_local_or_published_reading_guide(tmp_path: Path) -> None:
    diagram = build_diagram(ROWS[0])
    write_gallery([diagram], tmp_path)
    assert 'href="https://github.com/brn-dev/swarm-bots/blob/main/docs/policy_parameters/diagrams/README.md"' in (
        tmp_path / "index.html"
    ).read_text(encoding="utf-8")
    (tmp_path / "README.md").write_text("Local guide", encoding="utf-8")
    write_gallery([diagram], tmp_path)
    assert 'href="README.md"' in (tmp_path / "index.html").read_text(encoding="utf-8")


def test_depth_choices_export_to_distinct_files_and_gallery_entries(tmp_path: Path) -> None:
    rows = [row for row in ROWS if row["variant"] == "tmasac" and row["model_scale"]["name"].startswith("5M-")]
    diagrams = generate(rows, tmp_path)
    assert len(diagrams) == 2
    for depth in (2, 3):
        svg_path = tmp_path / f"5M-{depth}L/tmasac.svg"
        assert svg_path.is_file()
        root = ElementTree.fromstring(svg_path.read_text(encoding="utf-8"))
        assert f"Encoder block × {depth}" in "".join(root.itertext())
    gallery = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "5M-2L/tmasac" in gallery and "5M-3L/tmasac" in gallery


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
    assert (("future_actions", "transition") in {(edge.source, edge.target) for edge in d.edges}) is row["use_nop"]
    assert ("actions", "transition") not in {(edge.source, edge.target) for edge in d.edges}
    for role in ("actor", "critic", "shared_encoder", "next_obs_prediction"):
        assert (
            sum(node.parameters or 0 for node in d.nodes if node.parameter_role == role) == row["roles"][role]["total"]
        )
    assert sum(node.parameters or 0 for node in d.nodes) == row["trainable"]
    assert d.parameter_summary in "".join(root.itertext())
    for node in d.nodes:
        assert [item.text for item in node.content_layout[0] if isinstance(item, ContentText)] == node.lines
        if node.parameters is not None:
            assert node.parameter_label in mermaid_diagram(d)
            assert node.parameter_label in dot_diagram(d)
        if any(line.startswith("obs MLP (once):") for line in node.lines):
            assert_line_order(node, "obs MLP (once):", "Encoder block ×", "FF / block:", "output:")
        if any(line.startswith("local/action MLP (once):") for line in node.lines):
            communication = (
                "Global interaction · across agents"
                if any("expanded message width" in line for line in node.lines) else "attention / block:"
            )
            assert_line_order(
                node,
                "local/action MLP (once):",
                "global MLP (once):",
                "add local",
                "Encoder block ×",
                communication,
                "FF / block:",
                "output:",
            )
        temporal = next(
            (line.split(" / block:")[0] for line in node.lines if line.startswith(("LSTM / block:", "sLSTM / block:"))),
            None,
        )
        if temporal:
            policy_config = configuration(row)
            config_key = (
                "critic_encoder_config" if node.id == "critic_encoder"
                else "encoder_config" if "encoder_config" in policy_config
                else "actor_encoder_config"
            )
            cfg = policy_config[config_key]
            first, second = (
                (temporal, "attention") if cfg["temporal_model_order"] == "temporal_first" else ("attention", temporal)
            )
            if cfg.get("inter_module_mlp"):
                assert_line_order(node, first + " / block:", "inter FF / block:", second + " / block:", "FF / block:")
            else:
                assert_line_order(node, first + " / block:", second + " / block:", "FF / block:")
        if node.id == "transition":
            assert_line_order(
                node, "coembed (per step):", "Transition encoder block ×", "attention / block:", "FF / block:", "head:", "latent residual;"
            )
        if node.id == "state_bridge":
            assert node.lines[0] == "detach last actor state"
    if row["variant"] == "tmasac__ac_shared_observation_encoder":
        actor = next(node for node in d.nodes if node.id == "actor_encoder")
        assert_line_order(actor, "FF / block:", "head:", "output:")
        assert actor.lines[-1] == f"output: N × {configuration(row)['actor_head_config']['hidden_dims'][-1]}"


@pytest.mark.parametrize(
    "row",
    [
        row
        for row in ROWS
        if row["variant"] in ["tmasac", "tmatd3", "tmasac__slstm", "tmasac__lstm"]
    ],
    ids=lambda row: row["variant"] + "-" + row["model_scale"]["name"],
)
def test_transformer_q_heads_share_one_encoder_and_recurrent_inputs_are_detached(row):
    d = build_diagram(row)
    edges = {(edge.source, edge.target): edge for edge in d.edges}
    assert ("critic_encoder", "head1") in edges and ("critic_encoder", "head2") in edges
    assert len([node for node in d.nodes if node.id == "critic_encoder"]) == 1
    if row["variant"] in ["tmasac__lstm", "tmasac__slstm"]:
        assert edges["state_bridge", "critic_encoder"].kind == "state"
        assert ("actor_encoder", "actor_encoder") in edges
    else:
        assert "state_bridge" not in {node.id for node in d.nodes}


@pytest.mark.parametrize("row", [row for row in ROWS if row["variant"] in ["matd3__deepset_critic", "masac__deepset_critic"]])
def test_deepset_twin_critics_share_encoder_and_pool_inside_private_heads(row: dict[str, Any]) -> None:
    d = build_diagram(row)
    edges = {(edge.source, edge.target) for edge in d.edges}
    nodes = {node.id: node for node in d.nodes}
    assert ("q_inputs", "critic_encoder") in edges
    assert ("critic_encoder", "head1") in edges and ("critic_encoder", "head2") in edges
    assert ("critic_encoder", "nop_bridge") in edges
    assert not {"q1_network", "q2_network", "q_features", "critic_encoder2"} & nodes.keys()
    assert not any("Team pooling" in line or "final masked mean" in line for line in nodes["critic_encoder"].lines)
    for id in ("head1", "head2"):
        assert any("Set interaction" in line for line in nodes[id].lines)
        assert "final masked mean across agents" in nodes[id].lines
        assert any(line.startswith("regressor:") for line in nodes[id].lines)


@pytest.mark.parametrize("row", [row for row in ROWS if row["variant"] in ["mat_orig", "mat_qcx", "mat_qcx__lstm"]])
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
    if "qcx" in row["variant"] and row["model_scale"]["name"] == "5M-2L NOP1M":
        assert "FF / block: 192 → 256 → 192 (98,752)" in actor.lines


def test_default_tmasac_labels_show_exact_subtotals_and_whole_components() -> None:
    row = next(row for row in ROWS if row["variant"] == "tmasac" and row["model_scale"]["name"] == "5M-2L NOP1M")
    d = build_diagram(row)
    nodes = {node.id: node for node in d.nodes}
    assert "element MLP (once): 384 → 128 (49,280)" in nodes["head1"].lines
    assert "Set interaction × 2" in nodes["head1"].lines
    assert "z = global LayerNorm(h)" in nodes["head1"].lines
    assert "global residual: u = h + g" in nodes["head1"].lines
    assert "v = local LayerNorm(u)" in nodes["head1"].lines
    assert "local residual: h_next = u + local MLP(v)" in nodes["head1"].lines
    assert "regressor: 128 → 128 → 128 → 1 (33,153)" in nodes["head1"].lines
    assert nodes["head1"].parameters == 49_280 + 2 * 100_416 + 33_153
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
            assert interaction.repeat == 2
            messages, local = [item for item in interaction.content if isinstance(item, ComponentBlock)]
            assert local.content[0] == "v = local LayerNorm(u)"
            assert "local MLP:" in local.content[1]
            assert "masked mean in the expanded message space" in messages.content
            assert any(line.startswith("linear down-projection (bias-free):") for line in messages.lines)
            assert any(line.startswith("nonlinear expansion before pooling:") for line in messages.lines)
            assert any(line.startswith("nonlinear pooled-message processing:") for line in messages.lines)
            assert messages.content[-1] == "global residual: u = h + g"
            assert local.content[-1] == "local residual: h_next = u + local MLP(v)"
            assert_line_order(node, "global residual:", "v = local LayerNorm(u)", "local residual:")
            readout = next(item for item in node.content if isinstance(item, ComponentBlock) and item.repeat is None)
            assert "final masked mean across agents" in readout.content
            assert readout.content[-1].startswith("output:" if node.id.startswith("critic_encoder") else "regressor:")
            assert node.content.index(interaction) < node.content.index(readout)


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["variant"] + "-" + row["model_scale"]["name"])
def test_connections_and_labels_leave_components_unobstructed(row: dict[str, Any]) -> None:
    assert_connections_and_labels_leave_components_unobstructed(build_diagram(row))


def assert_connections_and_labels_leave_components_unobstructed(diagram: Diagram) -> None:
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


@pytest.mark.parametrize("vertical", [False, True])
def test_connection_labels_find_clear_space_near_segment_ends(vertical: bool) -> None:
    from swarmbots.tools.policy_diagram_layout import Bounds, place_label

    points = [(0, 100), (1000, 100)]
    canvas = Bounds(0, 0, 1000, 200)
    obstacles = [Bounds(200, 0, 800, 92), Bounds(200, 108, 800, 92)]
    if vertical:
        points = [(y, x) for x, y in points]
        canvas = Bounds(canvas.y, canvas.x, canvas.height, canvas.width)
        obstacles = [Bounds(box.y, box.x, box.height, box.width) for box in obstacles]

    label = place_label("global context after pooling", points, obstacles, [], canvas)

    assert not any(label.overlaps(box) for box in obstacles)
    assert canvas.x <= label.x and label.x + label.width <= canvas.x + canvas.width
    assert canvas.y <= label.y and label.y + label.height <= canvas.y + canvas.height


def test_layout_expands_for_a_critic_encoder_taller_than_its_readout() -> None:
    diagram = build_diagram(next(row for row in ROWS if row["variant"] == "mat_dec"))
    nodes = {node.id: node for node in diagram.nodes}
    encoder = nodes["critic_encoder"]
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


def test_ddpg_has_one_standard_head_and_uses_encoder_features_for_nop() -> None:
    row = next(row for row in ROWS if row["variant"] == "maddpg__deepset_critic" and row["model_scale"]["name"] == "5M-2L NOP1M")
    d = build_diagram(row)
    nodes = {node.id: node for node in d.nodes}
    edges = {(edge.source, edge.target) for edge in d.edges}
    assert "critic_encoder" in nodes and "head1" in nodes
    assert "head2" not in nodes and "q_features" not in nodes
    assert ("critic_encoder", "head1") in edges and ("critic_encoder", "nop_bridge") in edges
    assert "final masked mean across agents" in nodes["head1"].lines


def test_refresh_keeps_compact_inputs_and_writes_verbose_reports_to_generated(tmp_path, monkeypatch):
    from swarmbots.tools import diagram_policy_architectures as diagrams

    report = json.loads(gzip.decompress(REPORTS[0].read_bytes()))
    guide = tmp_path / "docs/policy_parameters/model_scale.md"
    guide.parent.mkdir(parents=True)
    guide.write_text("Sizing guide", encoding="utf-8")
    def inspect(benchmark_id, variants, *, model_scale):
        assert benchmark_id == "SwarmBots-WallEasy-v0"
        assert variants == list_variants()
        assert model_scale == "5M-2L NOP1M"
        return report

    monkeypatch.setattr(diagrams, "ROOT", tmp_path)
    monkeypatch.setattr(diagrams, "inspect_variants", inspect)
    rows = diagrams.load_reports(["5M-2L NOP1M"], ["tmasac"], refresh=True)
    assert [row["variant"] for row in rows] == ["tmasac"]
    docs = tmp_path / "docs/policy_parameters"
    assert json.loads(gzip.decompress((docs / "5M-2L/counts.json.gz").read_bytes())) == report
    assert not (docs / "report.md").exists()
    assert not (docs / "counts.csv").exists()
    generated = docs / "generated/5M-2L"
    assert (generated / "counts.csv").exists()
    assert (generated / "layer_layout.csv").exists()
    assert "[model-scale allocation](../../model_scale.md)" in (generated / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("report_path", REPORTS, ids=lambda path: path.parent.name)
@pytest.mark.parametrize("refresh", [False, True])
def test_partial_audit_rebuild_preserves_core_and_cached_hidden_presets(tmp_path, monkeypatch, report_path, refresh):
    from swarmbots.tools import diagram_policy_architectures as diagrams
    from swarmbots.tools.inspect_policy_parameters import write_compressed_json

    report = json.loads(gzip.decompress(report_path.read_bytes()))
    scale = report["policies"][0]["model_scale"]["name"]
    hidden_name = "tmasac__no_nop"
    hidden_row = {**next(row for row in report["policies"] if row["variant"] == "tmasac__slstm"), "variant": hidden_name}
    report["policies"] = [row for row in report["policies"] if row["variant"] in list_variants()]
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
@pytest.mark.usefixtures("single_thread")
@pytest.mark.parametrize("independent_encoders", [False, True])
@pytest.mark.parametrize("actor_state_input", [False, True])
def test_recurrent_deepset_diagrams_preserve_actor_state_inputs_and_parameter_totals(
    tmp_path: Path, independent_encoders: bool, actor_state_input: bool,
) -> None:
    from swarmbots.tools.inspect_policy_parameters import inspect_variants

    report = inspect_variants(
        "SwarmBots-WallEasy-v0", ("matd3__deepset_critic",), model_scale="2.5M",
        policy_kwargs={
            "td3_recurrent_actor": True,
            "critic_independent_encoders": independent_encoders,
            "td3_actor_state_critic_input_config": "auto" if actor_state_input else None,
        },
    )
    assert not report["errors"]
    row, = report["policies"]
    diagram, = generate(report["policies"], tmp_path)
    nodes = {node.id: node for node in diagram.nodes}
    edges = {(edge.source, edge.target) for edge in diagram.edges}
    assert ("state_bridge" in nodes) is actor_state_input
    if actor_state_input:
        assert nodes["state_bridge"].parameters == row["module_parameter_counts"]["critic.actor_state_encoder"]
        assert "detach last actor state" in nodes["state_bridge"].lines
        assert ("actor_encoder", "state_bridge") in edges
        assert ("state_bridge", "critic_encoder") in edges
        assert (("state_bridge", "critic_encoder2") in edges) is independent_encoders
    for role, counts in diagram.role_parameters.items():
        assert sum(node.parameters or 0 for node in diagram.nodes if node.parameter_role == role) == counts
    assert sum(node.parameters or 0 for node in diagram.nodes) == row["trainable"]
    for edge in diagram.edges:
        assert not any(
            crosses_bounds(start, end, node_bounds(node))
            for start, end in zip(edge_points(edge, nodes), edge_points(edge, nodes)[1:])
            for node in diagram.nodes
        )
    svg_path, = tmp_path.glob("*/matd3__deepset_critic.svg")
    assert ElementTree.fromstring(svg_path.read_text(encoding="utf-8")).tag.endswith("svg")


@pytest.mark.integration
@pytest.mark.usefixtures("single_thread")
def test_independent_deepset_encoders_feed_private_readouts_and_concatenated_nop(tmp_path: Path) -> None:
    from swarmbots.learn.nn_components.deep_set import DeepSetInteractionConfig
    from swarmbots.tools.inspect_policy_parameters import inspect_variants

    report = inspect_variants(
        "SwarmBots-WallEasy-v0", ("masac__deepset_critic", "matd3__deepset_critic"), model_scale="2.5M",
        policy_kwargs={
            "critic_independent_encoders": True,
            "deepset_interaction_config": DeepSetInteractionConfig(local_update="residual", normalize=False),
        },
    )
    assert not report["errors"]
    for diagram, row in zip(generate(report["policies"], tmp_path), report["policies"], strict=True):
        nodes = {node.id: node for node in diagram.nodes}
        edges = {(edge.source, edge.target) for edge in diagram.edges}
        assert ("q_inputs", "critic_encoder") in edges and ("q_inputs", "critic_encoder2") in edges
        assert ("critic_encoder", "head1") in edges and ("critic_encoder2", "head2") in edges
        assert ("critic_encoder", "head2") not in edges and ("critic_encoder2", "head1") not in edges
        assert ("critic_encoder", "q_features") in edges and ("critic_encoder2", "q_features") in edges
        assert ("q_features", "nop_bridge") in edges
        assert nodes["q_features"].parameters == 0
        assert sum(node.parameters or 0 for node in diagram.nodes) == row["trainable"]
        for id in ("critic_encoder", "critic_encoder2"):
            assert "z = global LayerNorm(h)" in nodes[id].lines
            assert "v = local LayerNorm(u)" in nodes[id].lines
            assert "local residual: h_next = u + FF(v)" in nodes[id].lines
        for edge in diagram.edges:
            assert not any(
                crosses_bounds(start, end, node_bounds(node))
                for start, end in zip(edge_points(edge, nodes), edge_points(edge, nodes)[1:])
                for node in diagram.nodes
            )


@pytest.mark.integration
@pytest.mark.usefixtures("single_thread")
@pytest.mark.parametrize("model_scale", ["2.5M", "5M-2L", "5M-3L", "10M"])
def test_expanding_and_gated_deepset_diagrams_show_reception_and_complete_counts(model_scale: str) -> None:
    from swarmbots.tools.inspect_policy_parameters import inspect_variants

    bases = ("maddpg__deepset_critic", "matd3__deepset_critic", "masac__deepset_critic")
    report = inspect_variants(
        "SwarmBots-WallEasy-v0", (*bases, *(base + "__gated" for base in bases)), model_scale=model_scale,
    )
    assert not report["errors"]
    for row in report["policies"]:
        diagram = build_diagram(row)
        nodes = {node.id: node for node in diagram.nodes}
        gated = row["variant"].endswith("__gated")
        encoder = nodes["critic_encoder"]
        assert any("expanded message width M=" in line for line in encoder.lines)
        assert any(line.startswith("nonlinear expansion before pooling:") for line in encoder.lines)
        assert any(line.startswith("nonlinear pooled-message processing:") for line in encoder.lines)
        assert any(line.startswith("linear down-projection (bias-free):") for line in encoder.lines)
        assert ("Per-token reception gate" in encoder.lines) is gated
        if gated:
            assert_line_order(
                encoder, "nonlinear expansion before pooling:", "masked mean in the expanded message space",
                "nonlinear pooled-message processing:", "Per-token reception gate",
                "multiply expanded message", "linear down-projection", "global residual:", "FF / block:",
            )
        assert ("head2" in nodes) is not row["variant"].startswith("maddpg")
        for node in nodes.values():
            if node.id.startswith("head"):
                assert "final masked mean across agents" in node.lines
        assert sum(node.parameters or 0 for node in nodes.values()) == row["trainable"]
        for edge in diagram.edges:
            assert not any(
                crosses_bounds(start, end, node_bounds(node))
                for start, end in zip(edge_points(edge, nodes), edge_points(edge, nodes)[1:])
                for node in nodes.values()
            )
        assert ElementTree.fromstring(svg_diagram(diagram)).tag.endswith("svg")


@pytest.mark.integration
@pytest.mark.usefixtures("single_thread")
@pytest.mark.parametrize("joint_obs_embedding", [False, True])
def test_diagrams_show_optional_input_encoders_and_token_norm_in_forward_order(
    joint_obs_embedding: bool,
) -> None:
    from swarmbots.learn.presets.transformer import MATNormalizationConfig
    from swarmbots.tools.inspect_policy_parameters import inspect_variants

    report = inspect_variants(
        "SwarmBots-WallEasy-v0",
        ("maddpg__deepset_critic", "matd3__deepset_critic", "masac__deepset_critic",
         "tmatd3", "tmasac", "mat_ind", "mat_ind__lstm"),
        model_scale="2.5M",
        policy_kwargs={
            "tmasac_separate_observation_action_encoders": True,
            "mat_normalization": MATNormalizationConfig(normalize_encoder_tokens=True),
            "mat_joint_obs_embedding": joint_obs_embedding,
        },
    )
    assert not report["errors"]
    for row in report["policies"]:
        diagram = build_diagram(row)
        nodes = {node.id: node for node in diagram.nodes}
        for node_id in ("actor_encoder", "critic_encoder"):
            node = nodes[node_id]
            assert_line_order(node, "token LayerNorm (once):", "Encoder block ×", "output:")
            token_norm, = [line for line in node.content if isinstance(line, str) and line.startswith("token LayerNorm")]
            block = next(item for item in node.content if isinstance(item, ComponentBlock) and item.title == "Encoder block")
            assert token_norm not in block.lines
        if row["variant"].startswith("mat_ind"):
            continue
        encoder = nodes["critic_encoder"]
        assert_line_order(
            encoder, "observation MLP (once):", "action MLP (once):",
            "concatenate observation and action features", "local/action MLP (once):",
            "token LayerNorm (once):", "Encoder block ×",
        )
        if joint_obs_embedding:
            assert_line_order(
                encoder, "concatenate observation and action features",
                "concatenate local/action and global inputs", "local/action MLP (once):",
            )
        assert sum(node.parameters or 0 for node in diagram.nodes) == row["trainable"]
        for edge in diagram.edges:
            points = edge_points(edge, nodes)
            assert not any(
                crosses_bounds(start, end, node_bounds(node))
                for start, end in zip(points, points[1:]) for node in diagram.nodes
            )
        assert ElementTree.fromstring(svg_diagram(diagram)).tag.endswith("svg")


@pytest.mark.integration
@pytest.mark.usefixtures("single_thread")
@pytest.mark.parametrize(
    ("separate_inputs", "normalize_tokens", "gated"), [(False, True, False), (True, False, True), (True, True, True)],
)
def test_independent_critic_diagrams_keep_late_global_context_labels_clear_with_optional_controls(
    separate_inputs: bool, normalize_tokens: bool, gated: bool,
) -> None:
    from swarmbots.learn.presets.transformer import MATNormalizationConfig
    from swarmbots.tools.inspect_policy_parameters import inspect_variants

    suffix = "__gated" if gated else ""
    report = inspect_variants(
        "SwarmBots-WallEasy-v0",
        tuple(base + suffix for base in ("matd3__deepset_critic", "masac__deepset_critic")),
        model_scale="2.5M",
        policy_kwargs={
            "baseline_critic_context_in_elements": False,
            "critic_independent_encoders": True,
            "mat_joint_obs_embedding": separate_inputs,
            "tmasac_separate_observation_action_encoders": separate_inputs,
            "mat_normalization": MATNormalizationConfig(normalize_encoder_tokens=normalize_tokens),
        },
    )
    assert not report["errors"]
    for row in report["policies"]:
        diagram = build_diagram(row)
        assert_connections_and_labels_leave_components_unobstructed(diagram)
        context_edges = [edge for edge in diagram.edges if edge.label == "global context after pooling"]
        assert {edge.target for edge in context_edges} == {"head1", "head2"}
        assert sum(node.parameters or 0 for node in diagram.nodes) == row["trainable"]


@pytest.mark.integration
@pytest.mark.usefixtures("single_thread")
@pytest.mark.parametrize("context_in_elements", [False, True])
@pytest.mark.parametrize("joint_obs_embedding", [False, True])
def test_critic_diagrams_describe_actual_global_context_placement(
    context_in_elements: bool, joint_obs_embedding: bool,
) -> None:
    from swarmbots.tools.inspect_policy_parameters import inspect_variants

    report = inspect_variants(
        "SwarmBots-WallEasy-v0", ("maddpg__deepset_critic", "masac__deepset_critic"), model_scale="2.5M",
        policy_kwargs={
            "baseline_critic_context_in_elements": context_in_elements,
            "mat_joint_obs_embedding": joint_obs_embedding,
        },
    )
    assert not report["errors"]
    for row in report["policies"]:
        diagram = build_diagram(row)
        encoder = next(node for node in diagram.nodes if node.id == "critic_encoder")
        assert ("add local and global token projections" in encoder.lines) is (
            context_in_elements and not joint_obs_embedding
        )
        assert ("concatenate local/action and global inputs before projection" in encoder.lines) is (
            context_in_elements and joint_obs_embedding
        )
        context_edges = [edge for edge in diagram.edges if edge.label == "global context after pooling"]
        assert bool(context_edges) is not context_in_elements
        assert all(edge.target.startswith("head") for edge in context_edges)
