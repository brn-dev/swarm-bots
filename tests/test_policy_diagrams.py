from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from functools import cache
from itertools import combinations, product
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
    Edge,
    Node,
    build_diagram,
    configuration,
    mermaid_diagram,
    png_preview,
    svg_diagram,
    dot_diagram,
    generate,
    write_gallery,
)
from swarmbots.learn import list_variants
from swarmbots.learn.presets.model_scale import DEFAULT_MODEL_SCALE, SCALE_LAYOUTS
from swarmbots.tools.policy_diagram_layout import connection_paths, crosses_bounds, edge_points, layout_diagram, node_bounds
from swarmbots.tools.inspect_policy_parameters import inspect_variants

PolicyRow = Callable[[str, str], dict[str, Any]]
SCALES = (DEFAULT_MODEL_SCALE, *(scale for scale in SCALE_LAYOUTS if scale != DEFAULT_MODEL_SCALE))


def policy_cases(*variants: str) -> list[Any]:
    return [
        pytest.param((variant, scale), id=f"{variant}-{scale}")
        for scale in SCALES
        for variant in (variants or list_variants(include_hidden=True))
    ]


@pytest.fixture(scope="session")
def single_thread() -> Iterator[None]:
    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous_threads)


@pytest.fixture(scope="session")
def policy_row(single_thread: None) -> PolicyRow:
    @cache
    def inspect(variant: str, model_scale: str) -> dict[str, Any]:
        report = inspect_variants("SwarmBots-WallEasy-v0", (variant,), model_scale=model_scale)
        assert not report["errors"]
        row, = report["policies"]
        return row

    return inspect


@pytest.fixture(scope="session")
def row(request: pytest.FixtureRequest, policy_row: PolicyRow) -> dict[str, Any]:
    return policy_row(*request.param)


@pytest.mark.integration
def test_gallery_resolves_local_or_published_reading_guide(tmp_path: Path, policy_row: PolicyRow) -> None:
    diagram = build_diagram(policy_row("tmasac", DEFAULT_MODEL_SCALE))
    write_gallery([diagram], tmp_path)
    assert 'href="https://github.com/brn-dev/swarm-bots/blob/main/docs/policy_parameters/diagrams/README.md"' in (
        tmp_path / "index.html"
    ).read_text(encoding="utf-8")
    (tmp_path / "README.md").write_text("Local guide", encoding="utf-8")
    write_gallery([diagram], tmp_path)
    assert 'href="README.md"' in (tmp_path / "index.html").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("selected", "expected"),
    [
        (
            ("10M-5L NOP2M", "5M-3L NOP1M", "10M-3L NOP2M", "2.5M NOP0.75M", "10M-4L NOP2M", "5M-2L NOP1M"),
            ("2.5M NOP0.75M", "5M-2L NOP1M", "5M-3L NOP1M", "10M-3L NOP2M", "10M-4L NOP2M", "10M-5L NOP2M"),
        ),
        (
            ("10M-5L NOP2M", "5M-2L NOP1M", "10M-3L NOP2M"),
            ("5M-2L NOP1M", "10M-3L NOP2M", "10M-5L NOP2M"),
        ),
    ],
)
def test_gallery_orders_selected_scales_by_budget_then_depth(
    tmp_path: Path, selected: tuple[str, ...], expected: tuple[str, ...],
) -> None:
    diagrams = [
        Diagram(
            variant="tmasac", scale=scale, shape={},
            role_parameters={"actor": 0, "critic": 0, "shared_encoder": 0},
        )
        for scale in selected
    ]
    write_gallery(diagrams, tmp_path)
    gallery = (tmp_path / "index.html").read_text(encoding="utf-8")
    options = gallery.split('<select id="scale">', 1)[1].split("</select>", 1)[0]
    select = ElementTree.fromstring("<select>" + options + "</select>")
    assert tuple(option.text for option in select) == expected


@pytest.mark.integration
@pytest.mark.parametrize(("budget", "depths"), [("5M", (2, 3)), ("10M", (3, 4, 5))])
def test_depth_choices_export_to_distinct_files_and_gallery_entries(
    tmp_path: Path, policy_row: PolicyRow, budget: str, depths: tuple[int, ...],
) -> None:
    rows = [policy_row("tmasac", scale) for scale in SCALES if scale.startswith(budget + "-")]
    diagrams = generate(rows, tmp_path)
    assert len(diagrams) == len(depths)
    gallery = (tmp_path / "index.html").read_text(encoding="utf-8")
    for depth in depths:
        name = f"{budget}-{depth}L/tmasac"
        svg_path = tmp_path / (name + ".svg")
        assert svg_path.is_file()
        root = ElementTree.fromstring(svg_path.read_text(encoding="utf-8"))
        assert f"Encoder block \u00d7 {depth}" in "".join(root.itertext())
        assert name in gallery


@pytest.mark.integration
@pytest.mark.parametrize(("budget", "maximum_main"), [("5M", 5_350_000), ("10M", 10_650_000)])
@pytest.mark.parametrize("variant", list_variants(include_hidden=True))
def test_fixed_presets_meet_main_budget_and_match_across_depths(
    policy_row: PolicyRow, variant: str, budget: str, maximum_main: int,
) -> None:
    scales = [scale for scale in SCALES if scale.startswith(budget + "-")]
    rows = [policy_row(variant, scale) for scale in scales]
    counts = [
        row["trainable_minus_nop"] - row["processing"]["actor_decoder_attention_parameters"] for row in rows
    ]
    reference_counts = [
        count + row["module_parameter_counts"]["critic.0.network"] if variant.startswith("maddpg__")
        else count + row["module_parameter_counts"]["critic"] if variant.startswith(("mat_", "mappo__"))
        else count
        for count, row in zip(counts, rows, strict=True)
    ]
    assert all(count >= SCALE_LAYOUTS[scales[0]].main_parameters for count in reference_counts)
    tolerance = 0.03 if budget == "10M" and "__gated_" in variant else 0.01
    assert all(abs(count / counts[0] - 1) < tolerance for count in counts[1:])
    if not variant.endswith(("__lstm", "__slstm")) and "__gated_" not in variant:
        assert all(count <= maximum_main for count in reference_counts)


def assert_line_order(node: Node, *prefixes: str) -> None:
    positions = [next(i for i, line in enumerate(node.lines) if line.startswith(prefix)) for prefix in prefixes]
    assert positions == sorted(positions), (node.id, prefixes, node.lines)


@pytest.mark.integration
@pytest.mark.parametrize("row", policy_cases(), indirect=True)
def test_all_core_diagrams_render_with_valid_dimensions_and_privilege_boundary(row: dict[str, Any]) -> None:
    d = build_diagram(row)
    root = ElementTree.fromstring(svg_diagram(d))
    assert root.tag.endswith("svg")
    assert mermaid_diagram(d).startswith("flowchart LR")
    assert dot_diagram(d).startswith("digraph policy")
    assert all("see layer table" not in " ".join(node.lines) for node in d.nodes)
    destinations = {"privileged"}
    edges = {(edge.source, edge.target) for edge in d.edges}
    for _ in range(len(d.nodes)):
        destinations |= {edge.target for edge in d.edges if edge.source in destinations}
    assert not destinations & {"actor", "actor_encoder", "actions"}
    assert (("future_actions", "transition") in edges) is row["use_nop"]
    assert ("actions", "transition") not in edges
    if row["use_nop"] and ("actor_config" in configuration(row) or "encoder_config" in configuration(row)):
        source = "representation" if row["variant"] == "ppo" else "critic_encoder"
        assert (source, "nop_bridge") in edges
        assert ("actor_encoder", "nop_bridge") not in edges
    if row["variant"] == "mappo__deepset_critic":
        assert row["roles"]["shared_encoder"]["total"] == 0
        assert {"actor_encoder", "critic_encoder"} <= {node.id for node in d.nodes}
        assert ("privileged", "critic_encoder") in edges
        assert ("privileged", "value") not in edges

    for role in ("actor", "critic", "shared_encoder", "next_obs_prediction"):
        assert (
            sum(node.parameters or 0 for node in d.nodes if node.parameter_role == role) == row["roles"][role]["total"]
        )
    assert sum(node.parameters or 0 for node in d.nodes) == row["trainable"]
    assert d.parameter_summary in "".join(root.itertext())
    if row["variant"] != "ppo":
        actor_encoder = next(node for node in d.nodes if node.id == "actor_encoder")
        assert f"Encoder block × {SCALE_LAYOUTS[row['model_scale']['name']].layers}" in actor_encoder.lines
        assert any("Depth label counts actor/critic encoder blocks" in note for note in d.notes)
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


@pytest.mark.integration
@pytest.mark.parametrize("row", policy_cases("mat_qcx", "mat_qcx__lstm"), indirect=True)
def test_qcx_diagram_includes_final_actor_projection_and_output_width(row: dict[str, Any]) -> None:
    diagram = build_diagram(row)
    actor = next(node for node in diagram.nodes if node.id == "actor")
    config = configuration(row)["decoder_config"]
    assert any(line.startswith("actor head (once):") for line in actor.lines)
    assert actor.lines[-1] == f"output: N × {config['actor_head_hidden_dims'][-1]}"
    assert "Decoder block × 1" in actor.lines
    assert_line_order(actor, "final decoder LayerNorm", "actor head (once):", "output:")
    encoder_width = configuration(row)["encoder_config"]["d_model"]
    decoder_width = config["d_model"]
    assert f"memory Q: {decoder_width} → {decoder_width}; K/V: {encoder_width} → {decoder_width}" in actor.lines
    extra = row["processing"]["actor_decoder_attention_parameters"]
    assert any(f"Decoder attention: {extra:,} additional parameters" in note for note in diagram.notes)


@pytest.mark.integration
@pytest.mark.parametrize("row", policy_cases("tmasac", "tmatd3", "tmasac__slstm", "tmasac__lstm"), indirect=True)
def test_transformer_q_heads_share_one_encoder_and_recurrent_inputs_are_detached(row: dict[str, Any]) -> None:
    d = build_diagram(row)
    edges = {(edge.source, edge.target): edge for edge in d.edges}
    assert ("critic_encoder", "head1") in edges and ("critic_encoder", "head2") in edges
    assert len([node for node in d.nodes if node.id == "critic_encoder"]) == 1
    if row["variant"] in ["tmasac__lstm", "tmasac__slstm"]:
        assert edges["state_bridge", "critic_encoder"].kind == "state"
        assert ("actor_encoder", "actor_encoder") in edges
    else:
        assert "state_bridge" not in {node.id for node in d.nodes}


@pytest.mark.integration
@pytest.mark.parametrize("row", policy_cases("matd3__deepset_critic", "masac__deepset_critic"), indirect=True)
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


@pytest.mark.integration
@pytest.mark.parametrize("row", policy_cases("mat_orig", "mat_qcx", "mat_qcx__lstm"), indirect=True)
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
    if "qcx" in row["variant"]:
        layers = {layer["module"]: layer for layer in row["layer_layout"]}
        first, second = (layers[f"decoder.layers.0.linear{index}"] for index in (1, 2))
        dimensions = first["dimensions"] + " -> " + second["dimensions"].split(" -> ")[-1]
        parameters = first["parameters"] + second["parameters"]
        assert f"FF / block: {dimensions.replace(' -> ', ' → ')} ({parameters:,})" in actor.lines


@pytest.mark.integration
def test_default_tmasac_labels_show_exact_subtotals_and_whole_components(policy_row: PolicyRow) -> None:
    row = policy_row("tmasac", DEFAULT_MODEL_SCALE)
    d = build_diagram(row)
    nodes = {node.id: node for node in d.nodes}
    layers = {layer["module"]: layer for layer in row["layer_layout"]}
    counts = row["module_parameter_counts"]
    for label, module in (("element MLP (once)", "element_encoder"), ("regressor", "set_decoder")):
        layer = layers[f"critic.q1.deepset.{module}"]
        dimensions = layer["dimensions"].replace(" -> ", " → ")
        assert f"{label}: {dimensions} ({layer['parameters']:,})" in nodes["head1"].lines
    assert "Set interaction × 2" in nodes["head1"].lines
    assert "z = global LayerNorm(h)" in nodes["head1"].lines
    assert "global residual: u = h + g" in nodes["head1"].lines
    assert "v = local LayerNorm(u)" in nodes["head1"].lines
    assert "local residual: h_next = u + local MLP(v)" in nodes["head1"].lines
    assert nodes["head1"].parameters == counts["critic.q1"]
    assert nodes["head2"].parameters == nodes["head1"].parameters
    assert nodes["critic_encoder"].parameters == counts["critic.encoder"]
    assert nodes["critic_encoder"].parameters + 2 * nodes["head1"].parameters == row["roles"]["critic"]["total"]
    assert f"Per block: {counts['critic.encoder.layers.0']:,} parameters" in nodes["critic_encoder"].lines


@pytest.mark.integration
@pytest.mark.parametrize("row", policy_cases(), indirect=True)
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


@pytest.mark.integration
@pytest.mark.parametrize("row", policy_cases(), indirect=True)
def test_connections_and_labels_leave_components_unobstructed(row: dict[str, Any]) -> None:
    diagram = build_diagram(row)
    assert_connections_and_labels_leave_components_unobstructed(diagram)
    nodes = {node.id: node for node in diagram.nodes}
    allowed_crossing = None
    if "state_bridge" in nodes:
        action_edge = next(edge for edge in diagram.edges if (edge.source, edge.target) == ("actions", "q_inputs"))
        state_edge = next(
            edge for edge in diagram.edges if (edge.source, edge.target) == ("state_bridge", "critic_encoder")
        )
        allowed_crossing = (action_edge, state_edge)
        assert max(y for _, y in edge_points(action_edge, nodes)) <= nodes["q_inputs"].y
    assert_connection_pairs_do_not_intersect(diagram, combinations(diagram.edges, 2), allowed_crossing=allowed_crossing)


@pytest.mark.integration
@pytest.mark.parametrize("row", policy_cases(
    "ppo", "mappo__deepset_critic", "mat_ind", "mat_dec", "mat_orig", "mat_qcx",
    "mat_ind__lstm", "mat_qcx__lstm", "mat_qcx__no_nop",
), indirect=True)
def test_on_policy_public_inputs_separate_actor_and_critic_branches(row: dict[str, Any]) -> None:
    diagram = build_diagram(row)
    nodes = {node.id: node for node in diagram.nodes}
    public_edges = [edge for edge in diagram.edges if edge.source == "public"]
    actor_edge, = [edge for edge in public_edges if nodes[edge.target].role in {"actor", "shared"}]
    actor_points = edge_points(actor_edge, nodes)
    actor_ys = [y for _, y in actor_points]
    assert max(actor_ys) - min(actor_ys) <= 1
    for edge in public_edges:
        if nodes[edge.target].role == "critic":
            critic_points = edge_points(edge, nodes)
            start, following = critic_points[:2]
            assert start[0] == following[0] and start[1] < following[1]
            assert min(y for _, y in critic_points) > max(actor_ys)


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
    assert_joint_critic_inputs_do_not_intersect_outputs(diagram)


def assert_joint_critic_inputs_do_not_intersect_outputs(diagram: Diagram) -> None:
    inputs = [edge for edge in diagram.edges if edge.target == "q_inputs"]
    outputs = [edge for edge in diagram.edges if edge.source == "q_inputs"]
    assert_connection_pairs_do_not_intersect(diagram, product(inputs, outputs))


def assert_connection_pairs_do_not_intersect(
    diagram: Diagram, pairs: Iterable[tuple[Edge, Edge]], *, allowed_crossing: tuple[Edge, Edge] | None = None,
) -> None:
    nodes = {node.id: node for node in diagram.nodes}
    crossings: set[tuple[float, float]] = set()
    for edge, other in pairs:
        points, other_points = edge_points(edge, nodes), edge_points(other, nodes)
        for start, end in zip(points, points[1:]):
            for other_start, other_end in zip(other_points, other_points[1:]):
                if any(
                    max(min(start[axis], end[axis]), min(other_start[axis], other_end[axis]))
                    > min(max(start[axis], end[axis]), max(other_start[axis], other_end[axis]))
                    for axis in (0, 1)
                ):
                    continue
                assert allowed_crossing in ((edge, other), (other, edge)), (
                    edge.source, edge.target, other.source, other.target, start, end, other_start, other_end,
                )
                assert (start[1] == end[1]) != (other_start[1] == other_end[1])
                horizontal, vertical = (
                    ((start, end), (other_start, other_end)) if start[1] == end[1]
                    else ((other_start, other_end), (start, end))
                )
                x, y = vertical[0][0], horizontal[0][1]
                assert min(horizontal[0][0], horizontal[1][0]) < x < max(horizontal[0][0], horizontal[1][0])
                assert min(vertical[0][1], vertical[1][1]) < y < max(vertical[0][1], vertical[1][1])
                crossings.add((x, y))
    assert len(crossings) <= 1


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


def _crossing_diagram(
    crossing_xs: tuple[int, ...] = (260,), *, reverse_arrow: bool = False, reverse_order: bool = False,
) -> Diagram:
    horizontal = Edge("right", "left", kind="causal", source_port="left", target_port="right") if reverse_arrow else (
        Edge("left", "right", kind="causal")
    )
    verticals = []
    for index, x in enumerate(crossing_xs, start=1):
        offset = index / (len(crossing_xs) + 1)
        port_x = 210 + 100 * offset
        verticals.append(Edge(
            "top", "bottom", kind="state", source_port="bottom", target_port="top",
            source_offset=offset, target_offset=offset,
            via=[(port_x, 200), (x, 200), (x, 380), (port_x, 380)] if x != port_x else [],
        ))
    return Diagram(
        "crossing", "test", {}, width=560, height=560,
        role_parameters={"actor": 0, "critic": 0, "shared_encoder": 0},
        nodes=[
            Node("left", "Left", [], "input", x=20, y=240, width=100),
            Node("right", "Right", [], "output", x=400, y=240, width=100),
            Node("top", "Top", [], "input", x=210, y=100, width=100),
            Node("bottom", "Bottom", [], "output", x=210, y=400, width=100),
        ],
        edges=[*verticals, horizontal] if reverse_order else [horizontal, *verticals],
    )


@pytest.mark.parametrize("reverse_arrow", [False, True])
@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("crossing_xs", [(260,), (255, 265)])
def test_crossed_dashed_arrows_show_a_solid_jump_in_both_renderers(
    tmp_path: Path, reverse_arrow: bool, reverse_order: bool, crossing_xs: tuple[int, ...],
) -> None:
    from PIL import Image

    diagram = _crossing_diagram(crossing_xs, reverse_arrow=reverse_arrow, reverse_order=reverse_order)
    horizontal = next(edge for edge in diagram.edges if edge.kind == "causal")
    nodes = {node.id: node for node in diagram.nodes}
    rendered = connection_paths(diagram)
    jumping, = [connection for connection in rendered if connection.bridges]
    assert jumping.edge is horizontal
    assert jumping.points[0] == edge_points(horizontal, nodes)[0]
    assert jumping.points[-1] == edge_points(horizontal, nodes)[-1]
    assert all(start != end for start, end in zip(jumping.points, jumping.points[1:]))

    root = ElementTree.fromstring(svg_diagram(diagram))
    solid_bridges = [
        line for line in root.findall(".//{*}polyline")
        if line.get("stroke") == "#6b7280" and line.get("stroke-dasharray") is None
    ]
    assert len(solid_bridges) == len(crossing_xs)
    for bridge, x in zip(solid_bridges, reversed(crossing_xs) if reverse_arrow else crossing_xs, strict=True):
        points = [tuple(map(float, point.split(","))) for point in bridge.get("points").split()]
        assert min(px for px, _ in points) < x < max(px for px, _ in points)
        assert min(y for _, y in points) < 281

    output = tmp_path / "crossing.png"
    png_preview(diagram, output)
    with Image.open(output) as preview:
        ink = (107, 114, 128)
        for x in crossing_xs:
            for side in (-1, 1):
                assert any(
                    preview.getpixel((x + side * dx, 281 - dy)) == ink
                    for dx in range(1, 5) for dy in range(1, 13)
                )
                assert preview.getpixel((x + side * 2, 281)) != ink


@pytest.mark.parametrize("reverse_order", [False, True])
def test_jumping_connections_do_not_paint_over_other_bridges(tmp_path: Path, reverse_order: bool) -> None:
    from PIL import Image

    diagram = _crossing_diagram()
    diagram.width = 760
    vertical = diagram.edges[1]
    vertical.kind = "data"
    vertical.via = [(260, 330), (350, 330), (350, 380), (260, 380)]
    diagram.nodes.extend([
        Node("extra_top", "Top", [], "input", x=600, y=100, width=100),
        Node("extra_bottom", "Bottom", [], "output", x=600, y=400, width=100),
    ])
    diagram.edges.append(Edge(
        "extra_top", "extra_bottom", kind="state", source_port="bottom", target_port="top",
        via=[(650, 210), (310, 210), (310, 385), (650, 385)],
    ))
    if reverse_order:
        diagram.edges.reverse()
    assert sum(bool(connection.bridges) for connection in connection_paths(diagram)) >= 2

    root = ElementTree.fromstring(svg_diagram(diagram))
    lines = root.findall(".//{*}polyline")
    path_positions = [index for index, line in enumerate(lines) if line.get("marker-end")]
    bridge_positions = [index for index, line in enumerate(lines) if line.get("stroke-linejoin") == "round"]
    assert max(path_positions) < min(bridge_positions)

    output = tmp_path / "multiple-jumps.png"
    png_preview(diagram, output)
    with Image.open(output) as preview:
        assert preview.getpixel((260, 271)) == (107, 114, 128)
        assert preview.getpixel((260, 273)) == (250, 251, 254)
        assert preview.getpixel((260, 277)) == (107, 114, 128)


@pytest.mark.integration
@pytest.mark.parametrize("row", policy_cases("tmasac__lstm", "tmasac__slstm"), indirect=True)
def test_recurrent_action_crossing_has_exactly_one_jump(row: dict[str, Any]) -> None:
    connections = connection_paths(build_diagram(row))
    assert [
        (connection.edge.source, connection.edge.target, len(connection.bridges))
        for connection in connections if connection.bridges
    ] == [("actions", "q_inputs", 1)]


@pytest.mark.integration
def test_layout_expands_for_a_critic_encoder_taller_than_its_readout(policy_row: PolicyRow) -> None:
    diagram = build_diagram(policy_row("mat_dec", DEFAULT_MODEL_SCALE))
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


@pytest.mark.integration
@pytest.mark.parametrize("row", policy_cases(), indirect=True)
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


@pytest.mark.integration
def test_ddpg_has_one_standard_head_and_uses_encoder_features_for_nop(policy_row: PolicyRow) -> None:
    row = policy_row("maddpg__deepset_critic", DEFAULT_MODEL_SCALE)
    d = build_diagram(row)
    nodes = {node.id: node for node in d.nodes}
    edges = {(edge.source, edge.target) for edge in d.edges}
    assert "critic_encoder" in nodes and "head1" in nodes
    assert "head2" not in nodes and "q_features" not in nodes
    assert ("critic_encoder", "head1") in edges and ("critic_encoder", "nop_bridge") in edges
    assert "final masked mean across agents" in nodes["head1"].lines


@pytest.mark.integration
@pytest.mark.parametrize("scale", SCALES)
def test_diagram_inspection_uses_current_selection_and_saves_only_local_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy_row: PolicyRow, scale: str,
) -> None:
    from swarmbots.tools import diagram_policy_architectures as diagrams

    guide = tmp_path / "docs/policy_parameters/model_scale.md"
    guide.parent.mkdir(parents=True)
    guide.write_text("Sizing guide", encoding="utf-8")
    stale = guide.parent / SCALE_LAYOUTS[scale].slug / "counts.json.gz"
    stale.parent.mkdir()
    stale.write_bytes(b"obsolete snapshot")
    variants = ("tmasac", "tmasac__no_nop")
    calls = []
    use_nop: bool | None = None

    def inspect(benchmark_id: str, selected: tuple[str, ...], *, model_scale: str) -> dict[str, Any]:
        calls.append(selected)
        return inspect_variants(benchmark_id, selected, model_scale=model_scale, use_nop=use_nop)

    monkeypatch.setattr(diagrams, "ROOT", tmp_path)
    monkeypatch.setattr(diagrams, "inspect_variants", inspect)
    rows = diagrams.inspect_reports([scale], variants)
    assert [row["variant"] for row in rows] == list(variants)
    assert calls == [variants]
    assert stale.read_bytes() == b"obsolete snapshot"
    generated = guide.parent / "generated" / SCALE_LAYOUTS[scale].slug
    with gzip.open(generated / "counts.json.gz", "rt", encoding="utf-8") as stream:
        report = json.load(stream)
    assert report["benchmark_id"] == "SwarmBots-WallEasy-v0"
    assert report["requested_variants"] == list(variants)
    assert report["policies"] == rows
    assert rows[0]["total"] == policy_row("tmasac", scale)["total"]
    assert (generated / "counts.csv").is_file()
    assert (generated / "layer_layout.csv").is_file()
    assert "[model-scale allocation](../../model_scale.md)" in (generated / "report.md").read_text(encoding="utf-8")
    assert not (guide.parent / "report.md").exists()

    output = generated / "counts.json.gz"
    use_nop = False
    selected = ("tmasac",)
    refreshed_rows = diagrams.inspect_reports([scale], selected)
    assert calls == [variants, selected]
    assert [row["variant"] for row in refreshed_rows] == list(selected)
    assert refreshed_rows[0]["use_nop"] is False
    assert refreshed_rows[0]["total"] < rows[0]["total"]
    with gzip.open(output, "rt", encoding="utf-8") as stream:
        refreshed_report = json.load(stream)
    assert refreshed_report["requested_variants"] == list(selected)
    assert refreshed_report["policies"] == refreshed_rows


def test_diagram_inspection_does_not_export_failed_policies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from swarmbots.tools import diagram_policy_architectures as diagrams

    def inspect(benchmark_id: str, variants: tuple[str, ...], *, model_scale: str) -> dict[str, Any]:
        return {"errors": [{"variant": "tmasac", "error": "construction failed"}]}

    monkeypatch.setattr(diagrams, "ROOT", tmp_path)
    monkeypatch.setattr(diagrams, "inspect_variants", inspect)
    with pytest.raises(RuntimeError, match="construction failed"):
        diagrams.inspect_reports([DEFAULT_MODEL_SCALE], ["tmasac"])
    assert not (tmp_path / "docs").exists()


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
    assert_joint_critic_inputs_do_not_intersect_outputs(diagram)
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
        assert_joint_critic_inputs_do_not_intersect_outputs(diagram)
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
@pytest.mark.parametrize("model_scale", SCALES)
def test_expanding_and_gated_deepset_diagrams_show_reception_and_complete_counts(
    model_scale: str, policy_row: PolicyRow,
) -> None:
    variants = (
        f"{algorithm}__{critic}"
        for critic in ("deepset_critic", "gated_deepset_critic")
        for algorithm in ("maddpg", "matd3", "masac")
    )
    for variant in variants:
        row = policy_row(variant, model_scale)
        diagram = build_diagram(row)
        nodes = {node.id: node for node in diagram.nodes}
        gated = row["variant"].endswith("__gated_deepset_critic")
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
                assert "Per-token reception gate" not in node.lines
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
def test_q_head_gating_ablation_diagrams_show_gates_and_complete_counts() -> None:
    report = inspect_variants(
        "SwarmBots-WallEasy-v0",
        tuple(f"{algorithm}__gated_deepset_critic" for algorithm in ("maddpg", "matd3", "masac")),
        model_scale="2.5M", policy_kwargs={"critic_deepset_head_gating": True},
    )
    assert not report["errors"]
    for row in report["policies"]:
        diagram = build_diagram(row)
        for node in diagram.nodes:
            if node.id.startswith("head"):
                assert "Per-token reception gate" in node.lines
        assert_connections_and_labels_leave_components_unobstructed(diagram)
        assert sum(node.parameters or 0 for node in diagram.nodes) == row["trainable"]
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

    critic = "gated_deepset_critic" if gated else "deepset_critic"
    report = inspect_variants(
        "SwarmBots-WallEasy-v0",
        tuple(f"{algorithm}__{critic}" for algorithm in ("matd3", "masac")),
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
