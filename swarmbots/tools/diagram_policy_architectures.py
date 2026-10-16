"""Generate simple, dimensioned policy dependency diagrams from audited presets.

Outputs SVG, Mermaid, Graphviz DOT, graph JSON, Markdown, and an offline gallery.
Use --refresh to reconstruct policies and refresh the parameter/layer audits.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from swarmbots.tools.inspect_policy_parameters import (
    allocation_doc_link,
    inspect_variants,
    save_report,
    write_compressed_json,
)
from swarmbots.learn import list_model_scales, list_variants
from swarmbots.learn.presets.model_scale import SCALE_LAYOUTS, normalize_model_scale
from swarmbots.tools.policy_diagram_layout import DiagramLane, EdgeLabel, edge_points, layout_diagram

ROOT = Path.cwd()
COLORS = {
    "input": ("#edf2f7", "#64748b"),
    "shared": ("#e8f1ff", "#3565a8"),
    "actor": ("#e9f5ee", "#377254"),
    "critic": ("#fff0df", "#996126"),
    "nop": ("#f0eafa", "#74569a"),
    "output": ("#eef2f6", "#596579"),
}


@dataclass
class ComponentBlock:
    title: str
    content: list[str | ComponentBlock]
    repeat: int | None = None
    parameters: int | None = None

    @property
    def heading(self) -> str:
        return self.title if self.repeat is None else f"{self.title} × {self.repeat}"

    @property
    def labels(self) -> list[str]:
        labels = [self.heading]
        if self.parameters is not None:
            scope = "Per block" if self.repeat is not None else "Subtotal"
            labels.append(f"{scope}: {self.parameters:,} parameters")
        if self.repeat is not None:
            labels.append("Independent weights per block")
        return labels

    @property
    def lines(self) -> list[str]:
        return [*self.labels, *content_lines(self.content)]


def content_lines(content: list[str | ComponentBlock]) -> Iterator[str]:
    for item in content:
        if isinstance(item, ComponentBlock):
            yield from item.labels
            yield from content_lines(item.content)
        else:
            yield item


@dataclass
class ContentText:
    text: str
    x: int
    y: int
    bold: bool = False


@dataclass
class ContentBox:
    block: ComponentBlock
    x: int
    y: int
    width: int
    height: int


def layout_content(
    content: list[str | ComponentBlock], x: int, y: int, width: int,
) -> tuple[list[ContentText | ContentBox], int]:
    """Share nested-box geometry between the SVG and raster renderers."""
    elements: list[ContentText | ContentBox] = []
    top = y
    for item in content:
        if isinstance(item, ComponentBlock):
            heading, heading_height = layout_content([item.heading], x + 10, y + 10, width - 20)
            for text in heading:
                text.bold = True
            details, details_height = layout_content(item.labels[1:], x + 10, y + 10 + heading_height, width - 20)
            labels, label_height = [*heading, *details], heading_height + details_height
            body, body_height = layout_content(item.content, x + 10, y + 10 + label_height, width - 20)
            height = 20 + label_height + body_height
            elements.extend([ContentBox(item, x, y, width, height), *labels, *body])
            y += height + 10
        else:
            for part in textwrap.wrap(item, int(width / 6.3), break_long_words=False) or [""]:
                elements.append(ContentText(part, x, y + 13))
                y += 18
    return elements, y - top


@dataclass
class Node:
    id: str
    title: str
    content: list[str | ComponentBlock]
    role: str
    x: int = 0
    y: int = 0
    width: int = 280
    height: int = 0
    parameters: int | None = None
    parameter_role: str | None = None

    def __post_init__(self) -> None:
        _, height = self.content_layout
        self.height = max(82, self.text_offset + height + 12)

    @property
    def lines(self) -> list[str]:
        return list(content_lines(self.content))

    @property
    def content_layout(self) -> tuple[list[ContentText | ContentBox], int]:
        return layout_content(self.content, self.x + 13, self.y + self.text_offset, self.width - 26)

    @property
    def text_offset(self) -> int:
        return 55 if self.parameters is not None else 34

    @property
    def parameter_label(self) -> str:
        return f"Total: {self.parameters:,} parameters" if self.parameters is not None else ""


@dataclass
class Edge:
    source: str
    target: str
    label: str = ""
    kind: str = "data"
    source_port: str = "right"
    target_port: str = "left"
    via: list[tuple[float, float]] = field(default_factory=list)
    source_offset: float = 0.5
    target_offset: float = 0.5
    label_box: EdgeLabel | None = None


@dataclass
class Diagram:
    variant: str
    scale: str
    shape: dict[str, Any]
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    main_parameters: int = 0
    nop_parameters: int = 0
    role_parameters: dict[str, int] = field(default_factory=dict)
    lanes: list[DiagramLane] = field(default_factory=list)
    width: int = 1330
    height: int = 1050

    def node(
        self, id: str, title: str, content: list[str | ComponentBlock], role: str,
    ) -> str:
        self.nodes.append(Node(id, title, content, role))
        return id

    def edge(self, source: str, target: str, label: str = "", kind: str = "data") -> None:
        self.edges.append(Edge(source, target, label, kind))

    @property
    def parameter_summary(self):
        return f"Main: {self.main_parameters:,} · NOP: {self.nop_parameters:,} · Total: {self.main_parameters + self.nop_parameters:,} parameters"

    @property
    def role_summary(self):
        return f"Actor: {self.role_parameters['actor']:,} · Critic: {self.role_parameters['critic']:,} · Shared: {self.role_parameters['shared_encoder']:,} (counted once)"


def configuration(row):
    return next(value for key, value in row["hyper_parameters"].items() if key.endswith("policy_config"))


def chain(row, *paths, fallback=""):
    for path in paths:
        found = next((layer for layer in row["layer_layout"] if layer["module"] == path), None)
        if found:
            return found["dimensions"].replace(" -> ", " → ") + f" ({found['parameters']:,})"
        # Stacked GLUs are audited as individual layers, including in existing
        # snapshots. Describe their forward order while counting the whole stack,
        # whose parameters also include any normalization between layers.
        stack_prefix = path + ".layers."
        layers = sorted(
            (
                layer for layer in row["layer_layout"]
                if layer["module"].startswith(stack_prefix)
                and layer["module"][len(stack_prefix):].isdigit()
            ),
            key=lambda layer: int(layer["module"][len(stack_prefix):]),
        )
        if layers:
            descriptions = [f"{layer['kind']} ({layer['dimensions']})" for layer in layers]
            dimensions = (
                f"{len(layers)} × {descriptions[0]}"
                if len(set(descriptions)) == 1
                else "; ".join(descriptions)
            )
            return dimensions.replace(" -> ", " → ") + f" ({module_count(row, path):,})"
    if fallback:
        return fallback
    raise ValueError(f"Missing audited layer {paths} for {row['variant']}")


def module_count(row, *paths):
    for path in paths:
        if path in row["module_parameter_counts"]:
            return row["module_parameter_counts"][path]
    raise ValueError(f"Missing audited module {paths} for {row['variant']}")


def deepset_communication_lines(
    row: dict[str, Any], prefix: str, features: int, *, normalize: bool,
) -> list[str | ComponentBlock]:
    dimensions = next(
        layer["dimensions"] for layer in row["layer_layout"] if layer["module"] == prefix + ".message_encoder"
    )
    message_width = int(dimensions.split(" -> ")[-1])
    content: list[str | ComponentBlock] = [
        "z = global LayerNorm(h)" if normalize else "z = h (no normalization)",
        f"token width D={features}; expanded message width M={message_width}",
        "nonlinear expansion before pooling: " + chain(row, prefix + ".message_encoder"),
        "masked mean in the expanded message space",
        "nonlinear pooled-message processing: " + chain(row, prefix + ".global_update"),
    ]
    if prefix + ".reception_gate" in row["module_parameter_counts"]:
        content.append(ComponentBlock("Per-token reception gate", [
            "concatenate normalized local token and shared global message",
            "gate MLP: " + chain(row, prefix + ".reception_gate"),
            "sigmoid: one independent gate per message feature",
            "multiply expanded message by token-specific gates before compression",
        ], parameters=module_count(row, prefix + ".reception_gate")))
    content.extend([
        "linear down-projection (bias-free): " + chain(row, prefix + ".message_projection"),
        f"received message g: N × {features}",
        "global residual: u = h + g",
    ])
    return content


def encoder_lines(
    row: dict[str, Any], config: dict[str, Any], prefix: str, projection: str = "",
) -> list[str | ComponentBlock]:
    """List input projections, the repeated block's forward order, then output."""
    width = config["d_model"]
    block = prefix + ".layers.0"
    is_deepset = block + ".interaction" in row["module_parameter_counts"]
    if is_deepset:
        communication = deepset_communication_lines(
            row, block + ".interaction", width, normalize=config["norm_first"],
        )
        if not config["norm_first"]:
            communication[-1] = "global residual: u = LayerNorm(h + g)"
        attention: str | ComponentBlock = ComponentBlock(
            "Global interaction · across agents", communication,
            parameters=module_count(row, block + ".interaction"),
        )
    else:
        attention = (
            f"attention / block: {config['nhead']} heads ({module_count(row, block + '.self_attn'):,})"
            if config["use_agent_attention"] else "agent attention: disabled"
        )
    lines: list[str | ComponentBlock] = []
    if projection:
        lines.append("obs MLP (once): " + projection)
        if prefix + ".global_obs_encoder" in row["module_parameter_counts"]:
            lines.extend(
                [
                    "global MLP (once): " + chain(row, prefix + ".global_obs_encoder"),
                    "add local and global token projections",
                ]
            )
    if config.get("normalize_tokens"):
        lines.append(f"token LayerNorm (once): {module_count(row, prefix + '.token_norm'):,} parameters")
    block_lines: list[str | ComponentBlock] = [f"feature width: D={width}"]
    temporal = config.get("temporal_model_cls")
    if temporal:
        kind = "sLSTM" if "SLSTM" in temporal else "LSTM"
        temporal_width = config.get("temporal_hidden_dim") or width
        temporal_lines = []
        if temporal_width != width:
            temporal_lines.append(f"memory input / block: {width} → {temporal_width}")
        temporal_lines.append(
            f"{kind} / block: hidden={temporal_width} ({module_count(row, block + '.temporal_model'):,})"
        )
        if config.get("use_temporal_output_projection", True) or temporal_width != width:
            temporal_lines.append(f"memory output / block: {temporal_width} → {width}")
        temporal_first = config["temporal_model_order"] == "temporal_first"
        block_lines.extend(temporal_lines if temporal_first else [attention])
        if config.get("inter_module_mlp"):
            block_lines.append("inter FF / block: " + chain(row, block + ".inter_module_feedforward"))
        block_lines.extend([attention] if temporal_first else temporal_lines)
    else:
        block_lines.append(attention)
    if is_deepset:
        block_lines.extend([
            "v = local LayerNorm(u)" if config["norm_first"] else "v = u; normalize after residual",
            "FF / block: " + chain(row, block + ".feedforward"),
            "local residual: h_next = u + FF(v)" if config["norm_first"]
            else "local residual: h_next = LayerNorm(u + FF(v))",
        ])
    else:
        block_lines.append("FF / block: " + chain(row, block + ".feedforward"))
    lines.append(ComponentBlock("Encoder block", block_lines, config["num_layers"], module_count(row, block)))
    if prefix + ".norm" in row["module_parameter_counts"]:
        lines.append(f"final LayerNorm (once): {module_count(row, prefix + '.norm'):,} parameters")
    lines.append(f"output: N × {width}")
    return lines


def critic_embedding_lines(
    row: dict[str, Any], config: dict[str, Any], prefix: str, *, context_in_encoder: bool = True,
) -> list[str]:
    lines: list[str] = []
    preencoder = prefix + ".observation_action_encoder"
    if preencoder in row["module_parameter_counts"]:
        lines.extend([
            "observation MLP (once): " + chain(row, preencoder + ".observation_encoder"),
            "action MLP (once): " + chain(row, preencoder + ".action_encoder"),
            "concatenate observation and action features",
        ])
    has_global_encoder = prefix + ".global_encoder" in row["module_parameter_counts"]
    if not has_global_encoder and context_in_encoder and config["joint_obs_embedding"] and (
        row["shape"]["global_obs_dim"] + row["shape"]["hidden_global_vars_dim"]
    ):
        lines.append("concatenate local/action and global inputs before projection")
    lines.append("local/action MLP (once): " + chain(row, prefix + ".local_action_encoder"))
    if has_global_encoder:
        lines.extend([
            "global context + active count",
            "global MLP (once): " + chain(row, prefix + ".global_encoder"),
            "add local and global token projections",
        ])
    return lines


def annotate_component_totals(d, row):
    """Assign each physical online parameter to exactly one displayed box."""
    nodes = {node.id: node for node in d.nodes}
    roles = {role: row["roles"][role]["total"] for role in ("actor", "critic", "shared_encoder", "next_obs_prediction")}
    d.role_parameters = roles

    def assign(id, count, role):
        nodes[id].parameters = count
        nodes[id].parameter_role = role

    assign("actions", module_count(row, "action_dist", "actor.action_net"), "actor")
    if "actor_encoder" in nodes:
        assign("actor_encoder", module_count(row, "actor_encoder", "_actor_encoder", "actor.encoder"), "actor")
    if "actor" in nodes:
        remaining = roles["actor"] - sum(node.parameters or 0 for node in d.nodes if node.parameter_role == "actor")
        assign("actor", remaining, "actor")
    else:
        # The shared-encoder SAC sketch combines the private encoder and MLP.
        assign("actor_encoder", roles["actor"] - nodes["actions"].parameters, "actor")
    if "representation" in nodes:
        role = "shared_encoder" if nodes["representation"].role == "shared" else "critic"
        assign("representation", roles[role] if role == "shared_encoder" else module_count(row, "encoder"), role)
    for id, prefix in (("critic_encoder", "critic.encoder"), ("critic_encoder2", "critic.encoder2")):
        if id in nodes:
            assign(id, module_count(row, id, prefix, "critic.0.encoder"), "critic")
    if "value" in nodes:
        used = sum(node.parameters or 0 for node in d.nodes if node.parameter_role == "critic")
        assign("value", roles["critic"] - used, "critic")
    if "state_bridge" in nodes:
        assign("state_bridge", module_count(row, "critic.actor_state_encoder"), "critic")
    for index in (1, 2):
        if f"head{index}" in nodes:
            assign(f"head{index}", module_count(row, f"critic.q{index}", "critic.0.network"), "critic")
        if f"q{index}_network" in nodes:
            assign(f"q{index}_network", module_count(row, f"critic.{index - 1}", f"critic.q{index}"), "critic")
    if "q_features" in nodes:
        assign("q_features", 0, "critic")
    if "nop_bridge" in nodes:
        prefix = "critic_nop." if "critic_nop" in row["module_parameter_counts"] else ""
        assign("nop_bridge", module_count(row, prefix + "pre_transition_transform"), "next_obs_prediction")
        assign("transition", module_count(row, prefix + "transition_model"), "next_obs_prediction")
        assign(
            "predictions",
            roles["next_obs_prediction"] - nodes["nop_bridge"].parameters - nodes["transition"].parameters,
            "next_obs_prediction",
        )
    for role, total in roles.items():
        if sum(node.parameters or 0 for node in d.nodes if node.parameter_role == role) != total:
            raise ValueError(f"Displayed {role} boxes do not reconcile for {d.variant}")
    if any(node.parameters is not None and node.parameters < 0 for node in d.nodes):
        raise ValueError(f"Negative component count for {d.variant}")


def deepset_encoder_lines(
    row: dict[str, Any], prefix: str, interaction: dict[str, Any],
) -> list[str | ComponentBlock]:
    layers = {layer["module"]: layer["dimensions"] for layer in row["layer_layout"]}
    features = int(layers[prefix + ".element_encoder"].split(" -> ")[-1])
    interaction_prefix = prefix + ".interactions."
    blocks = [
        path for path in row["module_parameter_counts"]
        if path.startswith(interaction_prefix) and path.removeprefix(interaction_prefix).isdigit()
    ]
    lines: list[str | ComponentBlock] = ["element MLP (once): " + chain(row, prefix + ".element_encoder")]
    if blocks:
        block = interaction_prefix + "0"
        update = interaction["local_update"]
        content: list[str | ComponentBlock] = [
            f"input: per-agent h · N × {features}",
            ComponentBlock(
                "Global interaction · across agents",
                deepset_communication_lines(row, block, features, normalize=interaction["normalize"]),
            ),
        ]
        if update == "residual":
            content.append("h_next = u; local processing omitted")
        else:
            content.append(ComponentBlock("Local update · per agent", [
                "v = local LayerNorm(u)" if interaction["normalize"] else "v = u (no normalization)",
                "local MLP: " + chain(row, block + ".local_encoder"),
                "local residual: h_next = u + local MLP(v)" if update == "residual_mlp" else "h_next = local MLP(v)",
            ]))
        content.append("zero masked agents; pass to next block")
        lines.append(ComponentBlock("Set interaction", content, len(blocks), module_count(row, block)))
    return [*lines, f"final features: N × {features} (before pooling)"]


def critic_lines(row: dict[str, Any], prefix: str) -> list[str | ComponentBlock]:
    policy_config = configuration(row)
    joint = policy_config.get("joint_critic_config")
    if joint and joint["kind"] == "deepset":
        critic_config = joint.get("head_config") or joint
    else:
        critic_config = policy_config.get("transformer_critic_config") or policy_config["critic_config"]
    interaction = (critic_config.get("deep_set_config") or critic_config)["interaction_config"]
    lines = deepset_encoder_lines(row, prefix + ".deepset", interaction)
    layers = {layer["module"]: layer["dimensions"] for layer in row["layer_layout"]}
    features = int(layers[prefix + ".deepset.element_encoder"].split(" -> ")[-1])
    readout_input = int(layers[prefix + ".deepset.set_decoder"].split(" -> ")[0])
    readout = ["final masked mean across agents"]
    if readout_input > features:
        readout.append(f"append global context: {readout_input - features} features")
    readout.append("regressor: " + chain(row, prefix + ".deepset.set_decoder"))
    return [*lines,
        ComponentBlock("Team readout · once", readout),
        "output: one team value",
    ]


def add_readout(d: Diagram, row: dict[str, Any], prefix: str, id: str, title: str) -> str:
    d.node(id, title, critic_lines(row, prefix), "critic")
    return id


def add_memory(d: Diagram, id: str, config: dict[str, Any]) -> None:
    if config.get("temporal_model_cls"):
        d.edge(
            id,
            id,
            "previous temporal state",
            "state",
        )


def add_actor_state_bridge(d: Diagram, row: dict[str, Any]) -> None:
    if "critic.actor_state_encoder" not in row["module_parameter_counts"]:
        return
    d.node(
        "state_bridge",
        "Actor-state bridge",
        ["detach last actor state", chain(row, "critic.actor_state_encoder")],
        "critic",
    )
    d.edge("actor_encoder", "state_bridge", "h/c or h/m", "state")
    for node in d.nodes:
        if node.id in {"critic_encoder", "critic_encoder2"}:
            d.edge("state_bridge", node.id, "append to critic inputs", "state")


def build_diagram(row: dict[str, Any]) -> Diagram:
    cfg = configuration(row)
    s = row["shape"]
    d = Diagram(
        row["variant"],
        row["model_scale"]["name"],
        s,
        main_parameters=row["trainable_minus_nop"],
        nop_parameters=row["roles"]["next_obs_prediction"]["total"],
    )
    n, local, glob, private, private_global, action = (
        s[key]
        for key in (
            "n_agents",
            "local_obs_dim",
            "global_obs_dim",
            "hidden_local_vars_dim",
            "hidden_global_vars_dim",
            "actions_per_agent",
        )
    )
    on_policy = "encoder_config" in cfg or "actor_config" in cfg
    shared = "actor_config" in cfg or cfg.get("shared_encoder_config") is not None
    d.node(
        "public",
        "Public observations",
        [f"local: N × {local}", f"global: {glob}", "actor-visible inputs"],
        "input",
    )
    d.node(
        "privileged",
        "Privileged context",
        [f"local: N × {private}", f"global: {private_global}", "critic only"],
        "input",
    )
    distribution = row.get("continuous_action_dist")
    output_label = (
        "Deterministic tanh actions"
        if distribution is None
        else (
            "Gaussian mean + std; sample" if distribution == "predicted_std_gaussian" else "Action distribution; sample"
        )
    )
    d.node(
        "actions",
        "Actions",
        [output_label, f"N × {action}", "autoregressive if decoder shown"],
        "output",
    )
    d.notes = [
        f"N={n}; batch B omitted. Fixed input/output widths are task-specific.",
        "Solid = data. Dashed = causal actions / detached state. Insets show sub-component scope; crossing lines do not join.",
        "Main sketch is online networks. Frozen target copies and optimizer updates are omitted.",
    ]
    source = "representation"
    if on_policy:
        if "actor_config" in cfg:
            ac = cfg["actor_config"]
            width = ac.get("shared_encoder_latent_dim", ac.get("shared_encoder_latent_dim_per_agent"))
            kind = "Joint MLP shared encoder" if "latent_pi_dim_per_agent" in ac else "Per-agent shared encoder"
            d.node(source, kind, [chain(row, "shared_encoder.mlp"), f"reshape/output: N × {width}"], "shared")
        else:
            ec = cfg["encoder_config"]
            width = ec["d_model"]
            source = "critic_encoder"
            cc = cfg["critic_encoder_config"]
            d.node(
                source,
                "Critic observation encoder",
                encoder_lines(row, cc, source, chain(row, source + ".local_obs_encoder")),
                "critic",
            )
            add_memory(d, source, cc)
        d.edge("public", source)
        actor_source = source
        if not shared:
            ac = cfg["encoder_config"]
            d.node(
                "actor_encoder",
                "Actor observation encoder",
                encoder_lines(row, ac, "actor_encoder", chain(row, "actor_encoder.local_obs_encoder")),
                "actor",
            )
            d.edge("public", "actor_encoder")
            add_memory(d, "actor_encoder", ac)
            actor_source = "actor_encoder"
        if "decoder_config" in cfg:
            dec = cfg["decoder_config"]
            dec_width = dec["d_model"]
            qcx = "qcx" in row["variant"]
            projection = chain(
                row, "decoder.input_projection", "encoder_decoder_projection", fallback=f"identity {width} (0)"
            )
            action_embedding = chain(row, "action_encoder", "decoder.action_encoder")
            lines = [
                f"{'query' if qcx else 'obs'} projection (once): {projection}",
                f"action embedding (once): {action_embedding}",
                f"{'memory' if qcx else 'obs context'} width: {width if qcx else dec_width}",
            ]
            block = "decoder.layers.0" if qcx else "decoder.blocks.0"
            block_lines = [f"D={dec_width} · {dec['nhead']} heads"]
            if qcx:
                block_lines.extend(
                    [
                        "context MLP / block: " + chain(row, block + ".context_encoder", fallback="identity (0)"),
                        f"causal attn / block: {module_count(row, block + '.query_context_attn'):,}",
                        f"memory attn / block: {module_count(row, block + '.memory_attn'):,}",
                        f"FF / block: {dec_width} → {dec['dim_feedforward']} → {dec_width} ({module_count(row, block + '.linear1') + module_count(row, block + '.linear2'):,})",
                    ]
                )
            else:
                block_lines.extend(
                    [
                        f"causal self-attn / block: {module_count(row, block + '.attn1'):,}",
                        f"obs-query attn / block: {module_count(row, block + '.attn2'):,}",
                        "FF / block: " + chain(row, block + ".mlp"),
                    ]
                )
            lines.append(ComponentBlock("Decoder block", block_lines, dec['num_layers'], module_count(row, block)))
            if not qcx:
                lines.append("decoder head (once): " + chain(row, "decoder.head"))
            lines.append(f"output: N × {dec_width if qcx else dec.get('latent_pi_dim') or dec_width}")
            d.node("actor", "QCX autoregressive decoder" if qcx else "Original MAT decoder", lines, "actor")
            d.edge(actor_source, "actor", f"N × {width}; {'obs + memory' if qcx else 'obs context'}")
            d.edge(
                "actions",
                "actor",
                "previous agents only",
                "causal",
            )
            d.notes.append(
                "Decoder loop is across agent order, not future observations; training may teacher-force actions."
            )
        else:
            d.node("actor", "Private actor MLP", [chain(row, "actor.mlp", "actor_head"), "per agent"], "actor")
            d.edge(actor_source, "actor", f"N × {cfg.get('actor_encoder_config', {}).get('d_model', width)}")
        d.edge("actor", "actions")
        if row["base_policy_class"] == "MATOrigPolicy":
            d.node(
                "value",
                "Token-wise V head",
                [
                    "fuse encoded + privileged inputs",
                    chain(row, "critic.value_head", fallback="see layer table"),
                    "one scalar per agent",
                    "masked mean → team V",
                ],
                "critic",
            )
        elif (
            "actor_config" in cfg
            and "latent_pi_dim_per_agent" in cfg["actor_config"]
            or cfg.get("critic_config", {}).get("deep_set_config", "absent") is None
        ):
            d.node(
                "value",
                "Flattened V critic",
                [
                    "flatten agent features + context",
                    chain(row, "critic.value_features"),
                    "value readout: " + chain(row, "critic.value_head"),
                    "scalar team V",
                ],
                "critic",
            )
        else:
            add_readout(d, row, "critic", "value", "Deep Set V critic")
        critic_width = width if shared else cfg["critic_encoder_config"]["d_model"]
        d.edge(source, "value", f"N × {critic_width}")
        d.edge("privileged", "value", "local + global context")
        d.node("v", "V", ["scalar per world"], "output")
        d.edge("value", "v")
        source = actor_source
    else:
        actor_cfg = cfg["actor_encoder_config"]
        acwidth = actor_cfg["d_model"]
        actor_source = "public"
        if shared:
            shared_cfg = cfg["shared_encoder_config"]
            d.node(
                source,
                "Shared observation encoder",
                encoder_lines(
                    row,
                    shared_cfg,
                    "shared_observation_encoder",
                    chain(row, "shared_observation_encoder.local_obs_encoder"),
                ),
                "shared",
            )
            d.edge("public", source)
            actor_source = source
            add_memory(d, source, shared_cfg)
        d.node(
            "actor_encoder",
            "Actor encoder",
            encoder_lines(
                row,
                actor_cfg,
                "_actor_encoder" if "_actor_encoder" in row["module_parameter_counts"] else "actor.encoder",
                chain(row, "_actor_encoder.local_obs_encoder", "actor.encoder.local_obs_encoder"),
            ),
            "actor",
        )
        add_memory(d, "actor_encoder", actor_cfg)
        d.edge(actor_source, "actor_encoder", f"N × {cfg['shared_encoder_config']['d_model']}" if shared else "public")
        actor_chain = chain(
            row,
            "actor_head.head",
            "actor.head.head",
            fallback=f"{acwidth} → {cfg['actor_head_config']['hidden_dims'][-1]}",
        )
        # TD3 nests the backbone inside actor; the audited actor MLP width is still explicit.
        if shared:
            actor_node = next(node for node in d.nodes if node.id == "actor_encoder")
            actor_node.content[-1:] = [
                "head: " + actor_chain,
                f"output: N × {cfg['actor_head_config']['hidden_dims'][-1]}",
            ]
            actor_node.__post_init__()
            d.edge("actor_encoder", "actions")
        else:
            d.node("actor", "Actor output MLP", [actor_chain, "distribution/tanh readout"], "actor")
            d.edge("actor_encoder", "actor", f"N × {acwidth}")
            d.edge("actor", "actions")
        d.node(
            "q_inputs",
            "Joint critic inputs",
            ["privileged local + global", f"actions: N × {action}"]
            if shared
            else ["public observations", "privileged local + global", f"actions: N × {action}"],
            "input",
        )
        if not shared:
            d.edge(
                "public",
                "q_inputs",
                "public",
            )
        d.edge("privileged", "q_inputs")
        d.edge(
            "actions",
            "q_inputs",
            "replay / policy",
            "causal",
        )
        joint = cfg.get("joint_critic_config")
        is_joint = joint is not None and cfg.get("critic_kind", joint["kind"]) != "transformer"
        if is_joint and joint["kind"] == "deepset":
            single_head = "critic.0.encoder" in row["module_parameter_counts"]
            encoder_config = joint["encoder_config"]
            element_width = encoder_config["d_model"]
            independent = joint["independent_encoders"] and not single_head
            for index in range(1, 3 if independent else 2):
                suffix = "" if index == 1 else "2"
                id = "critic_encoder" + suffix
                prefix = "critic.0.encoder.network" if single_head else "critic.encoder" + suffix + ".network"
                d.node(
                    id,
                    f"Independent critic encoder {index}" if independent else "ONE Deep Set critic encoder",
                    [
                        *critic_embedding_lines(row, encoder_config, prefix, context_in_encoder=joint["context_in_elements"]),
                        *encoder_lines(row, encoder_config, prefix),
                    ],
                    "critic",
                )
                d.edge("q_inputs", id)
            for index in range(1, 2 if single_head else 3):
                add_readout(
                    d, row, "critic.0.network" if single_head else f"critic.q{index}",
                    f"head{index}", f"Separate Q{index} head",
                )
                encoder = "critic_encoder2" if independent and index == 2 else "critic_encoder"
                d.edge(encoder, f"head{index}", f"N × {element_width}")
                if not joint["context_in_elements"] and glob + private_global:
                    d.edge("q_inputs", f"head{index}", "global context after pooling")
                d.node(f"q{index}", f"Q{index}", ["scalar per world"], "output")
                d.edge(f"head{index}", f"q{index}")
            source = "critic_encoder"
            if independent:
                source = "q_features"
                d.node(
                    source, "Final equivariant critic features",
                    [f"2 × (N × {element_width})", f"concatenate → N × {2 * element_width}"], "critic",
                )
                for id in ("critic_encoder", "critic_encoder2"):
                    d.edge(id, source, "tokens")
            d.notes.append(
                "DDPG has one standard Q head; twin-Q algorithms use two independent heads."
                if single_head else "The two Q heads have separate parameters; the shared Deep Set runs once."
                if not independent else "Independent critic encoders concatenate their pre-pooling features for NOP."
            )
        elif is_joint:
            count = 1 if row["variant"].startswith("maddpg") else 2
            for i in range(count):
                index = i + 1
                prefix = (
                    f"critic.{i}.network"
                    if row["base_policy_class"] in {"TD3Policy", "RecurrentTD3Policy"}
                    else f"critic.q{index}.network"
                )
                lines = ["flatten observations/actions/mask", chain(row, prefix), "scalar Q"]
                d.node(f"q{index}_network", f"Independent Q{index} network", lines, "critic")
                d.edge("q_inputs", f"q{index}_network")
                d.node(f"q{index}", f"Q{index}", ["scalar per world"], "output")
                d.edge(f"q{index}_network", f"q{index}")
        else:
            cc = cfg.get("critic_encoder_config")
            d.node(
                "critic_encoder",
                "ONE shared critic encoder",
                [
                    *critic_embedding_lines(row, cc, "critic.encoder"),
                    *encoder_lines(row, cc, "critic.encoder"),
                ],
                "critic",
            )
            d.edge("q_inputs", "critic_encoder")
            if shared:
                d.edge(source, "critic_encoder", f"N × {cfg['shared_encoder_config']['d_model']}")
            for i in [1, 2]:
                add_readout(d, row, f"critic.q{i}", f"head{i}", f"Separate Q{i} head")
                d.edge("critic_encoder", f"head{i}", f"N × {cc['d_model']}")
                d.node(f"q{i}", f"Q{i}", ["scalar per world"], "output")
                d.edge(f"head{i}", f"q{i}")
            source = (
                "representation"
                if shared and cfg["nop_config"]["latent_source"] == "SHARED_ENCODER"
                else "critic_encoder"
            )
            d.notes.append("The two Q heads have separate parameters; the shared transformer runs once.")
        add_actor_state_bridge(d, row)
    if row["use_nop"]:
        prefix = "" if on_policy else "critic_nop."
        bridge = chain(row, prefix + "pre_transition_transform")
        prediction_chain = chain(row, prefix + "pre_predictors_transform")
        layout = SCALE_LAYOUTS[d.scale]
        block = prefix + "transition_model.encoder.layers.0"
        d.node("nop_bridge", "NOP source projection", ["from source features; before pooling", bridge], "nop")
        d.edge(source, "nop_bridge", "latents")
        d.node(
            "transition",
            "NOP transition model",
            [
                "coembed (per step): " + chain(row, prefix + "transition_model.coembed"),
                ComponentBlock("Transition encoder block", [
                    f"D={layout.nop_width} · 4 heads",
                    f"attention / block: {module_count(row, block + '.self_attn'):,}",
                    f"FF / block: {layout.nop_width} → {layout.nop_ff} → {layout.nop_width} ({module_count(row, block + '.linear1') + module_count(row, block + '.linear2'):,})",
                ], layout.nop_layers, module_count(row, block)),
                "head: " + chain(row, prefix + "transition_model.head"),
                "latent residual; repeated future steps",
                "same transition weights at every step",
            ],
            "nop",
        )
        d.edge("nop_bridge", "transition", "z_t")
        d.node(
            "future_actions",
            "Rollout / replay actions",
            [f"sequence: K × N × {action}", "known training trajectory", "one action per predicted step"],
            "input",
        )
        d.edge(
            "future_actions",
            "transition",
            "action sequence",
        )
        heads = []
        for key, label in [
            ("local_scalars_predictor", "scalars"),
            ("local_angles_predictor", "angle delta"),
            ("local_rot6ds_predictor", "rotation delta"),
            ("local_binaries_predictor", "binary"),
        ]:
            item = chain(row, prefix + key, fallback="disabled")
            if item != "disabled":
                heads.append(label + ": " + item)
        d.node(
            "predictions",
            "NOP prediction heads",
            [
                "predictor MLP: " + prediction_chain,
                "parallel prediction heads:",
                *heads,
                "future observations supervise loss",
            ],
            "nop",
        )
        d.edge("transition", "predictions", "predicted next latents")
    else:
        d.notes.append("NOP is disabled for this preset.")
    d.notes.append(
        "Counts include weights/biases and normalization. Box totals are additive; inner counts are included subtotals."
    )
    annotate_component_totals(d, row)
    layout_diagram(d)
    validate_diagram(d)
    return d


def validate_diagram(d):
    ids = {node.id for node in d.nodes}
    if len(ids) != len(d.nodes):
        raise ValueError("Duplicate diagram node")
    for edge in d.edges:
        if edge.source not in ids or edge.target not in ids:
            raise ValueError(f"Unknown endpoint: {edge}")
    for node in d.nodes:
        if node.x < 0 or node.y < 0 or node.x + node.width > d.width or node.y + node.height > d.height - 20:
            raise ValueError(f"Node outside canvas: {node.id}")
    for i, left in enumerate(d.nodes):
        for right in d.nodes[i + 1 :]:
            if max(left.x, right.x) < min(left.x + left.width, right.x + right.width) and max(left.y, right.y) < min(
                left.y + left.height, right.y + right.height
            ):
                raise ValueError(f"Overlapping diagram nodes: {left.id}, {right.id}")


def svg_diagram(d: Diagram) -> str:
    esc = html.escape
    bits = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {d.width} {d.height}" role="img" font-family="Arial, sans-serif" fill="#182538">',
        f"<title>{esc(d.variant)} · {esc(d.scale)}</title>",
        "<desc>Dimensioned actor, critic, shared representation, recurrence and next-observation prediction dependencies.</desc>",
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8z" fill="#6b7280"/></marker></defs>',
        f'<rect width="{d.width}" height="{d.height}" fill="#fafbfe"/>',
        f'<text x="22" y="30" font-size="23" font-weight="700">{esc(d.variant)} · {esc(d.scale)}</text>',
        f'<text x="22" y="53" font-size="13">{esc(d.parameter_summary)} · WallEasy audit</text>',
        f'<text x="22" y="74" font-size="13">{esc(d.role_summary)}</text>',
        '<g font-family="Arial, sans-serif" fill="#182538">',
    ]
    nodes = {node.id: node for node in d.nodes}
    for lane in d.lanes:
        bits.extend([
            f'<rect x="35" y="{lane.y}" width="{d.width - 70}" height="{lane.height}" rx="12" fill="#f5f7fb" stroke="#e0e6ef"/>',
            f'<text x="50" y="{lane.y + 23}" font-size="14" font-weight="700" fill="#526176">{esc(lane.title)}</text>',
        ])
    for edge in d.edges:
        points = edge_points(edge, nodes)
        coords = " ".join(f"{x:g},{y:g}" for x, y in points)
        dash = ' stroke-dasharray="6 4"' if edge.kind != "data" else ""
        bits.append(f'<polyline points="{coords}" fill="none" stroke="#fafbfe" stroke-width="5"{dash}/>')
        bits.append(
            f'<polyline points="{coords}" fill="none" stroke="#6b7280" stroke-width="1.7" marker-end="url(#arrow)"{dash}/>'
        )
        if edge.label_box:
            box = edge.label_box
            bits.append(
                f'<rect x="{box.x:g}" y="{box.y:g}" width="{box.width:g}" height="{box.height:g}" rx="3" fill="#fafbfe"/>'
            )
            for i, label in enumerate(box.lines):
                bits.append(
                    f'<text x="{box.x + box.width / 2:g}" y="{box.y + 12 + i * 14:g}" font-size="11" text-anchor="middle">{esc(label)}</text>'
                )
    for node in d.nodes:
        fill, border = COLORS[node.role]
        bits.extend(
            [
                f'<rect x="{node.x}" y="{node.y}" width="{node.width}" height="{node.height}" rx="9" fill="{fill}" stroke="{border}" stroke-width="1.2"/>',
                f'<text x="{node.x + 13}" y="{node.y + 25}" font-size="16" font-weight="700">{esc(node.title)}</text>',
            ]
        )
        if node.parameter_label:
            bits.append(
                f'<text x="{node.x + 13}" y="{node.y + 46}" font-size="12.3" font-weight="700" fill="{border}">{esc(node.parameter_label)}</text>'
            )
        for element in node.content_layout[0]:
            if isinstance(element, ContentBox):
                dash = ' stroke-dasharray="5 3"' if element.block.repeat is not None else ""
                bits.append(
                    f'<g data-component="{esc(element.block.title)}" data-repeat="{element.block.repeat or 1}">'
                    f'<rect x="{element.x}" y="{element.y}" width="{element.width}" height="{element.height}" '
                    f'rx="6" fill="#ffffff" fill-opacity="0.6" stroke="{border}" stroke-width="1"{dash}/></g>'
                )
            else:
                weight = ' font-weight="700"' if element.bold else ""
                bits.append(
                    f'<text x="{element.x}" y="{element.y}" font-size="12.3"{weight}>{esc(element.text)}</text>'
                )
    for i, note in enumerate(d.notes):
        bits.append(f'<text x="22" y="{d.height - 20 - (len(d.notes) - i) * 16}" font-size="11">{esc(note)}</text>')
    bits.append("</g></svg>")
    return "\n".join(bits)


def png_preview(d: Diagram, path: Path) -> None:
    """Optional Pillow raster preview of the same graph geometry (not needed for SVG)."""
    from PIL import Image, ImageDraw, ImageFont

    font_path = Path("C:/Windows/Fonts/arial.ttf")

    def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
        selected = font_path.with_name("arialbd.ttf") if bold else font_path
        return ImageFont.truetype(str(selected) if selected.exists() else "DejaVuSans.ttf", size)

    def connection(points: list[tuple[float, float]], color: str, width: int, *, dashed: bool) -> None:
        if not dashed:
            draw.line(points, fill=color, width=width)
            return
        for start, end in zip(points, points[1:]):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = (dx * dx + dy * dy) ** 0.5
            for offset in range(0, int(length) + 1, 10):
                a, b = offset / length, min((offset + 6) / length, 1)
                draw.line(
                    [(start[0] + dx * a, start[1] + dy * a), (start[0] + dx * b, start[1] + dy * b)],
                    fill=color, width=width,
                )

    canvas = Image.new("RGB", (d.width, d.height), "#fafbfe")
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 8), f"{d.variant} · {d.scale}", fill="#182538", font=font(23))
    draw.text(
        (22, 37),
        d.parameter_summary,
        fill="#182538",
        font=font(13),
    )
    draw.text((22, 58), d.role_summary, fill="#182538", font=font(13))
    nodes = {node.id: node for node in d.nodes}
    for lane in d.lanes:
        draw.rounded_rectangle(
            (35, lane.y, d.width - 35, lane.y + lane.height), radius=12, fill="#f5f7fb", outline="#e0e6ef",
        )
        draw.text((50, lane.y + 9), lane.title, fill="#526176", font=font(14, bold=True))
    for edge in d.edges:
        points = edge_points(edge, nodes)
        connection(points, "#fafbfe", 6, dashed=edge.kind != "data")
        connection(points, "#6b7280", 2, dashed=edge.kind != "data")
        endx, endy = points[-1]
        beforex, beforey = points[-2]
        dx, dy = endx - beforex, endy - beforey
        length = max(1, (dx * dx + dy * dy) ** 0.5)
        ux, uy = dx / length, dy / length
        draw.polygon(
            [
                (endx, endy),
                (endx - 9 * ux + 4 * uy, endy - 9 * uy - 4 * ux),
                (endx - 9 * ux - 4 * uy, endy - 9 * uy + 4 * ux),
            ],
            fill="#6b7280",
        )
        if edge.label_box:
            box = edge.label_box
            draw.rounded_rectangle(
                (box.x, box.y, box.x + box.width, box.y + box.height), radius=3, fill="#fafbfe",
            )
            for i, label in enumerate(box.lines):
                draw.text(
                    (box.x + box.width / 2, box.y + 1 + i * 14),
                    label,
                    fill="#182538",
                    font=font(11),
                    anchor="mt",
                    stroke_fill="#fafbfe",
                    stroke_width=2,
                )
    for node in d.nodes:
        fill, border = COLORS[node.role]
        draw.rounded_rectangle(
            (node.x, node.y, node.x + node.width, node.y + node.height), radius=9, fill=fill, outline=border
        )
        draw.text((node.x + 13, node.y + 9), node.title, fill="#182538", font=font(16))
        if node.parameter_label:
            draw.text((node.x + 13, node.y + 32), node.parameter_label, fill=border, font=font(12))
        for element in node.content_layout[0]:
            if isinstance(element, ContentBox):
                draw.rounded_rectangle(
                    (element.x, element.y, element.x + element.width, element.y + element.height),
                    radius=6, fill="#ffffff", outline=border,
                )
            else:
                draw.text((element.x, element.y - 13), element.text, fill="#182538", font=font(12, bold=element.bold))
    for i, note in enumerate(d.notes):
        draw.text((22, d.height - 33 - (len(d.notes) - i) * 16), note, fill="#182538", font=font(11))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def content_groups(content: list[str | ComponentBlock]) -> Iterator[list[str] | ComponentBlock]:
    pending: list[str] = []
    for item in content:
        if isinstance(item, ComponentBlock):
            if pending:
                yield pending
                pending = []
            yield item
        else:
            pending.append(item)
    if pending:
        yield pending


def mermaid_diagram(d: Diagram) -> str:
    lines = ["flowchart LR", f"  %% {d.parameter_summary}", f"  %% {d.role_summary}"]
    role_ids: dict[str, list[str]] = {role: [] for role in COLORS}

    def label(labels: list[str]) -> str:
        return "<br/>".join(html.escape(line, quote=True) for line in labels)

    def container(id: str, labels: list[str], content: list[str | ComponentBlock], role: str) -> None:
        lines.extend([f'  subgraph {id}["{label(labels)}"]', "    direction TB"])
        children = []
        for index, group in enumerate(content_groups(content)):
            child = f"{id}_part{index}"
            children.append(child)
            if isinstance(group, ComponentBlock):
                container(child, group.labels, group.content, role)
            else:
                lines.append(f'    {child}["{label(group)}"]')
            role_ids[role].append(child)
        # Subgraphs preserve displayed scope; invisible links only set reading order.
        for previous, current in zip(children, children[1:]):
            lines.append(f"    {previous} ~~~ {current}")
        lines.append("  end")

    for node in d.nodes:
        labels = [node.title, *([node.parameter_label] if node.parameter_label else [])]
        if any(isinstance(item, ComponentBlock) for item in node.content):
            container(node.id, labels, node.content, node.role)
        else:
            lines.append(f'  {node.id}["{label([*labels, *node.lines])}"]')
        role_ids[node.role].append(node.id)
    for edge in d.edges:
        arrow = "-->" if edge.kind == "data" else "-.->"
        label = f'|"{html.escape(edge.label, quote=True)}"|' if edge.label else ""
        lines.append(f"  {edge.source} {arrow}{label} {edge.target}")
    for role, (fill, border) in COLORS.items():
        lines.append(f"  classDef {role} fill:{fill},stroke:{border},color:#182538")
        ids = ",".join(role_ids[role])
        if ids:
            lines.append(f"  class {ids} {role}")
    return "\n".join(lines) + "\n"


def dot_diagram(d: Diagram) -> str:
    lines = [
        "digraph policy {",
        '  rankdir=LR; graph [bgcolor="white"]; node [shape=box,style="rounded,filled",fontname="Arial"];',
        f"  label={json.dumps(d.parameter_summary + chr(10) + d.role_summary, ensure_ascii=False)}; labelloc=t;",
    ]
    def table(labels: list[str], content: list[str | ComponentBlock], *, border: int = 0) -> str:
        rows = []
        for index, text in enumerate(labels):
            text = html.escape(text, quote=True)
            if index == 0:
                text = f"<B>{text}</B>"
            rows.append(f'<TR><TD ALIGN="LEFT">{text}</TD></TR>')
        for group in content_groups(content):
            if isinstance(group, ComponentBlock):
                rows.append(f"<TR><TD>{table(group.labels, group.content, border=1)}</TD></TR>")
            else:
                rows.extend(f'<TR><TD ALIGN="LEFT">{html.escape(line, quote=True)}</TD></TR>' for line in group)
        return f'<TABLE BORDER="{border}" CELLBORDER="0" CELLSPACING="0" CELLPADDING="5">' + "".join(rows) + "</TABLE>"

    for node in d.nodes:
        labels = [node.title, *([node.parameter_label] if node.parameter_label else [])]
        label = table(labels, node.content)
        fill, color = COLORS[node.role]
        lines.append(f'  {node.id} [label=<{label}>,fillcolor="{fill}",color="{color}"];')
    for edge in d.edges:
        lines.append(
            f"  {edge.source} -> {edge.target} [label={json.dumps(edge.label)},style={'solid' if edge.kind == 'data' else 'dashed'}];"
        )
    return "\n".join([*lines, "}"]) + "\n"


def load_reports(scales, variants, refresh):
    rows = []
    for scale in scales:
        layout = SCALE_LAYOUTS[scale]
        folder = (
            ROOT
            / "docs/policy_parameters"
            / layout.slug
        )
        path = folder / "counts.json.gz"
        report = json.loads(gzip.decompress(path.read_bytes())) if path.exists() else None
        wanted = set(variants)
        if (
            refresh
            or report is None
            or not wanted <= {row["variant"] for row in report["policies"]}
            or any("module_parameter_counts" not in row for row in report["policies"] if row["variant"] in wanted)
        ):
            # A selection controls drawing, not the contents of the shared audit.
            # Rebuild the complete audit together so its metadata stays consistent.
            available = set(list_variants(include_hidden=True))
            cached_variants = [row["variant"] for row in report["policies"]] if report is not None else []
            audit_variants = tuple(dict.fromkeys([
                *list_variants(),
                *(name for name in cached_variants if name in available),
                *variants,
            ]))
            report = inspect_variants("SwarmBots-WallEasy-v0", audit_variants, model_scale=scale)
            if report["errors"]:
                raise RuntimeError(report["errors"])
            # Keep the compact audit input; verbose reports remain local outputs.
            generated = ROOT / "docs/policy_parameters/generated" / layout.slug
            save_report(
                report,
                generated,
                allocation_doc=allocation_doc_link(generated, ROOT / "docs/policy_parameters/model_scale.md"),
            )
            folder.mkdir(parents=True, exist_ok=True)
            write_compressed_json(report, path)
        rows.extend(row for row in report["policies"] if row["variant"] in wanted)
    return rows


def write_gallery(diagrams, output):
    entries = {}
    for d in diagrams:
        slug = SCALE_LAYOUTS[d.scale].slug
        entries[d.scale + "/" + d.variant] = {"svg": svg_diagram(d), "path": f"{slug}/{d.variant}", "notes": d.notes}
    data = json.dumps(entries, ensure_ascii=False).replace("</", r"<\/")
    variants = list(dict.fromkeys(d.variant for d in diagrams))
    scales = list(dict.fromkeys(d.scale for d in diagrams))

    def options(values):
        return "".join(f"<option>{html.escape(v)}</option>" for v in values)

    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Policy architecture diagrams</title><style>
body{font:15px Arial,sans-serif;margin:0;color:#182538;background:#eef2f7}header{padding:22px 28px;background:#fff;border-bottom:1px solid #d9e1eb}h1{font-size:24px;margin:0 0 14px}label{margin-right:18px}select{padding:8px;max-width:320px;font:inherit}nav{margin:12px 0 0}a{color:#3565a8;margin-right:18px}.legend{display:flex;gap:18px;margin:16px 0 0;font-size:13px}.legend span:before{content:"";display:inline-block;width:12px;height:12px;background:var(--color);margin-right:6px;border-radius:3px}main{padding:18px;overflow:auto}.canvas{background:#fafbfe;border-radius:12px;max-width:2100px;margin:auto;box-shadow:0 2px 14px #14213a13}.canvas svg{display:block;width:100%;height:auto;min-width:1300px}p{margin:12px 0 0;font-size:13px;color:#526176}
</style><header><h1>Policy architecture diagrams</h1><label>Variant <select id="variant">VARIANTS</select></label><label>Scale <select id="scale">SCALES</select></label><nav><a id="svg" download>SVG</a><a id="mermaid" download>Mermaid source</a><a id="dot" download>Graphviz source</a><a href="GUIDE_URL">Reading guide</a></nav><div class="legend"><span style="--color:#edf2f7">Inputs</span><span style="--color:#e8f1ff">Shared</span><span style="--color:#e9f5ee">Actor</span><span style="--color:#fff0df">Critic</span><span style="--color:#f0eafa">NOP</span></div><p>Solid arrows show data; dashed arrows show causal actions or detached state. Batch dimension and training-only target copies are omitted.</p></header><main><div id="canvas" class="canvas"></div></main><script>
const entries=DATA;const variant=document.getElementById('variant'),scale=document.getElementById('scale');function update(){const item=entries[scale.value+'/'+variant.value];document.getElementById('canvas').innerHTML=item.svg;for(const [id,ext] of [['svg','svg'],['mermaid','mmd'],['dot','dot']])document.getElementById(id).href=item.path+'.'+ext;location.hash=encodeURIComponent(scale.value)+'/'+variant.value;}const initial=decodeURIComponent(location.hash.slice(1)).split('/');if(initial.length===2&&entries[initial[0]+'/'+initial[1]]){scale.value=initial[0];variant.value=initial[1];}else{scale.value='5M-2L NOP1M';if(!scale.value)scale.selectedIndex=0;variant.value='tmasac';if(!variant.value)variant.selectedIndex=0;}variant.addEventListener('change',update);scale.addEventListener('change',update);update();
</script></html>"""
    guide = (
        "README.md" if (output / "README.md").is_file()
        else "https://github.com/brn-dev/swarm-bots/blob/main/docs/policy_parameters/diagrams/README.md"
    )
    page = page.replace("GUIDE_URL", html.escape(guide, quote=True))
    page = page.replace("VARIANTS", options(variants)).replace("SCALES", options(scales)).replace("DATA", data)
    (output / "index.html").write_text(page, encoding="utf-8")


def generate(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    diagrams = [build_diagram(row) for row in rows]
    for d in diagrams:
        slug = SCALE_LAYOUTS[d.scale].slug
        folder = output / slug
        folder.mkdir(exist_ok=True)
        base = folder / d.variant
        base.with_suffix(".svg").write_text(svg_diagram(d), encoding="utf-8")
        base.with_suffix(".mmd").write_text(mermaid_diagram(d), encoding="utf-8")
        base.with_suffix(".dot").write_text(dot_diagram(d), encoding="utf-8")
        base.with_suffix(".md").write_text(
            f"# {d.variant} · {d.scale}\n\n![Architecture]({d.variant}.svg)\n\n```mermaid\n{mermaid_diagram(d)}```\n",
            encoding="utf-8",
        )
    (output / "diagrams.json").write_text(
        json.dumps([asdict(d) for d in diagrams], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_gallery(diagrams, output)
    return diagrams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", help="default: core variants")
    parser.add_argument("--include-hidden", action="store_true")
    parser.add_argument("--scales", nargs="+", default=list(list_model_scales()), help="e.g. 2.5M 5M-2L 5M-3L 10M")
    parser.add_argument("--refresh", action="store_true", help="rebuild audited models from current presets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/policy_parameters/diagrams")
    args = parser.parse_args()
    scales = [normalize_model_scale(scale) for scale in args.scales]
    variants = args.variants or list(list_variants(include_hidden=args.include_hidden))
    unknown = set(variants) - set(list_variants(include_hidden=True))
    if unknown:
        parser.error("Unknown variants: " + ", ".join(sorted(unknown)))
    import torch

    torch.set_num_threads(1)
    diagrams = generate(load_reports(scales, variants, args.refresh), args.output_dir)
    print(f"Generated {len(diagrams)} architecture diagrams in {args.output_dir}")


if __name__ == "__main__":
    main()
