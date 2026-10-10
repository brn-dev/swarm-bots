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
from typing import Any

from swarmbots.tools.inspect_policy_parameters import (
    allocation_doc_link,
    inspect_variants,
    save_report,
    write_compressed_json,
)
from swarmbots.learn import list_model_scales, list_variants
from swarmbots.learn.presets.model_scale import SCALE_LAYOUTS, normalize_model_scale

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
class Node:
    id: str
    title: str
    lines: list[str]
    role: str
    x: int
    y: int
    width: int = 280
    height: int = 0
    parameters: int | None = None
    parameter_role: str | None = None

    def __post_init__(self):
        self.height = max(82, self.text_offset + 2 + len(self.wrapped_lines) * 18)

    @property
    def wrapped_lines(self):
        return [
            part
            for line in self.lines
            for part in textwrap.wrap(line, int((self.width - 26) / 6.3), break_long_words=False) or [""]
        ]

    @property
    def text_offset(self):
        return 68 if self.parameters is not None else 47

    @property
    def parameter_label(self):
        return f"Total: {self.parameters:,} parameters" if self.parameters is not None else ""


@dataclass
class Edge:
    source: str
    target: str
    label: str = ""
    kind: str = "data"
    source_port: str = "right"
    target_port: str = "left"
    via: list[tuple[int, int]] = field(default_factory=list)


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
    width: int = 1330
    height: int = 1050

    def node(self, id, title, lines, role, x, y, width=280):
        self.nodes.append(Node(id, title, lines, role, x, y, width))
        return id

    def edge(self, source, target, label="", kind="data", source_port="right", target_port="left", via=()):
        self.edges.append(Edge(source, target, label, kind, source_port, target_port, list(via)))

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


def encoder_lines(row, config, prefix, projection=""):
    """List input projections, the repeated block's forward order, then output."""
    width = config["d_model"]
    block = prefix + ".layers.0"
    attention = (
        f"attention / block: {config['nhead']} heads ({module_count(row, block + '.self_attn'):,})"
        if config["use_agent_attention"]
        else "agent attention: disabled"
    )
    lines = []
    if projection:
        lines.append("obs MLP (once): " + projection)
        if prefix + ".global_obs_encoder" in row["module_parameter_counts"]:
            lines.extend(
                [
                    "global MLP (once): " + chain(row, prefix + ".global_obs_encoder"),
                    "add local and global token projections",
                ]
            )
    lines.extend(
        [
            f"repeat {config['num_layers']} block{'s' if config['num_layers'] != 1 else ''} · D={width}",
            f"block total: {module_count(row, block):,} per block",
        ]
    )
    temporal = config.get("temporal_model_cls")
    if temporal:
        kind = "sLSTM" if "SLSTM" in temporal else "LSTM"
        temporal_line = f"{kind} / block: hidden={width} ({module_count(row, block + '.temporal_model'):,})"
        temporal_first = config["temporal_model_order"] == "temporal_first"
        lines.append(temporal_line if temporal_first else attention)
        if config.get("inter_module_mlp"):
            lines.append("inter FF / block: " + chain(row, block + ".inter_module_feedforward"))
        lines.append(attention if temporal_first else temporal_line)
    else:
        lines.append(attention)
    lines.append("FF / block: " + chain(row, block + ".feedforward"))
    lines.append(f"output: N × {width}")
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
    if "value" in nodes:
        used = sum(node.parameters or 0 for node in d.nodes if node.parameter_role == "critic")
        assign("value", roles["critic"] - used, "critic")
    if "critic_encoder" in nodes:
        assign("critic_encoder", module_count(row, "critic.encoder"), "critic")
    if "state_bridge" in nodes:
        assign("state_bridge", module_count(row, "critic.actor_state_encoder"), "critic")
    for index in (1, 2):
        if f"head{index}" in nodes:
            assign(f"head{index}", module_count(row, f"critic.q{index}"), "critic")
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


def layout_counted_diagram(d):
    """Leave room for exact count labels while retaining the four-column sketch."""
    nodes = {node.id: node for node in d.nodes}
    positions = {
        "public": (20, 230),
        "privileged": (20, 650),
        "actions": (1170, 110),
        "representation": (380, 370),
        "actor_encoder": (380, 110),
        "actor": (775, 110),
        "value": (775, 650),
        "v": (1170, 670),
        "q_inputs": (20, 850),
        "critic_encoder": (380, 650),
        "state_bridge": (380, 480),
        "q1_network": (775, 580),
        "q2_network": (775, 850),
        "head1": (775, 580),
        "head2": (775, 850),
        "q1": (1170, 590),
        "q2": (1170, 860),
        "q_features": (380, 1000),
        "nop_bridge": (380, 1150),
        "transition": (775, 1150),
        "predictions": (1170, 1150),
        "future_actions": (20, 1120),
    }
    if "actor_encoder" in nodes and nodes["actor_encoder"].x == 675:
        positions["actor_encoder"] = (775, 110)
    if "state_bridge" in nodes and nodes["state_bridge"].x == 675:
        positions["state_bridge"] = (775, 480)
    if "representation" in nodes and nodes["representation"].role == "critic":
        positions["representation"] = (380, 650)
    for node in d.nodes:
        node.x, node.y = positions[node.id]
        node.width = 280 if node.role == "input" else 300 if node.role == "output" else 320
        node.__post_init__()
    for edge in d.edges:
        if edge.source == edge.target:
            node = nodes[edge.source]
            edge.via = [(node.x + node.width - 18, node.y - 4), (node.x + 18, node.y - 4)]
        elif (edge.source, edge.target) == ("public", "q_inputs"):
            edge.via = [(345, port(nodes["public"], "right")[1]), (345, port(nodes["q_inputs"], "right")[1])]
        elif (edge.source, edge.target) == ("actions", "actor"):
            edge.via = [(1500, 95), (935, 95)]
        elif (edge.source, edge.target) == ("actions", "q_inputs"):
            edge.via = [(1500, port(nodes["actions"], "right")[1]), (1500, 1110), (160, 1110)]
        elif (edge.source, edge.target) == ("future_actions", "transition"):
            edge.via = [(345, port(nodes["future_actions"], "right")[1]), (345, 1128), (935, 1128)]
    d.width = 1530
    d.height = max(1450, max(node.y + node.height for node in d.nodes) + 50 + len(d.notes) * 16)


def critic_lines(row, prefix):
    return [
        "element MLP: " + chain(row, prefix + ".deepset.element_encoder"),
        "masked mean across agents",
        "regressor: " + chain(row, prefix + ".deepset.set_decoder"),
        "output: one team value",
    ]


def add_readout(d, row, prefix, id, title, x, y):
    d.node(id, title, critic_lines(row, prefix), "critic", x, y)
    return id


def add_memory(d, id, config):
    if config.get("temporal_model_cls"):
        node = next(node for node in d.nodes if node.id == id)
        d.edge(
            id,
            id,
            "previous temporal state",
            "state",
            "top",
            "top",
            [(node.x + node.width - 18, node.y - 4), (node.x + 18, node.y - 4)],
        )


def build_diagram(row):
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
    shared = on_policy and row["base_policy_class"] != "MATDecPolicy" or cfg.get("shared_encoder_config") is not None
    d.node(
        "public",
        "Public observations",
        [f"local: N × {local}", f"global: {glob}", "actor-visible inputs"],
        "input",
        20,
        185,
        250,
    )
    d.node(
        "privileged",
        "Privileged context",
        [f"local: N × {private}", f"global: {private_global}", "critic only"],
        "input",
        20,
        415,
        250,
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
        1010,
        95,
        270,
    )
    d.notes = [
        f"N={n}; batch B omitted. Fixed input/output widths are task-specific.",
        "Solid = data. Dashed = causal actions / detached state. Masks and normalization are implicit.",
        "Main sketch is online networks. Frozen target copies and optimizer updates are omitted.",
    ]
    source = "representation"
    if on_policy:
        if "actor_config" in cfg:
            ac = cfg["actor_config"]
            width = ac.get("shared_encoder_latent_dim", ac.get("shared_encoder_latent_dim_per_agent"))
            kind = "Joint MLP shared encoder" if "latent_pi_dim_per_agent" in ac else "Per-agent shared encoder"
            d.node(source, kind, [chain(row, "shared_encoder.mlp"), f"reshape/output: N × {width}"], "shared", 340, 225)
        else:
            ec = cfg["encoder_config"]
            width = ec["d_model"]
            d.node(
                source,
                "Shared observation encoder" if shared else "Critic observation encoder",
                encoder_lines(row, ec, "encoder", chain(row, "encoder.local_obs_encoder")),
                "shared" if shared else "critic",
                340,
                245 if shared else 410,
            )
            add_memory(d, source, ec)
        d.edge("public", source)
        actor_source = source
        if not shared:
            ac = cfg["actor_encoder_config"]
            d.node(
                "actor_encoder",
                "Actor observation encoder",
                encoder_lines(row, ac, "actor_encoder", chain(row, "actor_encoder.local_obs_encoder")),
                "actor",
                340,
                80,
            )
            d.edge("public", "actor_encoder")
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
                f"repeat {dec['num_layers']} blocks · D={dec_width} · {dec['nhead']} heads",
            ]
            block = "decoder.layers.0" if qcx else "decoder.blocks.0"
            lines.append(f"block total: {module_count(row, block):,} per block")
            if qcx:
                lines.extend(
                    [
                        "context MLP / block: " + chain(row, block + ".context_encoder", fallback="identity (0)"),
                        f"causal attn / block: {module_count(row, block + '.query_context_attn'):,}",
                        f"memory attn / block: {module_count(row, block + '.memory_attn'):,}",
                        f"FF / block: {dec_width} → {dec['dim_feedforward']} → {dec_width} ({module_count(row, block + '.linear1') + module_count(row, block + '.linear2'):,})",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"causal self-attn / block: {module_count(row, block + '.attn1'):,}",
                        f"obs-query attn / block: {module_count(row, block + '.attn2'):,}",
                        "FF / block: " + chain(row, block + ".mlp"),
                        "decoder head: " + chain(row, "decoder.head"),
                    ]
                )
            lines.append(f"output: N × {dec_width if qcx else dec.get('latent_pi_dim') or dec_width}")
            d.node("actor", "QCX autoregressive decoder" if qcx else "Original MAT decoder", lines, "actor", 675, 80)
            d.edge(actor_source, "actor", f"N × {width}; {'obs + memory' if qcx else 'obs context'}")
            d.edge(
                "actions",
                "actor",
                "previous agents only",
                "causal",
                "bottom",
                "top",
                [(1297, 65), (815, 65)],
            )
            d.notes.append(
                "Decoder loop is across agent order, not future observations; training may teacher-force actions."
            )
        else:
            d.node("actor", "Private actor MLP", [chain(row, "actor.mlp", "actor_head"), "per agent"], "actor", 675, 95)
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
                675,
                435,
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
                675,
                435,
            )
        else:
            add_readout(d, row, "critic", "value", "Deep Set V critic", 675, 435)
        d.edge(source, "value", f"N × {width}")
        d.edge("privileged", "value", "local + global context")
        d.node("v", "V", ["scalar per world"], "output", 1010, 455, 270)
        d.edge("value", "v")
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
                340,
                230,
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
            340 if not shared else 675,
            80,
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
            actor_node.lines[-1:] = [
                "head: " + actor_chain,
                f"output: N × {cfg['actor_head_config']['hidden_dims'][-1]}",
            ]
            actor_node.__post_init__()
            d.edge("actor_encoder", "actions")
        else:
            d.node("actor", "Actor output MLP", [actor_chain, "distribution/tanh readout"], "actor", 675, 95)
            d.edge("actor_encoder", "actor", f"N × {acwidth}")
            d.edge("actor", "actions")
        d.node(
            "q_inputs",
            "Joint critic inputs",
            ["privileged local + global", f"actions: N × {action}"]
            if shared
            else ["public observations", "privileged local + global", f"actions: N × {action}"],
            "input",
            20,
            555,
            250,
        )
        if not shared:
            public_node = next(node for node in d.nodes if node.id == "public")
            input_node = next(node for node in d.nodes if node.id == "q_inputs")
            d.edge(
                "public",
                "q_inputs",
                "public",
                source_port="right",
                target_port="right",
                via=[(295, public_node.y + public_node.height / 2), (295, input_node.y + input_node.height / 2)],
            )
        d.edge("privileged", "q_inputs", source_port="bottom", target_port="top")
        d.edge(
            "actions",
            "q_inputs",
            "replay / policy",
            "causal",
            "right",
            "bottom",
            [(1297, 146), (1297, 700), (145, 700)],
        )
        joint = cfg.get("joint_critic_config")
        is_joint = row["variant"].endswith("_deepset") or row["variant"].endswith("_mlp")
        if is_joint:
            count = 1 if row["variant"].startswith("maddpg") else 2
            source_nodes = []
            for i in range(count):
                index = i + 1
                prefix = (
                    f"critic.{i}.network"
                    if row["base_policy_class"] in {"TD3Policy", "RecurrentTD3Policy"}
                    else f"critic.q{index}.network"
                )
                y = 355 + i * 185
                if joint["kind"] == "mlp":
                    lines = ["flatten observations/actions/mask", chain(row, prefix), "scalar Q"]
                else:
                    lines = critic_lines(row, prefix)
                    lines.insert(0, "global context + active count")
                d.node(f"q{index}_network", f"Independent Q{index} network", lines, "critic", 675, y)
                d.edge("q_inputs", f"q{index}_network")
                d.node(f"q{index}", f"Q{index}", ["scalar per world"], "output", 1010, y + 10, 270)
                d.edge(f"q{index}_network", f"q{index}")
                source_nodes.append(f"q{index}_network")
            if joint["kind"] == "deepset":
                source = "q_features"
                element_width = joint["element_hidden_dims"][-1]
                d.node(
                    source,
                    "Pre-pooling critic features",
                    [f"{count} × (N × {element_width})", f"concatenate → N × {count * element_width}"],
                    "critic",
                    340,
                    660,
                )
                for id in source_nodes:
                    d.edge(id, source, "element features", source_port="left", target_port="right")
        else:
            cc = cfg.get("critic_encoder_config")
            d.node(
                "critic_encoder",
                "ONE shared critic encoder",
                [
                    "local/action MLP (once): " + chain(row, "critic.encoder.local_action_encoder"),
                    "global MLP (once): " + chain(row, "critic.encoder.global_encoder", fallback="none"),
                    "add local and global token projections",
                    *encoder_lines(row, cc, "critic.encoder"),
                ],
                "critic",
                340,
                415,
            )
            d.edge("q_inputs", "critic_encoder")
            if shared:
                d.edge(source, "critic_encoder", f"N × {cfg['shared_encoder_config']['d_model']}")
            if cfg.get("actor_state_critic_input_config"):
                bridge = chain(row, "critic.actor_state_encoder")
                d.node(
                    "state_bridge",
                    "Actor-state bridge",
                    ["detach last actor state", bridge],
                    "critic",
                    675 if shared else 340,
                    300,
                )
                d.edge("actor_encoder", "state_bridge", "h/c or h/m", "state", "bottom", "top")
                d.edge("state_bridge", "critic_encoder", "append to critic inputs", "state", "bottom", "top")
            for i in [1, 2]:
                y = 355 + (i - 1) * 185
                add_readout(d, row, f"critic.q{i}", f"head{i}", f"Separate Q{i} head", 675, y)
                d.edge("critic_encoder", f"head{i}", f"N × {cc['d_model']}")
                d.node(f"q{i}", f"Q{i}", ["scalar per world"], "output", 1010, y + 10, 270)
                d.edge(f"head{i}", f"q{i}")
            source = (
                "representation"
                if shared and cfg["nop_config"]["latent_source"] == "SHARED_ENCODER"
                else "critic_encoder"
            )
            d.notes.append("The two Q heads have separate parameters; the shared transformer runs once.")
    if row["use_nop"]:
        prefix = "" if on_policy else "critic_nop."
        bridge = chain(row, prefix + "pre_transition_transform")
        prediction_chain = chain(row, prefix + "pre_predictors_transform")
        layout = SCALE_LAYOUTS[d.scale]
        block = prefix + "transition_model.encoder.layers.0"
        d.node("nop_bridge", "NOP source projection", ["from source features; before pooling", bridge], "nop", 340, 780)
        d.edge(source, "nop_bridge", "latents", source_port="bottom", target_port="top")
        d.node(
            "transition",
            "NOP transition model",
            [
                "coembed (per step): " + chain(row, prefix + "transition_model.coembed"),
                f"repeat {layout.nop_layers} blocks · D={layout.nop_width} · 4 heads",
                f"block total: {module_count(row, block):,} per block",
                f"attention / block: {module_count(row, block + '.self_attn'):,}",
                f"FF / block: {layout.nop_width} → {layout.nop_ff} → {layout.nop_width} ({module_count(row, block + '.linear1') + module_count(row, block + '.linear2'):,})",
                "head: " + chain(row, prefix + "transition_model.head"),
                "latent residual; repeated future steps",
            ],
            "nop",
            675,
            780,
        )
        d.edge("nop_bridge", "transition", "z_t")
        d.node(
            "future_actions",
            "Rollout / replay actions",
            [f"sequence: K × N × {action}", "known training trajectory", "one action per predicted step"],
            "input",
            20,
            750,
            250,
        )
        action_node = next(node for node in d.nodes if node.id == "future_actions")
        d.edge(
            "future_actions",
            "transition",
            "action sequence",
            source_port="right",
            target_port="top",
            via=[(310, action_node.y + action_node.height / 2), (310, 760), (815, 760)],
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
            1010,
            780,
            270,
        )
        d.edge("transition", "predictions", "predicted next latents")
    else:
        d.notes.append("NOP is disabled for this preset.")
    d.notes.append(
        "Counts include weights/biases and normalization. Box totals are additive; inner counts are included subtotals."
    )
    annotate_component_totals(d, row)
    layout_counted_diagram(d)
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


def port(node, side):
    return {
        "left": (node.x, node.y + node.height / 2),
        "right": (node.x + node.width, node.y + node.height / 2),
        "top": (node.x + node.width / 2, node.y),
        "bottom": (node.x + node.width / 2, node.y + node.height),
    }[side]


def edge_points(edge, nodes):
    start = port(nodes[edge.source], edge.source_port)
    end = port(nodes[edge.target], edge.target_port)
    if edge.via:
        return [start, *edge.via, end]
    if edge.source_port in {"top", "bottom"} and edge.target_port in {"top", "bottom"}:
        middle = (start[1] + end[1]) / 2
        return [start, (start[0], middle), (end[0], middle), end]
    middle = (start[0] + end[0]) / 2
    return [start, (middle, start[1]), (middle, end[1]), end]


def label_position(edge, nodes, points):
    start, end = points[0], points[-1]
    if not edge.via and {edge.source_port, edge.target_port} == {"right", "left"}:
        return (start[0] + end[0]) / 2, end[1] - 8
    if edge.via:
        sections = [(a, b) for a, b in zip(points, points[1:]) if a[1] == b[1]]
        if sections:
            a, b = max(sections, key=lambda pair: abs(pair[0][0] - pair[1][0]))
            return (a[0] + b[0]) / 2, a[1] - 8
    before = points[-2]
    return (before[0] + end[0]) / 2, (before[1] + end[1]) / 2 - 8


def edge_label_lines(edge, points):
    width = abs(points[0][0] - points[-1][0])
    if not edge.via and {edge.source_port, edge.target_port} == {"right", "left"}:
        return textwrap.wrap(edge.label, max(8, int((width - 10) / 6)), break_long_words=False)
    return [edge.label]


def svg_diagram(d):
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
    for edge in d.edges:
        points = edge_points(edge, nodes)
        coords = " ".join(f"{x:g},{y:g}" for x, y in points)
        dash = ' stroke-dasharray="6 4"' if edge.kind != "data" else ""
        bits.append(
            f'<polyline points="{coords}" fill="none" stroke="#6b7280" stroke-width="1.7" marker-end="url(#arrow)"{dash}/>'
        )
        if edge.label:
            # Put a small white label on the final incoming segment.
            lx, ly = label_position(edge, nodes, points)
            labels = edge_label_lines(edge, points)
            w = max(map(len, labels)) * 6.0 + 8
            top = ly - (len(labels) - 1) * 14
            bits.append(
                f'<rect x="{lx - w / 2:g}" y="{top - 11:g}" width="{w:g}" height="{len(labels) * 14 + 1}" fill="#fafbfe"/>'
            )
            for i, label in enumerate(labels):
                bits.append(
                    f'<text x="{lx:g}" y="{top + i * 14:g}" font-size="11" text-anchor="middle">{esc(label)}</text>'
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
        for index, line in enumerate(node.wrapped_lines):
            bits.append(
                f'<text x="{node.x + 13}" y="{node.y + node.text_offset + index * 18}" font-size="12.3">{esc(line)}</text>'
            )
    for i, note in enumerate(d.notes):
        bits.append(f'<text x="22" y="{d.height - 20 - (len(d.notes) - i) * 16}" font-size="11">{esc(note)}</text>')
    bits.append("</g></svg>")
    return "\n".join(bits)


def png_preview(d, path):
    """Optional Pillow raster preview of the same graph geometry (not needed for SVG)."""
    from PIL import Image, ImageDraw, ImageFont

    font_path = Path("C:/Windows/Fonts/arial.ttf")

    def font(size):
        return ImageFont.truetype(str(font_path) if font_path.exists() else "DejaVuSans.ttf", size)

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
    for edge in d.edges:
        points = edge_points(edge, nodes)
        draw.line(points, fill="#6b7280", width=2)
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
        if edge.label:
            lx, ly = label_position(edge, nodes, points)
            labels = edge_label_lines(edge, points)
            for i, label in enumerate(labels):
                draw.text(
                    (lx, ly - 11 - (len(labels) - i - 1) * 14),
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
        for i, line in enumerate(node.wrapped_lines):
            draw.text((node.x + 13, node.y + node.text_offset - 14 + i * 18), line, fill="#182538", font=font(12))
    for i, note in enumerate(d.notes):
        draw.text((22, d.height - 33 - (len(d.notes) - i) * 16), note, fill="#182538", font=font(11))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def mermaid_diagram(d):
    lines = ["flowchart LR", f"  %% {d.parameter_summary}", f"  %% {d.role_summary}"]
    for node in d.nodes:
        label = html.escape(
            "<br/>".join([node.title, *([node.parameter_label] if node.parameter_label else []), *node.lines]),
            quote=True,
        ).replace("&lt;br/&gt;", "<br/>")
        lines.append(f'  {node.id}["{label}"]')
    for edge in d.edges:
        arrow = "-->" if edge.kind == "data" else "-.->"
        label = f'|"{html.escape(edge.label, quote=True)}"|' if edge.label else ""
        lines.append(f"  {edge.source} {arrow}{label} {edge.target}")
    for role, (fill, border) in COLORS.items():
        lines.append(f"  classDef {role} fill:{fill},stroke:{border},color:#182538")
        ids = ",".join(node.id for node in d.nodes if node.role == role)
        if ids:
            lines.append(f"  class {ids} {role}")
    return "\n".join(lines) + "\n"


def dot_diagram(d):
    lines = [
        "digraph policy {",
        '  rankdir=LR; graph [bgcolor="white"]; node [shape=box,style="rounded,filled",fontname="Arial"];',
        f"  label={json.dumps(d.parameter_summary + chr(10) + d.role_summary, ensure_ascii=False)}; labelloc=t;",
    ]
    for node in d.nodes:
        label = "\n".join([node.title, *([node.parameter_label] if node.parameter_label else []), *node.lines])
        fill, color = COLORS[node.role]
        lines.append(f'  {node.id} [label={json.dumps(label, ensure_ascii=False)},fillcolor="{fill}",color="{color}"];')
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
            / ("" if layout.main_parameters == 5_000_000 else f"{layout.main_parameters / 1e6:g}M")
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
            generated = ROOT / "docs/policy_parameters/generated" / f"{layout.main_parameters / 1e6:g}M"
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
        slug = f"{SCALE_LAYOUTS[d.scale].main_parameters / 1e6:g}M"
        entries[d.scale + "/" + d.variant] = {"svg": svg_diagram(d), "path": f"{slug}/{d.variant}", "notes": d.notes}
    data = json.dumps(entries, ensure_ascii=False).replace("</", r"<\/")
    variants = list(dict.fromkeys(d.variant for d in diagrams))
    scales = list(dict.fromkeys(d.scale for d in diagrams))

    def options(values):
        return "".join(f"<option>{html.escape(v)}</option>" for v in values)

    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Policy architecture diagrams</title><style>
body{font:15px Arial,sans-serif;margin:0;color:#182538;background:#eef2f7}header{padding:22px 28px;background:#fff;border-bottom:1px solid #d9e1eb}h1{font-size:24px;margin:0 0 14px}label{margin-right:18px}select{padding:8px;max-width:320px;font:inherit}nav{margin:12px 0 0}a{color:#3565a8;margin-right:18px}.legend{display:flex;gap:18px;margin:16px 0 0;font-size:13px}.legend span:before{content:"";display:inline-block;width:12px;height:12px;background:var(--color);margin-right:6px;border-radius:3px}main{padding:18px;overflow:auto}.canvas{background:#fafbfe;border-radius:12px;max-width:1440px;margin:auto;box-shadow:0 2px 14px #14213a13}.canvas svg{display:block;width:100%;height:auto;min-width:920px}p{margin:12px 0 0;font-size:13px;color:#526176}
</style><header><h1>Policy architecture diagrams</h1><label>Variant <select id="variant">VARIANTS</select></label><label>Scale <select id="scale">SCALES</select></label><nav><a id="svg" download>SVG</a><a id="mermaid" download>Mermaid source</a><a id="dot" download>Graphviz source</a><a href="GUIDE_URL">Reading guide</a></nav><div class="legend"><span style="--color:#edf2f7">Inputs</span><span style="--color:#e8f1ff">Shared</span><span style="--color:#e9f5ee">Actor</span><span style="--color:#fff0df">Critic</span><span style="--color:#f0eafa">NOP</span></div><p>Solid arrows show data; dashed arrows show causal actions or detached state. Batch dimension and training-only target copies are omitted.</p></header><main><div id="canvas" class="canvas"></div></main><script>
const entries=DATA;const variant=document.getElementById('variant'),scale=document.getElementById('scale');function update(){const item=entries[scale.value+'/'+variant.value];document.getElementById('canvas').innerHTML=item.svg;for(const [id,ext] of [['svg','svg'],['mermaid','mmd'],['dot','dot']])document.getElementById(id).href=item.path+'.'+ext;location.hash=encodeURIComponent(scale.value)+'/'+variant.value;}const initial=decodeURIComponent(location.hash.slice(1)).split('/');if(initial.length===2&&entries[initial[0]+'/'+initial[1]]){scale.value=initial[0];variant.value=initial[1];}else{scale.value='5M NOP1M';if(!scale.value)scale.selectedIndex=0;variant.value='tmasac';if(!variant.value)variant.selectedIndex=0;}variant.addEventListener('change',update);scale.addEventListener('change',update);update();
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
        slug = f"{SCALE_LAYOUTS[d.scale].main_parameters / 1e6:g}M"
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
    parser.add_argument("--scales", nargs="+", default=list(list_model_scales()), help="e.g. 2.5M 5M 10M")
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
