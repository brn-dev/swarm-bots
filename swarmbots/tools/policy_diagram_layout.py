"""Content-sized policy lanes and orthogonal routes around components and labels."""

from __future__ import annotations

import heapq
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swarmbots.tools.diagram_policy_architectures import Diagram, Edge, Node

Point = tuple[float, float]


@dataclass
class Bounds:
    x: float
    y: float
    width: float
    height: float

    def expanded(self, margin: float) -> Bounds:
        return Bounds(self.x - margin, self.y - margin, self.width + 2 * margin, self.height + 2 * margin)

    def overlaps(self, other: Bounds) -> bool:
        return (
            max(self.x, other.x) < min(self.x + self.width, other.x + other.width)
            and max(self.y, other.y) < min(self.y + self.height, other.y + other.height)
        )


@dataclass
class EdgeLabel(Bounds):
    lines: list[str]


@dataclass
class DiagramLane:
    title: str
    y: int
    height: int


def node_bounds(node: Node) -> Bounds:
    return Bounds(node.x, node.y, node.width, node.height)


def crosses_bounds(start: Point, end: Point, bounds: Bounds) -> bool:
    if start[0] == end[0]:
        return bounds.x < start[0] < bounds.x + bounds.width and max(min(start[1], end[1]), bounds.y) < min(
            max(start[1], end[1]), bounds.y + bounds.height,
        )
    return bounds.y < start[1] < bounds.y + bounds.height and max(min(start[0], end[0]), bounds.x) < min(
        max(start[0], end[0]), bounds.x + bounds.width,
    )


def port(node: Node, side: str, offset: float = 0.5) -> Point:
    return {
        "left": (node.x, node.y + node.height * offset),
        "right": (node.x + node.width, node.y + node.height * offset),
        "top": (node.x + node.width * offset, node.y),
        "bottom": (node.x + node.width * offset, node.y + node.height),
    }[side]


def edge_points(edge: Edge, nodes: dict[str, Node]) -> list[Point]:
    return [
        port(nodes[edge.source], edge.source_port, edge.source_offset),
        *edge.via,
        port(nodes[edge.target], edge.target_port, edge.target_offset),
    ]


def place_components(diagram: Diagram) -> None:
    nodes = {node.id: node for node in diagram.nodes}
    component_width = 560
    column_step = component_width + 120
    columns = (60, 480, 480 + column_step, 480 + 2 * column_step)
    for node in diagram.nodes:
        node.width = 280 if node.role == "input" else 300 if node.role == "output" else component_width
        node.__post_init__()

    actor_ids = ["public", "actor_encoder", "actor", "actions"]
    shared = "representation" in nodes and nodes["representation"].role == "shared"
    if shared:
        actor_ids.insert(1, "representation")
    actor_nodes = [nodes[id] for id in actor_ids if id in nodes]
    actor_height = max(node.height for node in actor_nodes)
    actor_top = 180
    for node in actor_nodes:
        column = 0 if node.id == "public" else 3 if node.id == "actions" else 2 if node.id == "actor" else 1
        if node.id == "actor_encoder" and shared:
            column = 2
        node.x = columns[column]
        node.y = actor_top + (actor_height - node.height) // 2
    diagram.lanes = [DiagramLane("Actor / public features", actor_top - 40, actor_height + 65)]

    critic_top = actor_top + actor_height + 180
    if "state_bridge" in nodes:
        bridge = nodes["state_bridge"]
        bridge.x, bridge.y = columns[1], actor_top + actor_height + 90
        critic_top = bridge.y + bridge.height + 90
    readout_ids = [id for id in ("value", "head1", "head2", "q1_network", "q2_network") if id in nodes]
    readout_height = sum(nodes[id].height for id in readout_ids) + 100 * (len(readout_ids) - 1)
    input_ids = [id for id in ("privileged", "q_inputs") if id in nodes]
    input_height = sum(nodes[id].height for id in input_ids) + 90 * (len(input_ids) - 1)
    encoder_ids = [id for id in ("critic_encoder", "critic_encoder2", "representation") if id in nodes and nodes[id].role == "critic"]
    encoder_height = sum(nodes[id].height for id in encoder_ids) + 100 * (len(encoder_ids) - 1)
    critic_height = max(readout_height, input_height, encoder_height)
    y = critic_top + (critic_height - readout_height) // 2
    for id in readout_ids:
        node = nodes[id]
        node.x, node.y = columns[2], y
        output = nodes["v" if id == "value" else "q1" if id in {"head1", "q1_network"} else "q2"]
        output.x, output.y = columns[3], y + (node.height - output.height) // 2
        y += node.height + 100
    y = critic_top + (critic_height - input_height) // 2
    for id in input_ids:
        node = nodes[id]
        node.x, node.y = columns[0], y
        y += node.height + 90
    y = critic_top + (critic_height - encoder_height) // 2
    for id in encoder_ids:
        node = nodes[id]
        node.x, node.y = columns[1], y
        y += node.height + 100
    critic_bottom = critic_top + critic_height
    if "q_features" in nodes:
        node = nodes["q_features"]
        node.x, node.y = columns[1], critic_bottom + 70
        critic_bottom = node.y + node.height
    diagram.lanes.append(DiagramLane("Critic", critic_top - 40, critic_bottom - critic_top + 65))

    if "transition" in nodes:
        nop_nodes = [nodes[id] for id in ("future_actions", "nop_bridge", "transition", "predictions")]
        nop_height = max(node.height for node in nop_nodes)
        nop_top = critic_bottom + 180
        for column, node in enumerate(nop_nodes):
            node.x, node.y = columns[column], nop_top + (nop_height - node.height) // 2
        diagram.lanes.append(DiagramLane("Next-observation prediction · training", nop_top - 40, nop_height + 65))
    diagram.width = columns[-1] + component_width + 120
    diagram.height = max(node.y + node.height for node in diagram.nodes) + 140 + len(diagram.notes) * 16


def assign_ports(diagram: Diagram) -> None:
    nodes = {node.id: node for node in diagram.nodes}
    endpoints: dict[tuple[str, str], list[tuple[Edge, str]]] = defaultdict(list)
    for edge in diagram.edges:
        source, target = nodes[edge.source], nodes[edge.target]
        edge.via = []
        edge.label_box = None
        if edge.source == edge.target:
            edge.source_port, edge.target_port = "top", "top"
        elif (edge.source, edge.target) == ("actions", "actor"):
            edge.source_port, edge.target_port = "right", "top"
        elif (edge.source, edge.target) == ("actions", "q_inputs"):
            edge.source_port, edge.target_port = "right", "right"
        elif (edge.source, edge.target) == ("public", "q_inputs"):
            edge.source_port, edge.target_port = "right", "right"
        elif edge.target == "q_features" and source.x == target.x:
            edge.source_port, edge.target_port = "right", "right"
        elif source.x == target.x:
            edge.source_port, edge.target_port = ("bottom", "top") if source.y < target.y else ("top", "bottom")
        elif source.x < target.x:
            edge.source_port, edge.target_port = "right", "left"
        else:
            edge.source_port, edge.target_port = "left", "right"
        endpoints[edge.source, edge.source_port].append((edge, "source"))
        endpoints[edge.target, edge.target_port].append((edge, "target"))
    for connections in endpoints.values():
        for index, (edge, end) in enumerate(connections, start=1):
            setattr(edge, end + "_offset", index / (len(connections) + 1))


def simplified(points: list[Point]) -> list[Point]:
    result: list[Point] = []
    for point in points:
        if result and point == result[-1]:
            continue
        if len(result) >= 2 and (
            result[-2][0] == result[-1][0] == point[0] or result[-2][1] == result[-1][1] == point[1]
        ):
            result[-1] = point
        else:
            result.append(point)
    return result


def crossing_cost(start: Point, end: Point, segments: list[tuple[Point, Point]]) -> float:
    cost = 0.0
    horizontal = start[1] == end[1]
    for previous_start, previous_end in segments:
        if horizontal == (previous_start[1] == previous_end[1]):
            fixed, moving = (1, 0) if horizontal else (0, 1)
            if start[fixed] == previous_start[fixed]:
                overlap = min(max(start[moving], end[moving]), max(previous_start[moving], previous_end[moving])) - max(
                    min(start[moving], end[moving]), min(previous_start[moving], previous_end[moving]),
                )
                if overlap > 0:
                    cost += 120 + overlap * 3
        else:
            h_start, h_end, v_start, v_end = (
                (start, end, previous_start, previous_end) if horizontal else (previous_start, previous_end, start, end)
            )
            if min(h_start[0], h_end[0]) < v_start[0] < max(h_start[0], h_end[0]) and min(
                v_start[1], v_end[1],
            ) < h_start[1] < max(v_start[1], v_end[1]):
                cost += 160
    return cost


def orthogonal_route(
    start: Point, end: Point, obstacles: list[Bounds], segments: list[tuple[Point, Point]], canvas: Bounds,
) -> list[Point]:
    """Search obstacle boundaries, preferring shorter routes with fewer bends and crossings."""
    xs = sorted({start[0], end[0], canvas.x, canvas.x + canvas.width, *(
        x for box in obstacles for x in (box.x, box.x + box.width) if canvas.x <= x <= canvas.x + canvas.width
    )})
    ys = sorted({start[1], end[1], canvas.y, canvas.y + canvas.height, *(
        y for box in obstacles for y in (box.y, box.y + box.height) if canvas.y <= y <= canvas.y + canvas.height
    )})
    origin = (xs.index(start[0]), ys.index(start[1]), -1)
    goal = (xs.index(end[0]), ys.index(end[1]))
    queue = [(0.0, 0.0, origin)]
    costs = {origin: 0.0}
    parents: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    segment_costs: dict[tuple[Point, Point], float | None] = {}
    while queue:
        _, cost, state = heapq.heappop(queue)
        if cost != costs[state]:
            continue
        i, j, direction = state
        if (i, j) == goal:
            route = [(xs[i], ys[j])]
            while state in parents:
                state = parents[state]
                route.append((xs[state[0]], ys[state[1]]))
            return simplified(list(reversed(route)))
        point = (xs[i], ys[j])
        for ni, nj, axis in ((i - 1, j, 0), (i + 1, j, 0), (i, j - 1, 1), (i, j + 1, 1)):
            if not (0 <= ni < len(xs) and 0 <= nj < len(ys)):
                continue
            neighbor = (xs[ni], ys[nj])
            key = (point, neighbor)
            if key not in segment_costs:
                segment_costs[key] = None if any(crosses_bounds(point, neighbor, box) for box in obstacles) else (
                    abs(point[0] - neighbor[0]) + abs(point[1] - neighbor[1]) + crossing_cost(point, neighbor, segments)
                )
            step_cost = segment_costs[key]
            if step_cost is None:
                continue
            next_cost = cost + step_cost + (36 if direction not in {-1, axis} else 0)
            next_state = (ni, nj, axis)
            if next_cost < costs.get(next_state, float("inf")):
                costs[next_state], parents[next_state] = next_cost, state
                remaining = abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                heapq.heappush(queue, (next_cost + remaining, next_cost, next_state))
    raise ValueError(f"No clear diagram route from {start} to {end}")


def place_label(
    text: str, points: list[Point], obstacles: list[Bounds], segments: list[tuple[Point, Point]], canvas: Bounds,
    max_line_width: int = 22,
) -> EdgeLabel:
    candidates = sorted(zip(points, points[1:]), key=lambda pair: (
        pair[0][1] != pair[1][1], -abs(pair[0][0] - pair[1][0]) - abs(pair[0][1] - pair[1][1]),
    ))
    for start, end in candidates:
        horizontal = start[1] == end[1]
        length = abs(start[0] - end[0]) + abs(start[1] - end[1])
        for line_width in (max_line_width, 16, 11):
            lines = textwrap.wrap(text, line_width, break_long_words=False)
            width, height = max(map(len, lines)) * 6 + 12, len(lines) * 14 + 4
            if length < (width if horizontal else height) + 10:
                continue
            axis = 0 if horizontal else 1
            half_extent = (width if horizontal else height) / 2
            lower = min(start[axis], end[axis]) + half_extent + 5
            upper = max(start[axis], end[axis]) - half_extent - 5
            centers = [start[axis] + (end[axis] - start[axis]) * fraction for fraction in (0.5, 0.3, 0.7)]
            boundaries = {lower, upper}
            for box in obstacles:
                origin, extent = (box.x, box.width) if horizontal else (box.y, box.height)
                boundaries.update((origin - half_extent - 5, origin + extent + half_extent + 5))
            centers.extend(sorted(boundaries, key=lambda position: abs(position - (lower + upper) / 2)))
            for center in dict.fromkeys(centers):
                if not lower <= center <= upper:
                    continue
                x = (center if horizontal else start[0]) - width / 2
                y = start[1] if horizontal else center
                tops = [y - height / 2] if not horizontal else [
                    top for gap in (8, 24) for top in (y - height - gap, y + gap)
                ]
                for top in tops:
                    label = EdgeLabel(x, top, width, height, lines)
                    if (
                        canvas.x <= x and x + width <= canvas.x + canvas.width
                        and canvas.y <= top and top + height <= canvas.y + canvas.height
                        and not any(label.expanded(5).overlaps(box) for box in obstacles)
                        and not any(crosses_bounds(a, b, label.expanded(4)) for a, b in segments)
                    ):
                        return label
    raise ValueError(f"No clear label position for {text!r}")


def route_connections(diagram: Diagram) -> None:
    nodes = {node.id: node for node in diagram.nodes}
    obstacles = [node_bounds(node) for node in diagram.nodes]
    segments: list[tuple[Point, Point]] = []
    canvas = Bounds(20, 100, diagram.width - 40, diagram.height - 200 - len(diagram.notes) * 16)
    directions = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}
    attachment_zones = []
    for edge in diagram.edges:
        for id, side, offset in (
            (edge.source, edge.source_port, edge.source_offset), (edge.target, edge.target_port, edge.target_offset),
        ):
            x, y = port(nodes[id], side, offset)
            dx, dy = directions[side]
            zone = Bounds(min(x, x + dx * 28), min(y, y + dy * 28), abs(dx * 28), abs(dy * 28))
            attachment_zones.append((edge, zone.expanded(18)))
    edges = sorted(diagram.edges, key=lambda edge: (
        2 if edge.source == "actions" else 1 if edge.target == "q_features" else 0,
        nodes[edge.source].y if edge.target == "q_features" else (
            abs(nodes[edge.source].x - nodes[edge.target].x) + abs(nodes[edge.source].y - nodes[edge.target].y)
        ),
    ))
    for edge in edges:
        start = port(nodes[edge.source], edge.source_port, edge.source_offset)
        end = port(nodes[edge.target], edge.target_port, edge.target_offset)
        dx, dy = directions[edge.source_port]
        first = (start[0] + dx * 28, start[1] + dy * 28)
        dx, dy = directions[edge.target_port]
        last = (end[0] + dx * 28, end[1] + dy * 28)
        stops = [first]
        if edge.source == edge.target:
            top = nodes[edge.source].y - 60
            stops.extend([(first[0], top), (last[0], top)])
        elif (edge.source, edge.target) == ("actions", "actor"):
            top = min(nodes["actor"].y, nodes["actions"].y) - 70
            stops.extend([(diagram.width - 30, first[1]), (diagram.width - 30, top), (last[0], top)])
        elif (edge.source, edge.target) == ("actions", "q_inputs"):
            rail = diagram.lanes[1].y - 25
            stops.extend([(diagram.width - 30, first[1]), (diagram.width - 30, rail), (last[0], rail)])
        elif edge.target == "q_features":
            returns = [connection for connection in diagram.edges if connection.target == "q_features"]
            left = nodes[edge.target].x + nodes[edge.target].width
            index = returns.index(edge) + 1
            channel = left + 40 * index if nodes[edge.source].x == nodes[edge.target].x else (
                left + (nodes[edge.source].x - left) * index / (len(returns) + 1)
            )
            stops.extend([(channel, first[1]), (channel, last[1])])
        stops.append(last)
        points = [start]
        for a, b in zip(stops, stops[1:]):
            points.extend(orthogonal_route(a, b, [box.expanded(18) for box in obstacles], segments, canvas))
        points.extend([last, end])
        points = simplified(points)
        if edge.label:
            label_obstacles = [*obstacles, *(zone for owner, zone in attachment_zones if owner is not edge)]
            edge.label_box = place_label(
                edge.label, points, label_obstacles, segments, canvas, 16 if edge.target == "q_features" else 22,
            )
            obstacles.append(edge.label_box)
        edge.via = points[1:-1]
        segments.extend(zip(points, points[1:]))


def layout_diagram(diagram: Diagram) -> None:
    place_components(diagram)
    assign_ports(diagram)
    route_connections(diagram)
