from __future__ import annotations

from collections import Counter

import onnx
from onnx import helper

import optimize_equivalent as oe


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    outputs = {output for node in current.graph.node for output in node.output if output}
    required = {
        "aH",
        "last_u8",
        "top_mrow",
        "bot_mrow",
        "p100",
        "dbestT",
        "dbestB",
        "Tle",
        "vTie",
    }
    if not required.issubset(outputs | {init.name for init in current.graph.initializer}):
        return current, Counter()

    remove_outputs = {"dbestT", "dbestB", "Tle", "vTie"}
    replacement = [
        helper.make_node("Add", ["aH", "aH"], ["task328_row_twice"], name="task328_row_twice"),
        helper.make_node(
            "Add",
            ["task328_row_twice", "p100"],
            ["task328_row_biased"],
            name="task328_row_biased",
        ),
        helper.make_node(
            "Add",
            ["last_u8", "bot_mrow"],
            ["task328_split_base"],
            name="task328_split_base",
        ),
        helper.make_node(
            "Add",
            ["task328_split_base", "p100"],
            ["task328_split_biased"],
            name="task328_split_biased",
        ),
        helper.make_node(
            "Sub",
            ["task328_split_biased", "top_mrow"],
            ["task328_split"],
            name="task328_split",
        ),
        helper.make_node(
            "LessOrEqual",
            ["task328_row_biased", "task328_split"],
            ["Tle"],
            name="Tle",
        ),
        helper.make_node(
            "Equal",
            ["task328_row_biased", "task328_split"],
            ["vTie"],
            name="vTie",
        ),
    ]

    inserted = False
    new_nodes: list[onnx.NodeProto] = []
    for node in current.graph.node:
        first_output = node.output[0] if node.output else ""
        if first_output in remove_outputs:
            if not inserted:
                new_nodes.extend(replacement)
                inserted = True
            continue
        new_nodes.append(onnx.NodeProto.FromString(node.SerializeToString()))

    if not inserted:
        return current, Counter()

    del current.graph.node[:]
    current.graph.node.extend(new_nodes)
    stats = Counter({"task328_corner_distance": 1})
    stats["dead_nodes"] += oe.prune_dead(current)
    stats["initializers"] += oe.prune_initializers(current)

    live_outputs = {
        output
        for node in current.graph.node
        for output in node.output
        if output
    }
    kept_value_info = [
        onnx.ValueInfoProto.FromString(item.SerializeToString())
        for item in current.graph.value_info
        if item.name in live_outputs
    ]
    del current.graph.value_info[:]
    current.graph.value_info.extend(kept_value_info)

    onnx.checker.check_model(current, full_check=True)
    onnx.shape_inference.infer_shapes(current, strict_mode=True)
    return current, stats
