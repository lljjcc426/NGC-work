from __future__ import annotations

from collections import Counter

import numpy as np
import onnx
from onnx import helper, numpy_helper

import optimize_equivalent as oe


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    outputs = {output for node in current.graph.node for output in node.output if output}
    required = {"Kvu", "TP1", "TP2", "Tsum", "T"}
    if not required.issubset(outputs):
        return current, Counter()

    current.graph.initializer.extend(
        [
            numpy_helper.from_array(
                np.zeros((1, 1, 1, 5), dtype=np.uint8),
                name="task209_zero_row",
            ),
            numpy_helper.from_array(
                np.array([3, 3, 0, 1], dtype=np.int64),
                name="task209_stack_pads",
            ),
        ]
    )

    remove_outputs = {"TP1", "TP2", "Tsum", "T"}
    replacement = [
        helper.make_node(
            "Concat",
            ["Kvu", "task209_zero_row", "Kvu"],
            ["task209_template_stack"],
            name="task209_template_stack",
            axis=2,
        ),
        helper.make_node(
            "Pad",
            [
                "task209_template_stack",
                "task209_stack_pads",
                "",
                "pad18_axes_157_28",
            ],
            ["task209_template_fill"],
            name="task209_template_fill",
        ),
        helper.make_node(
            "Add",
            ["task209_template_fill", "Tbase"],
            ["T"],
            name="T",
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
    stats = Counter({"task209_template_stack": 1})
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
