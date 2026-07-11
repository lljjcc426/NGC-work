from __future__ import annotations

from collections import Counter

import numpy as np
import onnx
from onnx import helper, numpy_helper

import optimize_equivalent as oe


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    outputs = {output for node in current.graph.node for output in node.output if output}
    required = {"labc16", "occf", "aconv", "cornf", "occb", "stampX"}
    if not required.issubset(outputs):
        return current, Counter()

    additions = {
        "task368_qscale": np.asarray(1.0, dtype=np.float32),
        "task368_u8_zero": np.asarray(0, dtype=np.uint8),
        "task368_i8_zero": np.asarray(0, dtype=np.int8),
        "task368_corner_kernel": np.asarray(
            [[[[0, -3], [-3, 1]]]], dtype=np.int8
        ),
    }
    existing = {item.name for item in current.graph.initializer}
    for name, value in additions.items():
        if name not in existing:
            current.graph.initializer.append(numpy_helper.from_array(value, name=name))

    remove_outputs = {"labc16", "occf", "aconv", "cornf"}
    nodes: list[onnx.NodeProto] = []
    for node in current.graph.node:
        first_output = node.output[0] if node.output else ""
        if first_output in remove_outputs:
            continue
        nodes.append(onnx.NodeProto.FromString(node.SerializeToString()))
        if first_output == "occb":
            nodes.extend(
                [
                    helper.make_node(
                        "Cast",
                        ["occb"],
                        ["task368_occ_u8"],
                        to=onnx.TensorProto.UINT8,
                        name="task368_occ_u8",
                    ),
                    helper.make_node(
                        "QLinearConv",
                        [
                            "task368_occ_u8",
                            "task368_qscale",
                            "task368_u8_zero",
                            "task368_corner_kernel",
                            "task368_qscale",
                            "task368_i8_zero",
                            "task368_qscale",
                            "task368_u8_zero",
                        ],
                        ["task368_corner_u8"],
                        kernel_shape=[2, 2],
                        pads=[1, 1, 0, 0],
                        name="task368_qlinear_corners",
                    ),
                    helper.make_node(
                        "Cast",
                        ["task368_corner_u8"],
                        ["cornf"],
                        to=onnx.TensorProto.FLOAT16,
                        name="task368_corner_f16",
                    ),
                ]
            )

    del current.graph.node[:]
    current.graph.node.extend(nodes)
    stats = Counter({"task368_qlinear_corners": 1})
    stats["dead_nodes"] += oe.prune_dead(current)
    stats["initializers"] += oe.prune_initializers(current)

    live = {output for node in current.graph.node for output in node.output if output}
    kept = [
        onnx.ValueInfoProto.FromString(item.SerializeToString())
        for item in current.graph.value_info
        if item.name in live
    ]
    del current.graph.value_info[:]
    current.graph.value_info.extend(kept)
    onnx.checker.check_model(current, full_check=True)
    onnx.shape_inference.infer_shapes(current, strict_mode=True)
    return current, stats
