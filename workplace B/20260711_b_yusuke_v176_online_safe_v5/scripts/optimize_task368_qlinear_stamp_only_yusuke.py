from __future__ import annotations

from collections import Counter

import numpy as np
import onnx
from onnx import helper, numpy_helper

import optimize_equivalent as oe


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    outputs = {output for node in current.graph.node for output in node.output if output}
    required = {"cornf", "patch", "patchf", "patchf2", "W", "stampX", "outU10"}
    if not required.issubset(outputs):
        return current, Counter()

    additions = {
        "task368_qscale": np.asarray(1.0, dtype=np.float32),
        "task368_u8_zero": np.asarray(0, dtype=np.uint8),
    }
    existing = {item.name for item in current.graph.initializer}
    for name, value in additions.items():
        if name not in existing:
            current.graph.initializer.append(numpy_helper.from_array(value, name=name))

    remove_outputs = {"patchf", "patchf2", "W", "stampX", "outU10"}
    nodes: list[onnx.NodeProto] = []
    for node in current.graph.node:
        first_output = node.output[0] if node.output else ""
        if first_output in remove_outputs:
            if first_output == "patchf":
                nodes.extend(
                    [
                        helper.make_node(
                            "Gather",
                            ["patch", "flipi"],
                            ["task368_patch_flip_r"],
                            axis=2,
                            name="task368_patch_flip_r",
                        ),
                        helper.make_node(
                            "Gather",
                            ["task368_patch_flip_r", "flipi"],
                            ["task368_W_u8"],
                            axis=3,
                            name="task368_patch_flip_c",
                        ),
                        helper.make_node(
                            "Cast",
                            ["cornf"],
                            ["task368_corner_u8"],
                            to=onnx.TensorProto.UINT8,
                            name="task368_corner_u8",
                        ),
                        helper.make_node(
                            "QLinearConv",
                            [
                                "task368_corner_u8",
                                "task368_qscale",
                                "task368_u8_zero",
                                "task368_W_u8",
                                "task368_qscale",
                                "task368_u8_zero",
                                "task368_qscale",
                                "task368_u8_zero",
                            ],
                            ["outU10"],
                            kernel_shape=[4, 4],
                            pads=[3, 3, 0, 0],
                            name="task368_qlinear_stamp",
                        ),
                    ]
                )
            continue
        nodes.append(onnx.NodeProto.FromString(node.SerializeToString()))

    del current.graph.node[:]
    current.graph.node.extend(nodes)
    stats = Counter({"task368_qlinear_stamp": 1})
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
