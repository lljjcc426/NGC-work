from __future__ import annotations

from collections import Counter

import numpy as np
import onnx
from onnx import helper, numpy_helper

import optimize_equivalent as oe


def _initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name=name)


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    outputs = {output for node in current.graph.node for output in node.output if output}
    required = {
        "labc16",
        "occf",
        "aconv",
        "cornf",
        "occb",
        "patch",
        "patchf",
        "patchf2",
        "W",
        "stampX",
        "outU10",
    }
    if not required.issubset(outputs):
        return current, Counter()

    initializer_names = {item.name for item in current.graph.initializer}
    additions = {
        "task368_qscale": np.asarray(1.0, dtype=np.float32),
        "task368_u8_zero": np.asarray(0, dtype=np.uint8),
        "task368_i8_zero": np.asarray(0, dtype=np.int8),
        "task368_corner_kernel": np.asarray(
            [[[[0, -3], [-3, 1]]]], dtype=np.int8
        ),
    }
    for name, value in additions.items():
        if name not in initializer_names:
            current.graph.initializer.append(_initializer(name, value))

    remove_outputs = {
        "labc16",
        "occf",
        "aconv",
        "cornf",
        "not5",
        "colb",
        "colu",
        "patchf",
        "patchf2",
        "W",
        "stampX",
        "outU10",
    }
    new_nodes: list[onnx.NodeProto] = []
    for node in current.graph.node:
        first_output = node.output[0] if node.output else ""
        if first_output in remove_outputs:
            if first_output == "cornf":
                continue
            if first_output == "patchf":
                new_nodes.extend(
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

        new_nodes.append(onnx.NodeProto.FromString(node.SerializeToString()))
        if first_output == "occb":
            new_nodes.extend(
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
                ]
            )
        elif first_output == "is5":
            new_nodes.append(
                helper.make_node(
                    "Where",
                    ["is5", "u0", "L"],
                    ["colu"],
                    name="task368_non_gray_labels",
                )
            )

    del current.graph.node[:]
    current.graph.node.extend(new_nodes)

    stats = Counter(
        {
            "task368_qlinear_corners": 1,
            "task368_qlinear_stamp": 1,
            "task368_bbox_where": 1,
        }
    )
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
