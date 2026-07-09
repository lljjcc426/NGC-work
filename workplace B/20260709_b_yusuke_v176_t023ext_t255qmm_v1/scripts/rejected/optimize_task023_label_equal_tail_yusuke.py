from __future__ import annotations

from collections import Counter

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import optimize_equivalent as oe


def _ensure_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> str:
    for init in model.graph.initializer:
        if init.name == name:
            old = numpy_helper.to_array(init)
            if old.shape == array.shape and np.array_equal(old, array):
                return name
            break
    model.graph.initializer.append(numpy_helper.from_array(array, name=name))
    return name


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    outputs = {out for node in current.graph.node for out in node.output}
    required = {
        "safe_name_130",
        "safe_name_131",
        "safe_name_132",
        "safe_name_133",
        "safe_name_134",
        "output",
    }
    if not required.issubset(outputs):
        return current, Counter()
    init_names = {init.name for init in current.graph.initializer}
    if not {
        "safe_name_11",
        "v163_shared_pad_pads_0_17",
        "pad18_pads_102_16",
        "pad18_axes_102_17",
    }.issubset(init_names):
        return current, Counter()

    two = _ensure_initializer(current, "task023_two_u8", np.array(2, dtype=np.uint8))
    eight = _ensure_initializer(current, "task023_eight_u8", np.array(8, dtype=np.uint8))
    outside = _ensure_initializer(
        current, "task023_outside_u8", np.array(255, dtype=np.uint8)
    )
    basis = _ensure_initializer(
        current,
        "task023_basis_u8",
        np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1),
    )

    remove = {
        "safe_name_16",
        "safe_name_17",
        "safe_name_132",
        "safe_name_133",
        "safe_name_134",
        "output",
    }

    inserted = False
    new_nodes: list[onnx.NodeProto] = []
    for node in current.graph.node:
        first = node.output[0] if node.output else ""
        if first in remove:
            continue
        new_nodes.append(node)
        if first == "safe_name_131":
            new_nodes.extend(
                [
                    helper.make_node(
                        "Mul",
                        ["safe_name_131", two],
                        ["task023_label2"],
                        name="task023_label2",
                    ),
                    helper.make_node(
                        "Mul",
                        ["safe_name_130", eight],
                        ["task023_label8"],
                        name="task023_label8",
                    ),
                    helper.make_node(
                        "Add",
                        ["task023_label2", "task023_label8"],
                        ["task023_label_inner"],
                        name="task023_label_inner",
                    ),
                    helper.make_node(
                        "Pad",
                        [
                            "task023_label_inner",
                            "v163_shared_pad_pads_0_17",
                            "",
                            "pad18_axes_102_17",
                        ],
                        ["task023_label_small"],
                        name="task023_label_small",
                        mode="constant",
                    ),
                    helper.make_node(
                        "Pad",
                        [
                            "task023_label_small",
                            "pad18_pads_102_16",
                            outside,
                            "pad18_axes_102_17",
                        ],
                        ["task023_label_full"],
                        name="task023_label_full",
                        mode="constant",
                    ),
                    helper.make_node(
                        "Equal",
                        ["task023_label_full", basis],
                        ["output"],
                        name="task023_output_equal",
                    ),
                ]
            )
            inserted = True

    if not inserted:
        return current, Counter()

    del current.graph.node[:]
    current.graph.node.extend(new_nodes)
    for output in current.graph.output:
        if output.name == "output":
            output.type.tensor_type.elem_type = TensorProto.BOOL
    del current.graph.value_info[:]

    stats = Counter({"task023_label_equal_tail": 1})
    stats["dead_nodes"] += oe.prune_dead(current)
    stats["initializers"] += oe.prune_initializers(current)
    onnx.checker.check_model(current, full_check=True)
    return current, stats
