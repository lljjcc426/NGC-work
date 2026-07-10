from __future__ import annotations

from collections import Counter

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

import optimize_equivalent as oe


def _ensure_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> str:
    for init in model.graph.initializer:
        if init.name == name:
            old = numpy_helper.to_array(init)
            if old.shape == array.shape and np.array_equal(old, array):
                return name
            raise RuntimeError(f"initializer name conflict: {name}")
    model.graph.initializer.append(numpy_helper.from_array(array, name=name))
    return name


def _cast_to(node: onnx.NodeProto, output: str, dtype: int) -> bool:
    if node.op_type != "Cast" or list(node.output) != [output]:
        return False
    for attr in node.attribute:
        if attr.name == "to":
            return int(attr.i) == dtype
    return False


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    outputs = {out for node in current.graph.node for out in node.output if out}
    required = {"safe_name_352", "safe_name_353", "safe_name_354", "safe_name_654", "safe_name_655"}
    if not required.issubset(outputs):
        return current, Counter()

    zero_u8 = _ensure_initializer(current, "task018_zero_u8_topk", np.array(0, dtype=np.uint8))
    stats = Counter()
    new_nodes: list[onnx.NodeProto] = []
    for node in current.graph.node:
        if _cast_to(node, "safe_name_353", TensorProto.FLOAT16):
            stats["drop_first_topk_cast"] += 1
            continue
        if _cast_to(node, "safe_name_654", TensorProto.FLOAT16):
            stats["drop_second_topk_cast"] += 1
            continue

        copied = onnx.NodeProto.FromString(node.SerializeToString())
        if copied.op_type == "TopK" and copied.input and copied.input[0] == "safe_name_353":
            copied.input[0] = "safe_name_352"
            stats["first_topk_u8"] += 1
        elif copied.op_type == "TopK" and copied.input and copied.input[0] == "safe_name_654":
            copied.input[0] = "safe_name_653"
            stats["second_topk_u8"] += 1
        elif copied.op_type == "Greater" and copied.input and copied.input[0] == "safe_name_655":
            copied.input[1] = zero_u8
            stats["greater_u8_zero"] += 1
        new_nodes.append(copied)

    if not stats:
        return current, Counter()

    del current.graph.node[:]
    current.graph.node.extend(new_nodes)
    del current.graph.value_info[:]
    stats["dead_nodes"] += oe.prune_dead(current)
    stats["initializers"] += oe.prune_initializers(current)
    onnx.checker.check_model(current, full_check=True)
    return current, stats
