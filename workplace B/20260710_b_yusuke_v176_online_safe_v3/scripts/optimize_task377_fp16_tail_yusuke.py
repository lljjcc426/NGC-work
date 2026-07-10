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


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    outputs = {out for node in current.graph.node for out in node.output if out}
    required = {"OHb", "OHf", "Xs", "X", "ab", "R", "output"}
    if not required.issubset(outputs):
        return current, Counter()

    sk = None
    for init in current.graph.initializer:
        if init.name == "SK":
            sk = numpy_helper.to_array(init)
            break
    if sk is None:
        return current, Counter()
    sk16 = _ensure_initializer(current, "task377_SK_f16", sk.astype(np.float16))

    stats = Counter()
    for node in current.graph.node:
        if node.op_type == "Cast" and node.output and node.output[0] in {"OHf", "R"}:
            for attr in node.attribute:
                if attr.name == "to" and int(attr.i) != TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT16
                    stats["tail_cast_f16"] += 1
        elif node.op_type == "Einsum" and node.output and node.output[0] == "Xs":
            if len(node.input) >= 2 and node.input[1] == "SK":
                node.input[1] = sk16
                stats["sk_f16"] += 1

    if not stats:
        return current, Counter()

    for out in current.graph.output:
        if out.name == "output":
            out.type.tensor_type.elem_type = TensorProto.FLOAT16
    del current.graph.value_info[:]
    stats["dead_nodes"] += oe.prune_dead(current)
    stats["initializers"] += oe.prune_initializers(current)
    onnx.checker.check_model(current, full_check=True)
    return current, stats
