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
            break
    model.graph.initializer.append(numpy_helper.from_array(array, name=name))
    return name


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    outputs = {out for node in current.graph.node for out in node.output}
    required = {"safe_name_150", "safe_name_151", "safe_name_152", "safe_name_153", "safe_name_154", "safe_name_155", "output"}
    if not required.issubset(outputs):
        return current, Counter()

    scale = _ensure_initializer(
        current, "task255_qmm_scale", np.array(1.0, dtype=np.float32)
    )
    zero_u8 = _ensure_initializer(current, "task255_zero_u8_matmul", np.array(0, dtype=np.uint8))
    stats = Counter()
    for node in current.graph.node:
        if node.output and node.output[0] in {"safe_name_152", "safe_name_153"} and node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i != TensorProto.UINT8:
                    attr.i = TensorProto.UINT8
                    stats["cast_matmul_inputs_u8"] += 1
        if node.output and node.output[0] == "safe_name_154" and node.op_type == "MatMul":
            node.op_type = "QLinearMatMul"
            del node.input[:]
            node.input.extend(
                [
                    "safe_name_152",
                    scale,
                    zero_u8,
                    "safe_name_153",
                    scale,
                    zero_u8,
                    scale,
                    zero_u8,
                ]
            )
            stats["qlinear_matmul_u8"] += 1
        if node.output and node.output[0] == "safe_name_155" and node.op_type == "Greater":
            if len(node.input) >= 2 and node.input[1] != zero_u8:
                node.input[1] = zero_u8
                stats["greater_zero_u8"] += 1

    if not stats:
        return current, Counter()

    del current.graph.value_info[:]
    stats["dead_nodes"] += oe.prune_dead(current)
    stats["initializers"] += oe.prune_initializers(current)
    onnx.checker.check_model(current, full_check=True)
    return current, stats
