from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def build(source: Path, output: Path) -> None:
    model = onnx.load(source)
    nodes = list(model.graph.node)
    by_type = {node.op_type: node for node in nodes}
    required = {"Slice", "Reshape", "Sub", "Concat", "Einsum"}
    if set(by_type) != required or len(nodes) != len(required):
        raise RuntimeError("task267 source graph no longer matches the expected five-node form")

    reshape = by_type["Reshape"]
    slice_node = by_type["Slice"]
    sub = by_type["Sub"]
    concat = by_type["Concat"]
    einsum = by_type["Einsum"]
    if sub.input[0] != reshape.output[0] or reshape.input[0] != slice_node.output[0]:
        raise RuntimeError("unexpected marker extraction chain")

    initializers = {item.name: item for item in model.graph.initializer}
    e0 = initializers.get("e0")
    if e0 is None:
        raise RuntimeError("missing e0 initializer")
    e0_array = numpy_helper.to_array(e0)
    if e0_array.shape != (1, 10):
        raise RuntimeError(f"unexpected e0 shape: {e0_array.shape}")
    e0.CopyFrom(numpy_helper.from_array(e0_array.reshape(1, 10, 1, 1), "e0"))

    sub.input[0] = slice_node.output[0]
    next(attr for attr in concat.attribute if attr.name == "axis").i = 2
    equation = next(attr for attr in einsum.attribute if attr.name == "equation")
    equation.s = b"bjrc,bktq,tr,tc,tj->bkrc"

    del model.graph.node[:]
    model.graph.node.extend(node for node in nodes if node is not reshape)
    del model.graph.initializer[:]
    model.graph.initializer.extend(
        item for item in initializers.values() if item.name != reshape.input[1]
    )
    del model.graph.value_info[:]

    model.producer_name = "ngc_task267_compact_dynamic_color"
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.source, args.output)
