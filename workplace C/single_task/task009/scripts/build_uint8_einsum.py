from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def build(source: Path, output: Path) -> None:
    model = onnx.load(source)
    nodes = list(model.graph.node)
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }

    onehot_cast = next(
        node
        for node in nodes
        if node.op_type == "Cast" and node.output == ["onehot16"]
    )
    for attribute in onehot_cast.attribute:
        if attribute.name == "to":
            attribute.i = TensorProto.UINT8
            break
    onehot_cast.output[0] = "onehot_u8"

    for name in ("cv16", "sepA"):
        tensor = initializers[name]
        value = numpy_helper.to_array(tensor)
        if not np.array_equal(value, value.astype(np.uint8)):
            raise ValueError(f"{name} is not exactly uint8-valued")
        tensor.CopyFrom(
            numpy_helper.from_array(value.astype(np.uint8), name=name)
        )

    replacements = {
        "h_between": "h_u8",
        "v_between": "v_u8",
    }
    removable_outputs = set(replacements)
    rewritten = []
    for node in nodes:
        if node.op_type == "Cast" and node.input and node.input[0] in replacements:
            if node.output[0] != replacements[node.input[0]]:
                raise ValueError(f"unexpected Cast target for {node.input[0]}")
            continue
        for index, name in enumerate(node.input):
            if name == "onehot16":
                node.input[index] = "onehot_u8"
        for index, name in enumerate(node.output):
            if name in replacements:
                node.output[index] = replacements[name]
        rewritten.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    del model.graph.value_info[:]
    model.producer_name = "ngc_task009_uint8_einsum"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
