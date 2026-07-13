from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


FLOAT_INITIALIZERS = {"k_16", "k_21", "k_56", "k_57"}


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))

    initializers = []
    for item in model.graph.initializer:
        if item.name in FLOAT_INITIALIZERS:
            initializers.append(
                numpy_helper.from_array(
                    numpy_helper.to_array(item).astype(np.float32), name=item.name
                )
            )
        elif item.name == "c_eight_u8":
            initializers.append(
                numpy_helper.from_array(np.array(8.0, dtype=np.float32), name="c_eight_f32")
            )
        else:
            initializers.append(item)
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)

    nodes = []
    for original in model.graph.node:
        output_name = original.output[0] if original.output else ""
        if output_name == "cyan_c" and original.op_type == "Cast":
            continue
        if output_name == "cyan8_c" and original.op_type == "Mul":
            nodes.append(
                helper.make_node(
                    "Mul",
                    ["gat_2_c", "c_eight_f32"],
                    ["cyan8_f32"],
                    name="scale_color8_float",
                )
            )
            nodes.append(
                helper.make_node(
                    "Cast",
                    ["cyan8_f32"],
                    ["cyan8_c"],
                    to=TensorProto.UINT8,
                    name="cast_color8_u8",
                )
            )
            continue
        if output_name == "r7u8" and original.op_type == "Reshape":
            node = deepcopy(original)
            node.input[0] = "gat_2_c"
            nodes.append(node)
            continue
        if output_name == "r7u8p" and original.op_type == "Pad":
            node = deepcopy(original)
            node.output[0] = "pad_9"
            nodes.append(node)
            continue
        if output_name == "pad_9" and original.op_type == "Cast":
            continue
        node = deepcopy(original)
        if node.op_type == "Cast":
            for attribute in node.attribute:
                if attribute.name == "to" and attribute.i == TensorProto.FLOAT16:
                    attribute.i = TensorProto.FLOAT
        nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source, args.output))


if __name__ == "__main__":
    main()
