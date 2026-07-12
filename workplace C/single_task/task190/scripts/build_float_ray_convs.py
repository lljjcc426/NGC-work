from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    old = list(model.graph.node)
    if any(old[index].op_type != "QLinearConv" for index in range(6, 10)):
        raise RuntimeError("unexpected task190 graph")

    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    replacements = []
    added_weights = set()
    for index in range(6, 10):
        node = old[index]
        source_name = node.input[0]
        weight_name = node.input[3]
        output_name = node.output[0]
        float_input = f"{source_name}_float"
        float_weight = f"{weight_name}_float"
        float_output = f"{output_name}_float"
        if float_weight not in added_weights:
            model.graph.initializer.append(
                numpy_helper.from_array(initializers[weight_name].astype(np.float32), name=float_weight)
            )
            added_weights.add(float_weight)
        replacements.extend(
            [
                helper.make_node("Cast", [source_name], [float_input], to=TensorProto.FLOAT, name=f"cast_{source_name}_float"),
                helper.make_node(
                    "Conv",
                    [float_input, float_weight],
                    [float_output],
                    name=f"float_{output_name}",
                    pads=next(onnx.helper.get_attribute_value(a) for a in node.attribute if a.name == "pads"),
                ),
                helper.make_node("Cast", [float_output], [output_name], to=TensorProto.UINT8, name=f"cast_{output_name}_u8"),
            ]
        )

    del model.graph.node[:]
    model.graph.node.extend(old[:6] + replacements + old[10:])
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model, full_check=True)
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
