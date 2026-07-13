from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    model.graph.initializer.append(
        numpy_helper.from_array(np.array(2, dtype=np.uint8), name="fill_color_two")
    )

    nodes = []
    for original in model.graph.node:
        output_name = original.output[0] if original.output else ""
        if original.op_type == "Cast" and output_name == "safe_name_117":
            continue
        if original.op_type == "Sub" and output_name == "safe_name_120":
            nodes.append(
                helper.make_node(
                    "Where",
                    ["safe_name_116", "fill_color_two", "safe_name_119"],
                    ["safe_name_120"],
                    name="direct_fill_color_two",
                )
            )
            continue
        nodes.append(deepcopy(original))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    kept_value_info = [item for item in model.graph.value_info if item.name != "safe_name_117"]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)

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
