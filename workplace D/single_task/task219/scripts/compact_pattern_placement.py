from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


PLACEMENT_INDICES = {"add_221", "add_297", "add_373", "add_449", "add_525"}


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    replacements = {
        "k_140": numpy_helper.from_array(np.array([1, 0, 1, 0], dtype=np.int64), name="k_140"),
        "k_143_i32": numpy_helper.from_array(np.array([1], dtype=np.int32), name="k_143_i32"),
    }
    kept_initializers = [replacements.get(item.name, item) for item in model.graph.initializer]
    kept_initializers.extend(
        [
            numpy_helper.from_array(np.array(0, dtype=np.int32), name="placement_clip_min"),
            numpy_helper.from_array(np.array(4, dtype=np.int32), name="placement_clip_max"),
        ]
    )
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)

    nodes = []
    for original in model.graph.node:
        node = deepcopy(original)
        output_name = node.output[0] if node.output else ""
        if output_name in PLACEMENT_INDICES and node.op_type == "Add":
            raw_name = f"{output_name}_raw"
            node.output[0] = raw_name
            nodes.append(node)
            nodes.append(
                helper.make_node(
                    "Clip",
                    [raw_name, "placement_clip_min", "placement_clip_max"],
                    [output_name],
                    name=f"clip_{output_name}",
                )
            )
        else:
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
