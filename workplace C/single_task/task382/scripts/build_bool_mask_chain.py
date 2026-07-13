from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))

    zero_bool_name = "zero_bool_1"
    model.graph.initializer.append(
        numpy_helper.from_array(np.array([False], dtype=np.bool_), name=zero_bool_name)
    )

    kept_nodes = []
    for node in model.graph.node:
        if node.op_type == "Cast" and list(node.input) == ["pat_b"] and list(node.output) == ["pat"]:
            continue
        if node.op_type == "Cast" and list(node.input) == ["oriented_u"] and list(node.output) == ["oriented_b"]:
            continue

        node = deepcopy(node)
        for index, name in enumerate(node.input):
            if name == "pat":
                node.input[index] = "pat_b"
            elif name == "zero_u1" and node.op_type in {"Concat", "Where"}:
                node.input[index] = zero_bool_name
            elif name == "oriented_b":
                node.input[index] = "oriented_u"
        kept_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)

    used = {name for node in model.graph.node for name in node.input if name}
    kept_initializers = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)

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
