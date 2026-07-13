from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    anchor = next(
        numpy_helper.to_array(item) for item in model.graph.initializer if item.name == "anchor"
    )
    model.graph.initializer.extend(
        [
            numpy_helper.from_array((255 - anchor).astype(np.uint8), name="anchor_negative"),
            numpy_helper.from_array(np.array(2, dtype=np.uint8), name="br_background_zp"),
            numpy_helper.from_array(np.array(255, dtype=np.uint8), name="anchor_negative_zp"),
        ]
    )

    nodes = []
    for original in model.graph.node:
        if original.op_type == "BitwiseAnd" and list(original.output) == ["R"]:
            continue
        node = deepcopy(original)
        if node.op_type == "QLinearConv" and list(node.output) == ["olive"]:
            node.input[0] = "BR"
            node.input[2] = "br_background_zp"
            node.input[3] = "anchor_negative"
            node.input[5] = "anchor_negative_zp"
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)

    used = {name for node in model.graph.node for name in node.input if name}
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

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
