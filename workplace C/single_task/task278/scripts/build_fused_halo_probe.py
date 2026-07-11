from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    nodes = list(model.graph.node)
    if nodes[4].op_type != "QLinearConv" or nodes[5].op_type != "MaxPool":
        raise RuntimeError("unexpected task278 source graph")

    anchor = next(item for item in model.graph.initializer if item.name == "anchor")
    kernel3 = numpy_helper.to_array(anchor)[0, 0].astype(np.uint16)
    kernel5 = np.zeros((5, 5), dtype=np.uint16)
    for row in range(3):
        for col in range(3):
            kernel5[row : row + 3, col : col + 3] += kernel3
    anchor.CopyFrom(numpy_helper.from_array(kernel5.astype(np.uint8)[None, None], name="anchor"))

    fused = nodes[4]
    fused.output[0] = "halo"
    del fused.attribute[:]
    fused.attribute.extend(
        [
            helper.make_attribute("kernel_shape", [5, 5]),
            helper.make_attribute("pads", [2, 2, 2, 2]),
        ]
    )
    del nodes[5]
    del model.graph.node[:]
    model.graph.node.extend(nodes)

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
