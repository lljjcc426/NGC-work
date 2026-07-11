from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    model.opset_import[0].version = 18
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([0, 0, 10, 8], dtype=np.int64), name="padv_compact"),
            numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="padv_axes"),
        ]
    )
    for node in model.graph.node:
        if node.op_type == "Pad" and node.output[0] == "vp":
            node.input[1] = "padv_compact"
            while len(node.input) < 3:
                node.input.append("")
            node.input.append("padv_axes")
    kept = [item for item in model.graph.initializer if item.name != "padv"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
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
