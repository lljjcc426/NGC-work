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
    if [node.op_type for node in nodes] != ["Slice", "Einsum"]:
        raise RuntimeError("unexpected task298 graph")

    row_selector = np.zeros((3, 30), dtype=np.float32)
    row_selector[np.arange(3), np.arange(3)] = 1.0
    col_selector = np.zeros((1, 30), dtype=np.float32)
    col_selector[0, 2] = 1.0
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(row_selector, name="row_selector"),
            numpy_helper.from_array(col_selector, name="col_selector"),
        ]
    )

    fused = helper.make_node(
        "Einsum",
        ["input", "input", "row_selector", "col_selector", "rot", "input", "row_selector", "col_selector"],
        ["output"],
        name="fused_color_strip_rotation",
        equation="nchw,ncij,ti,qj,tu,ndkl,uk,ql->ndhw",
    )
    del model.graph.node[:]
    model.graph.node.append(fused)
    removed = {"starts", "ends", "axes"}
    kept = [item for item in model.graph.initializer if item.name not in removed]
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
