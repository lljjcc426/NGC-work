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
    if old[0].output != ["counts2"] or old[1].output != ["grid_f"]:
        raise RuntimeError("unexpected task304 graph")

    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.arange(9, dtype=np.uint8).reshape(1, 9, 1, 1), name="major_codes"),
            numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="histogram_axes"),
        ]
    )
    histogram = [
        old[1],
        old[4],
        helper.make_node("Equal", ["grid3", "major_codes"], ["histogram_mask"], name="grid_color_bins"),
        helper.make_node("Cast", ["histogram_mask"], ["histogram_i32"], to=TensorProto.INT32, name="histogram_to_i32"),
        helper.make_node("ReduceSum", ["histogram_i32", "histogram_axes"], ["counts2"], keepdims=0, name="grid_histogram"),
        old[2],
        old[3],
    ]
    del model.graph.node[:]
    model.graph.node.extend(histogram + old[5:])
    removed = {"count_cw", "sel3_line"}
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
