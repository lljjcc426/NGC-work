from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def build(source: Path, output: Path) -> Path:
    model = deepcopy(onnx.load(source))
    old = list(model.graph.node)
    if old[1].op_type != "MatMul" or old[7].output != ["maroon8"]:
        raise RuntimeError("unexpected task381 graph")

    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array(3, dtype=np.int64), name="axis3_scalar"),
            numpy_helper.from_array(np.arange(9, -1, -1, dtype=np.int64), name="reverse10"),
        ]
    )
    spans = [
        helper.make_node("CumSum", ["red8_f", "axis3_scalar"], ["left_count"], name="seen_from_left"),
        helper.make_node("Gather", ["red8_f", "reverse10"], ["red_reversed"], axis=3, name="reverse_red"),
        helper.make_node("CumSum", ["red_reversed", "axis3_scalar"], ["right_count_reversed"], name="seen_from_right_reversed"),
        helper.make_node("Gather", ["right_count_reversed", "reverse10"], ["right_count"], axis=3, name="restore_right_counts"),
        helper.make_node("Greater", ["left_count", "half"], ["left_seen"], name="left_seen"),
        helper.make_node("Greater", ["right_count", "half"], ["right_seen"], name="right_seen"),
        helper.make_node("And", ["left_seen", "right_seen"], ["span"], name="bounded_span"),
        helper.make_node("Less", ["red8_f", "half"], ["not_red"], name="not_red"),
        helper.make_node("And", ["span", "not_red"], ["maroon8"], name="fill_between_red"),
    ]
    del model.graph.node[:]
    model.graph.node.extend(old[:1] + spans + old[8:])
    removed = {"powers", "powers_rev", "thr", "thr_rev"}
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
