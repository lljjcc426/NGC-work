from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(time_axis: int) -> onnx.ModelProto:
    axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes0")
    nodes = [
        helper.make_node(
            "ReduceL2",
            ["input"],
            ["side_f"],
            axes=[0, 1, 2, 3],
            keepdims=0,
        ),
        helper.make_node("Cast", ["side_f"], ["side_i"], to=TensorProto.INT64),
        helper.make_node("Unsqueeze", ["side_i", "axes0"], ["side_vec"]),
        helper.make_node(
            "ReverseSequence",
            ["input", "side_vec"],
            ["output"],
            batch_axis=0,
            time_axis=time_axis,
        ),
    ]
    value = lambda name: helper.make_tensor_value_info(
        name, TensorProto.FLOAT, [1, 10, 30, 30]
    )
    graph = helper.make_graph(nodes, "dynamic_axis_reverse", [value("input")], [value("output")], [axes])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task, axis in (("task150", 3), ("task155", 2)):
        path = args.output_dir / f"{task}.onnx"
        onnx.save_model(build(axis), path)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
