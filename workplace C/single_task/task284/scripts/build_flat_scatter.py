from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def build(source: Path, output: Path) -> None:
    model = onnx.load(source)
    nodes = list(model.graph.node)
    cast_index = next(
        index
        for index, node in enumerate(nodes)
        if node.op_type == "Cast" and node.output and node.output[0] == "idx"
    )
    scatter_index = next(
        index
        for index, node in enumerate(nodes)
        if node.op_type == "ScatterND" and node.input[1] == "idx"
    )
    if scatter_index != cast_index + 1 or nodes[scatter_index].output[0] != "output":
        raise RuntimeError("unexpected task284 parent tail")

    tail = [
        helper.make_node(
            "Cast", ["idxU"], ["idx_i32"], name="flat_idx_i32", to=onnx.TensorProto.INT32
        ),
        helper.make_node(
            "MatMul", ["idx_i32", "flat_strides_i32"], ["idx_linear_2d"], name="flat_idx_linear"
        ),
        helper.make_node(
            "Reshape", ["idx_linear_2d", "flat_shape"], ["idx_linear"], name="flat_idx_reshape"
        ),
        helper.make_node("Reshape", ["input", "flat_shape"], ["input_flat"], name="flat_input"),
        helper.make_node(
            "ScatterElements",
            ["input_flat", "idx_linear", "upd"],
            ["output_flat"],
            name="flat_scatter",
            axis=0,
        ),
        helper.make_node(
            "Reshape", ["output_flat", "output_shape"], ["output"], name="flat_output"
        ),
    ]
    del model.graph.node[:]
    model.graph.node.extend(nodes[:cast_index] + tail)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(
                np.asarray([[9000], [900], [30], [1]], dtype=np.int32),
                name="flat_strides_i32",
            ),
            numpy_helper.from_array(np.asarray([-1], dtype=np.int64), name="flat_shape"),
            numpy_helper.from_array(
                np.asarray([1, 10, 30, 30], dtype=np.int64), name="output_shape"
            ),
        ]
    )
    model.producer_name = "ngc_task284_flat_scatter"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
