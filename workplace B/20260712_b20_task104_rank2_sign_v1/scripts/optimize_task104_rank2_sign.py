from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def transform(model: onnx.ModelProto) -> onnx.ModelProto:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    nodes = list(current.graph.node)
    if [node.op_type for node in nodes] != ["Slice", "Einsum"]:
        raise RuntimeError("unexpected task104 graph structure")

    p = np.zeros((2, 2, 30), dtype=np.float32)
    p[0, :, 0:4] = np.asarray([[3], [-2]], dtype=np.float32)
    p[0, :, 4:8] = np.asarray([[-3], [-2]], dtype=np.float32)
    p[0, :, 8] = np.asarray([0, -3], dtype=np.float32)
    p[1, :, 0] = np.asarray([0, -3], dtype=np.float32)
    p[1, :, 1:5] = np.asarray([[-3], [-2]], dtype=np.float32)
    p[1, :, 5:9] = np.asarray([[3], [-2]], dtype=np.float32)
    metric = np.asarray([1, -1], dtype=np.float32)
    colors = np.zeros(10, dtype=np.float32)
    colors[0] = -1
    colors[3] = 1

    keep = {"st", "en", "ax", "sp"}
    initializers = [
        onnx.TensorProto.FromString(item.SerializeToString())
        for item in current.graph.initializer
        if item.name in keep
    ]
    initializers.extend(
        [
            numpy_helper.from_array(p, name="task104_rank2_p"),
            numpy_helper.from_array(metric, name="task104_metric"),
            numpy_helper.from_array(colors, name="task104_colors"),
        ]
    )
    replacement = [
        onnx.NodeProto.FromString(nodes[0].SerializeToString()),
        helper.make_node(
            "Einsum",
            [
                "x",
                "task104_rank2_p",
                "task104_rank2_p",
                "task104_metric",
                "task104_colors",
            ],
            ["output"],
            name="output",
            equation="zuab,aqr,bqc,q,l->zlrc",
        ),
    ]
    del current.graph.node[:]
    current.graph.node.extend(replacement)
    del current.graph.initializer[:]
    current.graph.initializer.extend(initializers)
    del current.graph.value_info[:]

    onnx.checker.check_model(current, full_check=True)
    onnx.shape_inference.infer_shapes(current, strict_mode=True)
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = transform(onnx.load(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
