from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def transform(model: onnx.ModelProto) -> onnx.ModelProto:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    nodes = list(current.graph.node)
    if [node.op_type for node in nodes] != [
        "Slice",
        "Slice",
        "Cast",
        "Cast",
        "Or",
        "Not",
        "And",
        "Concat",
        "Pad",
    ]:
        raise RuntimeError("unexpected task395 graph structure")

    sparse_name = "task395_offset_weight"
    weight_shape = [1, 10, 28, 28]
    flat_indices = np.asarray(
        [
            1 * 28 * 28 + 3 * 28,
            9 * 28 * 28,
        ],
        dtype=np.int64,
    )
    sparse_weight = helper.make_sparse_tensor(
        numpy_helper.from_array(
            np.ones(flat_indices.size, dtype=np.float32), name=sparse_name
        ),
        numpy_helper.from_array(flat_indices),
        weight_shape,
    )

    replacement = [
        helper.make_node(
            "Conv",
            ["input", sparse_name],
            ["task395_occupied_f"],
            name="task395_occupied_f",
            kernel_shape=[28, 28],
        ),
        helper.make_node(
            "Cast",
            ["task395_occupied_f"],
            ["occupied"],
            name="occupied",
            to=TensorProto.BOOL,
        ),
        onnx.NodeProto.FromString(nodes[5].SerializeToString()),
        onnx.NodeProto.FromString(nodes[6].SerializeToString()),
        onnx.NodeProto.FromString(nodes[7].SerializeToString()),
        helper.make_node(
            "Pad",
            ["small", "task395_full_pads"],
            ["output"],
            name="output",
            mode="constant",
        ),
    ]

    del current.graph.node[:]
    current.graph.node.extend(replacement)
    del current.graph.initializer[:]
    current.graph.initializer.append(
        numpy_helper.from_array(
            np.asarray([0, 0, 0, 0, 0, 7, 27, 27], dtype=np.int64),
            name="task395_full_pads",
        )
    )
    del current.graph.sparse_initializer[:]
    current.graph.sparse_initializer.append(sparse_weight)
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
