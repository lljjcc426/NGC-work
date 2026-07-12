from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper, numpy_helper


def transform(model: onnx.ModelProto) -> onnx.ModelProto:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    values = {
        initializer.name: [
            int(value) for value in numpy_helper.to_array(initializer).reshape(-1)
        ]
        for initializer in current.graph.initializer
    }
    expected = {
        "axes",
        "top_starts",
        "top_ends",
        "bottom_starts",
        "bottom_ends",
        "pads",
    }
    if set(values) != expected:
        raise RuntimeError(f"unexpected task395 initializers: {sorted(values)}")

    constants = [
        helper.make_node(
            "Constant",
            [],
            [name],
            name=f"task395_const_{name}",
            value_ints=value,
        )
        for name, value in values.items()
    ]
    nodes = constants + [
        onnx.NodeProto.FromString(node.SerializeToString())
        for node in current.graph.node
    ]
    del current.graph.node[:]
    current.graph.node.extend(nodes)
    del current.graph.initializer[:]
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
