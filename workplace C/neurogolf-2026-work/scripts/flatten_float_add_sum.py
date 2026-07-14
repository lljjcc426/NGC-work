from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto


_FLOAT_TYPES = {TensorProto.FLOAT16, TensorProto.FLOAT, TensorProto.DOUBLE}


def _type_map(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=False, data_prop=True
    )
    return {
        item.name: int(item.type.tensor_type.elem_type)
        for item in [
            *inferred.graph.input,
            *inferred.graph.value_info,
            *inferred.graph.output,
        ]
        if item.type.tensor_type.elem_type
    }


def flatten_float_add_sum(model: onnx.ModelProto) -> int:
    types = _type_map(model)
    graph_outputs = {item.name for item in model.graph.output}
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    consumer_count: dict[str, int] = {}
    for node in model.graph.node:
        for name in node.input:
            if name:
                consumer_count[name] = consumer_count.get(name, 0) + 1

    removable: set[int] = set()
    removed_outputs: set[str] = set()
    flattened = 0
    for node in model.graph.node:
        if (
            node.op_type not in {"Add", "Sum"}
            or not node.input
            or not node.output
            or types.get(node.output[0]) not in _FLOAT_TYPES
        ):
            continue
        first_name = node.input[0]
        producer = producers.get(first_name)
        if (
            producer is None
            or producer.op_type not in {"Add", "Sum"}
            or consumer_count.get(first_name, 0) != 1
            or first_name in graph_outputs
            or id(producer) in removable
        ):
            continue
        expanded = [*producer.input, *node.input[1:]]
        node.op_type = "Sum"
        del node.input[:]
        node.input.extend(expanded)
        removable.add(id(producer))
        removed_outputs.update(producer.output)
        flattened += 1

    if not flattened:
        return 0
    kept = [node for node in model.graph.node if id(node) not in removable]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    value_info = [
        item for item in model.graph.value_info if item.name not in removed_outputs
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(value_info)
    return flattened


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    args = parser.parse_args()
    model = onnx.load(args.input_model)
    flattened = flatten_float_add_sum(model)
    if not flattened:
        raise SystemExit("no left-associated float Add/Sum chain found")
    model.producer_name = "ngc_flatten_float_add_sum"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output_model)
    print(f"flattened={flattened} output={args.output_model}")


if __name__ == "__main__":
    main()
