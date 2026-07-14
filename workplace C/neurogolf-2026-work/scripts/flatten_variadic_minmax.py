from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto


_INTEGER_TYPES = {
    TensorProto.BOOL,
    TensorProto.INT8,
    TensorProto.INT16,
    TensorProto.INT32,
    TensorProto.INT64,
    TensorProto.UINT8,
    TensorProto.UINT16,
    TensorProto.UINT32,
    TensorProto.UINT64,
}


def _type_map(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=False, data_prop=True
    )
    result: dict[str, int] = {}
    for item in [
        *inferred.graph.input,
        *inferred.graph.value_info,
        *inferred.graph.output,
    ]:
        tensor_type = item.type.tensor_type
        if tensor_type.elem_type:
            result[item.name] = int(tensor_type.elem_type)
    return result


def flatten_variadic_minmax(model: onnx.ModelProto) -> int:
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
            node.op_type not in {"Max", "Min"}
            or not node.output
            or types.get(node.output[0]) not in _INTEGER_TYPES
        ):
            continue
        expanded: list[str] = []
        changed = False
        for name in node.input:
            producer = producers.get(name)
            if (
                producer is not None
                and producer.op_type == node.op_type
                and consumer_count.get(name, 0) == 1
                and name not in graph_outputs
                and id(producer) not in removable
            ):
                expanded.extend(producer.input)
                removable.add(id(producer))
                removed_outputs.update(producer.output)
                flattened += 1
                changed = True
            else:
                expanded.append(name)
        if changed:
            del node.input[:]
            node.input.extend(expanded)

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
    flattened = flatten_variadic_minmax(model)
    if not flattened:
        raise SystemExit("no integer Min/Max chain found")
    model.producer_name = "ngc_flatten_variadic_minmax"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output_model)
    print(f"flattened={flattened} output={args.output_model}")


if __name__ == "__main__":
    main()
