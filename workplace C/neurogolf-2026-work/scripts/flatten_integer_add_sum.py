from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto


_INTEGER_TYPES = {
    TensorProto.UINT8,
    TensorProto.UINT16,
    TensorProto.UINT32,
    TensorProto.UINT64,
    TensorProto.INT8,
    TensorProto.INT16,
    TensorProto.INT32,
    TensorProto.INT64,
}


def _type_map(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=False, data_prop=True
    )
    result = {
        item.name: int(item.type.tensor_type.elem_type)
        for item in [
            *inferred.graph.input,
            *inferred.graph.value_info,
            *inferred.graph.output,
        ]
        if item.type.tensor_type.elem_type
    }
    result.update(
        {item.name: int(item.data_type) for item in inferred.graph.initializer}
    )
    return result


def flatten_integer_add_sum(model: onnx.ModelProto) -> int:
    types = _type_map(model)
    graph_outputs = {item.name for item in model.graph.output}
    flattened = 0

    while True:
        producers = {
            output: node
            for node in model.graph.node
            for output in node.output
            if output
        }
        consumers: dict[str, list[onnx.NodeProto]] = {}
        for node in model.graph.node:
            for name in node.input:
                if name:
                    consumers.setdefault(name, []).append(node)

        changed = False
        for outer in model.graph.node:
            if (
                outer.op_type not in {"Add", "Sum"}
                or not outer.output
                or types.get(outer.output[0]) not in _INTEGER_TYPES
            ):
                continue
            for input_index, name in enumerate(list(outer.input)):
                inner = producers.get(name)
                if inner is None or inner.op_type not in {"Add", "Sum"}:
                    continue
                if name in graph_outputs or consumers.get(name) != [outer]:
                    continue
                if not inner.output or types.get(inner.output[0]) != types.get(outer.output[0]):
                    continue

                expanded = [
                    *outer.input[:input_index],
                    *inner.input,
                    *outer.input[input_index + 1 :],
                ]
                outer.op_type = "Sum"
                del outer.input[:]
                outer.input.extend(expanded)
                model.graph.node.remove(inner)
                flattened += 1
                changed = True
                break
            if changed:
                break
        if not changed:
            return flattened


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten exclusive integer Add/Sum chains."
    )
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.input_model)
    flattened = flatten_integer_add_sum(model)
    if not flattened:
        raise SystemExit("no exclusive integer Add/Sum chain found")
    model.producer_name = "ngc_flatten_integer_add_sum"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output_model)
    print(f"flattened={flattened} output={args.output_model}")


if __name__ == "__main__":
    main()
