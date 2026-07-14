from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto


def _metadata(
    model: onnx.ModelProto,
) -> tuple[dict[str, int], dict[str, tuple[int, ...]]]:
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=False, data_prop=True
    )
    types: dict[str, int] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for item in [
        *inferred.graph.input,
        *inferred.graph.value_info,
        *inferred.graph.output,
    ]:
        tensor = item.type.tensor_type
        if tensor.elem_type:
            types[item.name] = int(tensor.elem_type)
        dims = tensor.shape.dim
        if all(dim.HasField("dim_value") for dim in dims):
            shapes[item.name] = tuple(int(dim.dim_value) for dim in dims)
    return types, shapes


def _cast_target(node: onnx.NodeProto) -> int | None:
    return next(
        (int(attribute.i) for attribute in node.attribute if attribute.name == "to"),
        None,
    )


def collapse_exact_bool_sum_casts(model: onnx.ModelProto) -> int:
    nodes = list(model.graph.node)
    types, shapes = _metadata(model)
    producers = {
        output: index
        for index, node in enumerate(nodes)
        for output in node.output
        if output
    }
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        for name in node.input:
            if name:
                consumers.setdefault(name, []).append(index)

    removable: set[int] = set()
    collapsed = 0
    for outer_index, outer in enumerate(nodes):
        if outer.op_type != "Cast" or len(outer.input) != 1:
            continue
        reduce_index = producers.get(outer.input[0])
        if reduce_index is None:
            continue
        reduce_node = nodes[reduce_index]
        if reduce_node.op_type != "ReduceSum" or not reduce_node.input:
            continue
        inner_index = producers.get(reduce_node.input[0])
        if inner_index is None:
            continue
        inner = nodes[inner_index]
        source_shape = shapes.get(inner.input[0]) if inner.input else None
        if (
            inner.op_type != "Cast"
            or len(inner.input) != 1
            or types.get(inner.input[0]) != TensorProto.BOOL
            or _cast_target(inner) != TensorProto.FLOAT
            or _cast_target(outer) != TensorProto.FLOAT16
            or source_shape is None
            or int(np.prod(source_shape or (1,))) > 2048
            or len(consumers.get(inner.output[0], [])) != 1
            or len(consumers.get(reduce_node.output[0], [])) != 1
        ):
            continue
        for attribute in inner.attribute:
            if attribute.name == "to":
                attribute.i = TensorProto.FLOAT16
                break
        old_reduce_output = reduce_node.output[0]
        reduce_node.output[0] = outer.output[0]
        removable.add(outer_index)
        for value_info in model.graph.value_info:
            if value_info.name in {inner.output[0], outer.output[0]}:
                value_info.type.tensor_type.elem_type = TensorProto.FLOAT16
            elif value_info.name == old_reduce_output:
                value_info.name = outer.output[0]
                value_info.type.tensor_type.elem_type = TensorProto.FLOAT16
        collapsed += 1

    if not collapsed:
        return 0
    kept = [node for index, node in enumerate(nodes) if index not in removable]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return collapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    args = parser.parse_args()
    model = onnx.load(args.input_model)
    collapsed = collapse_exact_bool_sum_casts(model)
    if not collapsed:
        raise SystemExit("no exact bool float32 sum to float16 chain found")
    model.producer_name = "ngc_collapse_exact_bool_sum_casts"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output_model)
    print(f"collapsed={collapsed} output={args.output_model}")


if __name__ == "__main__":
    main()
