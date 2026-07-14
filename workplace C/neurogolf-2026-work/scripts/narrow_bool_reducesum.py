from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


_WIDE_TYPES = {
    TensorProto.UINT32,
    TensorProto.UINT64,
    TensorProto.INT32,
    TensorProto.INT64,
    TensorProto.FLOAT,
    TensorProto.DOUBLE,
}


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


def narrow_bool_reducesum(model: onnx.ModelProto) -> int:
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

    inserted_after: dict[int, onnx.NodeProto] = {}
    narrowed = 0
    for reduce_index, reduce_node in enumerate(nodes):
        if (
            reduce_node.op_type != "ReduceSum"
            or not reduce_node.input
            or len(reduce_node.output) != 1
        ):
            continue
        cast_index = producers.get(reduce_node.input[0])
        if cast_index is None:
            continue
        cast = nodes[cast_index]
        original_type = _cast_target(cast)
        source_shape = shapes.get(cast.input[0]) if cast.input else None
        if (
            cast.op_type != "Cast"
            or len(cast.input) != 1
            or len(cast.output) != 1
            or types.get(cast.input[0]) != TensorProto.BOOL
            or original_type not in _WIDE_TYPES
            or source_shape is None
            or int(np.prod(source_shape or (1,))) > 2048
            or consumers.get(cast.output[0]) != [reduce_index]
        ):
            continue

        for attribute in cast.attribute:
            if attribute.name == "to":
                attribute.i = TensorProto.FLOAT16
                break
        for value_info in model.graph.value_info:
            if value_info.name == cast.output[0]:
                value_info.type.tensor_type.elem_type = TensorProto.FLOAT16

        original_output = reduce_node.output[0]
        compact_output = f"{original_output}_f16_count"
        reduce_node.output[0] = compact_output
        inserted_after[reduce_index] = helper.make_node(
            "Cast",
            [compact_output],
            [original_output],
            to=original_type,
            name=f"{original_output}_restore_cast",
        )
        narrowed += 1

    if not narrowed:
        return 0
    rewritten: list[onnx.NodeProto] = []
    for index, node in enumerate(nodes):
        rewritten.append(node)
        if index in inserted_after:
            rewritten.append(inserted_after[index])
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return narrowed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Narrow exact Bool Cast -> ReduceSum chains through float16."
    )
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.input_model)
    narrowed = narrow_bool_reducesum(model)
    if not narrowed:
        raise SystemExit("no wide Bool Cast -> ReduceSum chain found")
    model.producer_name = "ngc_narrow_bool_reducesum"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output_model)
    print(f"narrowed={narrowed} output={args.output_model}")


if __name__ == "__main__":
    main()
