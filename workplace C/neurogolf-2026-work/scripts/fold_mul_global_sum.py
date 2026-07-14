from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper


_FLOAT_TYPES = {TensorProto.FLOAT16, TensorProto.FLOAT, TensorProto.DOUBLE}


def _metadata(
    model: onnx.ModelProto,
) -> tuple[dict[str, int], dict[str, tuple[int, ...]], dict[str, object]]:
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
    constants = {
        item.name: numpy_helper.to_array(item) for item in model.graph.initializer
    }
    return types, shapes, constants


def _attribute(node: onnx.NodeProto, name: str) -> onnx.AttributeProto | None:
    return next((item for item in node.attribute if item.name == name), None)


def fold_mul_global_sum(model: onnx.ModelProto) -> int:
    nodes = list(model.graph.node)
    types, shapes, constants = _metadata(model)
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
    replacements: dict[int, onnx.NodeProto] = {}
    folded = 0
    labels = "abcdefghijklmnopqrstuvwxyz"
    for reduce_index, reduce_node in enumerate(nodes):
        if reduce_node.op_type != "ReduceSum" or not reduce_node.input:
            continue
        keepdims = _attribute(reduce_node, "keepdims")
        if keepdims is not None and int(keepdims.i) != 0:
            continue
        if len(reduce_node.input) > 1 and reduce_node.input[1]:
            axes = constants.get(reduce_node.input[1])
            source_shape = shapes.get(reduce_node.input[0])
            if axes is None or source_shape is None:
                continue
            normalized = {int(axis) % len(source_shape) for axis in axes.reshape(-1)}
            if normalized != set(range(len(source_shape))):
                continue
        mul_index = producers.get(reduce_node.input[0])
        if mul_index is None:
            continue
        mul = nodes[mul_index]
        if (
            mul.op_type != "Mul"
            or len(mul.input) != 2
            or len(consumers.get(mul.output[0], [])) != 1
            or types.get(mul.output[0]) not in _FLOAT_TYPES
        ):
            continue
        left_shape = shapes.get(mul.input[0])
        right_shape = shapes.get(mul.input[1])
        if left_shape is None or left_shape != right_shape or len(left_shape) > len(labels):
            continue
        subscripts = labels[: len(left_shape)]
        replacements[reduce_index] = helper.make_node(
            "Einsum",
            list(mul.input),
            list(reduce_node.output),
            equation=f"{subscripts},{subscripts}->",
            name=reduce_node.name,
        )
        removable.add(mul_index)
        folded += 1

    if not folded:
        return 0
    rewritten = []
    for index, node in enumerate(nodes):
        if index in removable:
            continue
        rewritten.append(replacements.get(index, node))
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return folded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    args = parser.parse_args()
    model = onnx.load(args.input_model)
    folded = fold_mul_global_sum(model)
    if not folded:
        raise SystemExit("no foldable Mul -> global ReduceSum chain")
    model.producer_name = "ngc_fold_mul_global_sum"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output_model)
    print(f"folded={folded} output={args.output_model}")


if __name__ == "__main__":
    main()
