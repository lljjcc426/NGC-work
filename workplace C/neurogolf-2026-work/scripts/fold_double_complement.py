from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def _constant_values(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }


def _shape_map(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=False, data_prop=True
    )
    result: dict[str, tuple[int, ...]] = {}
    for item in [
        *inferred.graph.input,
        *inferred.graph.value_info,
        *inferred.graph.output,
    ]:
        dims = item.type.tensor_type.shape.dim
        if all(dim.HasField("dim_value") for dim in dims):
            result[item.name] = tuple(int(dim.dim_value) for dim in dims)
    return result


def fold_double_complement(model: onnx.ModelProto) -> int:
    constants = _constant_values(model)
    shapes = _shape_map(model)
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    graph_outputs = {item.name for item in model.graph.output}
    replacements: dict[str, str] = {}
    removable: set[int] = set()

    for node in model.graph.node:
        if (
            node.op_type != "Sub"
            or len(node.input) != 2
            or len(node.output) != 1
            or node.output[0] in graph_outputs
        ):
            continue
        outer_constant = constants.get(node.input[0])
        inner = producers.get(node.input[1])
        if (
            outer_constant is None
            or inner is None
            or inner.op_type != "Sub"
            or len(inner.input) != 2
        ):
            continue
        inner_constant = constants.get(inner.input[0])
        if inner_constant is None or not np.array_equal(
            outer_constant, inner_constant
        ):
            continue

        source = inner.input[1]
        source_shape = shapes.get(source)
        output_shape = shapes.get(node.output[0])
        if source_shape != output_shape:
            source_size = int(np.prod(source_shape or (1,)))
            output_size = int(np.prod(output_shape or (1,)))
            cast = producers.get(source)
            comparison = (
                producers.get(cast.input[0])
                if cast is not None and cast.op_type == "Cast" and cast.input
                else None
            )
            if (
                source_size != 1
                or output_size != 1
                or comparison is None
                or comparison.op_type
                not in {
                    "Equal",
                    "Greater",
                    "GreaterOrEqual",
                    "Less",
                    "LessOrEqual",
                }
            ):
                continue
            broadcast_name = next(
                (
                    name
                    for name in comparison.input
                    if name in constants and constants[name].size == 1
                ),
                "",
            )
            if not broadcast_name:
                continue
            broadcast = np.asarray(constants[broadcast_name]).reshape(
                outer_constant.shape
            )
            initializers[broadcast_name].CopyFrom(
                numpy_helper.from_array(broadcast, name=broadcast_name)
            )
            constants[broadcast_name] = broadcast
        replacements[node.output[0]] = inner.input[1]
        removable.add(id(node))

    if not replacements:
        return 0

    def resolve(name: str) -> str:
        while name in replacements:
            name = replacements[name]
        return name

    kept = []
    for node in model.graph.node:
        if id(node) in removable:
            continue
        for index, name in enumerate(node.input):
            if name:
                node.input[index] = resolve(name)
        kept.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept)

    removed_outputs = set(replacements)
    value_info = [
        item for item in model.graph.value_info if item.name not in removed_outputs
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(value_info)
    return len(removable)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model", type=Path)
    parser.add_argument("output_model", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.input_model)
    folded = fold_double_complement(model)
    if not folded:
        raise SystemExit("no matching double complement")
    model.producer_name = "ngc_fold_double_complement"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output_model)
    print(f"folded={folded} output={args.output_model}")


if __name__ == "__main__":
    main()
