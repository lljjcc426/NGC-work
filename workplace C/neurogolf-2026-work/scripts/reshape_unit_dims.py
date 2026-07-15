from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _shapes(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    result: dict[str, tuple[int, ...]] = {}
    for value in list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info):
        dims: list[int] = []
        for dim in value.type.tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                dims = []
                break
            dims.append(int(dim.dim_value))
        if dims:
            result[value.name] = tuple(dims)
    return result


def _removed_unit_axes(source: tuple[int, ...], target: tuple[int, ...]) -> tuple[int, ...] | None:
    count = len(source) - len(target)
    if count <= 0:
        return None
    unit_axes = [index for index, dim in enumerate(source) if dim == 1]
    for axes in combinations(unit_axes, count):
        removed = set(axes)
        if tuple(dim for index, dim in enumerate(source) if index not in removed) == target:
            return tuple(axes)
    return None


def _added_unit_axes(source: tuple[int, ...], target: tuple[int, ...]) -> tuple[int, ...] | None:
    axes = _removed_unit_axes(target, source)
    return axes


def _unique_name(model: onnx.ModelProto, base: str) -> str:
    names = {item.name for item in model.graph.initializer}
    names.update(name for node in model.graph.node for name in node.input if name)
    names.update(name for node in model.graph.node for name in node.output if name)
    if base not in names:
        return base
    index = 1
    while f"{base}_{index}" in names:
        index += 1
    return f"{base}_{index}"


def fold(model: onnx.ModelProto) -> int:
    shapes = _shapes(model)
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    uses = Counter(name for node in model.graph.node for name in node.input if name)
    opset = next((item.version for item in model.opset_import if item.domain in ("", "ai.onnx")), 0)
    removed_shapes: set[str] = set()
    new_initializers: list[onnx.TensorProto] = []
    rebuilt: list[onnx.NodeProto] = []
    changed = 0

    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Reshape" or len(node.input) < 2:
            rebuilt.append(node)
            continue
        source_shape = shapes.get(node.input[0])
        output_shape = shapes.get(node.output[0])
        shape_value = initializers.get(node.input[1])
        if source_shape is None or output_shape is None or shape_value is None:
            rebuilt.append(node)
            continue

        op_type: str | None = None
        axes: tuple[int, ...] | None = None
        if len(source_shape) > len(output_shape):
            op_type = "Squeeze"
            axes = _removed_unit_axes(source_shape, output_shape)
        elif len(source_shape) < len(output_shape):
            op_type = "Unsqueeze"
            axes = _added_unit_axes(source_shape, output_shape)
        if op_type is None or axes is None:
            rebuilt.append(node)
            continue

        inputs = [node.input[0]]
        kwargs: dict[str, object] = {}
        if opset >= 13:
            axes_name = _unique_name(model, f"ngc_unit_axes_{node_index}")
            inputs.append(axes_name)
            new_initializers.append(
                numpy_helper.from_array(np.asarray(axes, dtype=np.int64), name=axes_name)
            )
        else:
            kwargs["axes"] = list(axes)
        rebuilt.append(
            helper.make_node(
                op_type,
                inputs,
                list(node.output),
                name=node.name,
                **kwargs,
            )
        )
        if uses[node.input[1]] == 1:
            removed_shapes.add(node.input[1])
        changed += 1

    if changed:
        del model.graph.node[:]
        model.graph.node.extend(rebuilt)
        kept = [item for item in model.graph.initializer if item.name not in removed_shapes]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
        model.graph.initializer.extend(new_initializers)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(args.input)
    count = fold(model)
    if count <= 0:
        raise SystemExit(2)
    model.producer_name = "ngc_reshape_unit_dims"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
