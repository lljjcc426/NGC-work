from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _shapes(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    result: dict[str, tuple[int, ...]] = {}
    for value in list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info):
        dims = []
        for dim in value.type.tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                dims = []
                break
            dims.append(int(dim.dim_value))
        if dims:
            result[value.name] = tuple(dims)
    return result


def flatten(model: onnx.ModelProto) -> int:
    shapes = _shapes(model)
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    uses = Counter(name for node in model.graph.node for name in node.input if name)
    replaced_shapes: list[str] = []
    rebuilt: list[onnx.NodeProto] = []
    changed = 0
    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            rebuilt.append(node)
            continue
        source_shape = shapes.get(node.input[0])
        output_shape = shapes.get(node.output[0])
        shape_value = initializers.get(node.input[1])
        if (
            source_shape is None
            or output_shape is None
            or len(output_shape) != 2
            or shape_value is None
            or shape_value.size != 2
        ):
            rebuilt.append(node)
            continue
        axis = next(
            (
                index
                for index in range(len(source_shape) + 1)
                if (
                    int(np.prod(source_shape[:index], dtype=np.int64)),
                    int(np.prod(source_shape[index:], dtype=np.int64)),
                )
                == output_shape
            ),
            None,
        )
        if axis is None:
            rebuilt.append(node)
            continue
        rebuilt.append(
            helper.make_node(
                "Flatten",
                [node.input[0]],
                list(node.output),
                name=node.name,
                axis=axis,
            )
        )
        if uses[node.input[1]] == 1:
            replaced_shapes.append(node.input[1])
        changed += 1
    if changed:
        del model.graph.node[:]
        model.graph.node.extend(rebuilt)
        removed = set(replaced_shapes)
        kept = [item for item in model.graph.initializer if item.name not in removed]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(args.input)
    count = flatten(model)
    if count <= 0:
        raise SystemExit(2)
    model.producer_name = "ngc_reshape_to_flatten"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
