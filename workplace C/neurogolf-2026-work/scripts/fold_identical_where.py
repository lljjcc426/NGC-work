from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def _shapes(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    result: dict[str, tuple[int, ...]] = {}
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    ):
        dims: list[int] = []
        for dim in value.type.tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                dims = []
                break
            dims.append(int(dim.dim_value))
        result[value.name] = tuple(dims)
    return result


def fold(model: onnx.ModelProto) -> int:
    shapes = _shapes(model)
    replacements: dict[str, str] = {}
    kept: list[onnx.NodeProto] = []

    def source(name: str) -> str:
        while name in replacements:
            name = replacements[name]
        return name

    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name:
                node.input[index] = source(name)
        if (
            node.op_type == "Where"
            and len(node.input) == 3
            and len(node.output) == 1
            and node.input[1] == node.input[2]
            and shapes.get(node.input[1]) == shapes.get(node.output[0])
        ):
            replacements[node.output[0]] = source(node.input[1])
            continue
        kept.append(node)

    if not replacements:
        return 0

    for node in kept:
        for index, name in enumerate(node.input):
            if name:
                node.input[index] = source(name)
    for output in model.graph.output:
        output.name = source(output.name)

    removed = len(model.graph.node) - len(kept)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(args.input)
    count = fold(model)
    if count <= 0:
        raise SystemExit(2)
    model.producer_name = "ngc_fold_identical_where"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
