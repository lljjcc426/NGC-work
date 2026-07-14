from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def _tensor_types(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    return {value.name: value.type.tensor_type.elem_type for value in values}


def fold(model: onnx.ModelProto) -> int:
    nodes = list(model.graph.node)
    types = _tensor_types(model)
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        for name in node.input:
            consumers.setdefault(name, []).append(index)

    replacements: dict[int, onnx.NodeProto] = {}
    skipped: set[int] = set()
    count = 0
    for index, cast in enumerate(nodes):
        if cast.op_type != "Cast" or len(cast.input) != 1 or len(cast.output) != 1:
            continue
        target = next((attr.i for attr in cast.attribute if attr.name == "to"), None)
        cast_consumers = consumers.get(cast.output[0], [])
        if target != TensorProto.BOOL or len(cast_consumers) != 1:
            continue
        not_index = cast_consumers[0]
        not_node = nodes[not_index]
        if not_node.op_type != "Not" or len(not_node.output) != 1:
            continue
        source_type = types.get(cast.input[0])
        if source_type in {None, TensorProto.BOOL, TensorProto.STRING}:
            continue
        zero_name = f"fold_not_cast_zero_{count}"
        model.graph.initializer.append(
            helper.make_tensor(zero_name, source_type, [], [0])
        )
        replacements[index] = helper.make_node(
            "Equal",
            [cast.input[0], zero_name],
            [not_node.output[0]],
            name=f"fold_not_cast_equal_{count}",
        )
        skipped.add(not_index)
        count += 1

    if not count:
        return 0
    rewritten = []
    for index, node in enumerate(nodes):
        if index in skipped:
            continue
        rewritten.append(replacements.get(index, node))
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Fold Not(Cast(x, bool)) into Equal(x, 0).")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(args.input)
    count = fold(model)
    if not count:
        raise RuntimeError("no foldable Not(Cast(x, bool)) pattern")
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    reloaded = onnx.load(args.output)
    onnx.checker.check_model(reloaded, full_check=True)
    print(f"folded={count} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
