from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def _types(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    return {value.name: value.type.tensor_type.elem_type for value in values}


def fold(model: onnx.ModelProto) -> int:
    nodes = list(model.graph.node)
    types = _types(model)
    producers = {name: index for index, node in enumerate(nodes) for name in node.output}
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        for name in node.input:
            consumers.setdefault(name, []).append(index)
    removed: set[int] = set()
    replacements: dict[int, onnx.NodeProto] = {}
    zero_by_type: dict[int, str] = {}
    count = 0
    for index, mul in enumerate(nodes):
        if mul.op_type != "Mul" or len(mul.input) != 2:
            continue
        match = None
        for position, name in enumerate(mul.input):
            producer_index = producers.get(name)
            if producer_index is None:
                continue
            cast = nodes[producer_index]
            if (
                cast.op_type == "Cast"
                and len(cast.input) == 1
                and types.get(cast.input[0]) == TensorProto.BOOL
                and len(consumers.get(name, [])) == 1
            ):
                match = (position, producer_index, cast)
                break
        if match is None:
            continue
        position, cast_index, cast = match
        numeric_type = types.get(cast.output[0])
        if numeric_type in {None, TensorProto.BOOL, TensorProto.STRING}:
            continue
        zero_name = zero_by_type.get(numeric_type)
        if zero_name is None:
            zero_name = f"fold_mask_zero_{numeric_type}"
            model.graph.initializer.append(helper.make_tensor(zero_name, numeric_type, [], [0]))
            zero_by_type[numeric_type] = zero_name
        other = mul.input[1 - position]
        replacements[index] = helper.make_node(
            "Where",
            [cast.input[0], other, zero_name],
            list(mul.output),
            name=mul.name,
        )
        removed.add(cast_index)
        count += 1
    if not count:
        return 0
    rewritten = []
    for index, node in enumerate(nodes):
        if index in removed:
            continue
        rewritten.append(replacements.get(index, node))
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Fold Mul(Cast(bool_mask), x) into Where.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(args.input)
    count = fold(model)
    if not count:
        raise RuntimeError("no foldable cast-mask Mul")
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
