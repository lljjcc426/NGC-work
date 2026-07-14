from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def tensor_types(model: onnx.ModelProto) -> dict[str, int]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    return {value.name: value.type.tensor_type.elem_type for value in values}


def fold(model: onnx.ModelProto) -> int:
    nodes = list(model.graph.node)
    types = tensor_types(model)
    consumers: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        for name in node.input:
            consumers[name].append(index)
    removed: set[int] = set()
    replacements: dict[int, onnx.NodeProto] = {}
    zeros: dict[int, str] = {}
    claimed_uses: set[int] = set()
    folded = 0
    for cast_index, cast in enumerate(nodes):
        if cast.op_type != "Cast" or len(cast.input) != 1 or types.get(cast.input[0]) != TensorProto.BOOL:
            continue
        uses = consumers.get(cast.output[0], [])
        if len(uses) < 2 or any(nodes[index].op_type != "Mul" for index in uses):
            continue
        # If two shared masks meet in one Mul, fold one side and leave the
        # other Cast alive; deleting both would make either rewrite reference
        # a removed tensor.
        if any(index in claimed_uses for index in uses):
            continue
        dtype = types.get(cast.output[0])
        if dtype not in {
            TensorProto.UINT8, TensorProto.INT8, TensorProto.UINT16, TensorProto.INT16,
            TensorProto.UINT32, TensorProto.INT32, TensorProto.UINT64, TensorProto.INT64,
        }:
            continue
        zero = zeros.get(dtype)
        if zero is None:
            zero = f"shared_mask_zero_{dtype}"
            model.graph.initializer.append(helper.make_tensor(zero, dtype, [], [0]))
            zeros[dtype] = zero
        for use_index in uses:
            mul = nodes[use_index]
            position = list(mul.input).index(cast.output[0])
            value = mul.input[1 - position]
            replacements[use_index] = helper.make_node(
                "Where", [cast.input[0], value, zero], list(mul.output), name=mul.name
            )
        removed.add(cast_index)
        claimed_uses.update(uses)
        folded += 1
    if not folded:
        return 0
    rewritten = []
    for index, node in enumerate(nodes):
        if index in removed:
            continue
        rewritten.append(replacements.get(index, node))
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return folded


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove shared integer Cast masks used only by Mul nodes.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(args.input)
    count = fold(model)
    if not count:
        raise RuntimeError("no shared Cast mask found")
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    onnx.checker.check_model(onnx.load(args.output), full_check=True)
    print(f"folded={count} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
