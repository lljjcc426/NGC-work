from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper


def fold(model: onnx.ModelProto) -> int:
    nodes = list(model.graph.node)
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        for name in node.input:
            consumers.setdefault(name, []).append(index)
    removed: set[int] = set()
    replacements: dict[int, onnx.NodeProto] = {}
    count = 0
    for index, node in enumerate(nodes):
        if node.op_type != "Not" or len(node.input) != 1 or len(node.output) != 1:
            continue
        uses = consumers.get(node.output[0], [])
        if len(uses) != 1:
            continue
        where_index = uses[0]
        where = nodes[where_index]
        if where.op_type != "Where" or where.input[0] != node.output[0]:
            continue
        replacements[where_index] = helper.make_node(
            "Where",
            [node.input[0], where.input[2], where.input[1]],
            list(where.output),
            name=where.name,
        )
        removed.add(index)
        count += 1
    if count:
        rewritten = []
        for index, node in enumerate(nodes):
            if index in removed:
                continue
            rewritten.append(replacements.get(index, node))
        del model.graph.node[:]
        model.graph.node.extend(rewritten)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Fold Where(Not(mask), A, B).")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(args.input)
    count = fold(model)
    if not count:
        raise RuntimeError("no foldable Not-to-Where pattern")
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
