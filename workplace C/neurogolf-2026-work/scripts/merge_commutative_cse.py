from __future__ import annotations

import argparse
from pathlib import Path

import onnx


COMMUTATIVE = {
    "Add",
    "Mul",
    "And",
    "Or",
    "Xor",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "Max",
    "Min",
    "Equal",
}


def merge(model: onnx.ModelProto) -> int:
    graph_outputs = {item.name for item in model.graph.output}
    seen: dict[tuple, str] = {}
    aliases: dict[str, str] = {}
    kept: list[onnx.NodeProto] = []
    changed = 0

    def resolve(name: str) -> str:
        while name in aliases:
            name = aliases[name]
        return name

    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name:
                node.input[index] = resolve(name)
        if len(node.output) != 1 or node.op_type not in COMMUTATIVE or node.output[0] in graph_outputs:
            kept.append(node)
            continue
        signature = (
            node.domain,
            node.op_type,
            tuple(sorted(node.input)),
            tuple(sorted((item.name, item.SerializeToString()) for item in node.attribute)),
        )
        previous = seen.get(signature)
        if previous is None:
            seen[signature] = node.output[0]
            kept.append(node)
            continue
        aliases[node.output[0]] = previous
        changed += 1

    if changed:
        for node in kept:
            for index, name in enumerate(node.input):
                if name:
                    node.input[index] = resolve(name)
        del model.graph.node[:]
        model.graph.node.extend(kept)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(args.input)
    count = merge(model)
    if count <= 0:
        raise SystemExit(2)
    model.producer_name = "ngc_merge_commutative_cse"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
