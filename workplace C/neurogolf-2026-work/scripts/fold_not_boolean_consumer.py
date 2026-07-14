from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def fold(model: onnx.ModelProto) -> int:
    nodes = list(model.graph.node)
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        for name in node.input:
            consumers.setdefault(name, []).append(index)
    removed: set[int] = set()
    replacements: dict[int, onnx.NodeProto] = {}
    needs_false = False
    needs_true = False
    count = 0
    for index, node in enumerate(nodes):
        if node.op_type != "Not" or len(node.input) != 1 or len(node.output) != 1:
            continue
        uses = consumers.get(node.output[0], [])
        if len(uses) != 1:
            continue
        consumer_index = uses[0]
        consumer = nodes[consumer_index]
        if consumer.op_type not in {"And", "Or"} or len(consumer.input) != 2:
            continue
        other = consumer.input[1] if consumer.input[0] == node.output[0] else consumer.input[0]
        if node.output[0] not in consumer.input:
            continue
        if consumer.op_type == "And":
            inputs = [node.input[0], "fold_bool_false", other]
            needs_false = True
        else:
            inputs = [node.input[0], other, "fold_bool_true"]
            needs_true = True
        replacements[consumer_index] = helper.make_node(
            "Where", inputs, list(consumer.output), name=consumer.name
        )
        removed.add(index)
        count += 1
    if not count:
        return 0
    if needs_false:
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(False, dtype=np.bool_), "fold_bool_false")
        )
    if needs_true:
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(True, dtype=np.bool_), "fold_bool_true")
        )
    rewritten = []
    for index, node in enumerate(nodes):
        if index in removed:
            continue
        rewritten.append(replacements.get(index, node))
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Fold And/Or consumers of a single-use Not.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(args.input)
    count = fold(model)
    if not count:
        raise RuntimeError("no foldable Not boolean consumer")
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
