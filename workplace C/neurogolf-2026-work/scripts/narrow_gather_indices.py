from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import TensorProto


def narrow(model: onnx.ModelProto) -> int:
    consumers: dict[str, list[tuple[onnx.NodeProto, int]]] = defaultdict(list)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name:
                consumers[name].append((node, index))
    graph_outputs = {value.name for value in model.graph.output}

    changed = 0
    for node in model.graph.node:
        if node.op_type != "Cast" or not node.output:
            continue
        target = next((item for item in node.attribute if item.name == "to"), None)
        if target is None or target.i != TensorProto.INT64:
            continue
        output = node.output[0]
        uses = consumers.get(output, [])
        if output in graph_outputs or not uses:
            continue
        if not all(consumer.op_type == "Gather" and input_index == 1 for consumer, input_index in uses):
            continue
        target.i = TensorProto.INT32
        for value in model.graph.value_info:
            if value.name == output:
                value.type.tensor_type.elem_type = TensorProto.INT32
        changed += 1
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(args.input)
    count = narrow(model)
    if count <= 0:
        raise SystemExit(2)
    model.producer_name = "ngc_narrow_gather_indices"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
