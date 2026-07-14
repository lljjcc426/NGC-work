from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _constant_values(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    values = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        value = next((item for item in node.attribute if item.name == "value"), None)
        if value is not None:
            values[node.output[0]] = numpy_helper.to_array(value.t)
    return values


def _remove_dead_constants(model: onnx.ModelProto) -> None:
    used = {name for node in model.graph.node for name in node.input if name}
    kept_initializers = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)

    kept_nodes = []
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] not in used:
            continue
        kept_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)


def fold(model: onnx.ModelProto) -> int:
    values = _constant_values(model)
    changed = 0
    replacement: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.op_type != "Where" or len(node.input) != 3:
            replacement.append(node)
            continue
        yes = values.get(node.input[1])
        no = values.get(node.input[2])
        if yes is None or no is None or yes.size != 1 or no.size != 1:
            replacement.append(node)
            continue
        if yes.item() != 1 or no.item() != 0 or yes.dtype != no.dtype:
            replacement.append(node)
            continue
        try:
            to_type = helper.np_dtype_to_tensor_dtype(yes.dtype)
        except ValueError:
            replacement.append(node)
            continue
        replacement.append(
            helper.make_node(
                "Cast",
                [node.input[0]],
                list(node.output),
                name=node.name,
                to=to_type,
            )
        )
        changed += 1
    if changed:
        del model.graph.node[:]
        model.graph.node.extend(replacement)
        _remove_dead_constants(model)
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
    model.producer_name = "ngc_fold_where_bool_cast"
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
