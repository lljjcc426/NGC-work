from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

import optimize_task205_integral_not_yusuke as integral_not


def _replace_initializer(
    model: onnx.ModelProto, name: str, dtype: np.dtype
) -> bool:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name != name:
            continue
        values = numpy_helper.to_array(initializer).astype(dtype)
        model.graph.initializer[index].CopyFrom(numpy_helper.from_array(values, name))
        return True
    return False


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current, stats = integral_not.transform(model)
    if not stats.get("task205_integral_not"):
        return current, Counter()
    if not _replace_initializer(current, "safe_name_0", np.int64):
        return onnx.ModelProto.FromString(model.SerializeToString()), Counter()

    by_output = {
        output: node
        for node in current.graph.node
        for output in node.output
        if output
    }
    required = {
        "safe_name_9",
        "safe_name_39",
        "safe_name_40",
        "safe_name_42",
        "task205_bg_u8",
        "task205_fg_u8",
    }
    if not required.issubset(by_output):
        return onnx.ModelProto.FromString(model.SerializeToString()), Counter()

    nodes: list[onnx.NodeProto] = []
    remove_outputs = {"task205_bg_u8", "task205_fg_u8", "safe_name_39"}
    for node in current.graph.node:
        output = node.output[0] if node.output else ""
        if output in remove_outputs:
            continue
        copied = onnx.NodeProto.FromString(node.SerializeToString())
        if output == "safe_name_9":
            copied.input[1] = "safe_name_8"
        elif output == "safe_name_42":
            copied.input[1] = "safe_name_41"
        elif output == "safe_name_40":
            copied = helper.make_node(
                "Where",
                ["safe_name_9", "safe_name_4", "safe_name_38"],
                ["safe_name_40"],
                name="safe_name_40",
            )
        nodes.append(copied)

    del current.graph.node[:]
    current.graph.node.extend(nodes)
    del current.graph.value_info[:]

    onnx.checker.check_model(current, full_check=True)
    onnx.shape_inference.infer_shapes(current, strict_mode=True)
    stats["task205_where_exclusion"] += 1
    return current, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, stats = transform(onnx.load(args.input))
    if not stats.get("task205_where_exclusion"):
        raise RuntimeError("task205 Where exclusion pattern was not found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(dict(stats))


if __name__ == "__main__":
    main()
