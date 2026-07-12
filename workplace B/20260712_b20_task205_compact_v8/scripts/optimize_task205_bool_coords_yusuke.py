from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import onnx
from onnx import TensorProto, helper

import optimize_task205_where_exclusion_yusuke as where_exclusion


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current, stats = where_exclusion.transform(model)
    if not stats.get("task205_where_exclusion"):
        return current, Counter()

    by_output = {
        output: node
        for node in current.graph.node
        for output in node.output
        if output
    }
    required = {
        "task205_s60_f16_source_f16",
        "task205_s61_f16_source_f16",
        "task205_s60_f16_selected_3d",
        "task205_s61_f16_selected_3d",
        "safe_name_66",
        "safe_name_67",
        "safe_name_75",
        "safe_name_76",
        "safe_name_78",
        "safe_name_79",
        "output",
    }
    if not required.issubset(by_output):
        return onnx.ModelProto.FromString(model.SerializeToString()), Counter()

    nodes: list[onnx.NodeProto] = []
    remove_outputs = {
        "task205_s60_f16_source_f16",
        "task205_s61_f16_source_f16",
        "safe_name_66",
        "safe_name_67",
    }
    for node in current.graph.node:
        output = node.output[0] if node.output else ""
        if output in remove_outputs:
            continue
        copied = onnx.NodeProto.FromString(node.SerializeToString())
        if output == "task205_s60_f16_selected_3d":
            copied.input[0] = "task205_not_46"
        elif output == "task205_s61_f16_selected_3d":
            copied.input[0] = "task205_not_47"
        elif output == "safe_name_75":
            copied = helper.make_node(
                "And",
                ["safe_name_64", "task205_s60_f16_selected_3d"],
                ["safe_name_75"],
                name="safe_name_75",
            )
        elif output == "safe_name_76":
            copied.input[0] = "safe_name_64"
            nodes.append(copied)
            nodes.append(
                helper.make_node(
                    "Cast",
                    ["safe_name_76"],
                    ["task205_rows_kh"],
                    to=TensorProto.FLOAT16,
                    name="task205_rows_kh",
                )
            )
            continue
        elif output == "safe_name_78":
            copied = helper.make_node(
                "And",
                ["safe_name_65", "task205_s61_f16_selected_3d"],
                ["safe_name_78"],
                name="safe_name_78",
            )
        elif output == "safe_name_79":
            copied.input[0] = "safe_name_65"
            nodes.append(copied)
            nodes.append(
                helper.make_node(
                    "Cast",
                    ["safe_name_79"],
                    ["task205_cols_kw"],
                    to=TensorProto.FLOAT16,
                    name="task205_cols_kw",
                )
            )
            continue
        elif output == "output":
            copied.input[1] = "task205_rows_kh"
            copied.input[2] = "task205_cols_kw"
        nodes.append(copied)

    del current.graph.node[:]
    current.graph.node.extend(nodes)
    del current.graph.value_info[:]

    onnx.checker.check_model(current, full_check=True)
    onnx.shape_inference.infer_shapes(current, strict_mode=True)
    stats["task205_bool_coords"] += 1
    return current, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, stats = transform(onnx.load(args.input))
    if not stats.get("task205_bool_coords"):
        raise RuntimeError("task205 boolean coordinate pattern was not found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(dict(stats))


if __name__ == "__main__":
    main()
