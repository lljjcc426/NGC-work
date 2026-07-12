from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import optimize_task205_compact_coords_yusuke as compact_coords


def _set_attribute(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [attr for attr in node.attribute if attr.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


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
    current, stats = compact_coords.transform(model)
    if not stats.get("task205_compact_coords"):
        return current, Counter()

    by_output = {
        output: node
        for node in current.graph.node
        for output in node.output
        if output
    }
    required = {
        "safe_name_8",
        "safe_name_9",
        "safe_name_30",
        "safe_name_31",
        "safe_name_32",
        "safe_name_33",
        "safe_name_41",
        "safe_name_42",
        "safe_name_48",
        "safe_name_49",
        "safe_name_50",
        "safe_name_51",
        "safe_name_56",
        "safe_name_57",
        "safe_name_70",
        "task205_s60_f16_source_f16",
        "task205_s61_f16_source_f16",
        "task205_s70_f16",
    }
    if not required.issubset(by_output):
        return onnx.ModelProto.FromString(model.SerializeToString()), Counter()
    if not all(
        (
            _replace_initializer(current, "safe_name_0", np.uint8),
            _replace_initializer(current, "safe_name_1", np.int32),
            _replace_initializer(current, "safe_name_6", np.int32),
        )
    ):
        return onnx.ModelProto.FromString(model.SerializeToString()), Counter()

    for name in ("safe_name_30", "safe_name_31", "safe_name_32", "safe_name_33"):
        _set_attribute(by_output[name], "to", TensorProto.INT32)

    nodes: list[onnx.NodeProto] = []
    remove_outputs = {"safe_name_50", "safe_name_51", "safe_name_56", "safe_name_57"}
    for node in current.graph.node:
        output = node.output[0] if node.output else ""
        if output in remove_outputs:
            continue
        copied = onnx.NodeProto.FromString(node.SerializeToString())
        if output == "safe_name_8":
            nodes.append(copied)
            nodes.append(
                helper.make_node(
                    "Cast",
                    ["safe_name_8"],
                    ["task205_bg_u8"],
                    to=TensorProto.UINT8,
                    name="task205_bg_u8",
                )
            )
            continue
        if output == "safe_name_9":
            copied.input[1] = "task205_bg_u8"
        elif output == "safe_name_41":
            nodes.append(copied)
            nodes.append(
                helper.make_node(
                    "Cast",
                    ["safe_name_41"],
                    ["task205_fg_u8"],
                    to=TensorProto.UINT8,
                    name="task205_fg_u8",
                )
            )
            continue
        elif output == "safe_name_42":
            copied.input[1] = "task205_fg_u8"
        elif output == "safe_name_48":
            copied = helper.make_node(
                "Not", ["safe_name_46"], ["task205_not_46"], name="task205_not_46"
            )
        elif output == "safe_name_49":
            copied = helper.make_node(
                "Not", ["safe_name_47"], ["task205_not_47"], name="task205_not_47"
            )
        elif output == "task205_s60_f16_source_f16":
            copied.input[0] = "task205_not_46"
        elif output == "task205_s61_f16_source_f16":
            copied.input[0] = "task205_not_47"
        elif output == "task205_s60_f16_selected_3d":
            copied.input[1] = "safe_name_54"
        elif output == "task205_s61_f16_selected_3d":
            copied.input[1] = "safe_name_55"
        elif output == "safe_name_70":
            copied = helper.make_node(
                "Cast",
                ["safe_name_10"],
                ["task205_bg_f16"],
                to=TensorProto.FLOAT16,
                name="task205_bg_f16",
            )
        elif output == "task205_s70_f16":
            copied = helper.make_node(
                "Sub",
                ["task205_bg_f16", "task205_s43_f16"],
                ["task205_s70_f16"],
                name="task205_s70_f16",
            )
        nodes.append(copied)

    del current.graph.node[:]
    current.graph.node.extend(nodes)
    del current.graph.value_info[:]

    onnx.checker.check_model(current, full_check=True)
    onnx.shape_inference.infer_shapes(current, strict_mode=True)
    stats["task205_integral_not"] += 1
    return current, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, stats = transform(onnx.load(args.input))
    if not stats.get("task205_integral_not"):
        raise RuntimeError("task205 integral/Not patterns were not found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(dict(stats))


if __name__ == "__main__":
    main()
