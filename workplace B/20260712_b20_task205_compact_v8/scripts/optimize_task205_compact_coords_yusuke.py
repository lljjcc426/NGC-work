from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import onnx
from onnx import helper

import optimize_task205_compact_einsum_yusuke as compact_einsum


def _set_attribute(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [attr for attr in node.attribute if attr.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current, stats = compact_einsum.transform(model)
    if not stats.get("task205_compact_einsum"):
        return current, Counter()

    by_output = {
        output: node
        for node in current.graph.node
        for output in node.output
        if output
    }
    required = {
        "safe_name_44",
        "safe_name_45",
        "safe_name_73",
        "safe_name_74",
        "safe_name_76",
        "safe_name_77",
        "safe_name_79",
        "task205_s60_f16_selected_3d",
        "task205_s61_f16_selected_3d",
        "output",
    }
    if not required.issubset(by_output):
        return onnx.ModelProto.FromString(model.SerializeToString()), Counter()

    _set_attribute(by_output["safe_name_44"], "equation", "nchw,nc,nw->h")
    _set_attribute(by_output["safe_name_45"], "equation", "nchw,nc,nh->w")
    _set_attribute(by_output["task205_s60_f16_selected_3d"], "axis", 0)
    _set_attribute(by_output["task205_s61_f16_selected_3d"], "axis", 0)

    nodes: list[onnx.NodeProto] = []
    for node in current.graph.node:
        output = node.output[0] if node.output else ""
        if output in {"safe_name_74", "safe_name_77"}:
            continue
        copied = onnx.NodeProto.FromString(node.SerializeToString())
        if output == "safe_name_73":
            nodes.append(copied)
            nodes.append(
                helper.make_node(
                    "Unsqueeze",
                    ["safe_name_73"],
                    ["task205_color_nkc"],
                    axes=[0],
                    name="task205_color_nkc",
                )
            )
            continue
        if output == "safe_name_76":
            del copied.input[:]
            copied.input.extend(["safe_name_66", "safe_name_75"])
            _set_attribute(copied, "axis", 0)
        elif output == "safe_name_79":
            del copied.input[:]
            copied.input.extend(["safe_name_67", "safe_name_78"])
            _set_attribute(copied, "axis", 0)
        elif output == "output":
            copied.input[0] = "task205_color_nkc"
            _set_attribute(copied, "equation", "nkc,kh,kw->nchw")
        nodes.append(copied)

    del current.graph.node[:]
    current.graph.node.extend(nodes)
    del current.graph.value_info[:]

    onnx.checker.check_model(current, full_check=True)
    onnx.shape_inference.infer_shapes(current, strict_mode=True)
    stats["task205_compact_coords"] += 1
    return current, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, stats = transform(onnx.load(args.input))
    if not stats.get("task205_compact_coords"):
        raise RuntimeError("task205 compact coordinate pattern was not found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(dict(stats))


if __name__ == "__main__":
    main()
