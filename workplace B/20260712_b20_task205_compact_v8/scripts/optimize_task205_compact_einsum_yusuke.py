from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import onnx

import optimize_task205_broadcast_gather_yusuke as broadcast_gather


def _set_attribute(node: onnx.NodeProto, name: str, value: object) -> None:
    kept = [attr for attr in node.attribute if attr.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(onnx.helper.make_attribute(name, value))


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current, stats = broadcast_gather.transform(model)
    if not stats:
        return current, Counter()

    by_output = {
        output: node
        for node in current.graph.node
        for output in node.output
        if output
    }
    required = {
        "safe_name_41",
        "safe_name_42",
        "safe_name_44",
        "safe_name_45",
        "safe_name_70",
        "safe_name_71",
        "safe_name_72",
        "safe_name_73",
        "output",
    }
    if not required.issubset(by_output):
        return onnx.ModelProto.FromString(model.SerializeToString()), Counter()

    argmax = by_output["safe_name_41"]
    if argmax.op_type != "ArgMax":
        return onnx.ModelProto.FromString(model.SerializeToString()), Counter()
    _set_attribute(argmax, "keepdims", 1)

    _set_attribute(by_output["safe_name_44"], "equation", "nchw,nc,nw->nh")
    _set_attribute(by_output["safe_name_45"], "equation", "nchw,nc,nh->nw")

    remove_outputs = {"safe_name_71", "safe_name_72"}
    nodes: list[onnx.NodeProto] = []
    for node in current.graph.node:
        output = node.output[0] if node.output else ""
        if output in remove_outputs:
            continue
        copied = onnx.NodeProto.FromString(node.SerializeToString())
        if output == "safe_name_73":
            del copied.input[:]
            copied.input.extend(["task205_s43_f16", "task205_s70_f16"])
            _set_attribute(copied, "axis", 0)
        elif output == "output":
            _set_attribute(copied, "equation", "kc,nkh,nkw->nchw")
        nodes.append(copied)

    del current.graph.node[:]
    current.graph.node.extend(nodes)
    del current.graph.value_info[:]

    onnx.checker.check_model(current, full_check=True)
    onnx.shape_inference.infer_shapes(current, strict_mode=True)
    stats["task205_compact_einsum"] += 1
    return current, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, stats = transform(onnx.load(args.input))
    if not stats.get("task205_compact_einsum"):
        raise RuntimeError("task205 compact Einsum pattern was not found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    print(dict(stats))


if __name__ == "__main__":
    main()
