from __future__ import annotations

from collections import Counter

import onnx
from onnx import TensorProto, helper

import optimize_equivalent as oe


def _attr_i(node: onnx.NodeProto, name: str) -> int | None:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return None


def _is_cast_to(node: onnx.NodeProto, output: str, dtype: int) -> bool:
    return (
        node.op_type == "Cast"
        and len(node.input) == 1
        and len(node.output) == 1
        and node.output[0] == output
        and _attr_i(node, "to") == dtype
    )


def _set_value_info(
    model: onnx.ModelProto,
    name: str,
    elem_type: int,
    shape: list[int],
) -> None:
    matches = [item for item in model.graph.value_info if item.name == name]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate value_info: {name}")
    if matches:
        item = matches[0]
        item.type.tensor_type.elem_type = elem_type
        item.type.tensor_type.shape.ClearField("dim")
        for size in shape:
            item.type.tensor_type.shape.dim.add().dim_value = size
        return
    model.graph.value_info.append(helper.make_tensor_value_info(name, elem_type, shape))


def transform(model: onnx.ModelProto) -> tuple[onnx.ModelProto, Counter]:
    current = onnx.ModelProto.FromString(model.SerializeToString())
    outputs = {out for node in current.graph.node for out in node.output if out}
    initializers = {init.name for init in current.graph.initializer}
    required_outputs = {"g", "gf", "tv", "c", "score", "scoref", "sv", "mflat", "mf16", "mv"}
    if not required_outputs.issubset(outputs) or "u80" not in initializers:
        return current, Counter()

    stats = Counter()
    new_nodes: list[onnx.NodeProto] = []
    for node in current.graph.node:
        first_out = node.output[0] if node.output else ""
        if _is_cast_to(node, "gf", TensorProto.FLOAT16):
            stats["drop_g_to_f16"] += 1
            continue
        if _is_cast_to(node, "c", TensorProto.UINT8) and node.input[0] == "tv":
            stats["drop_tv_to_c"] += 1
            continue
        if _is_cast_to(node, "scoref", TensorProto.FLOAT16):
            stats["drop_score_to_f16"] += 1
            continue
        if _is_cast_to(node, "mf16", TensorProto.FLOAT16):
            stats["drop_mflat_to_f16"] += 1
            continue

        copied = onnx.NodeProto.FromString(node.SerializeToString())
        if copied.op_type == "TopK" and copied.input and copied.input[0] == "gf":
            copied.input[0] = "g"
            copied.output[0] = "c"
            stats["topk_g_u8"] += 1
        elif copied.op_type == "TopK" and copied.input and copied.input[0] == "scoref":
            copied.input[0] = "score"
            stats["topk_score_u8"] += 1
        elif copied.op_type == "TopK" and copied.input and copied.input[0] == "mf16":
            copied.input[0] = "mflat"
            stats["topk_mflat_u8"] += 1
        elif copied.op_type == "Greater" and copied.input and copied.input[0] in {"sv", "mv"}:
            copied.input[1] = "u80"
            stats["greater_u8_zero"] += 1

        # The original Cast tv->c is removed, so any direct use of tv must read c.
        for index, name in enumerate(copied.input):
            if name == "tv":
                copied.input[index] = "c"
                stats["rewire_tv_to_c"] += 1
        new_nodes.append(copied)

    if not stats:
        return current, Counter()

    del current.graph.node[:]
    current.graph.node.extend(new_nodes)
    stats["dead_nodes"] += oe.prune_dead(current)
    stats["initializers"] += oe.prune_initializers(current)

    live_outputs = {
        output
        for node in current.graph.node
        for output in node.output
        if output
    }
    kept_value_info = [
        onnx.ValueInfoProto.FromString(item.SerializeToString())
        for item in current.graph.value_info
        if item.name in live_outputs
    ]
    del current.graph.value_info[:]
    current.graph.value_info.extend(kept_value_info)
    for name, elem_type, shape in (
        ("c", TensorProto.UINT8, [31]),
        ("ti", TensorProto.INT64, [31]),
        ("sv", TensorProto.UINT8, [3]),
        ("si", TensorProto.INT64, [3]),
        ("mv", TensorProto.UINT8, [3, 9]),
        ("mi", TensorProto.INT64, [3, 9]),
    ):
        _set_value_info(current, name, elem_type, shape)
    onnx.checker.check_model(current, full_check=True)
    onnx.shape_inference.infer_shapes(current, strict_mode=True)
    return current, stats
