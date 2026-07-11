from __future__ import annotations

from collections import Counter

import onnx
from onnx import TensorProto, helper

import optimize_equivalent as oe


TARGET_CAST_OUTPUTS = {
    "target_red_flat",
    "source_hidden_flat",
    "source_visible_flat",
}


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
    required = TARGET_CAST_OUTPUTS | {
        "target_topk_values",
        "source_hidden_topk_values",
        "source_visible_topk_values",
    }
    if not required.issubset(outputs):
        return current, Counter()

    stats = Counter()
    for node in current.graph.node:
        if node.op_type != "Cast" or not node.output or node.output[0] not in TARGET_CAST_OUTPUTS:
            continue
        for attr in node.attribute:
            if attr.name == "to" and int(attr.i) != TensorProto.UINT8:
                attr.i = TensorProto.UINT8
                stats["mask_cast_u8"] += 1

    if not stats:
        return current, Counter()

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
        ("target_red_flat", TensorProto.UINT8, [225]),
        ("target_topk_values", TensorProto.UINT8, [3]),
        ("target_topk_indices", TensorProto.INT64, [3]),
        ("source_hidden_flat", TensorProto.UINT8, [225]),
        ("source_hidden_topk_values", TensorProto.UINT8, [5]),
        ("source_hidden_topk_indices", TensorProto.INT64, [5]),
        ("source_visible_flat", TensorProto.UINT8, [225]),
        ("source_visible_topk_values", TensorProto.UINT8, [6]),
        ("source_visible_topk_indices", TensorProto.INT64, [6]),
    ):
        _set_value_info(current, name, elem_type, shape)
    onnx.checker.check_model(current, full_check=True)
    onnx.shape_inference.infer_shapes(current, strict_mode=True)
    return current, stats
